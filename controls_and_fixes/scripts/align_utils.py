"""Option 1: operand-attention supervision utilities.

For each equation operand in a training solution we find its source:
  - a number that appears exactly once in the question  -> target = that number's span in the question
  - the result of an earlier equation in the solution    -> target = that result's span (most recent)
  - otherwise (constant, number word, ambiguous)         -> no target
The query is the token just before the operand's first token (where the model decides what number to write next).
Loss (per supervised head, per aligned operand):  -log( sum of attention on the target span ).
Random-span control: same queries, target replaced by a random other number span (different value) that is earlier.
"""
import random, re

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
TAIL = re.compile(r"[$(]*\s*\.?\d[\d\s.,$%()+\-*/x×÷]*$")      # contiguous arithmetic expression just before '='
OPER = re.compile(r"\d\s*%?\)*\s*[-+*/x×÷]\s*[$(]*\s*\.?\d")      # tail must contain an operator between numbers


def _val(s):
    try:
        return round(float(s.replace(",", "")), 6)
    except ValueError:
        return None


def _numbers(text, offset=0):
    out = []
    for m in NUM.finditer(text):
        s, e = m.start(), m.end()
        while e > s and text[e - 1] in ",.": e -= 1          # '5,' or '5.' at clause end
        v = _val(text[s:e])
        if v is not None: out.append((v, offset + s, offset + e))
    return out


def char_alignments(question, solution, random_ctrl=False, rng=None):
    """Return [(op_start, op_end, tgt_start, tgt_end, where)] with op_* in solution chars, tgt_* in question ('q')
    or solution ('a') chars."""
    qn = _numbers(question); qcount = {}
    for v, _, _ in qn: qcount[v] = qcount.get(v, 0) + 1
    prev = []                                                  # (value, start, end) of earlier results, in order
    pairs = []
    pos = 0
    for line in solution.split("\n"):
        for m in re.finditer("=", line):
            head = line[:m.start()]
            t = TAIL.search(head)
            if t and OPER.search(t.group(0)):
                for v, s, e in _numbers(t.group(0), pos + t.start()):
                    if qcount.get(v, 0) == 1:
                        _, qs, qe = next(x for x in qn if x[0] == v); tgt = (qs, qe, "q")
                    else:
                        hits = [x for x in prev if x[0] == v]
                        if not hits: continue
                        tgt = (hits[-1][1], hits[-1][2], "a")
                    if random_ctrl:
                        cands = [(x[1], x[2], "q") for x in qn if x[0] != v] + [(x[1], x[2], "a") for x in prev if x[0] != v and x[2] <= s]
                        if not cands: continue
                        tgt = (rng or random).choice(cands)
                    pairs.append((s, e) + tgt)
            r = re.match(r"\s*\$?\s*(" + NUM.pattern + ")", line[m.end():])
            if r:
                rs = pos + m.end() + r.start(1); rv = _numbers(r.group(1), rs)
                if rv: prev.append(rv[0])
        pos += len(line) + 1
    return pairs


def token_alignments(tokenizer, formatted, question, solution, max_length, add_special_tokens, K=32,
                     random_ctrl=False, rng=None):
    """Map char alignments to token indices of tokenizer(formatted, ...). Returns three length-K lists
    (query index, target start, target end exclusive), padded with -1."""
    qi = formatted.find(question); ai = formatted.find(solution, qi + len(question) if qi >= 0 else 0)
    q, s_, e_ = [], [], []
    if qi >= 0 and ai >= 0:
        enc = tokenizer(formatted, truncation=True, max_length=max_length, return_offsets_mapping=True,
                        add_special_tokens=add_special_tokens)
        offs = [(a, b) for a, b in enc["offset_mapping"]]
        starts = [a if b > a else -1 for a, b in offs]

        def tok_at(c):                                         # token containing char c
            for i, (a, b) in enumerate(offs):
                if b > a and a <= c < b: return i
            return None

        def toks_in(cs, ce):
            return [i for i, (a, b) in enumerate(offs) if b > a and a < ce and b > cs]

        for os_, oe, ts, te, where in char_alignments(question, solution, random_ctrl, rng):
            t = tok_at(ai + os_)
            if t is None or t < 1: continue
            base = qi if where == "q" else ai
            tt = toks_in(base + ts, base + te)
            if not tt or tt[-1] >= t: continue                # target must lie strictly before the operand
            q.append(t - 1); s_.append(tt[0]); e_.append(tt[-1] + 1)
            if len(q) == K: break
    pad = K - len(q)
    return q + [-1] * pad, s_ + [-1] * pad, e_ + [-1] * pad


def align_loss(attentions, heads, al_q, al_s, al_e):
    """attentions: tuple of (B, H, T, T) per layer (eager attention). heads: [(layer, head)].
    al_*: (B, K) long tensors, -1 = unused. Returns (loss, mean attention mass on target)."""
    import torch
    valid = al_q >= 0                                          # (B, K)
    if valid.sum() == 0:
        z = attentions[0].new_zeros(()).float(); return z, z
    B, K = al_q.shape; T = attentions[0].shape[-1]
    ar = torch.arange(T, device=al_q.device)
    mask = (ar[None, None, :] >= al_s[..., None]) & (ar[None, None, :] < al_e[..., None])    # (B, K, T)
    qidx = al_q.clamp(min=0)
    losses, masses = [], []
    for layer, head in heads:
        A = attentions[layer][:, head].float()                 # (B, T, T)
        rows = torch.gather(A, 1, qidx[..., None].expand(B, K, T))                            # (B, K, T)
        mass = (rows * mask).sum(-1).clamp(min=1e-6)           # (B, K)
        losses.append(-torch.log(mass)[valid]); masses.append(mass[valid])
    return torch.cat(losses).mean(), torch.cat(masses).mean().detach()
