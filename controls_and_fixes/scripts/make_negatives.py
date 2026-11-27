"""Candidate 1: error-taxonomy negatives (step level). GSM8K training data only, no sampling, no teacher.
For each correct training solution, pick solution steps of the form 'a OP b = c' (arithmetic verified) and build
minimal counterfactual steps that differ by exactly ONE typed error, matching the measured failure types:
  binding    : one operand replaced by another quantity from the question or an earlier result (result recomputed,
               so the arithmetic stays consistent and the only error is the choice of quantity)
  relation   : operator inverted (+ <-> -, * <-> /), result recomputed (only error is the relation direction)
  arithmetic : expression unchanged, result corrupted (only error is the calculation)
Control ('mismatch'): the rejected step is a correct step taken from a DIFFERENT problem (a generic negative).
Output JSONL rows: {messages (system,user), prefix (assistant text before the step), chosen, rejected, type}.
Usage: python make_negatives.py TRAIN_FILE OUT_TYPED OUT_MISMATCH [max_pairs_per_example=3] [seed=0]
"""
import json, random, re, sys

tf, out_t, out_m = sys.argv[1], sys.argv[2], sys.argv[3]
MAXP = int(sys.argv[4]) if len(sys.argv) > 4 else 3
rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
NUM = r"\d[\d,]*(?:\.\d+)?"
EQ = re.compile(r"(?<![\d.])(\$?)(" + NUM + r")\s*([-+*/x×÷])\s*(\$?)(" + NUM + r")\s*=\s*(\$?)(" + NUM + r")(?![\d])")
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "x": lambda a, b: a * b,
       "×": lambda a, b: a * b, "/": lambda a, b: a / b if b else None, "÷": lambda a, b: a / b if b else None}
INV = {"+": "-", "-": "+", "*": "/", "x": "/", "×": "÷", "/": "*", "÷": "×"}


def val(s): return float(s.replace(",", ""))


def fmt(v):
    if v is None or v != v or abs(v) > 1e9: return None
    if abs(v - round(v)) < 1e-9: return str(int(round(v)))
    r = round(v, 2)
    return None if abs(r - v) > 1e-9 else f"{r:g}"          # only exact (<= 2 decimals) results look like real steps


def nums(text): return [val(m) for m in re.findall(NUM, text.replace("$", " "))]


def split_solution(content):
    """Return (head, steps, tail): head = text up to and incl. 'Solution:\n' (or ''), steps = solution lines, tail = '#### ...'."""
    head = ""
    if "Solution:\n" in content:
        i = content.index("Solution:\n") + len("Solution:\n"); head, content = content[:i], content[i:]
    body, _, ans = content.rpartition("\n#### ")
    return head, body.split("\n"), "#### " + ans


def corrupt(line, m, kind, pool):
    """Change only the affected tokens in place (original spacing and symbols kept), so chosen and rejected differ
    only in the error itself."""
    a, op, b, c = val(m.group(2)), m.group(3), val(m.group(5)), val(m.group(7))
    sub = {}                                                   # group index -> replacement text
    if kind == "relation":
        op2 = INV[op]; c2 = fmt(OPS[op2](a, b))
        if c2 is None or val(c2) == c or val(c2) < 0: return None
        sub = {3: op2, 7: c2}
    elif kind == "binding":
        side = rng.choice([2, 5]); cands = [v for v in pool if v not in (a, b) and v > 0]
        if not cands: return None
        v = rng.choice(cands); vs = fmt(v)
        a2, b2 = (v, b) if side == 2 else (a, v)
        c2 = fmt(OPS[op](a2, b2))
        if vs is None or c2 is None or val(c2) == c or val(c2) < 0: return None
        sub = {side: vs, 7: c2}
    else:  # arithmetic
        for _ in range(6):
            c2 = fmt(rng.choice([c + rng.choice([1, 2, 3, 10]), c - rng.choice([1, 2, 3, 10]), c * 10, c / 10 if c >= 10 else c + 5]))
            if c2 is not None and val(c2) != c and val(c2) >= 0: break
        else: return None
        sub = {7: c2}
    out = line
    for g in sorted(sub, key=lambda g: m.start(g), reverse=True):
        out = out[:m.start(g)] + sub[g] + out[m.end(g):]
    return out


rows = [json.loads(l) for l in open(tf)]
all_steps = []                                               # pool of correct steps for the mismatch control
typed, counts = [], {"binding": 0, "relation": 0, "arithmetic": 0}
for r in rows:
    msgs = r["messages"]; head, steps, tail = split_solution(msgs[2]["content"])
    q = msgs[1]["content"]; cands = []
    for k, line in enumerate(steps):
        for m in EQ.finditer(line):
            a, op, b, c = val(m.group(2)), m.group(3), val(m.group(5)), val(m.group(7))
            res = OPS[op](a, b)
            if res is None or abs(res - c) > 1e-6 * max(1, abs(c)): continue        # only verified steps
            pool = set(nums(q)) | {v for s in steps[:k] for v in nums(s)}
            cands.append((k, m, pool)); break                                         # first equation of the line
        if line.strip(): all_steps.append(line)
    rng.shuffle(cands); made = 0; used = set()
    for kind in rng.sample(["binding", "relation", "arithmetic"], 3):
        for k, m, pool in cands:
            if (k, kind) in used: continue
            bad = corrupt(steps[k], m, kind, pool)
            if bad and bad != steps[k]:
                prefix = head + "".join(s + "\n" for s in steps[:k])
                typed.append({"messages": msgs[:2], "prefix": prefix, "chosen": steps[k] + "\n", "rejected": bad + "\n", "type": kind})
                counts[kind] += 1; used.add((k, kind)); made += 1; break
        if made >= MAXP: break
rng.shuffle(typed)
with open(out_t, "w") as f:
    for p in typed: f.write(json.dumps(p) + "\n")
with open(out_m, "w") as f:                                  # same prefixes/chosen, generic negative
    for p in typed:
        while True:
            s = rng.choice(all_steps)
            if s.strip() != p["chosen"].strip(): break
        f.write(json.dumps(dict(p, rejected=s + "\n", type="mismatch")) + "\n")
print(f"NEG {tf}: {len(rows)} solutions -> {len(typed)} typed pairs {counts}; mismatch control {len(typed)} pairs")
