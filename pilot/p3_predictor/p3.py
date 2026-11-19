"""Pilot P3: one-hop predictor (Phase 1 only, T = 1), size sweep, frozen backbone.

Stages
  dec   train the T = 1 oracle decoder on target states (the probe and the ceiling)
  pred  train a one-hop predictor z0 -> z*_1, score it through the frozen oracle decoder,
        and time one predictor step at batch size 1
Sizes   s = 4 layers, width 768 (about 38M in the layers)
        m = 4 layers, width 1536 (about 150M in the layers)
        top4 = trainable copy of the backbone's top four layers (about 0.24B)
"""
import argparse, copy, math, random, time
from fractions import Fraction
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import parse_gsm
from parse_gsm import validate, group_texts, safe_eval, EQ

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["dec", "pred", "ddec"])
ap.add_argument("--src", default="pred", choices=["pred", "copy"])   # ddec: states the decoder is fine-tuned on
ap.add_argument("--size", default="s", choices=["s", "m", "top4"])
ap.add_argument("--ntrain", type=int, default=150000)
ap.add_argument("--nval", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=0.0)        # 0 = the default for the stage and size
ap.add_argument("--k", type=float, default=4.0)         # scaled-cosine factor (proposal eq. 3)
ap.add_argument("--seed", type=int, default=82)
ap.add_argument("--dec_ckpt", default="p3_dec_T1.pt")
ap.add_argument("--nlat", type=int, default=220)        # latency questions; the first 20 are warm-up
ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
args = ap.parse_args()
T, R = 1, 2                                             # one hop; decoder tasks: step text, answer
random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"
if AMP:
    torch.backends.cuda.matmul.allow_tf32 = True

def amp():
    return torch.autocast(DEV, dtype=torch.bfloat16, enabled=AMP)

tok = AutoTokenizer.from_pretrained(args.model)
llm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if AMP else torch.float32).to(DEV).eval()
for p in llm.parameters():
    p.requires_grad_(False)
enc = llm.model
NLAYERS = len(enc.layers)
E = llm.get_input_embeddings().weight.detach()
DM = E.shape[1]
EF = E.float()                                          # fp32 copy for the output logits
NLID = tok("\n", add_special_tokens=False).input_ids[0]
BOS, EOS = tok.bos_token_id, tok.eos_token_id

def ids_of(s):
    return tok(s, add_special_tokens=False).input_ids

# ---------------------------------------------------------------- data
t0 = time.time()
valid = []
for q, c, a in parse_gsm.load_split():
    ex, _ = validate(q, c, a)
    if ex:
        ex["answer"] = a.replace(",", "").strip()
        valid.append(ex)
random.Random(args.seed).shuffle(valid)
val_all, val_one, train = [], [], []
for ex in valid:
    chain = group_texts(ex, T)[0]                       # T = 1: the single step is the whole chain
    tg = [ids_of(chain) + [EOS], ids_of(ex["answer"]) + [EOS]]
    if max(len(t) for t in tg) > 120:
        continue
    ex["chain"], ex["tgt"] = chain, tg
    ex["seg"] = [ids_of(ex["question"]), ids_of(chain)]
    if len(val_all) < args.nval:
        val_all.append(ex)
    elif ex["n"] == 1 and len(val_one) < args.nval:
        val_one.append(ex)
    elif len(train) < args.ntrain:
        train.append(ex)
    if len(train) >= args.ntrain and len(val_one) >= args.nval:
        break
n1 = sum(e["n"] == 1 for e in train)
print(f"data ready in {time.time() - t0:.0f}s: train {len(train)} (one-step {n1}), "
      f"val_all {len(val_all)}, val_one {len(val_one)}", flush=True)

@torch.no_grad()
def encode_batch(exs, want_mid=False, question_only=False):
    """One frozen pass. z: (B, 1 or 2, DM) = z0[, z*_1]; hx: prompt states; mid: input to the top four layers."""
    seqs, poss = [], []
    for ex in exs:
        ids, pos = [BOS], []
        for s in (ex["seg"][:1] if question_only else ex["seg"]):
            ids += s + [NLID]
            pos.append(len(ids) - 1)
        seqs.append(ids); poss.append(pos)
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), EOS, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    o = enc(input_ids=x.to(DEV), attention_mask=m.to(DEV), use_cache=False, output_hidden_states=want_mid)
    h = o.last_hidden_state
    z = torch.stack([F.normalize(h[i, poss[i]].float(), dim=-1) for i in range(len(seqs))])
    lens = torch.tensor([p[0] + 1 for p in poss], device=DEV)
    Lh = int(lens.max())
    hx = h[:, :Lh].float()
    hmask = torch.arange(Lh, device=DEV)[None, :] < lens[:, None]
    mid = o.hidden_states[NLAYERS - 4][:, :Lh].float() if want_mid else None
    return z, hx, hmask, lens, mid

# ---------------------------------------------------------------- modules
class Dec(nn.Module):
    """Same decoder as P2 run 4: 2 layers, width 768, tied frozen embeddings."""
    def __init__(self, E, R, d=768, heads=12, layers=2, drop=0.1):
        super().__init__()
        self.register_buffer("E", E, persistent=False)
        dm = E.shape[1]
        self.R = R
        self.tok_in, self.z_in, self.hx_in = nn.Linear(dm, d), nn.Linear(dm, d), nn.Linear(dm, d)
        self.task = nn.Embedding(R, d)
        self.pos = nn.Embedding(128, d)
        nn.init.normal_(self.task.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)
        self.zs, self.es = dm ** 0.5, 1.0 / float(E.float().std())
        self.hx_norm, self.z_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.hx_pre = nn.LayerNorm(dm)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(layer, layers)
        self.out_norm, self.out = nn.LayerNorm(d), nn.Linear(d, dm)

    def memory(self, zc, task, hx, hmask, rep=None):
        rep = self.R if rep is None else rep
        zt = self.z_norm(self.z_in(zc * self.zs) + self.task(task))
        h = self.hx_norm(self.hx_in(self.hx_pre(hx))).repeat_interleave(rep, 0)
        pad = ~hmask.repeat_interleave(rep, 0)
        mem = torch.cat([zt[:, None, :], h], 1)
        pad = torch.cat([torch.zeros(len(zt), 1, dtype=torch.bool, device=zt.device), pad], 1)
        return mem, pad

    def hidden(self, inp, mem, pad):
        N, L = inp.shape
        x = self.tok_in(self.E[inp].float() * self.es) + self.pos(torch.arange(L, device=inp.device))[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=inp.device), 1)
        h = self.dec(x, mem, tgt_mask=causal, memory_key_padding_mask=pad)
        return self.out(self.out_norm(h))

class Pred(nn.Module):
    """Proposal 3.2: 4-layer Transformer, cross-attention over H_X, self-attention over the latent
    history, k = 2 learned predictor tokens; new state = mean of their outputs, projected, L2-normalised;
    direct (non-residual) update."""
    def __init__(self, dm, d, layers=4, nq=2, drop=0.1, max_hist=8):
        super().__init__()
        self.nq = nq
        self.z_in, self.hx_in = nn.Linear(dm, d), nn.Linear(dm, d)
        self.hx_pre = nn.LayerNorm(dm)
        self.hx_norm, self.z_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.step = nn.Embedding(max_hist, d)
        nn.init.normal_(self.step.weight, std=0.02)
        self.zs = dm ** 0.5
        self.q = nn.Parameter(torch.randn(nq, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, d // 64, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.core = nn.TransformerDecoder(layer, layers)
        self.out_norm, self.out = nn.LayerNorm(d), nn.Linear(d, dm)

    def forward(self, zhist, hx, hmask):
        B, t, _ = zhist.shape
        zt = self.z_norm(self.z_in(zhist * self.zs) + self.step(torch.arange(t, device=zhist.device))[None])
        x = torch.cat([zt, self.q[None].expand(B, -1, -1).to(zt.dtype)], 1)
        mem = self.hx_norm(self.hx_in(self.hx_pre(hx)))
        h = self.core(x, mem, memory_key_padding_mask=~hmask)
        y = self.out(self.out_norm(h[:, -self.nq:].mean(1)))
        return F.normalize(y.float(), dim=-1)

class Top4(nn.Module):
    """Trainable copy of the backbone's top four layers and final norm. Input: the frozen backbone's
    hidden states entering those layers, question tokens only. At initialisation the output equals z0."""
    def __init__(self, llm):
        super().__init__()
        cfg = copy.deepcopy(llm.config)
        cfg.num_hidden_layers = 4
        self.m = type(llm.model)(cfg)
        self.m.layers.load_state_dict(llm.model.layers[-4:].state_dict())
        self.m.norm.load_state_dict(llm.model.norm.state_dict())
        self.m.embed_tokens = None
        self.m.float()

    def forward(self, mid, hmask, lens):
        h = self.m(inputs_embeds=mid, attention_mask=hmask.long(), use_cache=False).last_hidden_state
        y = h[torch.arange(len(lens), device=h.device), lens - 1]
        return F.normalize(y.float(), dim=-1)

def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6

dec = Dec(E, R).to(DEV)
print(f"decoder trainable parameters: {nparams(dec):.1f}M", flush=True)

# ---------------------------------------------------------------- decoding and scoring
@torch.no_grad()
def greedy(zc, task, hx, hmask, rep=None, max_new=100):
    mem, pad = dec.memory(zc, task, hx, hmask, rep)
    out = torch.full((zc.shape[0], 1), BOS, device=DEV)
    done = torch.zeros(zc.shape[0], dtype=torch.bool, device=DEV)
    for _ in range(max_new):
        h = dec.hidden(out, mem, pad)[:, -1]
        nxt = (h.float() @ EF.T).argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
        out = torch.cat([out, nxt[:, None]], 1)
        done |= nxt == EOS
        if bool(done.all()):
            break
    res = []
    for row in out[:, 1:].tolist():
        res.append(row[:row.index(EOS)] if EOS in row else row)
    return res

def last_val(text):
    eqs = EQ.findall(text)
    if not eqs:
        return None
    try:
        return safe_eval(eqs[-1].rsplit("=", 1)[1])
    except Exception:
        return None

def num(s):
    try:
        return Fraction(s.replace(",", "").replace("$", "").strip())
    except Exception:
        return None

def fmt(a):
    return "[" + ", ".join(f"{x:.3f}" for x in a) + "]"

pred = None
WANT_MID = False

def run_pred(z, hx, hm, lens, mid):
    with amp():
        if args.size == "top4":
            return pred(mid, hm, lens)
        return pred(z[:, :1], hx, hm)

@torch.no_grad()
def evaluate(exs, cond, chunk=64, show=0):
    """Decode the step text and the answer from a state chosen by cond.
    Returns per-problem arrays: step numeric match, answer match, cos(state, z*_1)."""
    dec.eval()
    if pred is not None:
        pred.eval()
    step_ok, ans_ok, cs = np.zeros(len(exs)), np.zeros(len(exs)), np.zeros(len(exs))
    for s in range(0, len(exs), chunk):
        part = exs[s:s + chunk]
        z, hx, hm, lens, mid = encode_batch(part, WANT_MID and cond.startswith("pred"))
        B = z.shape[0]
        roll = torch.roll(torch.arange(B, device=DEV), 1)
        base = cond.replace("_shuf", "")
        if base == "oracle":
            c = z[:, 1]
        elif base == "copy_z0":
            c = z[:, 0]
        elif base == "shuffled":
            c = z[:, 1][roll]
        else:
            c = run_pred(z, hx, hm, lens, mid)
        if cond.endswith("_shuf"):
            c = c[roll]
        cs[s:s + B] = (c * z[:, 1]).sum(-1).float().cpu().numpy()
        zc = c[:, None, :].expand(B, R, -1).reshape(B * R, -1)
        task = torch.arange(R, device=DEV).repeat(B)
        with amp():
            outs = greedy(zc, task, hx, hm)
        for i, ex in enumerate(part):
            text = tok.decode(outs[i * R], skip_special_tokens=True).strip()
            ref, got = last_val(ex["chain"]), last_val(text)
            step_ok[s + i] = float(got is not None and got == ref)
            g, a = num(tok.decode(outs[i * R + 1], skip_special_tokens=True)), num(ex["answer"])
            ans_ok[s + i] = float(g is not None and g == a)
            if show and s == 0 and i < show:
                print("   ref:", ex["chain"], "| got:", text, flush=True)
    return step_ok, ans_ok, cs

def lr_at(s, total, peak):
    warm = min(1000, max(1, total // 10))
    if s < warm:
        return peak * (s + 1) / warm
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

steps_per_epoch = len(train) // args.bs
total = steps_per_epoch * args.epochs

# ================================================================ stage: oracle decoder
if args.stage == "dec":
    peak = args.lr or 1e-4
    opt = torch.optim.AdamW(dec.parameters(), lr=peak, weight_decay=0.01, betas=(0.9, 0.98))
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        dec.train()
        random.shuffle(train)
        for b in range(steps_per_epoch):
            exs = train[b * args.bs:(b + 1) * args.bs]
            z, hx, hm, lens, _ = encode_batch(exs)
            B = z.shape[0]
            zc = z[:, 1][:, None, :].expand(B, R, -1).reshape(B * R, -1)
            task = torch.arange(R, device=DEV).repeat(B)
            tg = [ex["tgt"][k] for ex in exs for k in range(R)]
            N, L = len(tg), max(len(t) for t in tg)
            inp = torch.zeros(N, L, dtype=torch.long); lab = torch.full((N, L), -100)
            for i, t in enumerate(tg):
                inp[i, 0] = BOS
                inp[i, 1:len(t)] = torch.tensor(t[:-1])
                lab[i, :len(t)] = torch.tensor(t)
            inp, lab = inp.to(DEV), lab.to(DEV)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, peak)
            with amp():
                mem, pad = dec.memory(zc, task, hx, hm)
                hs = dec.hidden(inp, mem, pad)[lab != -100]
            logits = hs.float() @ EF.T
            loss = F.cross_entropy(logits, lab[lab != -100])
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} gn {gn.item():.2f} "
                      f"lr {lr_at(step, total, peak):.2e} {time.time() - t0:.0f}s", flush=True)
        so, ao, _ = evaluate(val_one[:500], "oracle")
        sa, aa, _ = evaluate(val_all[:500], "oracle")
        print(f"== epoch {ep + 1}: oracle step acc one-step {so.mean():.3f} | all {sa.mean():.3f} "
              f"| answer one-step {ao.mean():.3f} | all {aa.mean():.3f}", flush=True)
        torch.save(dec.state_dict(), args.dec_ckpt)
    print("\nFINAL EVALUATION, T = 1 oracle decoder", flush=True)
    for name, vs in (("one-step (n = 1)", val_one), ("all chain lengths", val_all)):
        print(f"{name}: {len(vs)} held-out problems")
        for cond in ("oracle", "copy_z0", "shuffled"):
            so, ao, _ = evaluate(vs, cond, show=3 if (cond == "oracle" and vs is val_one) else 0)
            print(f"  {cond:9s} step numeric match {so.mean():.3f} | answer acc {ao.mean():.3f}", flush=True)
            if cond == "oracle" and vs is val_all:
                for lo, hi, lab_ in ((1, 1, "n=1"), (2, 2, "n=2"), (3, 3, "n=3"), (4, 99, "n>=4")):
                    idx = [i for i, e in enumerate(vs) if lo <= e["n"] <= hi]
                    if idx:
                        print(f"     {lab_:5s} n={len(idx):4d} step {so[idx].mean():.3f} answer {ao[idx].mean():.3f}")
    print("saved", args.dec_ckpt)
    raise SystemExit

# ================================================================ stage: one-hop predictor
dec.load_state_dict(torch.load(args.dec_ckpt, map_location=DEV))
dec.eval()
for p in dec.parameters():
    p.requires_grad_(False)
if args.size == "top4":
    pred = Top4(llm).to(DEV)
    WANT_MID = True
else:
    pred = Pred(DM, 768 if args.size == "s" else 1536).to(DEV)
core = nparams(pred.core) if args.size != "top4" else nparams(pred.m.layers)
print(f"predictor size {args.size}: {nparams(pred):.1f}M trainable ({core:.1f}M in the Transformer layers)", flush=True)

# ---------------------------------------------------------------- stage: deployed decoder
if args.stage == "ddec":
    # Fine-tune a copy of the oracle decoder on states from a frozen source: the trained predictor
    # (--src pred) or the question state z0 (--src copy, the decoder-only control).
    if args.src == "pred":
        pred.load_state_dict(torch.load(f"p3_pred_{args.size}.pt", map_location=DEV))
    pred.eval()
    for p in pred.parameters():
        p.requires_grad_(False)
    for p in dec.parameters():
        p.requires_grad_(True)
    peak = args.lr or 5e-5
    opt = torch.optim.AdamW(dec.parameters(), lr=peak, weight_decay=0.01, betas=(0.9, 0.98))
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        random.shuffle(train)
        for b in range(steps_per_epoch):
            dec.train()
            exs = train[b * args.bs:(b + 1) * args.bs]
            z, hx, hm, lens, mid = encode_batch(exs, WANT_MID and args.src == "pred")
            with torch.no_grad():
                c = run_pred(z, hx, hm, lens, mid) if args.src == "pred" else z[:, 0]
            B = z.shape[0]
            zc = c.detach()[:, None, :].expand(B, R, -1).reshape(B * R, -1)
            task = torch.arange(R, device=DEV).repeat(B)
            tg = [ex["tgt"][k] for ex in exs for k in range(R)]
            N, L = len(tg), max(len(t) for t in tg)
            inp = torch.zeros(N, L, dtype=torch.long); lab = torch.full((N, L), -100)
            for i, t in enumerate(tg):
                inp[i, 0] = BOS
                inp[i, 1:len(t)] = torch.tensor(t[:-1])
                lab[i, :len(t)] = torch.tensor(t)
            inp, lab = inp.to(DEV), lab.to(DEV)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, peak)
            with amp():
                mem, pad = dec.memory(zc, task, hx, hm)
                hs = dec.hidden(inp, mem, pad)[lab != -100]
            loss = F.cross_entropy(hs.float() @ EF.T, lab[lab != -100])
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} gn {gn.item():.2f} "
                      f"lr {lr_at(step, total, peak):.2e} {time.time() - t0:.0f}s", flush=True)
    tag = f"{args.size}_{args.src}" if args.src == "pred" else "copy"
    torch.save(dec.state_dict(), f"p3_ddec_{tag}.pt")
    src = "pred" if args.src == "pred" else "copy_z0"
    print(f"\nDEPLOYED DECODER fine-tuned on {src} states" + (f" of predictor {args.size}" if args.src == "pred" else ""), flush=True)
    for name, vs in (("one-step (n = 1)", val_one), ("all chain lengths", val_all)):
        print(f"{name}: {len(vs)} held-out problems")
        for cond in (src, src + "_shuf", "oracle"):
            so, ao, _ = evaluate(vs, cond, show=3 if (cond == src and vs is val_one) else 0)
            print(f"  {cond:13s} step numeric match {so.mean():.3f} | answer acc {ao.mean():.3f}", flush=True)
            if cond == src and vs is val_all:
                for lo, hi, lab_ in ((1, 1, "n=1"), (2, 2, "n=2"), (3, 3, "n=3"), (4, 99, "n>=4")):
                    idx = [i for i, e in enumerate(vs) if lo <= e["n"] <= hi]
                    if idx:
                        print(f"     {lab_:5s} n={len(idx):4d} step {so[idx].mean():.3f} answer {ao[idx].mean():.3f}")
    print("saved", f"p3_ddec_{tag}.pt")
    raise SystemExit

peak = args.lr or {"s": 2e-4, "m": 1e-4, "top4": 2e-5}[args.size]
params = [p for p in pred.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=peak, weight_decay=0.01, betas=(0.9, 0.98))

def quick(vs):
    sp, ap_, cp = evaluate(vs, "pred")
    return sp.mean(), ap_.mean(), cp.mean()

with torch.no_grad():
    z_, hx_, hm_, ln_, mid_ = encode_batch(val_all[:16], WANT_MID)
    pred.eval()
    print(f"init check: cos(pred, z0) = {(run_pred(z_, hx_, hm_, ln_, mid_) * z_[:, 0]).sum(-1).mean().item():.4f}"
          " (top4 should be about 1.000)", flush=True)
s0, a0, c0 = quick(val_one[:200])
print(f"before training: one-step cos(pred, z*) {c0:.3f} | decoded step acc {s0:.3f}", flush=True)
step, t0 = 0, time.time()
for ep in range(args.epochs):
    pred.train()
    random.shuffle(train)
    for b in range(steps_per_epoch):
        exs = train[b * args.bs:(b + 1) * args.bs]
        z, hx, hm, lens, mid = encode_batch(exs, WANT_MID)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, peak)
        zh = run_pred(z, hx, hm, lens, mid)
        cos = (zh * z[:, 1]).sum(-1)
        loss = args.k * (1 - cos).mean()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} cos {cos.mean().item():.4f} gn {gn.item():.2f} "
                  f"lr {lr_at(step, total, peak):.2e} {time.time() - t0:.0f}s", flush=True)
    s1, a1, c1 = quick(val_one[:500])
    s2, a2, c2 = quick(val_all[:500])
    print(f"== epoch {ep + 1}: one-step cos {c1:.3f} step acc {s1:.3f} answer {a1:.3f} "
          f"| all cos {c2:.3f} step acc {s2:.3f} answer {a2:.3f}", flush=True)
    torch.save(pred.state_dict(), f"p3_pred_{args.size}.pt")

print(f"\nFINAL EVALUATION, one-hop predictor, size {args.size}", flush=True)
summary = {}
for name, vs in (("one-step (n = 1)", val_one), ("all chain lengths", val_all)):
    print(f"{name}: {len(vs)} held-out problems")
    r = {}
    for cond in ("oracle", "pred", "copy_z0"):
        r[cond] = evaluate(vs, cond, show=3 if (cond == "pred" and vs is val_one) else 0)
        print(f"  {cond:8s} step numeric match {r[cond][0].mean():.3f} | answer acc {r[cond][1].mean():.3f} "
              f"| cos to z*_1 {r[cond][2].mean():.3f}", flush=True)
    d_pred, d_copy = 1 - r["pred"][2].mean(), 1 - r["copy_z0"][2].mean()
    skill = 1 - d_pred / d_copy
    ratio = r["pred"][0].mean() / max(1e-9, r["oracle"][0].mean())
    print(f"  skill score S_1 = 1 - D/D_copy = {skill:.3f} | decoded accuracy as a share of the oracle = {ratio:.3f}")
    if vs is val_all:
        for lo, hi, lab_ in ((1, 1, "n=1"), (2, 2, "n=2"), (3, 3, "n=3"), (4, 99, "n>=4")):
            idx = [i for i, e in enumerate(vs) if lo <= e["n"] <= hi]
            if idx:
                print(f"     {lab_:5s} n={len(idx):4d} oracle {r['oracle'][0][idx].mean():.3f} "
                      f"pred {r['pred'][0][idx].mean():.3f} copy {r['copy_z0'][0][idx].mean():.3f}")
    summary[name] = (r["oracle"][0].mean(), r["pred"][0].mean(), r["copy_z0"][0].mean(), skill, ratio)
o, p_, c, sk, ra = summary["one-step (n = 1)"]
print(f"\nP3 size {args.size}: one-step decoded accuracy {p_:.3f} (oracle {o:.3f}, copy {c:.3f})")
print("  registered gate (>= 0.80):", "PASS" if p_ >= 0.80 else "FAIL")
print("  ratio reading (>= 8/9 of the oracle):", "PASS" if ra >= 8 / 9 else "FAIL")

# ---------------------------------------------------------------- latency, batch size 1
def timed(fn):
    if AMP:
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if AMP:
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000

@torch.no_grad()
def latency(exs):
    pred.eval(); dec.eval()
    rows = []
    task1 = torch.tensor([1], device=DEV)
    for ex in exs:
        (z, hx, hm, lens, mid), t_enc = timed(lambda: encode_batch([ex], WANT_MID, question_only=True))
        zh, t_pred = timed(lambda: run_pred(z, hx, hm, lens, mid))
        def dec_answer():
            with amp():
                return greedy(zh, task1, hx, hm, rep=1, max_new=20)
        _, t_dec = timed(dec_answer)
        ids = torch.tensor([[BOS] + ex["seg"][0] + [NLID]], device=DEV)
        n_cot = len(ex["seg"][1]) + 1 + len(ex["tgt"][1])
        n_ans = len(ex["tgt"][1])
        def gen(n):
            return lambda: llm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                                        min_new_tokens=n, do_sample=False, temperature=None, top_p=None,
                                        pad_token_id=EOS, use_cache=True)
        _, t_cot = timed(gen(n_cot))
        _, t_noc = timed(gen(n_ans))
        rows.append((t_enc, t_pred, t_dec, t_enc + t_pred + t_dec, t_cot, t_noc, n_cot))
    a = np.array(rows[20:])
    med = np.median(a, 0)
    print(f"\nLATENCY, batch size 1, {len(a)} one-step questions after 20 warm-up, median ms")
    print(f"  encoder {med[0]:.1f} | predictor step {med[1]:.2f} | decoder (answer) {med[2]:.1f} | latent total {med[3]:.1f}")
    print(f"  CoT-length generation proxy {med[4]:.1f} (median {med[6]:.0f} tokens) | answer-only proxy {med[5]:.1f}")
    print(f"  speed-up over the CoT proxy {med[4] / med[3]:.2f}x (bar: 2.5x)")

if len(val_one) >= 40:
    try:
        latency(val_one[:args.nlat])
    except Exception as err:
        print("latency measurement failed:", repr(err))
print("saved", f"p3_pred_{args.size}.pt")
