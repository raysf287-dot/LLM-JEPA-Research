"""Pilot P4: short free-running rollouts against the copy baseline, frozen backbone.

Stages
  dec   train the oracle decoder for horizon T on target states (the probe)
  pred  train a rollout predictor for horizon T (teacher-forced, then scheduled sampling, then
        free-running), roll it out on held-out problems and report D(t), the skill score S_t,
        R = D(T)/D(1), step-faithfulness a_t against the copy baseline, and the collapse check
Sizes   s = 4-layer cross-attentive predictor, width 768 (about 38M in the layers)
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
ap.add_argument("--stage", required=True, choices=["dec", "pred"])
ap.add_argument("--T", type=int, required=True)
ap.add_argument("--size", default="s", choices=["s", "top4"])
ap.add_argument("--ntrain", type=int, default=150000)
ap.add_argument("--nval", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=0)        # 0 = 5 for dec, 3 for pred
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=0.0)        # 0 = the default for the stage and size
ap.add_argument("--k", type=float, default=4.0)         # scaled-cosine factor (proposal eq. 3)
ap.add_argument("--seed", type=int, default=82)
ap.add_argument("--dec_ckpt", default="")               # default p4_dec_T{T}.pt
ap.add_argument("--dec_d", type=int, default=1024)      # decoder width (P2 run 6: 1024, 4 layers)
ap.add_argument("--dec_layers", type=int, default=4)
ap.add_argument("--nboot", type=int, default=10000)
ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
args = ap.parse_args()
T, R = args.T, args.T + 1                               # R decoder tasks: T steps + the answer
EPOCHS = args.epochs or (5 if args.stage == "dec" else 3)
DEC_CKPT = args.dec_ckpt or f"p4_dec_T{T}.pt"
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
EF = E.float()
NLID = tok("\n", add_special_tokens=False).input_ids[0]
BOS, EOS = tok.bos_token_id, tok.eos_token_id

def ids_of(s):
    return tok(s, add_special_tokens=False).input_ids

# ---------------------------------------------------------------- data (same split rule as p2_decoder.py)
t0 = time.time()
valid = []
for q, c, a in parse_gsm.load_split():
    ex, _ = validate(q, c, a)
    if ex:
        ex["answer"] = a.replace(",", "").strip()
        valid.append(ex)
random.Random(args.seed).shuffle(valid)
pool = []
for ex in valid:
    if ex["n"] < T:
        continue
    groups = group_texts(ex, T)
    tg = [ids_of(g) + [EOS] for g in groups] + [ids_of(ex["answer"]) + [EOS]]
    if max(len(t) for t in tg) > 120:
        continue
    ex["groups"], ex["tgt"] = groups, tg
    ex["seg"] = [ids_of(ex["question"])] + [ids_of(g) for g in groups]
    pool.append(ex)
    if len(pool) >= args.nval + args.ntrain:
        break
val, train = pool[:args.nval], pool[args.nval:]
print(f"data ready in {time.time() - t0:.0f}s: T={T}, problems with n >= {T}: train {len(train)}, val {len(val)}", flush=True)

@torch.no_grad()
def encode_batch(exs, want_mid=False):
    """One frozen pass. z: (B, T+1, DM) = z0, z*_1..z*_T; hx: prompt states; mid: input to the top four layers."""
    seqs, poss = [], []
    for ex in exs:
        ids, pos = [BOS], []
        for s in ex["seg"]:
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
    """Decoder of P2 run 5/6 (tied frozen embeddings, rescaled inputs)."""
    def __init__(self, E, R, d, heads, layers, drop=0.1):
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

    def memory(self, zc, task, hx, hmask):
        zt = self.z_norm(self.z_in(zc * self.zs) + self.task(task))
        h = self.hx_norm(self.hx_in(self.hx_pre(hx))).repeat_interleave(self.R, 0)
        pad = ~hmask.repeat_interleave(self.R, 0)
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
    history, 2 learned predictor tokens; new state = mean of their outputs, projected, L2-normalised."""
    def __init__(self, dm, d=768, layers=4, nq=2, drop=0.1, max_hist=8):
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
    hidden states entering those layers (question tokens only), followed by one projected token per
    earlier latent state. The new state is read at the last position."""
    def __init__(self, llm, dm):
        super().__init__()
        cfg = copy.deepcopy(llm.config)
        cfg.num_hidden_layers = 4
        self.m = type(llm.model)(cfg)
        self.m.layers.load_state_dict(llm.model.layers[-4:].state_dict())
        self.m.norm.load_state_dict(llm.model.norm.state_dict())
        self.m.embed_tokens = None
        self.m.float()
        self.z_in = nn.Linear(dm, dm)
        self.zs = dm ** 0.5

    def forward(self, mid, hmask, lens, zhist):
        B, t, _ = zhist.shape                           # states so far: z0 .. z_{t-1}
        ar = torch.arange(B, device=mid.device)
        x = mid
        if t > 1:
            x = torch.cat([mid, mid.new_zeros(B, t - 1, mid.shape[2])], 1)
            tokz = self.z_in(zhist[:, 1:] * self.zs).to(x.dtype)
            for j in range(t - 1):
                x = x.index_put((ar, lens + j), tokz[:, j])
        mask = torch.arange(x.shape[1], device=mid.device)[None, :] < (lens + t - 1)[:, None]
        h = self.m(inputs_embeds=x, attention_mask=mask.long(), use_cache=False).last_hidden_state
        y = h[ar, lens + t - 2]
        return F.normalize(y.float(), dim=-1)

def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6

dec = Dec(E, R, args.dec_d, args.dec_d // 64, args.dec_layers).to(DEV)
print(f"decoder trainable parameters: {nparams(dec):.1f}M", flush=True)

# ---------------------------------------------------------------- decoding and scoring
@torch.no_grad()
def greedy(zc, task, hx, hmask, max_new=100):
    mem, pad = dec.memory(zc, task, hx, hmask)
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

@torch.no_grad()
def decode_states(part, S, hx, hm):
    """S: (B, T+1, DM); S[:, t] is the state decoded for step t (index 0 is unused); the answer uses S[:, T].
    Returns step numeric match (B, T) and answer match (B)."""
    B = S.shape[0]
    C = torch.stack([S[:, min(k + 1, T)] for k in range(R)], 1).reshape(B * R, -1)
    task = torch.arange(R, device=DEV).repeat(B)
    with amp():
        outs = greedy(C, task, hx, hm)
    step_ok, ans_ok = np.zeros((B, T)), np.zeros(B)
    for i, ex in enumerate(part):
        for k in range(T):
            text = tok.decode(outs[i * R + k], skip_special_tokens=True).strip()
            ref, got = last_val(ex["groups"][k]), last_val(text)
            step_ok[i, k] = float(got is not None and got == ref)
        g, a = num(tok.decode(outs[i * R + T], skip_special_tokens=True)), num(ex["answer"])
        ans_ok[i] = float(g is not None and g == a)
    return step_ok, ans_ok

def lr_at(s, total, peak):
    warm = min(1000, max(1, total // 10))
    if s < warm:
        return peak * (s + 1) / warm
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

steps_per_epoch = len(train) // args.bs
total = steps_per_epoch * EPOCHS

# ================================================================ stage: oracle decoder
if args.stage == "dec":
    @torch.no_grad()
    def eval_dec(exs, cond, chunk=64):
        dec.eval()
        so, ao = [], []
        for s in range(0, len(exs), chunk):
            part = exs[s:s + chunk]
            z, hx, hm, _, _ = encode_batch(part)
            if cond == "copy_z0":
                S = z[:, :1].expand(-1, T + 1, -1)
            elif cond == "shuffled":
                S = z[torch.roll(torch.arange(z.shape[0], device=DEV), 1)]
            else:
                S = z
            a, b = decode_states(part, S, hx, hm)
            so.append(a); ao.append(b)
        return np.concatenate(so), np.concatenate(ao)

    peak = args.lr or 1e-4
    opt = torch.optim.AdamW(dec.parameters(), lr=peak, weight_decay=0.01, betas=(0.9, 0.98))
    step, t0 = 0, time.time()
    for ep in range(EPOCHS):
        random.shuffle(train)
        for b in range(steps_per_epoch):
            dec.train()
            exs = train[b * args.bs:(b + 1) * args.bs]
            z, hx, hm, _, _ = encode_batch(exs)
            B = z.shape[0]
            zc = torch.stack([z[:, min(k + 1, T)] for k in range(R)], 1).reshape(B * R, -1)
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
        so, ao = eval_dec(val[:500], "oracle")
        print(f"== epoch {ep + 1}: val(500) oracle step acc {fmt(so.mean(0))} | answer acc {ao.mean():.3f}", flush=True)
        torch.save(dec.state_dict(), DEC_CKPT)
    print(f"\nFINAL EVALUATION, T = {T} oracle decoder, {len(val)} held-out problems", flush=True)
    for cond in ("oracle", "copy_z0", "shuffled"):
        so, ao = eval_dec(val, cond)
        print(f"  {cond:9s} step numeric match per t {fmt(so.mean(0))} | answer acc {ao.mean():.3f}", flush=True)
    print("saved", DEC_CKPT)
    raise SystemExit

# ================================================================ stage: rollout predictor
dec.load_state_dict(torch.load(DEC_CKPT, map_location=DEV))
dec.eval()
for p in dec.parameters():
    p.requires_grad_(False)
WANT_MID = args.size == "top4"
pred = (Top4(llm, DM) if WANT_MID else Pred(DM)).to(DEV)
print(f"predictor size {args.size}: {nparams(pred):.1f}M trainable; oracle decoder {DEC_CKPT}", flush=True)

def run_pred(zhist, hx, hm, lens, mid):
    with amp():
        if WANT_MID:
            return pred(mid, hm, lens, zhist)
        return pred(zhist, hx, hm)

def rollout(z, hx, hm, lens, mid, p_free):
    """Predict z_1..z_T. Each fed-back state is the model's own prediction with probability p_free
    (detached), otherwise the true target. p_free = 1 is the free-running rollout used at inference."""
    B = z.shape[0]
    use = torch.rand(B, T, device=DEV) < p_free
    hist, outs = [z[:, 0]], []
    for t in range(1, T + 1):
        zh = run_pred(torch.stack(hist, 1), hx, hm, lens, mid)
        outs.append(zh)
        hist.append(torch.where(use[:, t - 1:t], zh.detach(), z[:, t]))
    return torch.stack(outs, 1)

def p_free_at(s):
    f = s / max(1, total)
    return 0.0 if f < 0.4 else (1.0 if f >= 0.7 else (f - 0.4) / 0.3)

@torch.no_grad()
def evaluate(exs, decode=True, chunk=64):
    pred.eval()
    out = {k: [] for k in ("cos_free", "cos_tf", "cos_copy", "zp", "zt",
                           "ok_oracle", "ok_pred", "ok_copy", "an_oracle", "an_pred", "an_copy")}
    for s in range(0, len(exs), chunk):
        part = exs[s:s + chunk]
        z, hx, hm, lens, mid = encode_batch(part, WANT_MID)
        zp = rollout(z, hx, hm, lens, mid, 1.0)
        ztf = rollout(z, hx, hm, lens, mid, 0.0)
        out["cos_free"].append((zp * z[:, 1:]).sum(-1).cpu().numpy())
        out["cos_tf"].append((ztf * z[:, 1:]).sum(-1).cpu().numpy())
        out["cos_copy"].append((z[:, :1] * z[:, 1:]).sum(-1).cpu().numpy())
        out["zp"].append(zp.cpu().numpy()); out["zt"].append(z[:, 1:].cpu().numpy())
        if decode:
            for name, S in (("oracle", z), ("pred", torch.cat([z[:, :1], zp], 1)), ("copy", z[:, :1].expand(-1, T + 1, -1))):
                a, b = decode_states(part, S, hx, hm)
                out["ok_" + name].append(a); out["an_" + name].append(b)
    return {k: np.concatenate(v) for k, v in out.items() if v}

peak = args.lr or {"s": 2e-4, "top4": 2e-5}[args.size]
params = [p for p in pred.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=peak, weight_decay=0.01, betas=(0.9, 0.98))
step, t0 = 0, time.time()
for ep in range(EPOCHS):
    random.shuffle(train)
    for b in range(steps_per_epoch):
        pred.train()
        exs = train[b * args.bs:(b + 1) * args.bs]
        z, hx, hm, lens, mid = encode_batch(exs, WANT_MID)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, peak)
        pf = p_free_at(step)
        zp = rollout(z, hx, hm, lens, mid, pf)
        cos = (zp * z[:, 1:]).sum(-1)
        loss = args.k * (1 - cos).mean()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} cos {fmt(cos.detach().mean(0).tolist())} p_free {pf:.2f} "
                  f"gn {gn.item():.2f} lr {lr_at(step, total, peak):.2e} {time.time() - t0:.0f}s", flush=True)
    r = evaluate(val[:500])
    print(f"== epoch {ep + 1}: free-running cos {fmt(r['cos_free'].mean(0))} | teacher-forced cos {fmt(r['cos_tf'].mean(0))} "
          f"| a_t {fmt(r['ok_pred'].mean(0))} | answer {r['an_pred'].mean():.3f}", flush=True)
    torch.save(pred.state_dict(), f"p4_pred_T{T}_{args.size}.pt")

# ---------------------------------------------------------------- final report
def mcnemar(x, y):
    """Exact two-sided McNemar test on paired 0/1 arrays."""
    b = int(((x == 1) & (y == 0)).sum()); c = int(((x == 0) & (y == 1)).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))
    return b, c, min(1.0, 2 * tail / 2 ** n)

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj

r = evaluate(val)
N = len(val)
d_free, d_tf, d_copy = 1 - r["cos_free"], 1 - r["cos_tf"], 1 - r["cos_copy"]
D, Dtf, Dc = d_free.mean(0), d_tf.mean(0), d_copy.mean(0)
S = 1 - D / Dc
rng = np.random.default_rng(args.seed)
boot = np.zeros((args.nboot, T))
for s in range(0, args.nboot, 500):
    idx = rng.integers(0, N, (min(500, args.nboot - s), N))
    boot[s:s + len(idx)] = 1 - d_free[idx].mean(1) / d_copy[idx].mean(1)
lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
tests = [mcnemar(r["ok_pred"][:, t], r["ok_copy"][:, t]) for t in range(T)]
padj = holm([x[2] for x in tests])
sd_ratio = r["zp"].std(0).mean(-1) / r["zt"].std(0).mean(-1)

print(f"\nFINAL EVALUATION, free-running rollout, T = {T}, size {args.size}, {N} held-out problems with n >= {T}")
print("  t | D(t) free | D(t) teacher-forced | D_copy(t) | skill S_t [95% CI] | a_t pred | a_t copy | a_t oracle | McNemar b/c, Holm p | std ratio")
for t in range(T):
    print(f"  {t + 1} | {D[t]:.4f} | {Dtf[t]:.4f} | {Dc[t]:.4f} | {S[t]:.3f} [{lo[t]:.3f}, {hi[t]:.3f}] | "
          f"{r['ok_pred'][:, t].mean():.3f} | {r['ok_copy'][:, t].mean():.3f} | {r['ok_oracle'][:, t].mean():.3f} | "
          f"{tests[t][0]}/{tests[t][1]}, {padj[t]:.2g} | {sd_ratio[t]:.3f}")
print(f"  drift ratio R = D(T)/D(1) = {D[-1] / D[0]:.3f} (teacher-forced {Dtf[-1] / Dtf[0]:.3f}; copy {Dc[-1] / Dc[0]:.3f})")
print(f"  answer accuracy from the final state: pred {r['an_pred'].mean():.3f} | copy {r['an_copy'].mean():.3f} | oracle {r['an_oracle'].mean():.3f}")
collapse = bool((sd_ratio < 0.07).any())
skill_ok = bool(lo[-1] > 0)
faith_ok = all(r["ok_pred"][:, t].mean() > r["ok_copy"][:, t].mean() and padj[t] < 0.05 for t in range(T))
print(f"\nP4 T = {T}, size {args.size}:")
print("  validity (no collapse, std ratio >= 0.07 at every t):", "OK" if not collapse else "COLLAPSE FLAGGED")
print("  S_T > 0 (95% bootstrap lower bound above zero):", "PASS" if skill_ok else "FAIL")
print("  a_t above the copy baseline at every t (Holm-corrected McNemar, 0.05):", "PASS" if faith_ok else "FAIL")
print("  gate:", "PASS" if (skill_ok and faith_ok and not collapse) else "FAIL")
print("saved", f"p4_pred_T{T}_{args.size}.pt")
