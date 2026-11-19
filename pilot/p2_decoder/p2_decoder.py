import argparse, math, random, time
from fractions import Fraction
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from parse_gsm import load_split, validate, group_texts, safe_eval, EQ

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=3)
ap.add_argument("--ntrain", type=int, default=50000)
ap.add_argument("--nval", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--d", type=int, default=768)
ap.add_argument("--seed", type=int, default=82)
ap.add_argument("--eval_only", default="")
ap.add_argument("--layer", type=int, default=-1)
ap.add_argument("--dec_layers", type=int, default=2)
args = ap.parse_args()
T, R = args.T, args.T + 1          # R tasks per problem: T steps + the answer
random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True

M = "meta-llama/Llama-3.2-1B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
llm = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
for p in llm.parameters():
    p.requires_grad_(False)
enc = llm.model
E = llm.get_input_embeddings().weight.detach()
NLID = tok("\n", add_special_tokens=False).input_ids[0]
BOS, EOS = tok.bos_token_id, tok.eos_token_id

def ids_of(s):
    return tok(s, add_special_tokens=False).input_ids

t0 = time.time()
rows = load_split()
valid = []
for q, c, a in rows:
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
print(f"data ready in {time.time() - t0:.0f}s: train {len(train)}, val {len(val)}, T={T}", flush=True)

@torch.no_grad()
def encode_batch(exs):
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
    o = enc(input_ids=x.cuda(), attention_mask=m.cuda(), use_cache=False, output_hidden_states=args.layer != -1)
    h = o.last_hidden_state
    hz = h if args.layer == -1 else enc.norm(o.hidden_states[args.layer])
    z = torch.stack([F.normalize(hz[i, poss[i]].float(), dim=-1) for i in range(len(seqs))])  # (B,T+1,2048): z0, z*_1..z*_T
    lens = torch.tensor([p[0] + 1 for p in poss], device="cuda")
    Lh = int(lens.max())
    hx = h[:, :Lh].float()
    hmask = torch.arange(Lh, device="cuda")[None, :] < lens[:, None]
    return z, hx, hmask

class Dec(nn.Module):
    def __init__(self, E, R, d, heads=12, layers=2, drop=0.1):
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

dec = Dec(E, R, args.d, heads=args.d // 64, layers=args.dec_layers).cuda()
print("decoder trainable parameters: %.1fM" % (sum(p.numel() for p in dec.parameters()) / 1e6), flush=True)
if args.eval_only:
    dec.load_state_dict(torch.load(args.eval_only)); args.epochs = 0

def conditioning(z, cond):
    B = z.shape[0]
    C = torch.stack([z[:, min(k + 1, T)] for k in range(R)], 1)       # z*_1..z*_T, then z*_T for the answer
    if cond == "copy_z0":
        C = z[:, 0:1].expand(B, R, -1)
    elif cond == "shuffled":
        C = C[torch.roll(torch.arange(B, device=z.device), 1)]
    return C.reshape(B * R, -1), torch.arange(R, device=z.device).repeat(B)

@torch.no_grad()
def greedy(zc, task, hx, hmask, max_new=100):
    mem, pad = dec.memory(zc, task, hx, hmask)
    out = torch.full((zc.shape[0], 1), BOS, device="cuda")
    done = torch.zeros(zc.shape[0], dtype=torch.bool, device="cuda")
    for _ in range(max_new):
        h = dec.hidden(out, mem, pad)[:, -1]
        nxt = (h.to(torch.bfloat16) @ dec.E.T).argmax(-1)
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

@torch.no_grad()
def evaluate(exs, cond, chunk=64, show=0):
    dec.eval()
    step_ok = np.zeros((len(exs), T)); text_ok = np.zeros((len(exs), T)); ans_ok = np.zeros(len(exs))
    for s in range(0, len(exs), chunk):
        part = exs[s:s + chunk]
        z, hx, hm = encode_batch(part)
        zc, task = conditioning(z, cond)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = greedy(zc, task, hx, hm)
        for i, ex in enumerate(part):
            for k in range(R):
                text = tok.decode(outs[i * R + k], skip_special_tokens=True).strip()
                if k < T:
                    ref, got = last_val(ex["groups"][k]), last_val(text)
                    step_ok[s + i, k] = float(got is not None and got == ref)
                    text_ok[s + i, k] = float(text == ex["groups"][k])
                    if show and s == 0 and i < show and k == T - 1:
                        print("   ref:", ex["groups"][k], "| got:", text)
                else:
                    g, a = num(text), num(ex["answer"])
                    ans_ok[s + i] = float(g is not None and g == a)
    dec.train()
    global LAST
    LAST = step_ok
    return step_ok.mean(0), text_ok.mean(0), float(ans_ok.mean())

def fmt(a):
    return "[" + ", ".join(f"{x:.3f}" for x in a) + "]"

steps_per_epoch = len(train) // args.bs
total = steps_per_epoch * args.epochs
opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
def lr_at(s, warm=1000):
    if s < warm:
        return args.lr * (s + 1) / warm
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

step, t0 = 0, time.time()
dec.train()
for ep in range(args.epochs):
    random.shuffle(train)
    for b in range(steps_per_epoch):
        exs = train[b * args.bs:(b + 1) * args.bs]
        z, hx, hm = encode_batch(exs)
        zc, task = conditioning(z, "oracle")
        tg = [ex["tgt"][k] for ex in exs for k in range(R)]
        N, L = len(tg), max(len(t) for t in tg)
        inp = torch.zeros(N, L, dtype=torch.long); lab = torch.full((N, L), -100)
        for i, t in enumerate(tg):
            inp[i, 0] = BOS
            inp[i, 1:len(t)] = torch.tensor(t[:-1])
            lab[i, :len(t)] = torch.tensor(t)
        inp, lab = inp.cuda(), lab.cuda()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem, pad = dec.memory(zc, task, hx, hm)
            h = dec.hidden(inp, mem, pad)
            sel = lab != -100
            hs = h[sel]
        with torch.autocast('cuda', enabled=False):
            logits = hs.float() @ dec.E.float().T
        loss = F.cross_entropy(logits, lab[sel])
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} gn {gn.item():.2f} lr {lr_at(step):.2e} {time.time() - t0:.0f}s", flush=True)
    sa, ta, aa = evaluate(val[:500], "oracle")
    print(f"== epoch {ep + 1}: val(500) oracle step acc {fmt(sa)} | answer acc {aa:.3f}", flush=True)
    torch.save(dec.state_dict(), f"p2_decoder_T{T}_ep{ep + 1}.pt")

torch.save(dec.state_dict(), f"p2_decoder_T{T}.pt")
print("\nFINAL EVALUATION on", len(val), "held-out problems", flush=True)
res = {}
for cond in ("oracle", "copy_z0", "shuffled"):
    sa, ta, aa = evaluate(val, cond, show=3 if cond == "oracle" else 0)
    res[cond] = (sa, ta, aa)
    if cond == 'oracle':
        ORA = LAST.copy()
    print(f"{cond:9s} step numeric-match per t {fmt(sa)} | full-text match {fmt(ta)} | answer acc {aa:.3f}", flush=True)
ok = bool((res["oracle"][0] >= 0.90).all())
print("P2 gate (oracle numeric match >= 90% at every t):", "PASS" if ok else "FAIL")

def isint(v): return Fraction(v).denominator == 1
def ref(i, k):
    return last_val(val[i]["groups"][k])
def strat(name, f):
    row = []
    for k in range(T):
        idx = [i for i in range(len(val)) if ref(i, k) is not None and f(ref(i, k))]
        row.append((len(idx), float(ORA[idx, k].mean()) if idx else float("nan")))
    print(f"{name:30s}", " | ".join(f"n={n:4d} acc={a:.3f}" for n, a in row))
print("\nSTRATIFIED oracle step accuracy by the reference result of that step")
strat("all", lambda v: True)
strat("integer", isint)
strat("non-integer", lambda v: not isint(v))
strat("integer, |v| < 100", lambda v: isint(v) and abs(v) < 100)
strat("integer, 100 <= |v| < 10000", lambda v: isint(v) and 100 <= abs(v) < 10000)
strat("|v| >= 10000", lambda v: abs(v) >= 10000)
cl = [i for i in range(len(val)) if all(ref(i, k) is not None and isint(ref(i, k)) and abs(ref(i, k)) < 10000 for k in range(T))]
print("clean subset (all steps integer, <10000): n =", len(cl), "per-t", fmt(ORA[cl].mean(0)))
