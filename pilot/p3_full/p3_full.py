"""P3 full training: one-hop JEPA predictor with an unfrozen encoder, scored on the GSM8K test set.

What is new compared with p3.py
  * the encoder is adapted with LoRA and trained jointly with the predictor and the decoder
  * the target comes from an EMA copy of the encoder (momentum tau), with a stop-gradient
  * Phase 1 (encoder frozen) is followed by Phase 2 (encoder trained), inside one run
  * the decoder is trained alongside, on the predicted state and on the true target state
  * the final report includes answer accuracy on the 1,319 GSM8K test questions

--size none trains the decoder-only control: no predictor and no latent loss; the decoder reads
the question state directly. It shows how much the decoder and the adapted encoder do unaided.
"""
import argparse, copy, math, random, time
from fractions import Fraction
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import parse_gsm
from parse_gsm import validate, group_texts, safe_eval, EQ

ap = argparse.ArgumentParser()
ap.add_argument("--size", required=True, choices=["none", "s", "m", "top4"])
ap.add_argument("--ntrain", type=int, default=0)             # 0 = every valid training problem
ap.add_argument("--nval", type=int, default=2000)
ap.add_argument("--examples", type=int, default=1000000)     # training examples seen in total (Phase 1 of the proposal: about 1M)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--warm_frac", type=float, default=0.5)      # share of the run with the encoder frozen (Phase 1)
ap.add_argument("--freeze_encoder", action="store_true")     # never unfreeze: full-budget frozen run
ap.add_argument("--lr_pred", type=float, default=0.0)        # 0 = default for the size
ap.add_argument("--lr_dec", type=float, default=1e-4)
ap.add_argument("--lr_lora", type=float, default=1e-4)
ap.add_argument("--k", type=float, default=4.0)              # scaled-cosine factor
ap.add_argument("--lam_dec", type=float, default=0.1)        # weight of the decoding loss (proposal: 0.1)
ap.add_argument("--tau", type=float, default=0.996)          # EMA momentum of the target encoder
ap.add_argument("--lora_r", type=int, default=128)
ap.add_argument("--lora_alpha", type=int, default=32)
ap.add_argument("--dec_d", type=int, default=768)
ap.add_argument("--dec_layers", type=int, default=2)
ap.add_argument("--seed", type=int, default=82)
ap.add_argument("--tag", default="")
ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
args = ap.parse_args()
T, R = 1, 2                                                  # one hop; decoder tasks: chain text, answer
TAG = args.tag or args.size
random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"
if AMP:
    torch.backends.cuda.matmul.allow_tf32 = True

def amp():
    return torch.autocast(DEV, dtype=torch.bfloat16, enabled=AMP)

tok = AutoTokenizer.from_pretrained(args.model)
# fp32 weights (LoRA and EMA need the precision); the forward pass still runs in bf16 under autocast
base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(DEV).eval()
for p in base.parameters():
    p.requires_grad_(False)
NLAYERS = len(base.model.layers)
E = base.get_input_embeddings().weight.detach()
DM = E.shape[1]
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
cap = args.ntrain or 10 ** 9
for ex in valid:                                             # same split rule as p3.py
    chain = group_texts(ex, T)[0]
    tg = [ids_of(chain) + [EOS], ids_of(ex["answer"]) + [EOS]]
    if max(len(t) for t in tg) > 120:
        continue
    ex["chain"], ex["tgt"] = chain, tg
    ex["q_ids"], ex["c_ids"] = ids_of(ex["question"]), ids_of(chain)
    if len(val_all) < args.nval:
        val_all.append(ex)
    elif ex["n"] == 1 and len(val_one) < args.nval:
        val_one.append(ex)
    elif len(train) < cap:
        train.append(ex)
    if len(train) >= cap and len(val_one) >= args.nval:
        break
test = []
for q, c, a in parse_gsm.load_split("gsm8k_test.json"):      # the 1,319 GSM8K test questions
    ex, _ = validate(q, c, a)
    t = {"question": q, "answer": str(a).replace(",", "").strip(), "q_ids": ids_of(q), "c_ids": None, "chain": None, "n": 0}
    if ex:
        t["chain"] = group_texts(ex, T)[0]; t["c_ids"] = ids_of(t["chain"]); t["n"] = ex["n"]
    test.append(t)
print(f"data ready in {time.time() - t0:.0f}s: train {len(train)}, val_all {len(val_all)}, val_one {len(val_one)}, "
      f"GSM8K test {len(test)} ({sum(t['c_ids'] is not None for t in test)} with a usable reference chain)", flush=True)

# ---------------------------------------------------------------- modules
class Dec(nn.Module):
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
    """4-layer cross-attentive predictor (proposal 3.2)."""
    def __init__(self, dm, d, layers=4, nq=2, drop=0.1):
        super().__init__()
        self.nq = nq
        self.z_in, self.hx_in = nn.Linear(dm, d), nn.Linear(dm, d)
        self.hx_pre = nn.LayerNorm(dm)
        self.hx_norm, self.z_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.zs = dm ** 0.5
        self.q = nn.Parameter(torch.randn(nq, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, d // 64, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.core = nn.TransformerDecoder(layer, layers)
        self.out_norm, self.out = nn.LayerNorm(d), nn.Linear(d, dm)

    def forward(self, z0, hx, hmask):
        B = z0.shape[0]
        zt = self.z_norm(self.z_in(z0 * self.zs))[:, None, :]
        x = torch.cat([zt, self.q[None].expand(B, -1, -1).to(zt.dtype)], 1)
        mem = self.hx_norm(self.hx_in(self.hx_pre(hx)))
        h = self.core(x, mem, memory_key_padding_mask=~hmask)
        y = self.out(self.out_norm(h[:, -self.nq:].mean(1)))
        return F.normalize(y.float(), dim=-1)

class Top4(nn.Module):
    """Trainable copy of the (un-adapted) backbone's top four layers and final norm."""
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
    return sum(p.numel() for p in m.parameters()) / 1e6

dec = Dec(E, R, args.dec_d, args.dec_d // 64, args.dec_layers).to(DEV)
pred = None
if args.size == "top4":
    pred = Top4(base).to(DEV)                                # built before LoRA is injected into the backbone
elif args.size in ("s", "m"):
    pred = Pred(DM, 768 if args.size == "s" else 1536).to(DEV)
WANT_MID = args.size == "top4"

# LoRA on the online encoder; the target encoder is an EMA copy
lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
online = get_peft_model(base, lcfg)
enc_on = online.base_model.model.model
target = copy.deepcopy(online).eval()
for p in target.parameters():
    p.requires_grad_(False)
enc_tg = target.base_model.model.model
lora_on = [(n, p) for n, p in online.named_parameters() if "lora_" in n]
tg_params = dict(target.named_parameters())
lora_tg = [tg_params[n] for n, _ in lora_on]
for _, p in lora_on:
    p.requires_grad_(False)                                  # Phase 1: encoder frozen
print(f"decoder {nparams(dec):.1f}M | predictor {nparams(pred) if pred is not None else 0:.1f}M ({args.size}) | "
      f"LoRA {sum(p.numel() for _, p in lora_on) / 1e6:.1f}M (rank {args.lora_r})", flush=True)

def encode(body, exs, chain, want_mid=False, grad=False):
    """One pass. Returns z0 (state at the question delimiter), z_last (state at the last delimiter),
    hx (question token states), hmask, lens, mid (input to the top four layers, question tokens)."""
    seqs, qlen, last = [], [], []
    for ex in exs:
        ids = [BOS] + ex["q_ids"] + [NLID]
        qlen.append(len(ids))
        if chain:
            ids = ids + ex["c_ids"] + [NLID]
        seqs.append(ids); last.append(len(ids) - 1)
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), EOS, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    with torch.set_grad_enabled(grad), amp():
        o = body(input_ids=x.to(DEV), attention_mask=m.to(DEV), use_cache=False, output_hidden_states=want_mid)
    h = o.last_hidden_state.float()
    ar = torch.arange(len(seqs), device=DEV)
    lens = torch.tensor(qlen, device=DEV)
    z0 = F.normalize(h[ar, lens - 1], dim=-1)
    zl = F.normalize(h[ar, torch.tensor(last, device=DEV)], dim=-1)
    Lh = int(lens.max())
    hx = h[:, :Lh]
    hmask = torch.arange(Lh, device=DEV)[None, :] < lens[:, None]
    mid = o.hidden_states[NLAYERS - 4][:, :Lh].float() if want_mid else None
    return z0, zl, hx, hmask, lens, mid

def run_pred(z0, hx, hm, lens, mid):
    if pred is None:
        return z0                                            # decoder-only control
    with amp():
        return pred(mid, hm, lens) if WANT_MID else pred(z0, hx, hm)

def dec_loss(states, hx, hm, exs):
    B = states.shape[0]
    zc = states[:, None, :].expand(B, R, -1).reshape(B * R, -1)
    task = torch.arange(R, device=DEV).repeat(B)
    tg = [ex["tgt"][k] for ex in exs for k in range(R)]
    N, L = len(tg), max(len(t) for t in tg)
    inp = torch.zeros(N, L, dtype=torch.long); lab = torch.full((N, L), -100)
    for i, t in enumerate(tg):
        inp[i, 0] = BOS
        inp[i, 1:len(t)] = torch.tensor(t[:-1])
        lab[i, :len(t)] = torch.tensor(t)
    inp, lab = inp.to(DEV), lab.to(DEV)
    with amp():
        mem, pad = dec.memory(zc, task, hx, hm)
        hs = dec.hidden(inp, mem, pad)[lab != -100]
    return F.cross_entropy(hs.float() @ E.T, lab[lab != -100])

@torch.no_grad()
def greedy(zc, task, hx, hmask, max_new=100):
    mem, pad = dec.memory(zc, task, hx, hmask)
    out = torch.full((zc.shape[0], 1), BOS, device=DEV)
    done = torch.zeros(zc.shape[0], dtype=torch.bool, device=DEV)
    for _ in range(max_new):
        h = dec.hidden(out, mem, pad)[:, -1]
        nxt = (h.float() @ E.T).argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
        out = torch.cat([out, nxt[:, None]], 1)
        done |= nxt == EOS
        if bool(done.all()):
            break
    return [row[:row.index(EOS)] if EOS in row else row for row in out[:, 1:].tolist()]

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
def evaluate(exs, conds, chunk=64):
    """Decode chain and answer from each state in conds ('pred', 'copy', 'oracle').
    Every problem in exs needs a reference chain when 'oracle' is requested or cosines are wanted.
    Returns {cond: (step_ok, ans_ok, cos_to_target)} and the stacked predicted and target states."""
    dec.eval()
    if pred is not None:
        pred.eval()
    has_chain = all(ex["c_ids"] is not None for ex in exs)
    res = {c: ([], [], []) for c in conds}
    ZH, ZT = [], []
    for s in range(0, len(exs), chunk):
        part = exs[s:s + chunk]
        z0, _, hx, hm, lens, mid = encode(enc_on, part, False, WANT_MID)
        zh = run_pred(z0, hx, hm, lens, mid)
        zt = encode(enc_tg, part, True)[1] if has_chain else None
        ZH.append(zh.float().cpu())
        if zt is not None:
            ZT.append(zt.cpu())
        B = z0.shape[0]
        task = torch.arange(R, device=DEV).repeat(B)
        for c in conds:
            st = {"pred": zh, "copy": z0, "oracle": zt}[c]
            zc = st[:, None, :].expand(B, R, -1).reshape(B * R, -1)
            with amp():
                outs = greedy(zc, task, hx, hm)
            for i, ex in enumerate(part):
                if ex["chain"] is not None:
                    got = last_val(tok.decode(outs[i * R], skip_special_tokens=True).strip())
                    res[c][0].append(float(got is not None and got == last_val(ex["chain"])))
                g, a = num(tok.decode(outs[i * R + 1], skip_special_tokens=True)), num(ex["answer"])
                res[c][1].append(float(g is not None and a is not None and g == a))
            if zt is not None:
                res[c][2].extend((st.float() * zt).sum(-1).cpu().tolist())
    out = {c: tuple(np.array(v) for v in res[c]) for c in conds}
    return out, torch.cat(ZH), (torch.cat(ZT) if ZT else None)

def mean(a):
    return float(a.mean()) if len(a) else float("nan")

# ---------------------------------------------------------------- optimiser and schedule
total = max(1, args.examples // args.bs)
warm_steps = total if args.freeze_encoder else int(total * args.warm_frac)
lr_pred = args.lr_pred or {"s": 2e-4, "m": 1e-4, "top4": 2e-5, "none": 0.0}[args.size]
groups = [{"params": list(dec.parameters()), "lr": args.lr_dec, "base": args.lr_dec, "name": "dec"},
          {"params": [p for _, p in lora_on], "lr": 0.0, "base": args.lr_lora, "name": "lora"}]
if pred is not None:
    groups.append({"params": list(pred.parameters()), "lr": lr_pred, "base": lr_pred, "name": "pred"})
opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.98))
all_params = [p for g in groups for p in g["params"]]

def sched(s):
    warm = min(1000, max(1, total // 10))
    if s < warm:
        return (s + 1) / warm
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

# reference statistics before any training: the encoder still equals the original backbone
ref, _, ZT0 = evaluate(val_all[:500], ["copy"])
STD0 = float(ZT0.std(0).mean())
print(f"start: copy cosine cos(z0, z*) {mean(ref['copy'][2]):.3f} | target spread (mean per-dimension std) {STD0:.5f}", flush=True)
print(f"schedule: {total} steps of {args.bs}; encoder frozen for the first {warm_steps} steps"
      + ("" if warm_steps >= total else f", then LoRA + EMA target (tau {args.tau})"), flush=True)

# ---------------------------------------------------------------- training
step, t0, order, ptr = 0, time.time(), list(range(len(train))), 0
random.shuffle(order)
milestones = {max(1, total * i // 4) for i in (1, 2, 3, 4)}
unfrozen = False
while step < total:
    if ptr + args.bs > len(order):
        random.shuffle(order); ptr = 0
    exs = [train[i] for i in order[ptr:ptr + args.bs]]; ptr += args.bs
    phase2 = step >= warm_steps
    if phase2 and not unfrozen:
        unfrozen = True
        for _, p in lora_on:
            p.requires_grad_(True)
        print(f"--- step {step}: encoder unfrozen (LoRA), target encoder now follows by EMA ---", flush=True)
    m = sched(step)
    for g in opt.param_groups:
        g["lr"] = g["base"] * m * (min(1.0, (step - warm_steps + 1) / 500) if g["name"] == "lora" else 1.0) if (g["name"] != "lora" or phase2) else 0.0
    dec.train()
    if pred is not None:
        pred.train()
    z0, _, hx, hm, lens, mid = encode(enc_on, exs, False, WANT_MID, grad=phase2)
    zt = encode(enc_tg, exs, True)[1]                         # stop-gradient target
    zh = run_pred(z0, hx, hm, lens, mid)
    cos = (zh * zt).sum(-1)
    l_lat = args.k * (1 - cos).mean() if pred is not None else zh.new_zeros(())
    l_dec = dec_loss(zh, hx, hm, exs) + dec_loss(zt, hx.detach(), hm, exs)   # second term trains the decoder only
    loss = l_lat + args.lam_dec * l_dec
    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True); step += 1; continue
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_([p for p in all_params if p.grad is not None], 1.0)
    opt.step()
    if phase2:
        with torch.no_grad():
            for (_, po), pt in zip(lora_on, lora_tg):
                pt.mul_(args.tau).add_(po.detach(), alpha=1 - args.tau)
    step += 1
    if step % 100 == 0:
        print(f"step {step}/{total} {'P2' if phase2 else 'P1'} lat {float(l_lat):.4f} dec {float(l_dec):.4f} cos {cos.mean().item():.4f} "
              f"gn {gn.item():.2f} {time.time() - t0:.0f}s", flush=True)
    if step in milestones:
        r1, _, _ = evaluate(val_one[:500], ["pred", "oracle"])
        r2, _, zt2 = evaluate(val_all[:500], ["pred", "copy"])
        print(f"== {100 * step // total}%: one-step step-acc pred {mean(r1['pred'][0]):.3f} oracle {mean(r1['oracle'][0]):.3f} "
              f"answer pred {mean(r1['pred'][1]):.3f} | all-lengths answer pred {mean(r2['pred'][1]):.3f} | cos pred {mean(r2['pred'][2]):.3f} "
              f"copy {mean(r2['copy'][2]):.3f} | target spread ratio {float(zt2.std(0).mean()) / STD0:.3f}", flush=True)

torch.save({"dec": dec.state_dict(), "pred": pred.state_dict() if pred is not None else None,
            "lora": {n: p.detach().cpu() for n, p in lora_on}, "args": vars(args)}, f"p3full_{TAG}.pt")

# ---------------------------------------------------------------- final report
print(f"\nFINAL EVALUATION, size {args.size}" + (" (decoder-only control)" if pred is None else ""), flush=True)
for name, vs in (("held-out GSM8k-Aug, one-step (n = 1)", val_one), ("held-out GSM8k-Aug, all chain lengths", val_all)):
    r, zh_, zt_ = evaluate(vs, ["oracle", "pred", "copy"])
    print(f"{name}: {len(vs)} problems")
    for c in ("oracle", "pred", "copy"):
        print(f"  {c:6s} step numeric match {mean(r[c][0]):.3f} | answer acc {mean(r[c][1]):.3f} | cos to target {mean(r[c][2]):.3f}", flush=True)
    d_p, d_c = 1 - mean(r["pred"][2]), 1 - mean(r["copy"][2])
    print(f"  skill score {1 - d_p / max(1e-9, d_c):.3f} | spread ratio: target {float(zt_.std(0).mean()) / STD0:.3f}, "
          f"predicted {float(zh_.std(0).mean()) / STD0:.3f} (collapse flag below 0.07)")
    if vs is val_one:
        p1 = mean(r["pred"][0])
        print(f"  registered P3 gate, one-step decoded accuracy at least 0.80: {'PASS' if p1 >= 0.80 else 'FAIL'} ({p1:.3f})")
    else:
        for lo, hi, lab in ((1, 1, "n=1"), (2, 2, "n=2"), (3, 3, "n=3"), (4, 99, "n>=4")):
            idx = [i for i, e in enumerate(vs) if lo <= e["n"] <= hi]
            if idx:
                print(f"     {lab:5s} n={len(idx):4d} answer acc: oracle {r['oracle'][1][idx].mean():.3f} pred {r['pred'][1][idx].mean():.3f} copy {r['copy'][1][idx].mean():.3f}")

rt, _, _ = evaluate(test, ["pred", "copy"])
sub = [t for t in test if t["c_ids"] is not None]
rs, _, _ = evaluate(sub, ["oracle", "pred"])
print(f"\nGSM8K TEST SET, answer accuracy (exact match of the final answer), {len(test)} questions")
print(f"  system (predicted state) {mean(rt['pred'][1]):.4f}  =  {int(rt['pred'][1].sum())} / {len(test)}")
print(f"  copy (question state)    {mean(rt['copy'][1]):.4f}  =  {int(rt['copy'][1].sum())} / {len(test)}")
print(f"  on the {len(sub)} questions with a usable reference chain: oracle (true state) {mean(rs['oracle'][1]):.4f} | system {mean(rs['pred'][1]):.4f}")
print(f"saved p3full_{TAG}.pt")
