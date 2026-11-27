"""Candidate 1 training stage: step-level preference optimisation on typed counterfactual steps.
Starts from an SFT model (Given + RFT). Loss per pair (prefix, chosen step, rejected step):
    L = -log sigmoid( beta * [ (log pi(c) - log ref(c)) - (log pi(r) - log ref(r)) ] )  +  alpha * NLL_pi(c) / |c|
pi = trainable policy (fp32 master weights, bf16 autocast), ref = frozen copy of the starting model.
Only the step tokens are scored; the prefix (system, question, Given block, earlier steps) is context.
Usage: python dpo_step.py --base ./ft_X --pairs pairs.jsonl --out ./ft_Y [--beta 0.1 --alpha 1.0 --lr 1e-6
        --epochs 1 --batch 16 --micro 8 --max_pairs 0 --seed 82 --device cuda]
"""
import argparse, json, math, random, time, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--micro", type=int, default=8)
ap.add_argument("--max_pairs", type=int, default=0); ap.add_argument("--seed", type=int, default=82)
ap.add_argument("--device", default="cuda"); ap.add_argument("--max_len", type=int, default=512)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device(a.device); amp = dev.type == "cuda"

tok = AutoTokenizer.from_pretrained(a.base)
pol = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32).to(dev)
ref = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16 if amp else torch.float32).to(dev).eval()
for p in ref.parameters(): p.requires_grad_(False)
pol.config.use_cache = False; pol.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

P = [json.loads(l) for l in open(a.pairs)]
random.shuffle(P)
if a.max_pairs: P = P[:a.max_pairs]


def encode(p):
    ctx = tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True) + p["prefix"]
    c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    out = []
    for comp in (p["chosen"], p["rejected"]):
        full = tok(ctx + comp, add_special_tokens=False)["input_ids"]
        n = len(c_ids)
        assert full[:n] == c_ids, "prefix tokenisation changed at the step boundary"
        if len(full) > a.max_len:                               # keep the step; drop the oldest context tokens
            drop = len(full) - a.max_len; full = full[drop:]; n = max(1, n - drop)
        out.append((full, n))
    return out


def batch_logps(model, items):
    """items: list of (ids, n_prefix). Returns (sum logp of completion tokens, n completion tokens) per item."""
    L = max(len(i) for i, _ in items); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(items), L), pad, dtype=torch.long); att = torch.zeros_like(ids); msk = torch.zeros_like(ids, dtype=torch.bool)
    for k, (i, n) in enumerate(items):
        ids[k, :len(i)] = torch.tensor(i); att[k, :len(i)] = 1; msk[k, n:len(i)] = True
    ids, att, msk = ids.to(dev), att.to(dev), msk.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lp = torch.gather(F.log_softmax(logits, -1), 2, ids[:, 1:, None]).squeeze(-1)
    m = msk[:, 1:].float()
    return (lp * m).sum(-1), m.sum(-1)


opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=0.0)
steps = math.ceil(len(P) / a.batch) * a.epochs; warm = max(1, int(0.05 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
pol.train(); t0 = time.time(); step = 0; acc_hist = []; margin_hist = []
for ep in range(a.epochs):
    for b0 in range(0, len(P), a.batch):
        B = P[b0:b0 + a.batch]; opt.zero_grad(set_to_none=True)
        for m0 in range(0, len(B), a.micro):
            enc = [encode(p) for p in B[m0:m0 + a.micro]]
            items = [e[0] for e in enc] + [e[1] for e in enc]; k = len(enc)
            with torch.no_grad():
                r_lp, _ = batch_logps(ref, items)
            p_lp, n_tok = batch_logps(pol, items)
            pc, pr, rc, rr = p_lp[:k], p_lp[k:], r_lp[:k], r_lp[k:]
            z = a.beta * ((pc - rc) - (pr - rr))
            loss = (-F.logsigmoid(z) + a.alpha * (-pc / n_tok[:k])).mean() * (k / len(B))
            loss.backward()
            acc_hist.append((z > 0).float().mean().item()); margin_hist.append(z.mean().item())
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if step % 50 == 1 or step == steps:
            print(f"[dpo] step {step}/{steps} loss {loss.item() * len(B) / k:.4f} pref-acc {sum(acc_hist[-50:]) / len(acc_hist[-50:]):.3f} "
                  f"margin {sum(margin_hist[-50:]) / len(margin_hist[-50:]):.3f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
pol = pol.to(torch.bfloat16); pol.save_pretrained(a.out); tok.save_pretrained(a.out)
print(f"[dpo] saved {a.out} after {step} steps, {time.time() - t0:.0f}s")
