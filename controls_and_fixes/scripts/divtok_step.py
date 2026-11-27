"""Candidate 1, revised loss: divergence-token contrast on typed counterfactual steps.
For a pair (context, correct step, corrupted step) the two token sequences are identical up to the first token where
they differ (the error decision point: which number or operator to write). At that position only:
    L_con = max(0, m - [log p(correct token) - log p(wrong token)])        (hinge: stops pushing once the margin is met)
plus the usual SFT loss on the correct step:
    L = alpha * NLL(correct step tokens) + lam * L_con
No reference model and no sequence-level push-down of the rejected text (avoids DPO's likelihood displacement on
near-identical pairs). One forward pass per pair (the wrong token's log-prob is read at the same position).
Usage: python divtok_step.py --base ./ft_X --pairs pairs.jsonl --out ./ft_Y [--margin 1.0 --lam 1.0 --alpha 1.0
        --lr 1e-6 --epochs 1 --batch 16 --micro 16 --max_pairs 0 --seed 82 --device cuda]
"""
import argparse, json, math, random, time, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=float, default=1.0); ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--micro", type=int, default=16); ap.add_argument("--max_pairs", type=int, default=0)
ap.add_argument("--seed", type=int, default=82); ap.add_argument("--device", default="cuda")
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--bos", type=int, default=1, help="1 = tokenise like finetune_step.py / gen_val.py (template BOS + tokenizer BOS)")
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device(a.device); amp = dev.type == "cuda"
tok = AutoTokenizer.from_pretrained(a.base)
pol = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32).to(dev)
pol.config.use_cache = False; pol.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

P = [json.loads(l) for l in open(a.pairs)]; random.shuffle(P)
if a.max_pairs: P = P[:a.max_pairs]


def encode(p):
    """-> (ids of context+correct step, n_context, divergence index d, wrong token id) or None."""
    ctx = tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True) + p["prefix"]
    n = len(tok(ctx, add_special_tokens=bool(a.bos))["input_ids"])
    c = tok(ctx + p["chosen"], add_special_tokens=bool(a.bos))["input_ids"]
    r = tok(ctx + p["rejected"], add_special_tokens=bool(a.bos))["input_ids"]
    d = next((i for i in range(min(len(c), len(r))) if c[i] != r[i]), None)
    if d is None or d < n: return None                        # identical, or tokenisation changed inside the context
    if len(c) > a.max_len:
        drop = len(c) - a.max_len; c = c[drop:]; n = max(1, n - drop); d -= drop
        if d < 1: return None
    return c, n, d, r[d]


enc = [e for e in (encode(p) for p in P) if e is not None]
print(f"[divtok] {len(enc)} usable pairs of {len(P)}", flush=True)
opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=0.0)
steps = math.ceil(len(enc) / a.batch) * a.epochs; warm = max(1, int(0.05 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
pol.train(); t0 = time.time(); step = 0; H = {"gap": [], "sat": [], "nll": []}
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
for ep in range(a.epochs):
    for b0 in range(0, len(enc), a.batch):
        B = enc[b0:b0 + a.batch]; opt.zero_grad(set_to_none=True)
        for m0 in range(0, len(B), a.micro):
            M = B[m0:m0 + a.micro]; L = max(len(x[0]) for x in M)
            ids = torch.full((len(M), L), pad, dtype=torch.long); att = torch.zeros_like(ids); msk = torch.zeros_like(ids, dtype=torch.bool)
            for k, (c, n, d, w) in enumerate(M):
                ids[k, :len(c)] = torch.tensor(c); att[k, :len(c)] = 1; msk[k, n:len(c)] = True
            ids, att, msk = ids.to(dev), att.to(dev), msk.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = pol(input_ids=ids, attention_mask=att).logits.float()
            lsm = F.log_softmax(logits[:, :-1], -1)
            tgt = torch.gather(lsm, 2, ids[:, 1:, None]).squeeze(-1); m = msk[:, 1:].float()
            nll = -(tgt * m).sum(-1) / m.sum(-1)                                        # per-pair mean NLL of the step
            pos = torch.tensor([d - 1 for _, _, d, _ in M], device=dev)                 # logits at d-1 predict token d
            wid = torch.tensor([w for *_, w in M], device=dev); cid = ids[torch.arange(len(M), device=dev), pos + 1]
            row = lsm[torch.arange(len(M), device=dev), pos]
            gap = row.gather(1, cid[:, None]).squeeze(1) - row.gather(1, wid[:, None]).squeeze(1)
            con = F.relu(a.margin - gap)
            loss = (a.alpha * nll + a.lam * con).mean() * (len(M) / len(B))
            loss.backward()
            H["gap"].append(gap.mean().item()); H["sat"].append((gap >= a.margin).float().mean().item()); H["nll"].append(nll.mean().item())
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if step % 50 == 1 or step == steps:
            w = lambda k: sum(H[k][-50:]) / len(H[k][-50:])
            print(f"[divtok] step {step}/{steps} gap {w('gap'):.3f} margin-met {w('sat'):.3f} nll {w('nll'):.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
pol = pol.to(torch.bfloat16); pol.save_pretrained(a.out); tok.save_pretrained(a.out)
print(f"[divtok] saved {a.out} after {step} steps, {time.time() - t0:.0f}s")
