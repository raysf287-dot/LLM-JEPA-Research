"""Baseline for P3 on GSM8K: the backbone itself, LoRA fine-tuned to answer directly.

--mode nocot  question -> answer            (the no-CoT baseline; closest to ordinary fine-tuning)
--mode cot    question -> equation chain -> answer   (CoT-SFT; also gives a measured CoT latency)

Same data split, seed, LoRA rank and training budget as p3_full.py. Scored by exact match of the
final answer on held-out GSM8k-Aug problems and on the 1,319 GSM8K test questions.
"""
import argparse, math, random, time
from fractions import Fraction
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import parse_gsm
from parse_gsm import validate, group_texts

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="nocot", choices=["nocot", "cot"])
ap.add_argument("--ntrain", type=int, default=0)
ap.add_argument("--nval", type=int, default=2000)
ap.add_argument("--examples", type=int, default=1000000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lora_r", type=int, default=128)
ap.add_argument("--lora_alpha", type=int, default=32)
ap.add_argument("--seed", type=int, default=82)
ap.add_argument("--nlat", type=int, default=220)
ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
args = ap.parse_args()
random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"

def amp():
    return torch.autocast(DEV, dtype=torch.bfloat16, enabled=AMP)

tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(DEV).eval()
for p in base.parameters():
    p.requires_grad_(False)
NLID = tok("\n", add_special_tokens=False).input_ids[0]
BOS, EOS = tok.bos_token_id, tok.eos_token_id

def ids_of(s):
    return tok(s, add_special_tokens=False).input_ids

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
for ex in valid:                                             # same split rule as p3.py and p3_full.py
    chain = group_texts(ex, 1)[0]
    c_ids, a_ids = ids_of(chain), ids_of(ex["answer"])
    if max(len(c_ids), len(a_ids)) + 1 > 120:
        continue
    ex["q_ids"] = ids_of(ex["question"])
    ex["t_ids"] = (c_ids + [NLID] if args.mode == "cot" else []) + a_ids + [EOS]
    if len(val_all) < args.nval:
        val_all.append(ex)
    elif ex["n"] == 1 and len(val_one) < args.nval:
        val_one.append(ex)
    elif len(train) < cap:
        train.append(ex)
    if len(train) >= cap and len(val_one) >= args.nval:
        break
test = [{"q_ids": ids_of(q), "answer": str(a).replace(",", "").strip()} for q, c, a in parse_gsm.load_split("gsm8k_test.json")]
print(f"data ready in {time.time() - t0:.0f}s: mode {args.mode}, train {len(train)}, val_all {len(val_all)}, val_one {len(val_one)}, GSM8K test {len(test)}", flush=True)

lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(base, lcfg)
llm = model.base_model.model
body, head = llm.model, llm.lm_head
lora = [p for n, p in model.named_parameters() if "lora_" in n]
print(f"LoRA {sum(p.numel() for p in lora) / 1e6:.1f}M trainable (rank {args.lora_r})", flush=True)
opt = torch.optim.AdamW(lora, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
total = max(1, args.examples // args.bs)

def sched(s):
    warm = min(1000, max(1, total // 10))
    if s < warm:
        return (s + 1) / warm
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

def num(s):
    try:
        return Fraction(s.replace(",", "").replace("$", "").strip())
    except Exception:
        return None

MAXNEW = 160 if args.mode == "cot" else 16

@torch.no_grad()
def generate(exs, chunk=64):
    """Greedy generation; returns the decoded answer string for each problem."""
    outs = []
    for s in range(0, len(exs), chunk):
        part = exs[s:s + chunk]
        seqs = [[BOS] + ex["q_ids"] + [NLID] for ex in part]
        L = max(len(x) for x in seqs)
        x = torch.full((len(seqs), L), EOS, dtype=torch.long); m = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, q in enumerate(seqs):                         # left padding, so generation continues from the last real token
            x[i, L - len(q):] = torch.tensor(q); m[i, L - len(q):] = 1
        with amp():
            g = model.generate(input_ids=x.to(DEV), attention_mask=m.to(DEV), max_new_tokens=MAXNEW, do_sample=False,
                               temperature=None, top_p=None, eos_token_id=EOS, pad_token_id=EOS, use_cache=True)
        for row in g[:, L:].tolist():
            row = row[:row.index(EOS)] if EOS in row else row
            text = tok.decode(row, skip_special_tokens=True).strip()
            outs.append(text.split("\n")[-1].strip() if text else "")
    return outs

def accuracy(exs):
    got = generate(exs)
    ok = np.array([float(num(g) is not None and num(e["answer"]) is not None and num(g) == num(e["answer"])) for g, e in zip(got, exs)])
    return ok

step, t0, order, ptr = 0, time.time(), list(range(len(train))), 0
random.shuffle(order)
milestones = {max(1, total * i // 4) for i in (1, 2, 3, 4)}
gn = 0.0
while step < total:
    if ptr + args.bs > len(order):
        random.shuffle(order); ptr = 0
    exs = [train[i] for i in order[ptr:ptr + args.bs]]; ptr += args.bs
    seqs = [[BOS] + ex["q_ids"] + [NLID] + ex["t_ids"] for ex in exs]
    L = max(len(x) for x in seqs)
    x = torch.full((len(seqs), L), EOS, dtype=torch.long); m = torch.zeros((len(seqs), L), dtype=torch.long)
    lab = torch.full((len(seqs), L), -100)
    for i, (q, ex) in enumerate(zip(seqs, exs)):
        x[i, :len(q)] = torch.tensor(q); m[i, :len(q)] = 1
        st = len(q) - len(ex["t_ids"])
        lab[i, st:len(q)] = torch.tensor(ex["t_ids"])
    x, m, lab = x.to(DEV), m.to(DEV), lab.to(DEV)
    for g in opt.param_groups:
        g["lr"] = args.lr * sched(step)
    with amp():
        h = body(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state
        sel = lab[:, 1:] != -100                             # position i predicts token i + 1
        logits = head(h[:, :-1][sel])
    loss = F.cross_entropy(logits.float(), lab[:, 1:][sel])
    opt.zero_grad(set_to_none=True)
    if torch.isfinite(loss):
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(lora, 1.0)
        opt.step()
    step += 1
    if step % 100 == 0:
        print(f"step {step}/{total} loss {loss.item():.4f} gn {float(gn):.2f} lr {args.lr * sched(step):.2e} {time.time() - t0:.0f}s", flush=True)
    if step in milestones:
        print(f"== {100 * step // total}%: answer accuracy one-step {accuracy(val_one[:500]).mean():.3f} | all lengths {accuracy(val_all[:500]).mean():.3f}", flush=True)

torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}, f"p3_{args.mode}_lora.pt")
name = "no-CoT fine-tuned backbone" if args.mode == "nocot" else "CoT-SFT backbone"
print(f"\nFINAL EVALUATION, {name}", flush=True)
a1, a2 = accuracy(val_one), accuracy(val_all)
print(f"held-out GSM8k-Aug: answer accuracy one-step {a1.mean():.3f} ({len(val_one)}) | all chain lengths {a2.mean():.3f} ({len(val_all)})")
for lo, hi, lb in ((1, 1, "n=1"), (2, 2, "n=2"), (3, 3, "n=3"), (4, 99, "n>=4")):
    idx = [i for i, e in enumerate(val_all) if lo <= e["n"] <= hi]
    if idx:
        print(f"     {lb:5s} n={len(idx):4d} answer acc {a2[idx].mean():.3f}")
at = accuracy(test)
print(f"\nGSM8K TEST SET, answer accuracy (exact match of the final answer), {len(test)} questions")
print(f"  {name}: {at.mean():.4f}  =  {int(at.sum())} / {len(test)}")

def timed(fn):
    if AMP:
        torch.cuda.synchronize()
    t = time.perf_counter(); out = fn()
    if AMP:
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000
try:
    ts = [timed(lambda: generate([ex]))[1] for ex in test[:args.nlat]]
    print(f"\nLATENCY, batch size 1, {len(ts) - 20} test questions after 20 warm-up: median {np.median(ts[20:]):.1f} ms per question")
except Exception as err:
    print("latency measurement failed:", repr(err))
print(f"saved p3_{args.mode}_lora.pt")
