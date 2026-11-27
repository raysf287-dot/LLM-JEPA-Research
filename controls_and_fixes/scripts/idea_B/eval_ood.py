"""Out-of-distribution evaluation (evaluation only, never used for model selection): SVAMP (test, 300),
GSM-Hard (1,319), MultiArith (test, 180). Greedy, LIMIT new tokens (default 128), same system prompt as GSM8K training.
Correct if the '#### ' answer equals the target numerically (tolerance 1e-4 relative).
Usage: python eval_ood.py MODEL_DIR TAG [LIMIT=128]         -> 'OOD TAG svamp x | gsmhard y | multiarith z'
       python eval_ood.py --check                -> prints the schema and one example of each dataset
"""
import json, re, sys, torch
from datasets import load_dataset

SYS = "Answer the math question, show steps."


def load(name):
    if name == "svamp":
        d = load_dataset("ChilleD/SVAMP", split="test")
        q = lambda r: (r.get("question_concat") or (r["Body"].strip() + " " + r["Question"].strip())).strip()
        return [(q(r), float(r["Answer"])) for r in d]
    if name == "gsmhard":
        d = load_dataset("reasoning-machines/gsm-hard", split="train")
        return [(r["input"].strip(), float(r["target"])) for r in d]
    if name == "multiarith":
        d = load_dataset("ChilleD/MultiArith", split="test")
        return [(r["question"].strip(), float(r["final_ans"])) for r in d]


if sys.argv[1] == "--check":
    for n in ("svamp", "gsmhard", "multiarith"):
        D = load(n); print(n, len(D), D[0])
    sys.exit()

from transformers import AutoTokenizer, AutoModelForCausalLM
path, tag = sys.argv[1], sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 128
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
res = {}
for n in ("svamp", "gsmhard", "multiarith"):
    D = load(n)
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True) for q, _ in D]
    order = sorted(range(len(D)), key=lambda i: len(prompts[i])); outs = [None] * len(D)
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        x = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = m.generate(**x, do_sample=False, max_new_tokens=LIMIT, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for j, i in enumerate(idx): outs[i] = tok.decode(o[j, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    ok = 0
    with open(f"logs/ood_{tag}_{n}.jsonl", "w") as f:
        for (q, y), t in zip(D, outs):
            mm = re.search(r"\n?#### ([^\n]+)", t); p = None
            if mm:
                try: p = float(mm.group(1).replace(",", "").replace("$", "").strip())
                except ValueError: p = None
            c = p is not None and abs(p - y) <= 1e-4 * max(1, abs(y)); ok += c
            f.write(json.dumps({"question": q, "target": y, "output": t, "pred": p, "correct": c}) + "\n")
    res[n] = ok / len(D) * 100
print(f"OOD {tag} " + " | ".join(f"{k} {v:.1f}" for k, v in res.items()))
