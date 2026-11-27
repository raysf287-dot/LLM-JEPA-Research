"""Generate on the 1,500 held-out validation examples and SAVE every output.
Usage: python gen_val.py ./ft_V_base logs/valgen_V_base [limits, default 128,256]
Writes <prefix>_<limit>.jsonl for each limit (one JSON object per validation example).
Validation only; never touches gsm8k_test.jsonl.
"""
import json, re, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path, prefix = sys.argv[1], sys.argv[2]
LIMITS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "128,256").split(",")]
strict_pat = re.compile(r"\n#### (.+)$")
rows = [json.loads(l)["messages"] for l in open("gsm8k_val.jsonl")]
gold_text = [r[2]["content"] for r in rows]
gold = [strict_pat.search(g).group(1).strip() for g in gold_text]
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
prompts = [tok.apply_chat_template([x for x in r if x["role"] != "assistant"], tokenize=False, add_generation_prompt=True) for r in rows]
order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))

for lim in LIMITS:
    t0 = time.time(); outs = [None] * len(rows); eos = [None] * len(rows)
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        x = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            o = m.generate(**x, do_sample=False, max_new_tokens=lim, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        new = o[:, x["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            outs[i] = tok.decode(new[j], skip_special_tokens=True).strip()
            eos[i] = bool((new[j] == tok.eos_token_id).any())
    ok = 0
    with open(f"{prefix}_{lim}.jsonl", "w") as f:
        for i in range(len(rows)):
            mt = strict_pat.search(outs[i]); pred = mt.group(1).strip() if mt else None
            c = pred == gold[i]; ok += c
            f.write(json.dumps({"idx": i, "question": rows[i][1]["content"], "gold_solution": gold_text[i], "gold": gold[i],
                                "output": outs[i], "pred": pred, "correct": c, "eos_emitted": eos[i]}) + "\n")
    print(f"RESULT {path} limit {lim}: strict {ok/len(rows):.4f} ({ok}/{len(rows)}) | {time.time()-t0:.0f}s", flush=True)
