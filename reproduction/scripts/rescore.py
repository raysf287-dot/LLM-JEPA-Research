import json, re, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
strict_pat = re.compile(r"\n#### (.+)$"); first_pat = re.compile(r"#### ([^\n]+)")
rows = [json.loads(l)["messages"] for l in open("gsm8k_test.jsonl")]
gts = [strict_pat.search(r[2]["content"]).group(1) for r in rows]
for path in sys.argv[1:]:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    prompts = [tok.apply_chat_template([x for x in r if x["role"] != "assistant"], tokenize=False, add_generation_prompt=True) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i])); outs = [None] * len(rows); lim = 0
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        x = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            o = m.generate(**x, do_sample=False, max_new_tokens=128, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        new = o[:, x["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            outs[i] = tok.decode(new[j], skip_special_tokens=True).strip()
            lim += int(not (new[j] == tok.eos_token_id).any())
    strict = sum(bool(strict_pat.search(g)) and strict_pat.search(g).group(1) == gt for g, gt in zip(outs, gts))
    first = sum(bool(first_pat.search(g)) and first_pat.search(g).group(1).strip() == gt for g, gt in zip(outs, gts))
    n = len(rows)
    print(f"RESULT {path}: strict {strict/n:.4f} ({strict}/{n}) | first-marker {first/n:.4f} ({first}/{n}) | never_emitted_eos {lim} | {time.time()-t0:.0f}s", flush=True)
    json.dump(outs, open(f"logs/rescore_{path.strip('./').replace('/', '_')}.json", "w"))
    del m; torch.cuda.empty_cache()
