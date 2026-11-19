import json, re, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
p = sys.argv[1]; n = 60
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16).cuda().eval()
rows = [json.loads(l)["messages"] for l in open("gsm8k_test.jsonl")][:n]
pat = re.compile(r"\n#### (.+)$"); has = end = ok = limit = 0; shown = None
for r in rows:
    x = tok.apply_chat_template(r[:2], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    out = m.generate(**x, max_new_tokens=128, do_sample=False, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
    new = out[0][x["input_ids"].shape[1]:]
    g = tok.decode(new, skip_special_tokens=True)
    gt = pat.search(r[2]["content"]).group(1); e = pat.search(g)
    has += "####" in g; end += bool(e); ok += bool(e and e.group(1) == gt); f = re.search(r"#### (.+?)(?:\n|$)", g); fok = globals().get("fok", 0) + bool(f and f.group(1) == gt); limit += len(new) >= 128
    if shown is None and not e: shown = g[-220:]
print("first-marker correct:", fok); print(f"{p}: n={n} has_marker={has} marker_at_end={end} correct={ok} hit_128_limit={limit}")
print("FIRST FAILURE TAIL:", repr(shown))
