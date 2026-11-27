import json, random, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
rows = [json.loads(l) for l in open("gsm8k_train.jsonl")]
idx = list(range(len(rows))); random.Random(0).shuffle(idx); val = set(idx[:1500])
strip = lambda s: re.sub(r"<<[^>]*>>", "", s)
NUMR = r"\$?-?\d[\d,]*(?:\.\d+)?%?"
EQ = re.compile(r"[\$\(\d][\d.,$%()+\-*/x×÷^\s]*?(?:\s*=\s*" + NUMR + r")+")
def split(c):
    b, a = c.rsplit("\n#### ", 1); return b, a
def v_eq(b):
    e = [m.group(0).strip() for l in b.split("\n") for m in EQ.finditer(l)]
    return "\n".join(e) if e else b
def v_short(b):
    k = [l for l in b.split("\n") if "=" in l]
    return "\n".join(k) if k else b
outs = {n: open(f"gsm8k_trainsplit_{n}.jsonl", "w") for n in ("short", "eq")}
va = open("gsm8k_val.jsonl", "w"); tn = open("gsm8k_trainsplit_noanno.jsonl", "w")
L = {"noanno": [], "short": [], "eq": []}; fb = {"short": 0, "eq": 0}; shown = 0
for i, d in enumerate(rows):
    if i in val: va.write(json.dumps(d) + "\n"); continue
    b, a = split(strip(d["messages"][2]["content"]))
    for name, fn in (("noanno", lambda x: x), ("short", v_short), ("eq", v_eq)):
        nb = fn(b)
        if name != "noanno" and nb == b and fn(b) == b and name in fb and (name == "eq" and not EQ.search(b) or name == "short" and "=" not in b): fb[name] += 1
        c = nb + "\n#### " + a
        L[name].append(len(tok(c)["input_ids"]))
        e = json.loads(json.dumps(d)); e["messages"][2]["content"] = c
        (tn if name == "noanno" else outs[name]).write(json.dumps(e) + "\n")
        if shown < 6 and i % 997 == 0: print(f"--- {name}\n{c}")
    if i % 997 == 0: shown += 3
import numpy as np
for n, v in L.items(): v = np.array(v); print(n, "n", len(v), "mean tokens", round(v.mean(), 1), "share >128", round((v > 128).mean() * 100, 1), "%")
print("fallbacks (no equation found):", fb)
