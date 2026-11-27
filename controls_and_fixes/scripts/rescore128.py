"""Re-score saved greedy generations at a 128-token limit.
Greedy decoding is deterministic, so a 128-token run produces the first 128 tokens of a longer greedy run. We
truncate each saved output to 128 tokens and re-extract the answer exactly as gen_val.py / eval_ood.py do.
Usage: python rescore128.py [glob ...]   (default: logs/valgen_*_256.jsonl logs/ood_*_*.jsonl)
"""
import glob, json, re, sys
from collections import defaultdict
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
strict = re.compile(r"\n#### (.+)$")
pats = sys.argv[1:] or ["logs/valgen_*_256.jsonl", "logs/ood_*_*.jsonl"]
files = sorted({f for p in pats for f in glob.glob(p)})
val, ood = {}, defaultdict(dict)
for f in files:
    R = [json.loads(l) for l in open(f)]
    ok = 0
    for r in R:
        ids = tok(r["output"], add_special_tokens=False)["input_ids"]
        t = tok.decode(ids[:128], skip_special_tokens=True).strip()
        if "gold" in r:                                            # GSM8K validation (gen_val.py rule)
            m = strict.search(t); ok += bool(m) and m.group(1).strip() == r["gold"]
        else:                                                      # OOD (eval_ood.py rule)
            m = re.search(r"\n?#### ([^\n]+)", t); p = None
            if m:
                try: p = float(m.group(1).replace(",", "").replace("$", "").strip())
                except ValueError: p = None
            ok += p is not None and abs(p - r["target"]) <= 1e-4 * max(1, abs(r["target"]))
    acc = ok / len(R) * 100
    if "valgen_" in f: val[f.split("valgen_")[1].rsplit("_", 1)[0]] = acc
    else:
        tag, ds = f.split("ood_")[1].rsplit(".", 1)[0].rsplit("_", 1); ood[tag][ds] = acc
print(f"{'run':14s} {'GSM8K val@128':>13s} {'svamp':>6s} {'gsmhard':>7s} {'multiarith':>10s}")
for t in sorted(set(val) | set(ood)):
    o = ood.get(t, {})
    print(f"{t:14s} {val.get(t, float('nan')):13.1f} {o.get('svamp', float('nan')):6.1f} {o.get('gsmhard', float('nan')):7.1f} {o.get('multiarith', float('nan')):10.1f}")
