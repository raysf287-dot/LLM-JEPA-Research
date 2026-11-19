import random, re, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from parse_gsm import load_split, validate, group_bounds

M = "meta-llama/Llama-3.2-1B-Instruct"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
K, TRIES, T_LIST = 5, 50, (2, 3)
NUM = re.compile(r"\d+(?:\.\d+)?|\.\d+")

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
NL = tok("\n", add_special_tokens=False).input_ids
assert len(NL) == 1, NL
NL = NL[0]

def encode_segments(segs):
    ids, pos = [tok.bos_token_id], []
    for s in segs:
        ids += tok(s, add_special_tokens=False).input_ids + [NL]
        pos.append(len(ids) - 1)
    return ids, pos

@torch.no_grad()
def states(segs):
    ids, pos = encode_segments(segs)
    x = torch.tensor([ids], device="cuda")
    h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
    return torch.nn.functional.normalize(h[pos].float(), dim=-1).cpu().numpy()

def deps(eqs):
    """deps[j] = set of i<j whose result appears as a number in the left side of j."""
    parts = [e[2:-2].rsplit("=", 1) for e in eqs]
    vals = []
    for _, r in parts:
        try:
            vals.append(float(r.replace(",", "")))
        except ValueError:
            vals.append(None)
    out = []
    for j, (l, _) in enumerate(parts):
        nums = {float(x) for x in NUM.findall(l)}
        out.append({i for i in range(j) if vals[i] is not None and vals[i] in nums})
    return out

def alt_orders(eqs, rng):
    n, d = len(eqs), deps(eqs)
    base, found = tuple(range(n)), set()
    for _ in range(TRIES):
        remaining, done = set(range(n)), []
        while remaining:
            ready = sorted(j for j in remaining if d[j] <= set(done))
            j = rng.choice(ready)
            done.append(j)
            remaining.remove(j)
        o = tuple(done)
        if o != base:
            found.add(o)
        if len(found) >= K:
            break
    return sorted(found)

rows = load_split()
valid = [ex for ex, _ in (validate(q, c, a) for q, c, a in rows) if ex]
random.Random(82).shuffle(valid)
rng = random.Random(82)
nrng = np.random.default_rng(82)

for T in T_LIST:
    pick = [e for e in valid if e["n"] >= T][:N]
    ref, within_pp = [], []
    for ex in pick:
        eqs, bounds = ex["eqs"], group_bounds(ex["n"], T)
        def grp(order):
            e2 = [eqs[i] for i in order]
            return [" ".join(e2[a:b]) for a, b in bounds]
        zr = states([ex["question"]] + grp(range(ex["n"])))[1:]
        ref.append(zr)
        alts = alt_orders(eqs, rng)
        if alts:
            dists = [[1 - float(zr[t] @ states([ex["question"]] + grp(o))[1 + t]) for t in range(T)] for o in alts]
            within_pp.append(np.mean(dists, axis=0))
    R = np.stack(ref)
    pairs = nrng.integers(0, len(R), size=(5000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    between = np.array([1 - np.einsum("nd,nd->n", R[pairs[:, 0], t], R[pairs[:, 1], t]).mean() for t in range(T)])
    print(f"\nT={T}: {len(pick)} problems with n>={T}; with an alternative valid ordering: {len(within_pp)} ({len(within_pp) / len(pick):.1%})")
    if within_pp:
        w = np.mean(within_pp, axis=0)
        ratio = float(w.mean() / between.mean())
        print("(iii) within-problem cosine distance per t:", w.round(4), "| between-problem:", between.round(4))
        print("      ratio per t:", (w / between).round(3), "| overall ratio:", round(ratio, 3),
              "| proposed gate <=0.5:", "PASS" if ratio <= 0.5 else "FAIL")
