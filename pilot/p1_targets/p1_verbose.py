import json, random, re, sys
from collections import Counter
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from parse_gsm import load_split, validate, group_bounds, group_texts

M = "meta-llama/Llama-3.2-1B-Instruct"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
T_LIST = (2, 3)
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$])")

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

def nl_groups(sents, T):
    return [" ".join(sents[a:b]) for a, b in group_bounds(len(sents), T)]

d = json.load(open(hf_hub_download("zen-E/GSM8k-Aug-NL", "gsm8k_aug_nl.json", repo_type="dataset"), encoding="utf-8"))
nl_count = Counter(d["question"])
rows = load_split()
cc = Counter(q for q, _, _ in rows)
comp = {}
for q, c, a in rows:
    if cc[q] == 1:
        ex, _ = validate(q, c, a)
        if ex:
            comp[q] = ex
pairs = []
for q, c in zip(d["question"], d["cot"]):
    if nl_count[q] == 1 and q in comp:
        sents = [s.strip() for s in SPLIT.split(c.strip()) if s.strip()]
        pairs.append((comp[q], sents))
random.Random(82).shuffle(pairs)
print(f"paired problems (unique questions, valid compact chain): {len(pairs)}")
nrng = np.random.default_rng(82)

def stats(Z):
    cos0 = np.einsum("nd,ntd->nt", Z[:, 0], Z[:, 1:])
    D = np.diff(Z, axis=1)
    D = D / (np.linalg.norm(D, axis=-1, keepdims=True) + 1e-12)
    return cos0, np.einsum("ntd,ntd->nt", D[:, 1:], D[:, :-1])

for T in T_LIST:
    pool = [p for p in pairs if p[0]["n"] >= T and len(p[1]) >= T]
    pick = pool[:N]
    aligned = float(np.mean([p[0]["n"] == len(p[1]) for p in pick]))
    Zc = np.stack([states([ex["question"]] + group_texts(ex, T)) for ex, _ in pick])
    Zv = np.stack([states([ex["question"]] + nl_groups(s, T)) for ex, s in pick])
    c0c, dcc = stats(Zc)
    c0v, dcv = stats(Zv)
    cross = 1 - np.einsum("nd,nd->n", Zc[:, -1], Zv[:, -1])
    pr = nrng.integers(0, len(Zc), size=(5000, 2))
    pr = pr[pr[:, 0] != pr[:, 1]]
    between = 1 - np.einsum("nd,nd->n", Zc[pr[:, 0], -1], Zc[pr[:, 1], -1])
    print(f"\nT={T}: {len(pick)} problems (n_equations>={T} and n_sentences>={T}); sentence count == equation count: {aligned:.1%}")
    print("(i)  mean cos(z0,z*_t)   compact:", round(float(c0c.mean()), 4), "| verbose:", round(float(c0v.mean()), 4))
    print("(ii) displacement cosine compact:", round(float(dcc.mean()), 4), "| verbose:", round(float(dcv.mean()), 4))
    print("(v)  cross-format distance of z*_T (compact vs verbose):", round(float(cross.mean()), 4),
          "| between-problem (compact):", round(float(between.mean()), 4),
          "| ratio:", round(float(cross.mean() / between.mean()), 3))
