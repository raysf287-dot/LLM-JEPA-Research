import random, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from parse_gsm import load_split, validate, group_bounds, group_texts

M = "meta-llama/Llama-3.2-1B-Instruct"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
T_LIST = (2, 3)

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

rows = load_split()
valid = [ex for ex, _ in (validate(q, c, a) for q, c, a in rows) if ex]
random.Random(82).shuffle(valid)

ids, pos = encode_segments([valid[0]["question"]] + group_texts(valid[0], 2))
print("delimiter tokens:", tok.convert_ids_to_tokens([ids[i] for i in pos]))

for T in T_LIST:
    pick = [e for e in valid if e["n"] >= T][:N]
    Z, Zo = [], []
    for ex in pick:
        g = group_texts(ex, T)
        Z.append(states([ex["question"]] + g))
        a, b = group_bounds(ex["n"], T)[-1]
        last = " ".join(ex["eqs"][a:b - 1])
        go = g[:-1] + ([last] if last else [])
        Zo.append(states([ex["question"]] + go)[-1])
    Z, Zo = np.stack(Z), np.stack(Zo)
    cos0 = np.einsum("nd,ntd->nt", Z[:, 0], Z[:, 1:])
    D = np.diff(Z, axis=1)
    D = D / (np.linalg.norm(D, axis=-1, keepdims=True) + 1e-12)
    dcos = np.einsum("ntd,ntd->nt", D[:, 1:], D[:, :-1])
    omit = np.einsum("nd,nd->n", Z[:, -1], Zo)
    print(f"\nT={T}: {len(pick)} problems with n>={T}")
    print("(i)  mean cos(z0,z*_t) per t:", cos0.mean(0).round(4), "| mean over t:", round(float(cos0.mean()), 4),
          "| gate <=0.9:", "PASS" if cos0.mean() <= 0.9 else "FAIL")
    print("(ii) mean cos of consecutive target displacements:", round(float(dcos.mean()), 4),
          "| >=0.3 turns the tube on:", bool(dcos.mean() >= 0.3))
    print("(iv) mean cos(z*_T, z*_T without final equation):", round(float(omit.mean()), 4), "(reported only)")
    np.savez(f"p1_T{T}.npz", Z=Z, Zo=Zo)
