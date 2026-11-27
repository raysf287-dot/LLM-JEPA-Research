"""Option 1 head probe and diagnostic (teacher-forced on gold solutions, annotations stripped).
For each aligned operand (see align_utils), measure every head's attention at the query token:
  mass = attention summed over the operand's source span;  hit = argmax attention lies in the span.
Select mode (base model, training split):
  python probe_heads.py --model meta-llama/Llama-3.2-1B-Instruct --file gsm8k_trainsplit_noanno.jsonl --n 500 \
      --layers 4-12 --top 4 --out heads.txt
Diagnostic mode (fine-tuned model, validation file, fixed heads):
  python probe_heads.py --model ./ft_A --file gsm8k_val.jsonl --n 500 --heads 6:3,8:11
"""
import argparse, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from align_utils import token_alignments

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--file", required=True)
ap.add_argument("--n", type=int, default=500); ap.add_argument("--layers", default="4-12")
ap.add_argument("--top", type=int, default=4); ap.add_argument("--out", default=None); ap.add_argument("--heads", default="")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
L, H = m.config.num_hidden_layers, m.config.num_attention_heads
mass = torch.zeros(L, H, dtype=torch.float64); hit = torch.zeros(L, H, dtype=torch.float64); n_pairs = 0
rows = [json.loads(l)["messages"] for l in open(a.file)][:a.n]
for msgs in rows:
    msgs = [dict(x) for x in msgs]
    msgs[2]["content"] = re.sub(r"<<[^>]*>>", "", msgs[2]["content"])          # same format as the stripped training data
    fm = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    q, s, e = token_alignments(tok, fm, msgs[1]["content"], msgs[2]["content"], 512, True)
    pairs = [(qq, ss, ee) for qq, ss, ee in zip(q, s, e) if qq >= 0]
    if not pairs: continue
    ids = tok(fm, truncation=True, max_length=512, return_tensors="pt", add_special_tokens=True).to("cuda")
    with torch.no_grad():
        att = m(**ids, output_attentions=True).attentions                       # L x (1, H, T, T)
    A = torch.stack([x[0].float() for x in att])                                 # (L, H, T, T)
    for qq, ss, ee in pairs:
        row = A[:, :, qq, :]                                                     # (L, H, T)
        mass += row[:, :, ss:ee].sum(-1).double().cpu()
        am = row.argmax(-1)
        hit += ((am >= ss) & (am < ee)).double().cpu(); n_pairs += 1
mass /= max(n_pairs, 1); hit /= max(n_pairs, 1)
print(f"[probe] {a.model} on {a.file}: {len(rows)} examples, {n_pairs} aligned operands")
if a.heads:
    hs = [tuple(int(x) for x in h.split(":")) for h in a.heads.split(",")]
    for l, h in hs: print(f"[probe] head {l}:{h}  mass {mass[l, h]:.3f}  argmax-in-span {hit[l, h]:.3f}")
    print(f"[probe] supervised heads mean: mass {sum(mass[l, h] for l, h in hs) / len(hs):.3f} "
          f"argmax-in-span {sum(hit[l, h] for l, h in hs) / len(hs):.3f}")
else:
    lo, hi = (int(x) for x in a.layers.split("-"))
    cand = sorted(((mass[l, h].item(), l, h) for l in range(lo, hi + 1) for h in range(H)), reverse=True)
    for v, l, h in cand[:10]: print(f"[probe] candidate {l}:{h}  mass {v:.3f}  argmax-in-span {hit[l, h]:.3f}")
    sel = ",".join(f"{l}:{h}" for _, l, h in cand[:a.top])
    print(f"[probe] selected heads {sel}")
    if a.out: open(a.out, "w").write(sel)
