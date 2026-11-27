"""Rejection-sampling fine-tuning (RFT, Yuan et al. 2023) data builder. Training split only.
Sample K solutions per training question from a fine-tuned model, keep those whose '#### answer' matches the gold
answer, de-duplicate by their equation sequence (and against the original solution), and write
  <out> = original training examples + kept samples.
Usage: python sample_rft.py MODEL_DIR TRAIN_FILE OUT_FILE [K=4] [TEMP=0.7] [LEN=0]
LEN > 0 (length-aware RFT for a LEN-token generation limit): keep only correct samples of <= LEN tokens, and replace an
original solution longer than LEN tokens by the shortest kept sample for that question (if there is one).
"""
import json, re, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path, tf, out = sys.argv[1], sys.argv[2], sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 4
TEMP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
LEN = int(sys.argv[6]) if len(sys.argv) > 6 else 0
pat = re.compile(r"\n#### (.+)$")
rows = [json.loads(l) for l in open(tf)]
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
torch.manual_seed(0)


def ans(t):
    x = pat.search(t.strip()); return x.group(1).strip().replace(",", "") if x else None


def key(t):  # equation sequence after 'Solution:' (or whole text), spaces removed
    body = t.split("Solution:")[-1].split("####")[0]
    return tuple(re.sub(r"\s+", "", e) for e in re.findall(r"[\d$.,()+\-*/x×÷%\s]+=\s*\$?[\d.,]*\d", body))


prompts = [tok.apply_chat_template([x for x in r["messages"] if x["role"] != "assistant"], tokenize=False,
                                   add_generation_prompt=True) for r in rows]
order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
kept = [[] for _ in rows]; t0 = time.time(); n_correct = 0
for s in range(0, len(order), 32):
    idx = order[s:s + 32]
    x = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        o = m.generate(**x, do_sample=True, temperature=TEMP, top_p=0.95, max_new_tokens=384, num_return_sequences=K,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    new = o[:, x["input_ids"].shape[1]:]
    for j, i in enumerate(idx):
        gold = ans(rows[i]["messages"][2]["content"]); seen = {key(rows[i]["messages"][2]["content"])}
        for k in range(K):
            t = tok.decode(new[j * K + k], skip_special_tokens=True).strip()
            if ans(t) is not None and gold is not None and ans(t) == gold:
                n_correct += 1; kk = key(t)
                if LEN and len(tok(t, add_special_tokens=False)["input_ids"]) > LEN - 1: continue   # must end within LEN (incl. end token)
                if kk not in seen: seen.add(kk); kept[i].append(t)
    if s % 640 == 0: print(f"[rft] {s}/{len(rows)} questions, {time.time() - t0:.0f}s", flush=True)

n_repl = 0
with open(out, "w") as f:
    for r, ks in zip(rows, kept):
        if LEN and ks and len(tok(r["messages"][2]["content"], add_special_tokens=False)["input_ids"]) > LEN - 1:
            short = min(ks, key=lambda t: len(tok(t, add_special_tokens=False)["input_ids"]))
            ks.remove(short); r = json.loads(json.dumps(r)); r["messages"][2]["content"] = short; n_repl += 1
        f.write(json.dumps(r) + "\n")
    for r, ks in zip(rows, kept):
        for t in ks:
            e = json.loads(json.dumps(r)); e["messages"][2]["content"] = t; f.write(json.dumps(e) + "\n")
n_kept = sum(len(k) for k in kept)
print(f"RFT {path} (LEN={LEN}, {n_repl} long originals replaced by a short correct sample): {len(rows)} questions x {K} samples, correct {n_correct / (len(rows) * K) * 100:.1f}%, "
      f"kept {n_kept} distinct new solutions ({sum(bool(k) for k in kept)} questions), file {out} has {len(rows) + n_kept} examples")
