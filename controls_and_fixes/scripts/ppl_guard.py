"""General-text guard: perplexity on WikiText-2 (raw, test split), first 100 chunks of 512 tokens.
Usage: python ppl_guard.py MODEL_PATH TAG   -> prints 'PPL TAG <value>'
"""
import math, sys, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

path, tag = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(path)
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
nll, n = 0.0, 0
for i in range(100):
    x = ids[i * 512:(i + 1) * 512].unsqueeze(0).cuda()
    with torch.no_grad():
        out = m(input_ids=x, labels=x)
    nll += out.loss.item() * (x.shape[1] - 1); n += x.shape[1] - 1
print(f"PPL {tag} {math.exp(nll / n):.3f}")
