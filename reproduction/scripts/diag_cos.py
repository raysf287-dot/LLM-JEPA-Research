import json, sys, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
path, k = sys.argv[1], int(sys.argv[2]); n = 200
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
rows = [json.loads(l)["messages"] for l in open("gsm8k_train.jsonl")][:n]
suf = "".join(f"<|predictor_{i}|>" for i in range(k, 0, -1))
def emb(msg, s=""):
    m = [dict(msg)]; m[0]["content"] += s
    ids = tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False), return_tensors="pt").input_ids.cuda()
    with torch.no_grad(): h = model(ids, output_hidden_states=True).hidden_states[-1][0]
    return h[-2].float()
U = torch.stack([emb(m[1], suf) for m in rows]); A = torch.stack([emb(m[2]) for m in rows])
def offdiag(X):
    X = F.normalize(X, dim=-1); S = X @ X.T; return ((S.sum() - S.diag().sum()) / (len(X) * (len(X) - 1))).item()
print(f"{path}: matched cos {F.cosine_similarity(U, A).mean():.4f} | mismatched {F.cosine_similarity(U, A.roll(1, 0)).mean():.4f} | "
      f"user-user {offdiag(U):.4f} | sol-sol {offdiag(A):.4f} | norms U {U.norm(dim=-1).mean():.0f} A {A.norm(dim=-1).mean():.0f}")
