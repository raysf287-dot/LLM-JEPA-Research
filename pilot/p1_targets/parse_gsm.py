import ast, json, operator as op, re
from collections import Counter
from fractions import Fraction

EQ = re.compile(r"<<([^<>]*)>>")
OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
TOL = Fraction(1, 1000)

def safe_eval(expr):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return Fraction(str(n.value))
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        raise ValueError("unsupported")
    return ev(ast.parse(expr.replace(",", "").strip(), mode="eval"))

def validate(question, cot, answer):
    """Return (example, None) or (None, reason)."""
    found = EQ.findall(cot)
    if not found:
        return None, "no_equations"
    if EQ.sub("", cot).strip():
        return None, "extra_text"
    eqs = []
    for e in found:
        if "=" not in e:
            return None, "bad_equation"
        lhs, rhs = e.rsplit("=", 1)
        eqs.append(f"<<{lhs.strip()}={rhs.strip()}>>")
    try:
        ans = Fraction(answer.replace(",", "").replace("$", "").strip())
    except (ValueError, ZeroDivisionError):
        return None, "bad_answer"
    lhs_last, rhs_last = found[-1].rsplit("=", 1)
    try:
        val, stated = safe_eval(lhs_last), safe_eval(rhs_last)
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return None, "bad_equation"
    if min(abs(val - ans), abs(stated - ans)) > TOL * max(1, abs(ans)):
        return None, "answer_mismatch"
    return {"question": question, "eqs": eqs, "n": len(eqs)}, None

def group_bounds(n, T):
    """g_t = ceil(t*n/T), g_0 = 0; returns (start, end) per step."""
    g = [0] + [-(-(t * n) // T) for t in range(1, T + 1)]
    return [(g[t - 1], g[t]) for t in range(1, T + 1)]

def group_texts(ex, T):
    return [" ".join(ex["eqs"][a:b]) for a, b in group_bounds(ex["n"], T)]

def noop_mask(n, T):
    return [a == b for a, b in group_bounds(n, T)]

def prefixes(ex, T):
    """P_1..P_T as strings; a no-op step repeats the previous prefix."""
    out, cur = [], ex["question"] + "\n"
    for a, b in group_bounds(ex["n"], T):
        if b > a:
            cur = cur + " ".join(ex["eqs"][a:b]) + "\n"
        out.append(cur)
    return out

def load_split(name="gsm8k_aug_train.json"):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("zen-E/GSM8k-Aug", name, repo_type="dataset")
    d = json.load(open(p, encoding="utf-8"))
    return list(zip(d["question"], d["cot"], d["answer"]))

if __name__ == "__main__":
    for name in ("gsm8k_aug_train.json", "gsm8k_test.json"):
        rows = load_split(name)
        kept, why, hist = 0, Counter(), Counter()
        for q, c, a in rows:
            ex, r = validate(q, c, a)
            if ex:
                kept += 1
                hist[ex["n"]] += 1
            else:
                why[r] += 1
        print(name, "total", len(rows), "kept", kept, "discarded", dict(why))
        print("  chain length n:", dict(sorted(hist.items())))
        for T in (2, 3):
            ge = sum(v for k, v in hist.items() if k >= T)
            print(f"  share with n>={T}: {ge / kept:.3f}")
