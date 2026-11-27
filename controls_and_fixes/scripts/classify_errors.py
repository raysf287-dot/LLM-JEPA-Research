"""Heuristic error classification for GSM8K validation generations (validation only).
Usage: python classify_errors.py valgen_V_base_128.jsonl [valgen_V_base_256.jsonl] [out_prefix]
Input rows (from gen_val.py): idx, question, gold_solution (with <<...>> annotations), gold, output, pred, correct, eos_emitted.
Writes <out_prefix>_labels.csv (one row per example) and <out_prefix>_summary.txt.

Each wrong answer gets the FIRST error found, checked in this order:
  truncated        no '####' and no end token (ran out of tokens); '+fixed@256' if the 256-token run is correct
  no_answer        stopped without a '#### answer' line
  arithmetic       a written equation whose left side does not evaluate to its right side
  unsupported_num  an operand that is not in the question, not an earlier result and not a common constant
                   (misread/hallucinated quantity = operand-selection error)
  wrong_operation  same operands as a gold step but a different operator
  wrong_setup      operands grounded and arithmetic right, but the step result is not a gold intermediate value
                   (wrong combination of quantities = binding/plan error)
  missing_step     every step result is a gold value but the chain stops before the gold answer
  answer_copy      the last computed value equals the gold answer but the '####' value differs
  other            none of the above (no equations parsed, or checks pass but answer wrong)
These are heuristics: audit a sample of each category by hand before trusting the proportions.
"""
import csv, json, re, sys, warnings
warnings.filterwarnings("ignore")
from collections import Counter, defaultdict

CONST = {0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 24, 30, 31, 52, 60, 100, 1000, 365, 0.5, 0.25, 0.1}
CUES = {"twice/double": r"\b(twice|double|doubles|doubled)\b", "half": r"\bhalf\b", "each/per": r"\b(each|per|every)\b",
        "remaining/left": r"\b(remain|remaining|left|rest)\b", "percent": r"(%|\bpercent)", "times": r"\btimes\b",
        "more/less than": r"\b(more|less|fewer|older|younger|taller|shorter)\s+than\b", "fraction": r"\b\d+/\d+\b|\b(third|quarter|fifth)s?\b",
        "money": r"\$", "time units": r"\b(hour|minute|week|day|month|year)s?\b"}
NUM = r"-?\d[\d,]*(?:\.\d+)?"


def num(s):
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        v = float(s); return round(v, 4)
    except ValueError:
        return None


WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "dozen": 12, "half": 0.5, "twice": 2, "double": 2,
         "triple": 3, "thrice": 3, "quarter": 0.25, "third": 3, "fourth": 4, "fifth": 5}


def nums_in(text):
    vals = {v for v in (num(m) for m in re.findall(NUM, text.replace("$", " "))) if v is not None}
    vals |= {float(v) for w, v in WORDS.items() if re.search(r"\b" + w + r"s?\b", text, re.I)}
    for t, o in re.findall(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(one|two|three|four|five|six|seven|eight|nine)\b", text, re.I):
        vals.add(float(WORDS[t.lower()] + WORDS[o.lower()]))
    for v, w in re.findall(r"(\d+(?:\.\d+)?)\s*(million|billion)", text, re.I):
        vals.add(round(float(v) * (1e6 if w.lower() == "million" else 1e9), 4))
    if re.search(r"\ba million\b", text, re.I): vals.add(1e6)
    return vals | {round(v / 100, 4) for v in vals} | {round(v * 100, 4) for v in vals}


def clean_expr(seg):
    e = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", seg)          # 'off-road'
    e = re.sub(r"(?<=[\d\s)])[x×](?=[\s\d$(.])", "*", e).replace("÷", "/").replace("$", "").replace(",", "")
    e = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", e)
    e = re.sub(r"[^0-9.+\-*/()\s]", " ", e)
    e = re.sub(r"\.(?!\d)", " ", e)                              # full stops, keep '.5'
    e = re.sub(r"([-+*/])(\s*[-+*/])+", r"\1", e)              # '8 / / 2' -> '8 / 2'
    e = re.sub(r"^[\s+*/)]+|[\s+\-*/(]+$", "", e).strip()
    return e


def evaluate(e):
    if not re.search(r"\d", e): return None
    try:
        return round(float(eval(e, {"__builtins__": {}})), 4)
    except Exception:
        return None


VERBAL = [(r"\bequals?\b|\bis equal to\b", "="), (r"\bdivided by\b", "/"), (r"\bmultiplied by\b", "*"),
          (r"(?<=\d)\s+times\s+(?=\$?\d)", " * "), (r"(?<=\d)\s+minus\s+(?=\$?\d)", " - "), (r"(?<=\d)\s+plus\s+(?=\$?\d)", " + ")]
ALGEBRA = re.compile(r"\d[A-Z]\b|\d[xX](?![\d\s$.(])|\b[A-Z]\s*[-+*/=]\s*[\dA-Z($]|[-+*/]\s*[A-Z]\b")   # '2S', '3x', 'S + 2'


def parse_eqs(text):
    """Return (operands, operators, lhs_value or None, rhs_value) for each '=' in the text, in order.
    Handles chains 'a + b = c + d = e' by comparing neighbouring sides."""
    out = []
    for a, b in VERBAL: text = re.sub(a, b, text, flags=re.I)
    text = re.sub(r"(?<=[A-Za-z\d])\s*/\s*(?=[A-Za-z])[A-Za-z]+", " ", text)   # units: 'pieces/house', '$12/hour'
    for line in text.split("\n"):
        if ALGEBRA.search(line): continue
        parts = line.split("=")
        for i in range(len(parts) - 1):
            left = parts[i]
            cut = max(left.rfind(": "), left.rfind(", "), left.rfind("; "), left.rfind(". "))
            if cut >= 0 and re.search(r"\d", left[cut + 1:]): left = left[cut + 1:]
            m = re.match(r"\s*\$?\s*[-\d.(][\d\s.,$%()+\-*/x×÷]*", parts[i + 1])   # leading numeric part of the right side
            if not m: continue
            rv = evaluate(clean_expr(m.group(0).replace("%", "")))
            if rv is None: continue
            tail = re.search(r"[$(]*\s*\.?\d[\d\s.,$%()+\-*/x×÷]*$", left.rstrip())   # contiguous expression just before '='
            e = clean_expr(tail.group(0)) if tail and re.search(r"\d\s*[-+*/x×÷]\s*[$(]*\.?\d", tail.group(0)) else clean_expr(left)
            ops = sorted(re.findall(r"(?<=[\d)\s])\s*([+*/]|-(?=\s*[\d(.]))", e))
            operands = [num(v) for v in re.findall(r"\d*\.?\d+", e)]
            val = evaluate(e) if ops else None
            out.append((operands, ops, val, rv))
    return out


def gold_steps(sol):
    steps = []
    for a in re.findall(r"<<([^>]*)>>", sol):
        if "=" not in a: continue
        lhs, rhs = a.rsplit("=", 1)
        steps.append(([num(v) for v in re.findall(r"\d+(?:\.\d+)?", lhs)], sorted(re.findall(r"(?<=[\d)])\s*([-+*/])", lhs)), num(rhs)))
    return steps


def close(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-2 * max(1, abs(b))


def close_pct(a, b):  # '100% - 40% = 60%' evaluates to 0.6 on the left
    return close(a, b) or close(a * 100, b) or close(a, b * 100)


def classify(r, r256):
    if r["correct"]: return "correct"
    out = r["output"]
    if "####" not in out:
        if not r["eos_emitted"]:
            return "truncated+fixed@256" if (r256 and r256["correct"]) else "truncated"
        return "no_answer"
    body = out.split("####")[0]
    eqs = parse_eqs(body)
    gs = gold_steps(r["gold_solution"]); gold = num(r["gold"])
    gvals = {g[2] for g in gs} | {e[3] for e in parse_eqs(re.sub(r"<<[^>]*>>", "", r["gold_solution"]).split("####")[0])}  # some gold steps lack annotations
    known = nums_in(r["question"]) | CONST
    for operands, ops, val, rhs in eqs:
        if val is None:                      # could not parse the left side: only record the result
            known.add(rhs); continue
        if not close_pct(val, rhs):
            return "arithmetic"
        if any(o is not None and o not in known and not any(close(o, k) for k in known) for o in operands):
            return "unsupported_num"
        if not any(close_pct(rhs, g) for g in gvals) and not close_pct(rhs, gold):
            same = [g for g in gs if sorted(x for x in g[0] if x is not None) == sorted(x for x in operands if x is not None)]
            return "wrong_operation" if same and all(g[1] != ops for g in same) else "wrong_setup"
        known.add(rhs)
    if eqs:
        if close_pct(eqs[-1][3], gold): return "answer_copy"
        return "missing_step"
    return "other"


def main():
    f128 = sys.argv[1]; f256 = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".jsonl") else None
    prefix = sys.argv[-1] if not sys.argv[-1].endswith(".jsonl") else f128.rsplit(".", 1)[0]
    R = [json.loads(l) for l in open(f128)]
    R256 = {x["idx"]: x for x in (json.loads(l) for l in open(f256))} if f256 else {}
    rows, cat = [], Counter(); by_steps = defaultdict(lambda: [0, 0]); by_cue = defaultdict(lambda: [0, 0])
    for r in R:
        c = classify(r, R256.get(r["idx"])); cat[c] += 1
        k = len(gold_steps(r["gold_solution"])); by_steps[min(k, 6)][0] += r["correct"]; by_steps[min(k, 6)][1] += 1
        cues = [n for n, p in CUES.items() if re.search(p, r["question"], re.I)]
        for n in cues: by_cue[n][0] += r["correct"]; by_cue[n][1] += 1
        rows.append({"idx": r["idx"], "label": c, "gold_steps": k, "cues": "|".join(cues), "gold": r["gold"], "pred": r["pred"],
                     "question": r["question"], "output": r["output"], "gold_solution": r["gold_solution"]})
    with open(prefix + "_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len(R); wrong = n - cat["correct"]; L = []
    L.append(f"{f128}: {n} examples, accuracy {cat['correct']/n*100:.1f}% ({cat['correct']}), wrong {wrong}")
    L.append("\nFirst-error category (share of wrong answers):")
    for k, v in cat.most_common():
        if k != "correct": L.append(f"  {k:22s} {v:5d}  {v/wrong*100:5.1f}%")
    L.append("\nAccuracy by number of gold steps (6 = 6 or more):")
    for k in sorted(by_steps): c, t = by_steps[k]; L.append(f"  {k} steps  n={t:4d}  acc {c/t*100:5.1f}%")
    L.append("\nAccuracy by wording cue in the question (overall " + f"{cat['correct']/n*100:.1f}%):")
    for k, (c, t) in sorted(by_cue.items(), key=lambda x: x[1][0] / x[1][1]): L.append(f"  {k:16s} n={t:4d}  acc {c/t*100:5.1f}%")
    s = "\n".join(L); print(s); open(prefix + "_summary.txt", "w").write(s + "\n")


if __name__ == "__main__":
    main()
