"""Program-grounded number resampling for GSM8K (training split only).

Each GSM8K solution carries calculator annotations <<expr=result>>, so it is an executable program. We
  1. parse the chain (exact rational arithmetic) and check it reproduces every annotated result,
  2. bind each operand literal to a question number ('in'), an earlier result ('res') or a fixed constant,
  3. resample the question numbers that are safe to change (unique in the question, not a protected constant,
     not part of a fraction/ratio/time/year/ordinal), re-run the chain, and keep only variants whose results stay
     positive, integral (if the original was), within 0.1x-10x of the original, and conflict-free,
  4. rewrite the question (bound spans only) and the solution (annotations stripped; numbers replaced by value map),
  5. verify the rewritten solution with an independent parser (classify_errors.parse_eqs): same number of equations,
     every equation evaluates, and the final answer equals the program's final result.
No outside model is used; validation rows (split seed 0, first 1,500 shuffled indices) are never touched.
Usage: python resample_gsm.py gsm8k_train.jsonl OUT.jsonl [variants_per_problem=3] [seed=0] [--audit N]
Writes OUT.jsonl (original + resampled rows in the noanno training format), OUT.meta.jsonl and OUT.stats.txt.
"""
import json, math, os, random, re, sys
from collections import Counter
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_errors import parse_eqs

PROTECTED = {Fraction(x) for x in ("0", "1", "2", "7", "12", "24", "30", "31", "52", "60", "100", "365", "1000", "1/2", "1/4")}
POW10 = {Fraction(10) ** k for k in range(1, 10)} | {Fraction(1, 10 ** k) for k in range(1, 5)}
ANN = re.compile(r"<<([^<>]*)>>")
NUM = re.compile(r"(?<![\w.])(\$?)((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(%?)")
ORD = re.compile(r"(st|nd|rd|th)\b")
SYSMSG = "Answer the math question, show steps."


# ------------------------------------------------------------------ exact arithmetic
def fr(s):
    return Fraction(s.replace(",", ""))


def decimals(x, cap=6):
    for d in range(cap + 1):
        if (x * 10 ** d).denominator == 1: return d
    return None


def etokens(s):
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"(?<=[\d)])\s*[xX×]\s*(?=[\d(.])", "*", s).replace("÷", "/")
    s = re.sub(r"(\d+(?:\.\d+)?|\.\d+)\s*%", r"\1/100", s)
    toks = re.findall(r"\d+(?:\.\d+)?|\.\d+|[()+\-*/]", s)
    if re.sub(r"\d+(?:\.\d+)?|\.\d+|[()+\-*/]|\s", "", s): return None
    return [("num", fr(t)) if t[0].isdigit() or t[0] == "." else ("op", t) for t in toks]


def evaluate(toks):
    """toks: list of ('num', Fraction) / ('op', str). Recursive descent, exact. None on error."""
    pos = [0]

    def peek(): return toks[pos[0]] if pos[0] < len(toks) else None

    def atom():
        t = peek()
        if t is None: raise ValueError
        if t == ("op", "("):
            pos[0] += 1; v = expr()
            if peek() != ("op", ")"): raise ValueError
            pos[0] += 1; return v
        if t == ("op", "-"): pos[0] += 1; return -atom()
        if t[0] == "num": pos[0] += 1; return t[1]
        raise ValueError

    def term():
        v = atom()
        while peek() in (("op", "*"), ("op", "/")):
            o = peek()[1]; pos[0] += 1; b = atom()
            if o == "/":
                if b == 0: raise ZeroDivisionError
                v = v / b
            else: v = v * b
        return v

    def expr():
        v = term()
        while peek() in (("op", "+"), ("op", "-")):
            o = peek()[1]; pos[0] += 1; b = term(); v = v + b if o == "+" else v - b
        return v

    try:
        v = expr()
        return v if pos[0] == len(toks) else None
    except (ValueError, ZeroDivisionError):
        return None


# ------------------------------------------------------------------ parsing one example
def qnumbers(q):
    out = []
    for m in NUM.finditer(q):
        s, e = m.start(), m.end()
        try: v = fr(m.group(2))
        except (ValueError, ZeroDivisionError): continue
        out.append({"v": v, "s": s, "e": e, "pre": m.group(1), "body": m.group(2), "pct": bool(m.group(3))
                    or bool(re.match(r"\s*(percent|per cent)\b", q[e:e + 10]))})
    return out


def eligible(q, n, cnt):
    s, e = n["s"], n["e"]
    if cnt[n["v"]] != 1 or n["v"] in PROTECTED: return False
    if re.search(r"[/:]\s*$", q[max(0, s - 2):s]) or re.match(r"\s*[/:]", q[e:e + 2]): return False      # fractions, ratios, times
    if ORD.match(q[e:e + 3]): return False
    if n["v"].denominator == 1 and 1900 <= n["v"] <= 2100 and not n["pre"]: return False                   # years
    return True


def parse_example(row):
    q = row["messages"][1]["content"]; sol = row["messages"][2]["content"]
    if "\n#### " not in sol: return None, "no_answer_line"
    body, ans = sol.rsplit("\n#### ", 1)
    anns = ANN.findall(body)
    if not anns: return None, "no_annotations"
    steps = []
    for a in anns:
        if a.count("=") != 1: return None, "bad_annotation"
        lhs, rhs = a.split("=")
        t = etokens(lhs); r = etokens(rhs.replace("%", ""))
        if t is None or r is None or len(r) != 1 or r[0][0] != "num" or not any(x[0] == "op" for x in t) and len(t) != 1: return None, "unparsable_annotation"
        v = evaluate(t)
        if v is None or v != r[0][1]: return None, "annotation_mismatch"
        steps.append({"toks": t, "res": v})
    try: final = fr(re.sub(r"[$\s]", "", ans))
    except (ValueError, ZeroDivisionError): return None, "answer_not_numeric"
    if final != steps[-1]["res"]: return None, "answer_not_last_result"
    qn = qnumbers(q); cnt = Counter(n["v"] for n in qn)
    elig = [i for i, n in enumerate(qn) if eligible(q, n, cnt)]
    ev = {qn[i]["v"]: i for i in elig}
    # bind literals
    # Pass 1: a literal equal to an earlier result is that result, unless the value is also a protected constant or a
    # question number (then decided in pass 2). Pass 2 uses the chain's structure: every intermediate result is
    # consumed by a later step, so an otherwise-unconsumed result is the natural owner of an ambiguous literal.
    qvals = {n["v"] for n in qn}
    for k, st in enumerate(steps):
        st["bind"] = [None] * len(st["toks"])
        for p, t in enumerate(st["toks"]):
            if t[0] != "num": continue
            f = t[1]; earlier = [j for j in range(k) if steps[j]["res"] == f]
            if earlier and f not in PROTECTED and f not in qvals: st["bind"][p] = ("res", earlier[-1])
            elif not earlier and f in ev: st["bind"][p] = ("in", ev[f])
            elif not earlier:
                if f not in PROTECTED and f != 0 and any(v != 0 and (f / v in POW10 or v / f in POW10) for v in {steps[j]["res"] for j in range(k)} | set(ev)):
                    return None, "scaled_literal"                                # e.g. 420,000,000 derived from 420 (unit change)
                st["bind"][p] = ("const", f)
    consumed = {b[1] for st in steps for b in st["bind"] if b and b[0] == "res"}
    for k, st in enumerate(steps):
        for p, t in enumerate(st["toks"]):
            if t[0] != "num" or st["bind"][p] is not None: continue
            f = t[1]; free = [j for j in range(k) if steps[j]["res"] == f and j not in consumed]
            if len(free) == 1: st["bind"][p] = ("res", free[0]); consumed.add(free[0])
            elif len(free) > 1 or f in qvals: return None, "ambiguous_literal"
            else: st["bind"][p] = ("const", f)
    allowed = qvals | {st["res"] for st in steps} | {t[1] for st in steps for t in st["toks"] if t[0] == "num"} | PROTECTED
    for t in NUM.finditer(ANN.sub("", body)):                                    # numbers in the prose that the program does not explain
        try: v = fr(t.group(2))
        except (ValueError, ZeroDivisionError): continue
        if v not in allowed and not ORD.match(body[t.end():t.end() + 3]): return None, "unexplained_prose_number"
    used = sorted({b[1] for st in steps for b in st["bind"] if b and b[0] == "in"})
    if not used: return None, "no_eligible_inputs"
    return {"q": q, "body": body, "ans": ans, "steps": steps, "qn": qn, "used": used, "final": final,
            "words": word_values(q) | word_values(ANN.sub("", body))}, None


def run_chain(ex, newin):
    res = []
    for st in ex["steps"]:
        toks = []
        for t, b in zip(st["toks"], st["bind"]):
            if b is None: toks.append(t)
            elif b[0] == "in": toks.append(("num", newin[b[1]]))
            elif b[0] == "res": toks.append(("num", res[b[1]]))
            else: toks.append(("num", b[1]))
        v = evaluate(toks)
        if v is None: return None
        res.append(v)
    return res


# ------------------------------------------------------------------ sampling and checks
def propose(n, rng):
    v = n["v"]; d = decimals(v)
    if n["pct"] and d == 0 and 0 < v <= 100: return Fraction(rng.choice([p for p in range(5, 100, 5) if p != v] or [5]))
    if d == 0:
        lo, hi = max(2, int(v * 0.5)), int(math.ceil(v * 1.6)) + 2
        if v >= 10 and v % 10 == 0: c = [x for x in range(lo - lo % 10 + 10, hi + 1, 10) if x != v]
        elif v >= 10 and v % 5 == 0: c = [x for x in range(lo - lo % 5 + 5, hi + 1, 5) if x != v]
        else: c = [x for x in range(lo, hi + 1) if x != v]
        return Fraction(rng.choice(c)) if c else None
    x = Fraction(round(float(v) * rng.uniform(0.6, 1.5) * 10 ** d), 10 ** d)
    return x if x != v and x > 0 else None


def good_results(ex, res):
    for st, r in zip(ex["steps"], res):
        o = st["res"]
        if (o > 0) != (r > 0) or (o != 0 and (abs(r / o) > 10 or abs(r / o) < Fraction(1, 10))) or r == 0 and o != 0: return False
        do, dn = decimals(o), decimals(r)
        if dn is None or do is None or dn > max(do, 2 if do else 0): return False
    return True


def fmt(x, orig_tok_decimals, commas):
    d = decimals(x)
    if d is None: return None
    d = max(d, orig_tok_decimals)
    s = f"{float(x):,.{d}f}" if commas else f"{float(x):.{d}f}"
    return s


def valmap(ex, newin, res):
    """old value -> new value for changed inputs and results; None on conflict."""
    m = {}
    items = [(ex["qn"][i]["v"], newin[i]) for i in ex["used"]] + [(st["res"], r) for st, r in zip(ex["steps"], res)]
    for o, n in items:
        if o == n: continue
        if o in m and m[o] != n: return None
        m[o] = n
    fixed = {n["v"] for i, n in enumerate(ex["qn"]) if i not in ex["used"]} | {b[1] for st in ex["steps"] for b in st["bind"] if b and b[0] == "const"}
    return None if any(o in fixed for o in m) else m


def replace_tokens(text, m, spans=None):
    """Replace numeric tokens whose exact value is in m (or only the given char spans). Returns (new_text, n_replaced) or None."""
    out, last, n = [], 0, 0
    for t in NUM.finditer(text):
        try: v = fr(t.group(2))
        except (ValueError, ZeroDivisionError): continue
        if spans is not None and (t.start(), t.end()) not in spans: continue
        if v not in m: continue
        if ORD.match(text[t.end():t.end() + 3]): return None
        body = t.group(2); d0 = len(body.split(".")[1]) if "." in body else 0
        s = fmt(m[v], d0, "," in body)
        if s is None: return None
        out.append(text[last:t.start()]); out.append(t.group(1) + s + t.group(3)); last = t.end(); n += 1
    out.append(text[last:])
    return "".join(out), n


FAIL = Counter()


NW = {w: Fraction(v) for w, v in zip(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred thousand dozen".split(),
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000, 12])}
NWRE = re.compile(r"\b(?:(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ](one|two|three|four|five|six|seven|eight|nine)|(" + "|".join(NW) + r"))s?\b", re.I)


def word_values(text):
    """Values of spelled-out numbers ('three', 'twenty-five', 'a dozen') in a text."""
    out = set()
    for m in NWRE.finditer(text):
        out.add(NW[m.group(1).lower()] + NW[m.group(2).lower()] if m.group(1) else NW[m.group(3).lower()])
    return out


BOUNDS = [(r"(\d+(?:\.\d+)?)\s+days?\s+(?:a|per|each|every)\s+week", 7), (r"(\d+(?:\.\d+)?)\s+hours?\s+(?:a|per|each|every)\s+day", 24),
          (r"(\d+(?:\.\d+)?)\s+months?\s+(?:a|per|each|every)\s+year", 12), (r"(\d+(?:\.\d+)?)\s+minutes?\s+(?:an|per|each|every)\s+hour", 60)]


def implausible(q):
    """Natural cycles that a resampled number must not exceed (8 days a week, 30 hours a day, ...)."""
    return any(float(x) > b for p, b in BOUNDS for x in re.findall(p, q))


TRIES = 400


def build_variant(ex, rng, tries=None):
    tries = tries or TRIES
    for _ in range(tries):
        newin = {}
        for i in ex["used"]:
            x = propose(ex["qn"][i], rng)
            if x is None: break
            newin[i] = x
        else:
            res = run_chain(ex, newin)
            if res is None: FAIL['chain_error'] += 1; continue
            if not good_results(ex, res): FAIL['results_not_plausible'] += 1; continue
            m = valmap(ex, newin, res)
            if m is None: FAIL['value_conflict'] += 1; continue
            if ex["words"] & set(m): FAIL['number_word_conflict'] += 1; continue          # 'three' in the prose would go stale
            q2 = replace_tokens(ex["q"], {ex["qn"][i]["v"]: newin[i] for i in ex["used"]}, {(ex["qn"][i]["s"], ex["qn"][i]["e"]) for i in ex["used"]})
            body = ANN.sub("", ex["body"])
            s2 = replace_tokens(body, m)
            if q2 is None or s2 is None or q2[1] != len(ex["used"]): FAIL['rewrite_failed'] += 1; continue
            if implausible(q2[0]): FAIL['implausible_bound'] += 1; continue
            ans = replace_tokens("\n#### " + ex["ans"], m)
            if ans is None: FAIL['rewrite_failed'] += 1; continue
            sol2 = s2[0] + ans[0]
            # independent verification with the error-analysis parser
            e0, e1 = parse_eqs(body), parse_eqs(s2[0])
            if len(e0) != len(e1): FAIL['eq_count_changed'] += 1; continue
            bad = False
            for ops, _o, lv, rv in e1:
                if lv is not None and abs(lv - rv) > 0.011 * max(1, abs(rv)): bad = True
            if bad: FAIL['equation_check_failed'] += 1; continue
            if Fraction(re.sub(r"[$,\s]", "", ans[0].split("#### ")[1])) != res[-1]: FAIL['answer_mismatch'] += 1; continue
            return {"q": q2[0], "sol": sol2, "newin": {i: str(v) for i, v in newin.items()}, "final": str(res[-1])}
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src, out = args[0], args[1]; V = int(args[2]) if len(args) > 2 else 3; seed = int(args[3]) if len(args) > 3 else 0
    audit = int(sys.argv[sys.argv.index("--audit") + 1]) if "--audit" in sys.argv else 0
    rows = [json.loads(l) for l in open(src)]
    idx = list(range(len(rows))); random.Random(0).shuffle(idx); val = set(idx[:1500])
    rng = random.Random(seed); why = Counter(); nvar = Counter(); fo = open(out, "w"); fm = open(out.replace(".jsonl", ".meta.jsonl"), "w"); shown = []
    ntrain = 0
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    global TRIES
    if "--tries" in sys.argv: TRIES = int(sys.argv[sys.argv.index("--tries") + 1])
    for i, r in enumerate(rows):
        if i in val: continue
        if limit and ntrain >= limit: break
        ntrain += 1
        base = json.loads(json.dumps(r)); base["messages"][2]["content"] = ANN.sub("", r["messages"][2]["content"])
        fo.write(json.dumps(base) + "\n")
        ex, reason = parse_example(r)
        if ex is None: why[reason] += 1; continue
        seen = set(); got = 0
        nv = V
        if "--balance" in sys.argv:                           # long chains are under-represented among valid variants: give them more
            n = len(ex["steps"]); nv = 2 if n <= 2 else (3 if n <= 4 else 5)
        for _ in range(nv):
            v = build_variant(ex, rng)
            if v is None: break
            key = tuple(sorted(v["newin"].items()))
            if key in seen: continue
            seen.add(key); got += 1
            e = json.loads(json.dumps(r)); e["messages"][1]["content"] = v["q"]; e["messages"][2]["content"] = v["sol"]
            fo.write(json.dumps(e) + "\n"); fm.write(json.dumps({"src_idx": i, "orig_q": ex["q"], "final": v["final"]}) + "\n")
            if len(shown) < audit and rng.random() < 0.02: shown.append((i, ex["q"], base["messages"][2]["content"], v["q"], v["sol"]))
        nvar[got] += 1
        if got == 0: why["no_valid_variant"] += 1
    fo.close(); fm.close()
    ok = sum(v for k, v in nvar.items() if k > 0); tot = sum(v * k for k, v in nvar.items())
    L = [f"train problems {ntrain}; parsed program + >=1 valid variant: {ok} ({ok / ntrain * 100:.1f}%); variants written {tot}",
         "variants per problem (of those parsed): " + ", ".join(f"{k}:{v}" for k, v in sorted(nvar.items())), "reasons a problem produced nothing:"]
    L += [f"  {k:26s} {v:5d}  {v / ntrain * 100:5.1f}%" for k, v in why.most_common()]
    L += ["failed attempts by check (counts over all tries):"] + [f"  {k:26s} {v:7d}" for k, v in FAIL.most_common()]
    open(out.replace(".jsonl", ".stats.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
    for i, q0, s0, q1, s1 in shown:
        print(f"\n===== train row {i}\nQ0: {q0}\nS0: {s0.replace(chr(10), ' | ')}\nQ1: {q1}\nS1: {s1.replace(chr(10), ' | ')}")


if __name__ == "__main__":
    main()
