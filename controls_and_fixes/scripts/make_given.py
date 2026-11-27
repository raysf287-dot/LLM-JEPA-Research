"""Option 0 (control): 'Given' format. Builds gsm8k_trainsplit_given.jsonl from gsm8k_trainsplit_noanno.jsonl
(the stripped training split made by make_formats.py; same 5,973 examples, same order).
Only the assistant target changes: a 'Given:' block of the question clauses that contain a quantity,
then 'Solution:' and the unchanged stripped solution and '#### answer'. Questions, validation and test files are untouched.
Usage: python make_given.py [in_file] [out_file]
"""
import json, re, sys

IN = sys.argv[1] if len(sys.argv) > 1 else "gsm8k_trainsplit_noanno.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "gsm8k_trainsplit_given.jsonl"
QWORDS = (r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|"
          r"hundred|thousand|million|dozen|half|halves|twice|double|triple|thrice|third|quarter|percent|once)\b")
LEAD = re.compile(r"^(and|but|then|so|if|while|when|after|before|also|however|meanwhile)\b[\s,]*", re.I)


def clauses(q):
    q = re.sub(r"\s+", " ", q.strip())
    sents = re.split(r"(?<=[.?!])\s+(?=[A-Z$\d])", q)          # sentence split, keeps '3.5' and '$1.50'
    out = []
    for s in sents:
        for c in re.split(r",\s+|;\s+", s):                      # clause split; '1,000' is safe (no space)
            c = c.strip()
            while LEAD.match(c): c = LEAD.sub("", c, count=1)
            c = c.strip(" .?!")
            if not c: continue
            if re.search(r"\d", c) or re.search(QWORDS, c, re.I):
                c = c[0].upper() + c[1:]
                if c not in out: out.append(c)
    return out


n = empty = 0; lens = []
with open(IN) as f, open(OUT, "w") as g:
    for line in f:
        d = json.loads(line); q = d["messages"][1]["content"]; sol = d["messages"][2]["content"]
        cl = clauses(q)
        if not cl: empty += 1
        given = "Given:\n" + "\n".join("- " + c for c in cl) + "\nSolution:\n" if cl else ""
        d["messages"][2]["content"] = given + sol
        g.write(json.dumps(d) + "\n"); n += 1; lens.append((len(given + sol), len(sol)))
        if n <= 3: print(f"--- example {n}\nQ: {q}\nTARGET:\n{given + sol}\n")
import statistics as st
print(f"wrote {n} examples to {OUT}; {empty} with no quantity clause (left unchanged)")
print(f"target length chars: given {st.mean(a for a, b in lens):.0f} vs stripped {st.mean(b for a, b in lens):.0f} "
      f"(x{st.mean(a for a, b in lens) / st.mean(b for a, b in lens):.2f})")
