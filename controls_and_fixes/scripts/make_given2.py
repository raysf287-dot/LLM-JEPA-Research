"""Step 1 formats, built from gsm8k_trainsplit_noanno.jsonl (same 5,973 examples, same order). Only the assistant
target changes. Questions, validation and test files are untouched.
  GG: Given (quantity clauses) + Goal (the question sentence) + Solution
  GR: Given + Relations (comparatives rewritten with explicit direction, precision-first rules) + Goal + Solution
Usage: python make_given2.py [in_file]   -> writes gsm8k_trainsplit_gg.jsonl and gsm8k_trainsplit_gr.jsonl
"""
import json, re, sys

# clause extraction copied from make_given.py (importing it would run that script)
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



IN = sys.argv[1] if len(sys.argv) > 1 else "gsm8k_trainsplit_noanno.jsonl"
NUMW = {"twice": "2", "double": "2", "triple": "3", "thrice": "3", "half": "1/2", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
        "twelve": "12", "fifteen": "15", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100"}
N = r"(\$?\d[\d,]*(?:\.\d+)?|" + "|".join(k for k in NUMW if k not in ("twice", "double", "triple", "thrice", "half")) + r")"
NAME = r"([A-Z][a-z]+(?:'s)?)"          # a capitalised name (precision-first: relations between named entities only)
STOP = {"The", "A", "An", "If", "Each", "Every", "In", "On", "At", "After", "Before", "When", "While", "There", "This",
        "That", "It", "He", "She", "They", "His", "Her", "Their", "How", "What", "Then", "But", "And", "So", "Of"}
MORE = r"(more|longer|taller|older|heavier|faster|bigger|larger|higher)"
LESS = r"(fewer|less|shorter|younger|lighter|slower|smaller|lower)"


def num(s):
    s = s.lower(); return NUMW.get(s, s.replace("$", "").replace(",", ""))


def relations(q):
    out = []
    X = r"(?P<x>[A-Z][a-z]+)(?:'s)?"; Y = r"(?P<y>[A-Z][a-z]+)(?:'s)?"
    R1 = re.compile(X + r"\b[^.?!,;]*?\b(?P<n>" + N[1:-1] + r")\s+(?:[a-z$]+\s+){0,3}?(?:(?P<more>" + MORE[1:-1] + r")|(?P<less>"
                    + LESS[1:-1] + r"))\b[^.?!,;]*?\bthan\s+(?:\w+\s+){0,2}?" + Y)
    R2 = re.compile(X + r"\b[^.?!,;]*?\b(?P<k>twice|half|triple|double|(?:" + N[1:-1] + r")\s+times)\s+as\s+"
                    r"(?:many|much|long|old|tall|fast|far|big|large|heavy)\b[^.?!,;]*?\bas\s+(?:\w+\s+){0,2}?" + Y)
    for c in re.split(r"(?<=[.?!,;])\s+", q):
        m = R1.search(c)
        if m and m["x"] not in STOP and m["y"] not in STOP and m["x"] != m["y"]:
            out.append(f"{m['x']} = {m['y']} {'+' if m['more'] else '-'} {num(m['n'])}"); continue
        m = R2.search(c)
        if m and m["x"] not in STOP and m["y"] not in STOP and m["x"] != m["y"]:
            out.append(f"{m['x']} = {num(m['k'].lower().replace(' times', ''))} × {m['y']}")
    return out


def goal(q):
    s = re.split(r"(?<=[.?!])\s+(?=[A-Z$\d])", re.sub(r"\s+", " ", q.strip()))
    qs = [x for x in s if x.endswith("?")] or [s[-1]]
    return " ".join(qs)


stats = {"n": 0, "rel_ex": 0, "rel": 0}
with open(IN) as f, open("gsm8k_trainsplit_gg.jsonl", "w") as gg, open("gsm8k_trainsplit_gr.jsonl", "w") as gr:
    for line in f:
        d = json.loads(line); q = d["messages"][1]["content"]; sol = d["messages"][2]["content"]
        cl = clauses(q); rel = relations(q); gl = goal(q)
        given = ("Given:\n" + "\n".join("- " + c for c in cl) + "\n") if cl else ""
        rels = ("Relations:\n" + "\n".join("- " + r for r in rel) + "\n") if rel else ""
        a = json.loads(line); a["messages"][2]["content"] = given + f"Goal: {gl}\nSolution:\n" + sol
        b = json.loads(line); b["messages"][2]["content"] = given + rels + f"Goal: {gl}\nSolution:\n" + sol
        gg.write(json.dumps(a) + "\n"); gr.write(json.dumps(b) + "\n")
        stats["n"] += 1; stats["rel_ex"] += bool(rel); stats["rel"] += len(rel)
print(f"wrote {stats['n']} examples: gsm8k_trainsplit_gg.jsonl, gsm8k_trainsplit_gr.jsonl; "
      f"relations in {stats['rel_ex']} examples ({stats['rel']} lines)")
