"""Paired comparison of two sets of validation generations (same questions, several seeds).
Usage: python paired_stats.py "logs/valgen_A_s*_128.jsonl" "logs/valgen_B_s*_128.jsonl"
Per question: mean over seeds of (B correct - A correct); 95% CI by bootstrap over questions (10,000 resamples).
Also prints each file's accuracy. Seeds are matched by sorted file name, so use the same seed order in both globs."""
import glob, json, random, sys
fa, fb = sorted(glob.glob(sys.argv[1])), sorted(glob.glob(sys.argv[2]))
assert fa and len(fa) == len(fb), f"need equally many files: {len(fa)} vs {len(fb)}"
load = lambda f: [int(json.loads(l)["correct"]) for l in open(f)]
A, B = [load(f) for f in fa], [load(f) for f in fb]
for f, x in zip(fa + fb, A + B): print(f"{f}: {sum(x) / len(x) * 100:.1f}")
n = len(A[0]); d = [sum(b[i] - a[i] for a, b in zip(A, B)) / len(A) for i in range(n)]
m = sum(d) / n * 100; rng = random.Random(0)
bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n * 100 for _ in range(10000))
print(f"mean A {sum(map(sum, A)) / len(A) / n * 100:.2f}  mean B {sum(map(sum, B)) / len(B) / n * 100:.2f}  paired diff B-A {m:+.2f} pts  95% CI [{bs[250]:+.2f}, {bs[9750]:+.2f}]")
