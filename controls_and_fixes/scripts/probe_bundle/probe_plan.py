"""Plan/binding probe (pre-build gate for latent plan tokens). Validation only; the GSM8K test set is never read.

Question: does the question's hidden state already contain the plan (operators, step count) and the binding (which
question numbers are used), so that the failure is in USING it? Or does the state lack it?

Targets come from the gold solutions (annotations stripped) parsed with the SAME parser as the error analysis
(classify_errors.parse_eqs), so gold and generated plans are parsed identically:
  n      number of equations that contain an operator (capped at 6)
  first  operator string of the first such equation (rare classes -> 'other')
  op+ op- op* op/   operator present anywhere in the solution  ('opset' = all four correct)
  used   per question number: does it appear as an operand? (binding / distractor flag)
Probes: logistic regression on frozen states (last prompt token and mean over question tokens; 3 layers; number-token
states for 'used'). Layer/pooling/C are chosen on a held-out 20% of TRAIN, never on validation.
Baselines: majority, TF-IDF on question text, shuffled labels. Generated plan: parsed from the saved V_base outputs.

Pre-specified gates (opset exact match, validation, items with a parsed gold plan):
  G1  probe - generated >= 5.0 points       -> information is in the state, generation under-uses it (latent plan tokens promising)
  G3  probe <  generated                    -> a linear readout at one position is below the written plan (prefer idea B).
                                               WEAK evidence: generation sees the whole question token by token, so a linear probe is a
                                               conservative instrument; only G1 is strong evidence for latent plan tokens.
  --  otherwise                             -> inconclusive
  G2  probe - TF-IDF <= 2.0 points (reported separately) -> state adds little over the surface text

Usage: python probe_plan.py --model ./ft_V_base --tag Vbase --gen logs/valgen_V_base_128.jsonl
       python probe_plan.py --model meta-llama/Llama-3.2-1B-Instruct --tag frozen        (also run: no train memorisation)
Options: --train gsm8k_trainsplit_noanno.jsonl --val gsm8k_val.jsonl --limit N (test) --reuse (reuse cached features)
"""
import argparse, json, os, re, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_errors import parse_eqs
from align_utils import _numbers

G_UNDERUSE, G_SURFACE = 5.0, 2.0
OPS = ["+", "-", "*", "/"]
PLAN_T = ["n", "first", "op+", "op-", "op*", "op/"]


# ----------------------------------------------------------------------------------------- targets
def parse_plan(text):
    return [e for e in parse_eqs(re.sub(r"<<[^>]*>>", "", text).split("####")[0]) if e[1]]


def plan_of(eqs):
    return {"n": min(len(eqs), 6), "first": "".join(eqs[0][1]) if eqs else "",
            **{"op" + o: int(any(o in e[1] for e in eqs)) for o in OPS}}


def used_set(eqs):
    return {round(v, 4) for e in eqs for v in e[0] if v is not None}


def is_used(v, S):
    return round(v, 4) in S or round(v / 100, 4) in S


def build(rows):
    D = {"Q": [r[1]["content"] for r in rows], "eqs": [parse_plan(r[2]["content"]) for r in rows]}
    D["ok"] = np.array([len(e) > 0 for e in D["eqs"]])
    D["plan"] = [plan_of(e) for e in D["eqs"]]
    D["nums"] = [_numbers(q) for q in D["Q"]]
    D["used"] = [[int(is_used(v, used_set(e))) for v, _, _ in nm] for nm, e in zip(D["nums"], D["eqs"])]
    return D


# ----------------------------------------------------------------------------------------- features
def extract(model_dir, rows, D, layers, bs):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    nl = m.config.num_hidden_layers
    layers = layers or sorted({round(nl * f) for f in (0.5, 0.75, 1.0)})
    prompts = [tok.apply_chat_template([x for x in r if x["role"] != "assistant"], tokenize=False, add_generation_prompt=True) for r in rows]
    N = len(rows); d = m.config.hidden_size
    last = {l: np.zeros((N, d), np.float16) for l in layers}; mean = {l: np.zeros((N, d), np.float16) for l in layers}
    off = np.cumsum([0] + [len(x) for x in D["nums"]]); num = {l: np.zeros((off[-1], d), np.float16) for l in layers}
    order = sorted(range(N), key=lambda i: len(prompts[i])); t0 = time.time()
    for s in range(0, N, bs):
        idx = order[s:s + bs]
        x = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, return_offsets_mapping=True, truncation=True, max_length=640)
        offs = x.pop("offset_mapping").numpy(); lens = x["attention_mask"].sum(1).numpy()
        with torch.no_grad():
            hs = m(**{k: v.to(dev) for k, v in x.items()}, output_hidden_states=True).hidden_states
        for b, i in enumerate(idx):
            q = D["Q"][i]; qi = prompts[i].find(q)
            if qi < 0: raise SystemExit(f"question {i} not found verbatim in the chat prompt")
            st, en = offs[b, :lens[b], 0], offs[b, :lens[b], 1]

            def tk(c): return int(max(0, min(lens[b] - 1, np.searchsorted(st, c, side="right") - 1)))
            qa, qb = tk(qi), tk(qi + len(q) - 1)
            for l in layers:
                H = hs[l][b].float().cpu().numpy()
                last[l][i] = H[lens[b] - 1]; mean[l][i] = H[qa:qb + 1].mean(0)
                for k, (_, a, e) in enumerate(D["nums"][i]): num[l][off[i] + k] = H[tk(qi + e - 1)]
        if (s // bs) % 20 == 0: print(f"[probe] {s}/{N} {time.time() - t0:.0f}s", flush=True)
    return layers, last, mean, num, off


# ----------------------------------------------------------------------------------------- probes
def lrfit(Xtr, ytr, Xte, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(set(ytr)) < 2: return np.full(len(Xte), ytr[0])
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(C=C, max_iter=200, tol=1e-3).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def targets(D, idx, keep):
    return {"n": np.array([D["plan"][i]["n"] for i in idx]), "first": np.array([D["plan"][i]["first"] if D["plan"][i]["first"] in keep else "other" for i in idx]),
            **{t: np.array([D["plan"][i][t] for i in idx]) for t in PLAN_T[2:]}}


def exact(P, T):
    return np.mean(np.all([P[t] == T[t] for t in PLAN_T[2:]], axis=0)) * 100


def plan_fit(Xtr, Ttr, Xte, C):
    return {t: lrfit(Xtr, Ttr[t], Xte, C) for t in PLAN_T}


def holdout_score(Xtr, Ttr, tr, ho, C):
    P = plan_fit(Xtr[tr], {t: v[tr] for t, v in Ttr.items()}, Xtr[ho], C)
    return np.mean([np.mean(P[t] == Ttr[t][ho]) for t in PLAN_T])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--gen", required=True)
    ap.add_argument("--train", default="gsm8k_trainsplit_noanno.jsonl"); ap.add_argument("--val", default="gsm8k_val.jsonl")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--reuse", action="store_true"); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--layers", default=""); ap.add_argument("--out", default="probe_results")
    a = ap.parse_args()
    from joblib import Parallel, delayed
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    import scipy.sparse as sp
    rd = lambda p: [json.loads(l)["messages"] for l in open(p)][:a.limit or None]
    TR, VA = rd(a.train), rd(a.val); G = [json.loads(l) for l in open(a.gen)][:len(VA)]
    assert len(G) == len(VA) and all(g["question"] == v[1]["content"] for g, v in zip(G, VA)), "gen file does not align with the validation file"
    Dtr, Dva = build(TR), build(VA)
    cache = f"probe_cache_{a.tag}.npz"; layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    if a.reuse and os.path.exists(cache):
        z = np.load(cache); layers = [int(x) for x in z["layers"]]
        F = {k: {l: z[f"{k}_{l}"] for l in layers} for k in ("trl", "trm", "trn", "val", "vam", "van")}; offtr, offva = z["offtr"], z["offva"]
    else:
        layers, trl, trm, trn, offtr = extract(a.model, TR, Dtr, layers, a.bs)
        _, val_, vam, van, offva = extract(a.model, VA, Dva, layers, a.bs)
        F = {"trl": trl, "trm": trm, "trn": trn, "val": val_, "vam": vam, "van": van}
        np.savez(cache, layers=layers, offtr=offtr, offva=offva, **{f"{k}_{l}": F[k][l] for k in F for l in layers})
    itr, iva = np.where(Dtr["ok"])[0], np.where(Dva["ok"])[0]
    cnt = {}
    for i in itr: cnt[Dtr["plan"][i]["first"]] = cnt.get(Dtr["plan"][i]["first"], 0) + 1
    keep = {k for k, v in cnt.items() if v >= 20}
    Ttr, Tva = targets(Dtr, itr, keep), targets(Dva, iva, keep)
    print(f"[probe] train items with parsed plan {len(itr)}/{len(TR)}, validation {len(iva)}/{len(VA)}; layers {layers}", flush=True)
    rs = np.random.RandomState(0); perm = rs.permutation(len(itr)); ho, tr = perm[:len(itr) // 5], perm[len(itr) // 5:]
    # ---- config selection on a held-out slice of train
    cfgs = [(l, p, C) for l in layers for p in ("last", "mean") for C in (0.001, 0.01)]
    feat = lambda k, l, p, I: (F["trl" if k == "tr" else "val"][l] if p == "last" else F["trm" if k == "tr" else "vam"][l])[I].astype(np.float32)
    sc = Parallel(n_jobs=min(4, os.cpu_count() or 1))(delayed(holdout_score)(feat("tr", l, p, itr), Ttr, tr, ho, C) for l, p, C in cfgs)
    (bl, bp, bC), bs_ = cfgs[int(np.argmax(sc))], max(sc)
    print(f"[probe] selected on train holdout: layer {bl} pooling {bp} C {bC} (mean acc {bs_ * 100:.1f})", flush=True)
    Xtr, Xva = feat("tr", bl, bp, itr), feat("va", bl, bp, iva)
    Pp = plan_fit(Xtr, Ttr, Xva, bC)
    shuf = {t: np.random.RandomState(1).permutation(v) for t, v in Ttr.items()}; Ps = plan_fit(Xtr, shuf, Xva, bC)
    # ---- TF-IDF baseline
    qs = lambda D, I: [D["Q"][i] for i in I]
    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(qs(Dtr, itr))
    Ttf, Tte = tv.transform(qs(Dtr, itr)), tv.transform(qs(Dva, iva))

    def tf_fit(Xa, Ya, Xb, C):
        from sklearn.linear_model import LogisticRegression
        return {t: (np.full(Xb.shape[0], Ya[t][0]) if len(set(Ya[t])) < 2 else LogisticRegression(C=C, max_iter=300).fit(Xa, Ya[t]).predict(Xb)) for t in PLAN_T}
    tfC = max((1.0, 10.0), key=lambda C: np.mean([np.mean(v == Ttr[t][ho]) for t, v in tf_fit(Ttf[tr], {t: v[tr] for t, v in Ttr.items()}, Ttf[ho], C).items()]))
    Pt = tf_fit(Ttf, Ttr, Tte, tfC)
    maj = {t: np.full(len(iva), max(set(Ttr[t]), key=list(Ttr[t]).count)) for t in PLAN_T}
    # ---- generated plan (same parser)
    gp = [plan_of(parse_plan(G[i]["output"])) for i in iva]
    Pg = {"n": np.array([x["n"] for x in gp]), "first": np.array([x["first"] if x["first"] in keep else "other" for x in gp]), **{t: np.array([x[t] for x in gp]) for t in PLAN_T[2:]}}
    wrong = np.array([not G[i]["correct"] for i in iva])
    R = {"n_val": int(len(iva)), "config": {"layer": bl, "pool": bp, "C": bC, "tfidf_C": tfC}}
    rows = [("majority", maj), ("shuffled labels", Ps), ("TF-IDF (question text)", Pt), ("probe (frozen state)", Pp), ("generated by model", Pg)]
    L = [f"Plan probe, model {a.model} (tag {a.tag}); validation items with parsed gold plan: {len(iva)}; probe config layer {bl}/{bp}/C={bC}",
         f"{'':26s} {'opset exact':>11s} {'n steps':>8s} {'1st op':>7s}   {'opset exact | V_base wrong':>26s}"]
    for name, P in rows:
        r = {"opset": exact(P, Tva), "n": float(np.mean(P["n"] == Tva["n"]) * 100), "first": float(np.mean(P["first"] == Tva["first"]) * 100),
             "opset_wrong": exact({t: v[wrong] for t, v in P.items()}, {t: v[wrong] for t, v in Tva.items()}) if wrong.any() else float("nan")}
        R[name] = r; L.append(f"{name:26s} {r['opset']:11.1f} {r['n']:8.1f} {r['first']:7.1f}   {r['opset_wrong']:26.1f}")
    d_gen = R["probe (frozen state)"]["opset"] - R["generated by model"]["opset"]; d_tf = R["probe (frozen state)"]["opset"] - R["TF-IDF (question text)"]["opset"]
    # ---- robustness: probe trained inside validation only (5-fold CV), no train/val state shift
    kf = KFold(5, shuffle=True, random_state=0); Pcv = {t: np.empty(len(iva), dtype=Tva[t].dtype) for t in PLAN_T}
    for a_, b_ in kf.split(iva):
        out = plan_fit(Xva[a_], {t: v[a_] for t, v in Tva.items()}, Xva[b_], bC)
        for t in PLAN_T: Pcv[t][b_] = out[t]
    R["probe val-CV opset"] = exact(Pcv, Tva)
    # ---- binding: is each question number used in the solution?
    def numrows(D, I, off): return np.concatenate([np.arange(off[i], off[i + 1]) for i in I]) if len(I) else np.array([], int)
    ntr, nva = numrows(Dtr, itr, offtr), numrows(Dva, iva, offva)
    ytr_b = np.array([u for i in itr for u in Dtr["used"][i]]); yva_b = np.array([u for i in iva for u in Dva["used"][i]])
    qtr = np.array([i for i in itr for _ in Dtr["used"][i]]); hq = set(itr[ho])
    hmask = np.array([q in hq for q in qtr])
    bsc = [(l, C, float(np.mean(lrfit(F["trn"][l][ntr[~hmask]].astype(np.float32), ytr_b[~hmask], F["trn"][l][ntr[hmask]].astype(np.float32), C) == ytr_b[hmask])))
           for l in layers for C in (0.001, 0.01)]
    bl2, bC2, _ = max(bsc, key=lambda x: x[2])
    pb = lrfit(F["trn"][bl2][ntr].astype(np.float32), ytr_b, F["van"][bl2][nva].astype(np.float32), bC2)

    def surf(D, I):
        txt, dense = [], []
        for i in I:
            q = D["Q"][i]; nm = D["nums"][i]
            for k, (v, s, e) in enumerate(nm):
                txt.append(q[max(0, s - 40):s] + " NUM " + q[e:e + 30])
                dense.append([s / max(1, len(q)), k / max(1, len(nm)), len(nm) / 10, float(k == len(nm) - 1), np.log10(abs(v) + 1), float(v == int(v)), float("%" in q[e:e + 8]), float("$" in q[max(0, s - 2):e + 2])])
        return txt, np.array(dense)
    ttr, dtr = surf(Dtr, itr); tva, dva = surf(Dva, iva)
    tv2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(ttr)
    from sklearn.linear_model import LogisticRegression
    Xs_tr, Xs_va = sp.hstack([tv2.transform(ttr), sp.csr_matrix(dtr)]).tocsr(), sp.hstack([tv2.transform(tva), sp.csr_matrix(dva)]).tocsr()
    ps = LogisticRegression(C=1.0, max_iter=300).fit(Xs_tr, ytr_b).predict(Xs_va)
    pg = np.array([int(is_used(v, used_set(parse_plan(G[i]["output"])))) for i in iva for v, _, _ in Dva["nums"][i]])
    bal = lambda p, y: float((np.mean(p[y == 1] == 1) + np.mean(p[y == 0] == 0)) / 2 * 100) if (y == 1).any() and (y == 0).any() else float("nan")
    L += ["", f"Binding: is each question number used in the solution? numbers {len(yva_b)}, used {yva_b.mean() * 100:.1f}% (layer {bl2}, C {bC2}); "
              f"only {int((1 - yva_b.mean()) * len(yva_b))} unused numbers, so read the balanced column and expect low power",
          f"{'':26s} {'accuracy':>9s} {'balanced':>9s}"]
    for name, p in (("majority", np.full(len(yva_b), int(ytr_b.mean() >= 0.5))), ("surface text + position", ps), ("probe (number-token state)", pb), ("generated by model", pg)):
        R["bind " + name] = {"acc": float(np.mean(p == yva_b) * 100), "bal": bal(p, yva_b)}; L.append(f"{name:26s} {R['bind ' + name]['acc']:9.1f} {R['bind ' + name]['bal']:9.1f}")
    # ---- can the state predict whether V_base is right? (5-fold CV inside validation; informational)
    yc = np.array([int(G[i]["correct"]) for i in iva]); auc = {}
    if 0 < yc.sum() < len(yc):
        for name, X in (("probe state", StandardScaler_(Xva)), ("TF-IDF", Tte), ("gold step count (oracle)", np.eye(7)[Tva["n"]])):
            oof = np.zeros(len(yc))
            for a_, b_ in kf.split(iva): oof[b_] = LogisticRegression(C=0.01 if name == "probe state" else 1.0, max_iter=300).fit(X[a_], yc[a_]).predict_proba(X[b_])[:, 1]
            auc[name] = float(roc_auc_score(yc, oof))
        L += ["", "Predicting whether V_base answers correctly (AUC, 5-fold CV in validation): " + ", ".join(f"{k} {v:.3f}" for k, v in auc.items())]
    R["auc_correct"] = auc
    verdict = "G1: information is in the state and generation under-uses it -> latent plan tokens promising" if d_gen >= G_UNDERUSE else \
              ("G3: linear readout is below the written plan -> prefer idea B (weak evidence: a one-position linear probe is a conservative instrument)" if d_gen < 0 else "inconclusive: probe is between the generated plan and +5 points")
    L += ["", f"Probe - generated = {d_gen:+.1f} pts; probe - TF-IDF = {d_tf:+.1f} pts; probe trained inside validation (CV) opset {R['probe val-CV opset']:.1f}",
          f"VERDICT: {verdict}" + ("  [G2 also holds: probe adds <= 2 pts over the surface text]" if d_tf <= G_SURFACE else ""),
          "Caveats: plans parsed by a heuristic parser (same for gold and generated); if this run used a model trained on the train split, train states may be memorised -> also run the frozen base model and compare."]
    R["verdict"] = verdict; R["d_gen"], R["d_tfidf"] = d_gen, d_tf
    print("\n".join(L)); open(f"{a.out}_{a.tag}.md", "w").write("```\n" + "\n".join(L) + "\n```\n"); json.dump(R, open(f"{a.out}_{a.tag}.json", "w"), indent=1)


def StandardScaler_(X):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)


if __name__ == "__main__":
    main()
