"""Accuracy comparison of every method tried (test and validation). Data are copied from
reproduction/results_stage1_output.txt and controls_and_fixes/results_stage2_output.txt."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "text.color": TXT, "axes.edgecolor": GRID})

def pct(*v): return [100 * x for x in v]

# ---------- Figure A: test accuracy, 128-token limit (seeds as dots, bar = mean) ----------
A = [  # (group, colour, label, seeds)
 ("Original pipeline (end token masked, annotations kept)", BLUE, [
  ("Regular", pct(.3434, .3487, .3434)),
  ("LLM-JEPA", pct(.3677, .3806, .3692)),
  ("Shuffled targets (control)", pct(.3609, .3753, .3594)),
  ("Random target (control)", pct(.3472, .3359, .3419)),
  ("Constant target (control)", pct(.3328, .3298, .3442)),
  ("λ = 0, JEPA term off (control)", pct(.3290)),
  ("Step-level targets", pct(.3313)),
  ("Stop-gradient (never stops, see note)", pct(.1638))]),
 ("End-token fixed, annotations kept", ORANGE, [
  ("Regular + fix", pct(.3647, .3776, .3632)),
  ("LLM-JEPA + fix", pct(.3723, .3715, .3745)),
  ("Stop-gradient + fix", pct(.3723))]),
 ("End-token fixed, annotations stripped", AQUA, [
  ("Regular + fix", pct(.4033, .3958, .3927)),
  ("LLM-JEPA + fix", pct(.3988, .3950, .3942)),
  ("Regular + fix, 256-token limit", pct(.4230))]),
]
fig, ax = plt.subplots(figsize=(11, 7.6), dpi=200)
y, ticks, labels = 0, [], []
for g, col, rows in A:
    ax.text(30.1, y - 0.15, g, fontsize=9.5, fontweight="bold", color=TXT, va="center")
    y += 1
    for lab, v in rows:
        m = np.mean(v)
        if m < 30:      # stop-gradient run: far off scale
            ax.annotate("16.38 (off scale)", (30.35, y), fontsize=8.5, color=TXT2, va="center")
            ax.plot([30.05], [y], marker="<", color=col, ms=7)
        else:
            ax.barh(y, m - 30, left=30, height=0.5, color=col, alpha=0.9, edgecolor=SURF, linewidth=2, zorder=2)
            if len(v) > 1:
                ax.scatter(v, [y] * len(v), s=22, color=SURF, edgecolor=TXT, linewidth=0.9, zorder=4)
            ax.text(max(v) + 0.2, y, f"{m:.2f}" + ("" if len(v) > 1 else " (1 seed)"), fontsize=8.5, color=TXT2, va="center")
        ticks.append(y); labels.append(lab); y += 1
    y += 0.6
for x, txt, ls in [(32.36, "paper regular 32.36", ":"), (36.36, "paper LLM-JEPA 36.36", "--")]:
    ax.axvline(x, color=TXT2, lw=1, ls=ls, zorder=1)
    ax.text(x, -1.0, txt, fontsize=8, color=TXT2, ha="center", va="bottom", bbox=dict(fc=SURF, ec="none", pad=1.5), zorder=5)
ax.set_xlim(30, 44); ax.set_ylim(y, -1.4)
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("GSM8K test accuracy (%), Llama-3.2-1B-Instruct, 128 new tokens; bar = mean, dots = individual seeds", fontsize=9, color=TXT2)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.suptitle("Test accuracy of every method tried\nThe end-token fix and annotation stripping move results; the JEPA variants do not",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.text(0.01, 0.005, "Stop-gradient (original): 1,318 of 1,319 outputs never stopped; first-marker rescoring gives 36.8. 3 seeds unless marked; the paper's numbers are from arXiv 2509.14252.",
         fontsize=7.5, color=TXT2)
fig.tight_layout(rect=(0, 0.02, 1, 0.94)); fig.savefig("Fig_A_test_accuracy_all_methods.png"); plt.close(fig)

# ---------- Figure B: validation accuracy (1,500 held-out problems, stripped + end-token fix) ----------
B = [
 ("Learning rate (4 epochs)", BLUE, [("1e-5", 47.8), ("2e-5  (chosen baseline)", 49.0), ("4e-5", 43.9), ("8e-5", 38.5)]),
 ("Epochs (lr 2e-5)", ORANGE, [("2 epochs", 46.8), ("3 epochs", 47.5), ("4 epochs  (chosen baseline)", 49.0), ("6 epochs", 46.5)]),
 ("JEPA variants", AQUA, [("B1 InfoNCE, last layer", 48.1), ("B2 layer 12 + EMA centring", 48.5), ("B3 predictor head + stop-gradient", 47.2)]),
 ("Training format and loss", YELLOW, [("Equations only", 41.1), ("Prose control (clean subset)", 47.9), ("Answer-only loss", 50.6)]),
]
fig, ax = plt.subplots(figsize=(10, 6.8), dpi=200)
y, ticks, labels = 0, [], []
ax.axvspan(49.0 - 1.3, 49.0 + 1.3, color="#d9d8d2", alpha=0.55, zorder=0)
ax.axvline(49.0, color=TXT2, lw=1, ls="--", zorder=1)
for g, col, rows in B:
    ax.text(36.1, y - 0.15, g, fontsize=9.5, fontweight="bold", va="center"); y += 1
    for lab, v in rows:
        ax.barh(y, v - 36, left=36, height=0.5, color=col, alpha=0.9, edgecolor=SURF, linewidth=2, zorder=2)
        ax.text(v + 0.2, y, f"{v:.1f}", fontsize=8.5, color=TXT2, va="center", bbox=dict(fc=SURF, ec="none", pad=1), zorder=5)
        ticks.append(y); labels.append(lab); y += 1
    y += 0.6
ax.text(49.0, -1.0, "plain baseline 49.0 (shaded band: ±1 s.e., about 1.3)", fontsize=8, color=TXT2, ha="center", va="bottom", bbox=dict(fc=SURF, ec="none", pad=1.5), zorder=5)
ax.set_xlim(36, 53); ax.set_ylim(y, -1.4)
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Validation accuracy (%), 1,500 held-out training problems (single seed; screening only, not comparable to test scores)", fontsize=9, color=TXT2)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.suptitle("Validation screening of every method tried\nNo JEPA variant beats the plain baseline; only answer-only loss is above it (unconfirmed on test)",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("Fig_B_validation_accuracy_all_methods.png"); plt.close(fig)
print("ok")
