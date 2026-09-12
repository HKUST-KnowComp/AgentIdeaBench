"""Figure for the metric-critique experiment. Reads reports/e11_ref_overlap/metric_critique.json.
Three panels:
  (1) centroid-cosine dynamic range is pinned near 0.88 within a topic (floor problem)
  (2) sensitivity: replace all papers -> cosine barely moves, Jaccard collapses
  (3) honest set metrics DO separate the two searches from a random split
Writes reports/figures/summary/metric_critique.png . /usr/bin/python3.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
D = json.loads((ROOT / "reports" / "e11_ref_overlap" / "metric_critique.json").read_text())
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

# ---- Panel 1: dynamic range / floor ----
cats = ["Different\nsubfield", "STATIC vs ACTIVE\n(observed)", "Same subfield,\nrandom split", "Identical\nset"]
vals = [D["cross_topic"]["centroid_cos_mean"],
        D["observed"]["centroid_cos_mean"],
        D["same_topic_floor"]["centroid_cos_mean"], 1.0]
colors = ["#9aa0a6", "#d1495b", "#4c78a8", "#59a14f"]
bars = ax[0].bar(range(4), vals, color=colors, width=0.68)
for b, v in zip(bars, vals):
    ax[0].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
ax[0].axhspan(D["same_topic_floor"]["centroid_cos_mean"], 1.0, color="#4c78a8", alpha=0.06)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(cats, fontsize=9)
ax[0].set_ylim(0, 1.12); ax[0].set_ylabel("centroid-cosine similarity")
ax[0].set_title("Panel 1 — the metric saturates within a topic", fontsize=11, fontweight="bold")
ax[0].annotate("observed 0.87 sits AT the\nsame-subfield floor (0.88)\n→ no signal beyond \"same topic\"",
               xy=(1, vals[1]), xytext=(0.15, 0.42), fontsize=8.3,
               arrowprops=dict(arrowstyle="->", color="#d1495b"))

# ---- Panel 2: sensitivity curve ----
S = D["sensitivity"]
fr = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
cos = [S[str(f)]["cos_mean"] for f in fr]
jac = [S[str(f)]["jac_mean"] for f in fr]
swap = [f * 100 for f in fr]
ax[1].plot(swap, cos, "-o", color="#d1495b", lw=2.4, label="centroid-cosine")
ax[1].plot(swap, jac, "-s", color="#333333", lw=2.4, label="Jaccard (actual papers)")
ax[1].set_ylim(-0.05, 1.08); ax[1].set_xlabel("% of papers swapped for random same-topic papers")
ax[1].set_ylabel("similarity to original set")
ax[1].legend(loc="center left", fontsize=9)
ax[1].set_title("Panel 2 — swap every paper, cosine barely moves", fontsize=11, fontweight="bold")
ax[1].annotate(f"all papers replaced\ncosine still {cos[-1]:.2f}", xy=(100, cos[-1]),
               xytext=(45, 0.66), fontsize=8.3, arrowprops=dict(arrowstyle="->", color="#d1495b"))
ax[1].annotate("Jaccard = 0\n(nothing shared)", xy=(100, 0.0), xytext=(45, 0.12), fontsize=8.3,
               arrowprops=dict(arrowstyle="->", color="#333333"))

# ---- Panel 3: honest set metrics discriminate ----
AM = D["alt_metrics"]
labels = ["mutual-kNN\n(nbrs in other set)", "energy distance\n(point clouds)"]
obs = [AM["mutual_knn_other_frac"]["observed_static_vs_active"],
       AM["energy_dist"]["observed_static_vs_active"]]
nul = [AM["mutual_knn_other_frac"]["same_topic_random_split"],
       AM["energy_dist"]["same_topic_random_split"]]
import numpy as np
x = np.arange(2); w = 0.36
b1 = ax[2].bar(x - w / 2, obs, w, color="#d1495b", label="STATIC vs ACTIVE (observed)")
b2 = ax[2].bar(x + w / 2, nul, w, color="#4c78a8", label="same-subfield random split (null)")
for bs in (b1, b2):
    for b in bs:
        ax[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.2f}",
                   ha="center", fontsize=8.5, fontweight="bold")
ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=9)
ax[2].set_ylim(0, 0.63); ax[2].legend(fontsize=8.3, loc="upper right")
ax[2].set_title("Panel 3 — honest metrics DO see the difference", fontsize=11, fontweight="bold")
ax[2].annotate("lower than null → the two\nsearches really are separated", xy=(0 - w / 2, obs[0]),
               xytext=(0.05, 0.12), fontsize=8.0, arrowprops=dict(arrowstyle="->", color="#d1495b"))

fig.suptitle("Why mean-embedding cosine is a poor richness/diversity metric",
             fontsize=13.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "metric_critique.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "metric_critique.png")
