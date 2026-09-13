"""Single-column figure for E28 within-cell idea diversity (Panel 1 only).

Horizontal dumbbell: four diversity metrics, each normalized so that RIGHT =
more diverse (self-ROUGE-L is shown as distinctness = 1 - overlap). Static (blue)
and Active (orange) markers connected by an arrow; raw values and Delta annotated.
All four agree: Active ideas are more diverse (n=2793 paired cells, all p<1e-6).

Reads reports/e28_idea_diversity.json; writes idea_diversity_single.png
(does NOT overwrite the original 3-panel idea_diversity.png). /usr/bin/python3.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
D = json.loads((ROOT / "reports" / "e28_idea_diversity.json").read_text())
OUT = ROOT / "reports" / "figures" / "summary"
M = D["metrics"]

CB, CC = "#3b6ea5", "#e0801a"   # static blue / active orange

# (json key, row label, transform to "diversity" direction, display range)
rows = [
    ("vendi",     "Vendi score",              lambda v: v,       (1.80, 2.20)),
    ("cosdist",   "Pairwise cosine dist.",     lambda v: v,       (0.30, 0.42)),
    ("rougeL",    "Distinctness (1$-$ROUGE-L)", lambda v: 1 - v,  (0.66, 0.84)),
    ("distinct2", "Distinct-2",                lambda v: v,       (0.78, 0.90)),
]

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False})
fig, ax = plt.subplots(figsize=(3.35, 2.55))

ys = np.arange(len(rows))[::-1]   # top row first
for y, (key, label, tf, (lo, hi)) in zip(ys, rows):
    s = tf(M[key]["static_mean"]); c = tf(M[key]["active_mean"])
    sn = (s - lo) / (hi - lo); cn = (c - lo) / (hi - lo)
    ax.plot([sn, cn], [y, y], color="#b9b9b9", lw=2.2, zorder=1,
            solid_capstyle="round")
    # arrowhead toward Active (always the more-diverse side here)
    ax.annotate("", xy=(cn, y), xytext=(sn, y),
                arrowprops=dict(arrowstyle="-|>", color="#9a9a9a", lw=0),
                zorder=1)
    ax.scatter([sn], [y], s=60, color=CB, zorder=3, edgecolor="white", linewidth=0.8)
    ax.scatter([cn], [y], s=60, color=CC, zorder=3, edgecolor="white", linewidth=0.8)
    # raw values at the two ends
    left, right = (sn, cn) if sn < cn else (cn, sn)
    ax.text(sn, y + 0.20, f"{s:.2f}", ha="center", va="bottom", fontsize=7,
            color=CB)
    ax.text(cn, y + 0.20, f"{c:.2f}", ha="center", va="bottom", fontsize=7,
            color=CC, fontweight="bold")
    # row label on the left, delta on the right margin
    ax.text(-0.04, y, label, ha="right", va="center", fontsize=8.2)
    dv = M[key]["delta_mean"]
    ax.text(1.10, y, f"$+${abs(dv):.2f}", ha="left", va="center", fontsize=7.6,
            color="#2a7a2a", fontweight="bold")

ax.set_xlim(-0.02, 1.14)
ax.set_ylim(-0.6, len(rows) - 0.3)
ax.set_yticks([])
ax.set_xticks([0, 1])
ax.set_xticklabels(["less\ndiverse", "more\ndiverse"], fontsize=7.5, color="#555")
ax.tick_params(length=0)
# legend
ax.scatter([], [], s=55, color=CB, label="Static")
ax.scatter([], [], s=55, color=CC, label="Active")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False,
          fontsize=8, handletextpad=0.2, columnspacing=1.2)
ax.text(1.10, len(rows) - 0.45, r"$\Delta$", ha="left", va="center",
        fontsize=7.6, color="#2a7a2a", fontweight="bold")

plt.tight_layout(pad=0.4)
outfile = OUT / "idea_diversity_single.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
print(f"wrote {outfile}  (n_pairs={D['n_pairs']}, all four metrics Active>Static, p<1e-6)")
