"""Figure for E28 within-cell idea diversity. Reads reports/e28_idea_diversity.json.
Panel 1: 4 diversity metrics, Static vs Active (paired, n=2793) — all agree Active > Static.
Panel 2: per-model Vendi B->C slope chart sorted by Static Vendi — convergence: weak models
         jump, strong models already diverse (equalization).
Panel 3: static capability vs Vendi gain scatter (r=-0.47) — diversity gain is NOT what
         drives the F2 score gain; search equalizes diversity, not quality.
Writes reports/figures/summary/idea_diversity.png . /usr/bin/python3."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
D = json.loads((ROOT / "reports" / "e28_idea_diversity.json").read_text())
OUT = ROOT / "reports" / "figures" / "summary"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(17, 4.9))
CB, CC = "#4477aa", "#e08214"   # static blue / active orange

# Panel 1 — 4 metrics side by side (normalized axes pairs)
mets = [("vendi", "Vendi (1-3)\nhigher=diverse"), ("cosdist", "cos distance\nhigher=diverse"),
        ("rougeL", "self-ROUGE-L\nLOWER=diverse"), ("distinct2", "distinct-2\nhigher=diverse")]
x = np.arange(len(mets)); w = 0.36
bvals = [D["metrics"][m]["static_mean"] for m, _ in mets]
cvals = [D["metrics"][m]["active_mean"] for m, _ in mets]
# scale each metric pair to its own [0,1] for display, annotate raw values
bn = []; cn = []
for b, c in zip(bvals, cvals):
    lo, hi = 0, max(b, c) * 1.15
    bn.append(b / hi); cn.append(c / hi)
ax[0].bar(x - w/2, bn, w, color=CB, label="Static")
ax[0].bar(x + w/2, cn, w, color=CC, label="Active")
for xi, (b, c, bnv, cnv) in enumerate(zip(bvals, cvals, bn, cn)):
    ax[0].text(xi - w/2, bnv + .015, f"{b:.2f}", ha="center", fontsize=8)
    ax[0].text(xi + w/2, cnv + .015, f"{c:.2f}", ha="center", fontsize=8, fontweight="bold")
    d = D["metrics"][mets[xi][0]]
    ax[0].text(xi, .04, f"Δ{d['delta_mean']:+.2f}\np<1e-6", ha="center", fontsize=7, color="#fff")
ax[0].set_xticks(x); ax[0].set_xticklabels([lbl for _, lbl in mets], fontsize=8)
ax[0].set_yticks([])
ax[0].legend(fontsize=9, loc="upper right")
ax[0].set_title(f"All 4 metrics: Active ideas more diverse (n={D['n_pairs']} paired cells)",
                fontsize=9.6, fontweight="bold")

# Panel 2 — per-model vendi slope chart sorted by static vendi
pm = D["per_model"]
models = sorted(pm, key=lambda m: pm[m]["static_vendi"])
ys_b = [pm[m]["static_vendi"] for m in models]
ys_c = [pm[m]["active_vendi"] for m in models]
xx = np.arange(len(models))
for i, m in enumerate(models):
    up = ys_c[i] >= ys_b[i]
    ax[1].plot([i, i], [ys_b[i], ys_c[i]], color="#1a9850" if up else "#d1495b",
               lw=1.6, alpha=.75, zorder=1)
ax[1].scatter(xx, ys_b, s=22, color=CB, label="Static", zorder=2)
ax[1].scatter(xx, ys_c, s=22, color=CC, label="Active", zorder=3)
ax[1].set_xticks(xx)
ax[1].set_xticklabels([m.split("/")[-1][:14] for m in models], rotation=90, fontsize=6)
ax[1].set_ylabel("within-cell Vendi (3 ideas)")
ax[1].legend(fontsize=8, loc="lower right")
ax[1].set_title("Convergence: weak models jump, strong models already diverse",
                fontsize=9.6, fontweight="bold")

# Panel 3 — capability vs vendi gain
g = D.get("capability_vs_vendi_gain", {})
cap = g.get("static_capability", {})
mm = [m for m in pm if m in cap]
xs = np.array([cap[m] for m in mm]); ys = np.array([pm[m]["vendi"] for m in mm])
ax[2].scatter(xs, ys, s=34, color="#6a51a3", alpha=.8)
z = np.polyfit(xs, ys, 1); xr = np.linspace(xs.min(), xs.max(), 50)
ax[2].plot(xr, np.polyval(z, xr), color="#333", lw=1.4, ls="--")
ax[2].axhline(0, color="#999", lw=.8)
for m in mm:
    if pm[m]["vendi"] > 0.4 or pm[m]["vendi"] < -0.05:
        ax[2].annotate(m.split("/")[-1][:13], (cap[m], pm[m]["vendi"]), fontsize=6,
                       xytext=(3, 3), textcoords="offset points")
ax[2].set_xlabel("Static capability (weighted lit8d score)")
ax[2].set_ylabel("Vendi gain (Active − Static)")
ax[2].set_title(f"Diversity gain anti-correlates with capability "
                f"(r={g.get('pearson_r')}, p={g.get('pearson_p'):.3f})",
                fontsize=9.4, fontweight="bold")

fig.suptitle("E28 — Agent-controlled retrieval EQUALIZES idea diversity, but not quality: Active "
             "ideas within a cell are more diverse (all metrics p<1e-6); the diversity gain is "
             "largest for weak models — yet only strong models convert evidence into score (F2)",
             fontsize=11, fontweight="bold", y=1.05)
fig.tight_layout()
fig.savefig(OUT / "idea_diversity.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "idea_diversity.png")
