"""Figure for E26 scaffolds (world-model + analogy). Reads reports/e26_worldmodel.json.
Panel 1: per-dimension delta for wm vs analogy — wm moves feasibility, analogy moves
originality (each moves the dim it targets). Panel 2: per-model delta for analogy —
strong capability-gating (biggest model +1.07, small models hurt).
Writes reports/figures/summary/worldmodel.png . /usr/bin/python3."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
J = json.loads((ROOT / "reports" / "e26_worldmodel.json").read_text())
WM, AN = J["wm"], J["analogy"]
OUT = ROOT / "reports" / "figures" / "summary"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))

# Panel 1: per-dimension delta, wm vs analogy
DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
wmv = [WM["per_dim_delta"][d] for d in DIMS]
anv = [AN["per_dim_delta"][d] for d in DIMS]
x = np.arange(5); w = 0.38
b1 = ax[0].bar(x - w/2, wmv, w, color="#4c78a8", label=f"world-model (overall {WM['delta_mean']:+.2f}, n.s.)")
b2 = ax[0].bar(x + w/2, anv, w, color="#e08214", label=f"analogy (overall {AN['delta_mean']:+.2f}, n.s.)")
for bs in (b1, b2):
    for b in bs:
        v = b.get_height()
        ax[0].text(b.get_x()+b.get_width()/2, v + (0.01 if v >= 0 else -0.02), f"{v:+.2f}",
                   ha="center", va="bottom" if v >= 0 else "top", fontsize=7.8, fontweight="bold")
ax[0].axhline(0, color="#333", lw=0.8)
ax[0].set_xticks(x); ax[0].set_xticklabels([d[:5] for d in DIMS], fontsize=9)
ax[0].set_ylim(-0.28, 0.45); ax[0].set_ylabel("scaffold − base (per dimension)")
ax[0].legend(fontsize=8, loc="upper right")
ax[0].set_title("Each scaffold moves the dimension it targets", fontsize=11, fontweight="bold")
ax[0].annotate("world-model → feasibility", xy=(1-w/2, wmv[1]), xytext=(1.1, 0.36),
               fontsize=7.8, color="#4c78a8", arrowprops=dict(arrowstyle="->", color="#4c78a8"))
ax[0].annotate("analogy → originality", xy=(0+w/2, anv[0]), xytext=(0.2, 0.40),
               fontsize=7.8, color="#e08214", arrowprops=dict(arrowstyle="->", color="#e08214"))

# Panel 2: per-model delta for analogy (capability gating)
pm = AN["per_model_delta"]
order = ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "xiaomi/mimo-v2.5-pro", "qwen/qwen3.5-397b-a17b"]
order = [m for m in order if m in pm] + [m for m in pm if m not in order]
names = [m.split("/")[-1] for m in order]; vals = [pm[m] for m in order]
col = ["#59a14f" if v > 0 else "#d1495b" for v in vals]
b = ax[1].bar(range(len(vals)), vals, color=col, width=0.6)
for bb, v in zip(b, vals):
    ax[1].text(bb.get_x()+bb.get_width()/2, v + (0.02 if v >= 0 else -0.04), f"{v:+.2f}",
               ha="center", va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
ax[1].axhline(0, color="#333", lw=0.8)
ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
ax[1].set_ylabel("analogy − base (weighted score)")
ax[1].set_ylim(-0.55, 1.25)
ax[1].set_title("Analogy is capability-gated: only the largest model gains", fontsize=11, fontweight="bold")
ax[1].annotate("strongest model\n+1.07", xy=(len(vals)-1, vals[-1]), xytext=(len(vals)-2.1, 0.85),
               fontsize=8.3, arrowprops=dict(arrowstyle="->", color="#59a14f"))

fig.suptitle("Targeting originality (analogy) moves originality — but the gain is "
             "capability-gated and offset by feasibility", fontsize=12.5, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(OUT / "worldmodel.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "worldmodel.png")
