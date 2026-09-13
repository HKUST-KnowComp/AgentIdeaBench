"""Figure for E27 closed-loop SWM (4-backbone honest version). Reads reports/e27_swm.json.
Panel 1: pooled Active+SWM - Active delta for the three designs run on all 4 backbones
         (S2/S4/S4b) with win rates + p. S1/S3 were qwen-9b-only ablations, omitted here.
Panel 2: per-backbone delta for S2/S4/S4b across the 4 backbones — shows S4b best on the
         qwen family, S4 best on the (saturated) deepseek family.
Panel 3: per-FAMILY per-dimension S4b delta (qwen vs deepseek) — the capability gate:
         qwen gains originality+feasibility together; deepseek trades originality for feasibility.
Writes reports/figures/summary/swm.png . /usr/bin/python3.

Requires the per-family per-dim numbers, recomputed here from swm_scores (read-only)."""
import json, sqlite3, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as W
from experiments.e10_idea_anchor_calibration import trimmed_mean
import config as cfg
WSUM = sum(W.values())

D = json.loads((ROOT / "reports" / "e27_swm.json").read_text())
OUT = ROOT / "reports" / "figures" / "summary"; OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b",
          "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"]
MLAB = ["qwen\n9b", "qwen\n27b", "v4\nflash", "v4\npro"]
QWEN = MODELS[:2]; DEEP = MODELS[2:]
CONDS = ["swm_S2", "swm_S4", "swm_S4b"]
SHORT = {"swm_S2": "S2\nmulti-role", "swm_S4": "S4\ndynamic", "swm_S4b": "S4b\ndyn+hard"}
COL = {"swm_S2": "#e08214", "swm_S4": "#7fbf7b", "swm_S4b": "#1a9850"}

# ---- recompute per-family per-dim S4b - base (read-only) ----
conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
cell = defaultdict(lambda: defaultdict(list))
for r in conn.execute("SELECT gen_model,subdomain,condition,scores_json "
                      "FROM swm_scores WHERE scores_json IS NOT NULL"):
    s = json.loads(r["scores_json"]); k = (r["gen_model"], r["subdomain"], r["condition"])
    for d in DIMS:
        cell[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
conn.close()
def _tm(v): return trimmed_mean(v) if len(v) >= 2 else (v[0] if v else None)
dimc = {k: {d: _tm(cell[k][d]) for d in DIMS} for k in cell}
byms = defaultdict(dict)
for (m, sub, c), dd in dimc.items():
    byms[(m, sub)][c] = dd
def fam_perdim(models, cond):
    out = {}
    for d in DIMS:
        v = [byms[k][cond][d] - byms[k]["active_base"][d] for k in byms if k[0] in models
             and "active_base" in byms[k] and cond in byms[k]
             and byms[k]["active_base"][d] is not None and byms[k][cond][d] is not None]
        out[d] = float(np.mean(v)) if v else 0.0
    return out

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

# Panel 1: pooled delta per design (S2/S4/S4b)
vals = [D[c]["delta_mean"] for c in CONDS]
b = ax[0].bar(range(len(CONDS)), vals, color=[COL[c] for c in CONDS], width=0.62)
for bb, c, v in zip(b, CONDS, vals):
    r = D[c]
    star = " *" if (r.get("wilcoxon_p") or 1) < 0.05 else ""
    ax[0].text(bb.get_x()+bb.get_width()/2, v + 0.006, f"{v:+.2f}{star}", ha="center",
               va="bottom", fontweight="bold", fontsize=9.5)
    ax[0].text(bb.get_x()+bb.get_width()/2, 0.006,
               f"n={r['n_pairs']}\nwin {int(r['delta_win_rate']*100)}%\np={r.get('wilcoxon_p')}",
               ha="center", va="bottom", fontsize=6.8, color="#fff")
ax[0].axhline(0, color="#333", lw=0.8)
ax[0].set_xticks(range(len(CONDS))); ax[0].set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)
ax[0].set_ylabel("Active+SWM − Active (weighted, pooled)")
ax[0].set_title("Pooled over 4 backbones: S4b best & only significant (*p<.05)",
                fontsize=9.3, fontweight="bold")

# Panel 2: per-backbone delta grouped by design
x = np.arange(len(MODELS)); w = 0.80 / len(CONDS)
for i, c in enumerate(CONDS):
    pmd = D[c].get("per_model_detail", {})
    dv = [pmd.get(m, {}).get("delta_mean", 0.0) for m in MODELS]
    off = (i - (len(CONDS)-1)/2) * w
    bar = ax[1].bar(x + off, dv, w, color=COL[c], label=SHORT[c].replace("\n", " "))
    for bb, v in zip(bar, dv):
        ax[1].text(bb.get_x()+bb.get_width()/2, v + (0.008 if v >= 0 else -0.03), f"{v:+.2f}",
                   ha="center", va="bottom" if v >= 0 else "top", fontsize=6.3)
ax[1].axhline(0, color="#333", lw=0.8)
ax[1].axvspan(-0.5, 1.5, color="#1a9850", alpha=0.05)
ax[1].axvspan(1.5, 3.5, color="#d73027", alpha=0.05)
ax[1].set_xticks(x); ax[1].set_xticklabels(MLAB, fontsize=8.5)
ax[1].set_ylabel("Active+SWM − Active (weighted)")
ax[1].legend(fontsize=7.6, loc="upper right")
ax[1].set_title("S4b best on qwen (headroom); saturates on deepseek-v4",
                fontsize=9.6, fontweight="bold")

# Panel 3: per-family per-dim S4b - base
xd = np.arange(5); wd = 0.38
dq = fam_perdim(QWEN, "swm_S4b"); dd = fam_perdim(DEEP, "swm_S4b")
vq = [dq[d] for d in DIMS]; vd = [dd[d] for d in DIMS]
ax[2].bar(xd - wd/2, vq, wd, color="#1a9850", label="qwen family")
ax[2].bar(xd + wd/2, vd, wd, color="#d73027", label="deepseek-v4 family")
for xi, v in zip(xd - wd/2, vq):
    ax[2].text(xi, v + (0.01 if v >= 0 else -0.03), f"{v:+.2f}", ha="center",
               va="bottom" if v >= 0 else "top", fontsize=6.6)
for xi, v in zip(xd + wd/2, vd):
    ax[2].text(xi, v + (0.01 if v >= 0 else -0.03), f"{v:+.2f}", ha="center",
               va="bottom" if v >= 0 else "top", fontsize=6.6, fontweight="bold")
ax[2].axhline(0, color="#333", lw=0.8)
ax[2].set_xticks(xd); ax[2].set_xticklabels([d[:5] for d in DIMS], fontsize=9)
ax[2].set_ylabel("S4b − base (per dimension)")
ax[2].legend(fontsize=8, loc="lower left")
ax[2].set_title("The gate: qwen gains originality+feasibility;\ndeepseek trades originality for feasibility",
                fontsize=9.3, fontweight="bold")

fig.suptitle("Closed-loop Scientific World Model (S4b: dynamic panel + hard-constrained aggregation) is the best "
             "design pooled\n(+0.245, p=0.040, only one significant), but its lift is capability-gated like search "
             "itself: strong on qwen, saturated on deepseek-v4",
             fontsize=10.8, fontweight="bold", y=1.06)
fig.tight_layout()
fig.savefig(OUT / "swm.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "swm.png")
