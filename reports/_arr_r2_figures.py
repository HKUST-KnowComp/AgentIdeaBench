"""ARR round-2 paper figures (review #10): replace the quartile-binned gate
bar chart with a continuous scatter, and add the cutoff-interaction figure the
paper's cutoff section lacked. Reads reports/e22_new_axis_stats.json (per-model
static/active/cutoff, identical to e37's validated aggregation), e37 (split-half
r + discrimination), e30 (track x cutoff interaction). Read-only; writes two
new PNGs to reports/figures/summary/. /usr/bin/python3.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(20260802)

E22 = json.loads((ROOT / "reports" / "e22_new_axis_stats.json").read_text())
E37 = json.loads((ROOT / "reports" / "e37_review_r2_stats.json").read_text())
E30 = json.loads((ROOT / "reports" / "e30_review_stats.json").read_text())
BLUE, ORANGE, GREEN, RED, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"


def _open_weight_rows():
    return [r for r in E22["per_model"] if not r["model"].startswith("google/gemini-")]


def _boot_band(x, y, xs, n=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    preds = np.empty((n, len(xs)))
    for b in range(n):
        idx = np.random.randint(0, len(x), len(x))
        s, i = np.polyfit(x[idx], y[idx], 1)
        preds[b] = s * xs + i
    return np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)


def fig_gate_scatter():
    rows = [r for r in _open_weight_rows()
            if r["static"] is not None and r["active"] is not None]
    x = np.array([r["static"] for r in rows])
    y = np.array([r["active"] - r["static"] for r in rows])   # Active-Static gain
    g = E37["gate_split_half"]
    naive_r = g["naive_pearson_r"]
    sh = g["split_half"]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    # fit line + bootstrap CI band
    s, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    lo, hi = _boot_band(x, y, xs)
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.15, zorder=1)
    ax.plot(xs, s * xs + b, color=BLUE, lw=2.4, zorder=2)
    # points colored by sign of the gain
    cols = [GREEN if v >= 0 else RED for v in y]
    ax.scatter(x, y, c=cols, s=42, alpha=0.85, edgecolor="white", lw=0.6, zorder=3)
    ax.set_xlabel("Static score  (a model's capability given curated references)")
    ax.set_ylabel("Active gain  (Active $-$ Static)")
    ax.set_title("The capability gate: the Active gain rises with static ability")
    txt = (f"Pearson $r = {naive_r:.2f}$  ($n={g['n_models']}$)\n"
           f"split-half $r = {sh['pearson_r_mean']:.2f}$ "
           f"[{sh['pearson_r_ci95'][0]:.2f}, {sh['pearson_r_ci95'][1]:.2f}]")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREY, alpha=0.9))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "f2_capability_gate_scatter.png", dpi=150)
    plt.close(fig)
    print("wrote f2_capability_gate_scatter.png  (n=%d, r=%.2f, split-half %.2f %s)"
          % (len(x), naive_r, sh["pearson_r_mean"], sh["pearson_r_ci95"]))


def fig_cutoff_interaction():
    rows = [r for r in _open_weight_rows() if r.get("cutoff") is not None]
    inter = E30["f3_interaction"]["model_level"]
    sl = E22["F3_slopes_vs_cutoff"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, col, lab in [("static", BLUE, "Static"), ("active", ORANGE, "Active")]:
        pts = [(r["cutoff"], r[key]) for r in rows if r[key] is not None]
        x = np.array([a for a, _ in pts]); y = np.array([b for _, b in pts])
        ax.scatter(x, y, c=col, s=34, alpha=0.75, edgecolor="white", lw=0.5, zorder=3)
        s, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        lo, hi = _boot_band(x, y, xs)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.13, zorder=1)
        ax.plot(xs, s * xs + b, color=col, lw=2.4, zorder=2,
                label=f"{lab}  (+{sl[key]['slope_per_yr']:.2f}/yr)")
    ci = inter["interaction_ci95_cluster_bootstrap_models"]
    ax.set_xlabel("Model knowledge cutoff (year)")
    ax.set_ylabel("Score")
    ax.set_title("Both settings improve with cutoff; the gap widens")
    txt = (f"track$\\times$cutoff interaction\n"
           f"$+{inter['beta_interaction_per_yr']:.2f}$/yr "
           f"[{ci[0]:.2f}, {ci[1]:.2f}]\nperm. $p={inter['permutation_p_two_sided']:.4f}$")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREY, alpha=0.9))
    ax.legend(loc="lower right", fontsize=9.5, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "cutoff_interaction.png", dpi=150)
    plt.close(fig)
    print("wrote cutoff_interaction.png  (interaction +%.2f/yr %s)"
          % (inter["beta_interaction_per_yr"], ci))


if __name__ == "__main__":
    fig_gate_scatter()
    fig_cutoff_interaction()
    print("done ->", OUT)
