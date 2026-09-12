"""Upgraded single-column capability-gate figure (ARR round-2b review request:
keep only the static-ability gate panel and raise the visual spec; the cutoff
panel is relegated to the appendix). Reads reports/e22_new_axis_stats.json
(per-model static/active) and reports/e37_review_r2_stats.json (naive +
split-half r). Read-only inputs; writes ONE NEW png (new filename, no existing
raw figure is overwritten): reports/figures/summary/f2_capability_gate_single.png.
/usr/bin/python3.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(20260802)

E22 = json.loads((ROOT / "reports" / "e22_new_axis_stats.json").read_text())
E37 = json.loads((ROOT / "reports" / "e37_review_r2_stats.json").read_text())
BLUE, GREEN, RED, GREY = "#2b6cb0", "#2f855a", "#c53030", "#718096"

SHORT = {
    "google/gemma-2-27b-it": "gemma-2-27b",
    "z-ai/glm-5.1": "glm-5.1",
}


def _boot_band(x, y, xs, n=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    preds = np.empty((n, len(xs)))
    for b in range(n):
        idx = np.random.randint(0, len(x), len(x))
        s, i = np.polyfit(x[idx], y[idx], 1)
        preds[b] = s * xs + i
    return np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)


def main():
    rows = [r for r in E22["per_model"]
            if not r["model"].startswith("google/gemini-")
            and r.get("static") is not None and r.get("active") is not None]
    x = np.array([r["static"] for r in rows])
    y = np.array([r["active"] - r["static"] for r in rows])   # Active-Static gain
    g = E37["gate_split_half"]
    sh = g["split_half"]

    fig, ax = plt.subplots(figsize=(3.4, 3.15))
    ax.axhline(0, color="black", lw=0.7, zorder=1)
    # least-squares fit + bootstrap 95% band
    s, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    lo, hi = _boot_band(x, y, xs)
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.14, zorder=1)
    ax.plot(xs, s * xs + b, color=BLUE, lw=2.0, zorder=2)
    # points: green helped, red hurt
    cols = [GREEN if v >= 0 else RED for v in y]
    ax.scatter(x, y, c=cols, s=34, alpha=0.9, edgecolor="white", lw=0.5, zorder=3)
    # quartile-mean overlay makes the monotone climb explicit
    order = np.argsort(x)
    q = np.array_split(order, 4)
    qx = [float(x[idx].mean()) for idx in q]
    qy = [float(y[idx].mean()) for idx in q]
    ax.plot(qx, qy, color=GREY, lw=1.0, ls="--", zorder=2)
    ax.scatter(qx, qy, marker="D", s=44, facecolor="white", edgecolor=GREY,
               lw=1.4, zorder=4)
    # label the two extreme models the text names
    imin, imax = int(np.argmin(y)), int(np.argmax(y))
    ax.annotate(SHORT.get(rows[imin]["model"], rows[imin]["model"].split("/")[-1]),
                (x[imin], y[imin]), textcoords="offset points", xytext=(7, 1),
                fontsize=7, ha="left", va="center", color="#333")
    ax.annotate(SHORT.get(rows[imax]["model"], rows[imax]["model"].split("/")[-1]),
                (x[imax], y[imax]), textcoords="offset points", xytext=(-7, -1),
                fontsize=7, ha="right", va="center", color="#333")
    txt = (f"$r = {g['naive_pearson_r']:.2f}$ ($n={g['n_models']}$)\n"
           f"split-half $r = {sh['pearson_r_mean']:.2f}$")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                    ec=GREY, alpha=0.9))
    ax.set_xlabel("Static score (capability)", fontsize=8.5)
    ax.set_ylabel("Active $-$ Static gain", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.18)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "f2_capability_gate_single.png", dpi=200)
    plt.close(fig)
    print("wrote f2_capability_gate_single.png  n=%d r=%.2f split-half=%.2f quartile_gains=%s"
          % (len(x), g["naive_pearson_r"], sh["pearson_r_mean"],
             ["%+.2f" % v for v in qy]))


if __name__ == "__main__":
    main()
