"""Capability-gate scatter, v2 (visual-overhaul round, 2026-08-03).

Restyles f2_capability_gate_single.png to the paper-wide visual language:
sign is encoded by marker fill (filled = Active helped, hollow = hurt)
instead of green/red, everything else keeps the round-2c design (fit +
bootstrap CI band, quartile-mean diamonds, extreme-model labels, corner
stats). Vector PDF sized for one column at natural scale.

Reads reports/e22_new_axis_stats.json + reports/e37_review_r2_stats.json.
Read-only inputs; writes f2_capability_gate_v2.{pdf,png} (new filenames).
Run with the base conda python.
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
np.random.seed(20260803)

E22 = json.loads((ROOT / "reports" / "e22_new_axis_stats.json").read_text())
E37 = json.loads((ROOT / "reports" / "e37_review_r2_stats.json").read_text())
BLUE, GREY = "#2b6cb0", "#64748b"

SHORT = {"google/gemma-2-27b-it": "gemma-2-27b", "z-ai/glm-5.1": "glm-5.1"}

# ACLPUB legibility floor: nothing below 8pt, 8.5pt axis labels. The canvas
# width equals the \columnwidth the figure is included at, so these are the
# point sizes in the compiled paper.
plt.rcParams.update({
    "font.size": 8.0, "axes.labelsize": 8.5, "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def boot_band(x, y, xs, n=3000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    preds = np.empty((n, len(xs)))
    for b in range(n):
        i = np.random.randint(0, len(x), len(x))
        s, c = np.polyfit(x[i], y[i], 1)
        preds[b] = s * xs + c
    return np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)


def main():
    # Restrict to the paper's primary roster. Excluding only the gemini prefix
    # was enough while e22_new_axis_stats.json held the 30 primary models, but
    # that file was refreshed on 2026-08-30 to carry all 67 scored models,
    # including the closed azure ladder the paper holds out; unfiltered it
    # plots 59 points at r=+0.20 instead of the published 28 at r=+0.69.
    roster = set(json.loads(
        (ROOT / "reports" / "primary_roster.json").read_text())["models"])
    rows = [r for r in E22["per_model"]
            if r["model"] in roster
            and not r["model"].startswith("google/gemini-")
            and r.get("static") is not None and r.get("active") is not None]
    x = np.array([r["static"] for r in rows])
    y = np.array([r["active"] - r["static"] for r in rows])
    g = E37["gate_split_half"]
    sh = g["split_half"]

    fig, ax = plt.subplots(figsize=(3.03, 2.30), constrained_layout=True)
    ax.axhline(0, color="#475569", lw=0.9, zorder=1)
    s, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    lo, hi = boot_band(x, y, xs)
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.13, zorder=1)
    ax.plot(xs, s * xs + b, color=BLUE, lw=2.2, zorder=2)
    # sign by marker fill, one hue: filled = helped, hollow = hurt
    pos, neg = y >= 0, y < 0
    ax.scatter(x[pos], y[pos], facecolor=BLUE, edgecolor="white", lw=0.5,
               s=26, alpha=0.9, zorder=3)
    ax.scatter(x[neg], y[neg], facecolor="white", edgecolor=BLUE, lw=1.1,
               s=26, zorder=3)
    # quartile means (diamonds) make the monotone climb explicit
    order = np.argsort(x)
    q = np.array_split(order, 4)
    qx = [float(x[i].mean()) for i in q]
    qy = [float(y[i].mean()) for i in q]
    ax.plot(qx, qy, color=GREY, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.scatter(qx, qy, marker="D", s=38, facecolor="#f1f5f9", edgecolor="#334155",
               lw=1.2, zorder=4)
    # extreme models named in the text
    imin, imax = int(np.argmin(y)), int(np.argmax(y))
    ax.annotate(SHORT.get(rows[imin]["model"], rows[imin]["model"].split("/")[-1]),
                (x[imin], y[imin]), textcoords="offset points", xytext=(7, 1),
                fontsize=8.0, ha="left", va="center", color="#334155")
    ax.annotate(SHORT.get(rows[imax]["model"], rows[imax]["model"].split("/")[-1]),
                (x[imax], y[imax]), textcoords="offset points", xytext=(-7, -1),
                fontsize=8.0, ha="right", va="center", color="#334155")
    ax.text(0.03, 0.97,
            f"$r = {g['naive_pearson_r']:.2f}$ ($n={g['n_models']}$)\n"
            f"split-half $r = {sh['pearson_r_mean']:.2f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.0,
            color="#334155", linespacing=1.4)
    ax.set_xlabel("Static score (capability)")
    ax.set_ylabel("Active $-$ Static gain")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.15, lw=0.5)
    # headroom so the r / split-half note clears the highest points
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * 0.13)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"f2_capability_gate_v2.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote f2_capability_gate_v2.pdf/.png  n=%d r=%.2f split-half=%.2f "
          "quartiles=%s" % (len(x), g["naive_pearson_r"], sh["pearson_r_mean"],
                            ["%+.2f" % v for v in qy]))


if __name__ == "__main__":
    main()
