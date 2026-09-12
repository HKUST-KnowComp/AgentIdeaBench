r"""Per-dimension Active-Static deltas (single-column paper figure, 2026-08-03).

Splits out the left half of the former two-panel fig4_mechanism: where the
Active gain lands across the five scoring dimensions. The decomposition of
*where the gain comes from* is a different question and now lives in its own
figure (_fig_gain_decomposition.py).

Data: reports/e39_perdim_ci.json -- per-model paired deltas over the 28
open-weight paired models with paired bootstrap 95% CIs; the point estimates
reproduce the published +1.21/+0.63/+0.58/+0.23/-0.14 exactly.

The figure carries data and labels only; definitions, units and significance
tests are stated in the caption and the body text, not inside the axes.
Marker encoding is the paper's: filled = significant, hollow = not significant
(no red/green). Canvas is 3.03in = ACL \columnwidth with the axes in figure
coordinates, so point sizes are 1:1 in the compiled paper (include at
width=\columnwidth).
Writes fig_perdim_deltas.{pdf,png} to reports/figures/summary/. Read-only.
Run with the base conda python.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

E39 = json.loads((ROOT / "reports" / "e39_perdim_ci.json").read_text())

ORANGE, INK, GREY = "#dd6b20", "#1f2937", "#64748b"

# ACLPUB asks for figure text that stays legible at print size. Because the
# canvas width equals the \columnwidth the figure is included at, these point
# sizes are the point sizes in the compiled paper: 8pt floor for any text,
# 8.5pt axis labels, 9.5pt title.
plt.rcParams.update({
    "font.size": 8.0, "axes.labelsize": 8.5, "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.5, "axes.titlesize": 9.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

DIM_LABEL = {"feasibility": "Feasibility", "clarity": "Clarity",
             "specificity": "Specificity", "impact": "Impact",
             "originality": "Originality"}


def main():
    pd_ = E39["per_dimension"]
    order = sorted(pd_, key=lambda d: -pd_[d]["delta_mean"])

    fig = plt.figure(figsize=(3.03, 1.78))
    ax = fig.add_axes([0.262, 0.245, 0.690, 0.630])
    xlim = (-0.62, 1.92)
    span = xlim[1] - xlim[0]

    ax.axvline(0, color="#94a3b8", lw=0.9, zorder=1)
    ys = list(range(len(order)))[::-1]
    for y, d in zip(ys, order):
        s = pd_[d]
        sig = s["wilcoxon_p"] < 0.05
        col = ORANGE if sig else GREY
        lo, hi = s["delta_ci95"]
        ax.plot([lo, hi], [y, y], color=col, lw=1.5, solid_capstyle="round",
                alpha=0.9, zorder=2)
        ax.plot([s["delta_mean"]], [y], marker="o", ms=5.2,
                mfc=col if sig else "white", mec=col, mew=1.3, zorder=3)
        ax.text(hi + span * 0.022, y, f"${s['delta_mean']:+.2f}$", fontsize=8.0,
                color=col, va="center", ha="left",
                fontweight="bold" if sig else "normal")

    ax.set_yticks(ys)
    ax.set_yticklabels([DIM_LABEL[d] for d in order], color=INK)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Active $-$ Static (score points)")
    # title spans the canvas, not the (narrower) axes, so it does not overflow
    fig.text(0.5, 0.912, "Grounding Rises, Originality Stays Flat",
             fontsize=9.5, color=INK, ha="center", va="bottom")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.15, lw=0.5)
    ax.set_axisbelow(True)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_perdim_deltas.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote fig_perdim_deltas.pdf/.png  (n_models={E39['n_models']})")
    for d in order:
        s = pd_[d]
        print(f"  {d:<12} {s['delta_mean']:+.2f} "
              f"[{s['delta_ci95'][0]:+.2f},{s['delta_ci95'][1]:+.2f}] "
              f"p={s['wilcoxon_p']:.3f}")


if __name__ == "__main__":
    main()
