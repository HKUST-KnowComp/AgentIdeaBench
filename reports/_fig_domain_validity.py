r"""Appendix figure: per-discipline critic validity (visual-overhaul round, 2026-08-03).

Draws the three score tiers the domain-validity paragraph describes -- weak-probe
control generators, the strongest model's Active ideas, and rewritten landmark
papers -- for each discipline, with the anchor-minus-model gap that decides which
disciplines the paper treats as reliable regimes.

Data: reports/e40_domain_validity.json (read-only recomputation whose gaps
reproduce the published +1.50 / +0.88 / +0.43 / +0.38 / +0.28). Chemistry has no
weak-control items, which the figure shows as a missing marker rather than
silently dropping the row.

Landmark anchors are ink (human ground truth), model Active is orange, weak
controls are gray. Canvas is 3.03in = ACL \columnwidth with the axes in figure
coordinates, so point sizes are 1:1 in the compiled paper
(include at width=\columnwidth).
Writes fig_domain_validity.{pdf,png} to reports/figures/summary/.
Run with the base conda python.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

E40 = json.loads((ROOT / "reports" / "e40_domain_validity.json").read_text())

ORANGE, INK, MUTED, GREY = "#dd6b20", "#1f2937", "#475569", "#94a3b8"

# ACLPUB legibility floor: nothing below 8pt, 8.5pt axis labels, 9.5pt title.
plt.rcParams.update({
    "font.size": 8.0, "axes.labelsize": 8.5, "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.5, "axes.titlesize": 9.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

LABEL = {"CS": "Computer sci.", "Physics": "Physics", "Medicine": "Medicine",
         "Chemistry": "Chemistry", "Biology": "Biology"}


def main():
    rows = E40["per_discipline"]
    order = sorted(rows, key=lambda d: -rows[d]["gap_anchor_minus_model"])

    fig = plt.figure(figsize=(3.03, 2.45))
    ax = fig.add_axes([0.315, 0.300, 0.595, 0.520])
    ys = list(range(len(order)))[::-1]

    for y, d in zip(ys, order):
        r = rows[d]
        ax.plot([r["model_active_mean"], r["anchor_top10_mean"]], [y, y],
                color="#cbd5e1", lw=1.6, solid_capstyle="round", zorder=1)
        if r["weak_control_mean"] is not None:
            ax.plot([r["weak_control_mean"]], [y], marker="o", ms=4.2,
                    mfc=GREY, mec="white", mew=0.6, zorder=3)
        ax.plot([r["model_active_mean"]], [y], marker="o", ms=5.0, mfc=ORANGE,
                mec="white", mew=0.7, zorder=3)
        ax.plot([r["anchor_top10_mean"]], [y], marker="o", ms=5.0, mfc=INK,
                mec="white", mew=0.7, zorder=3)
        ax.text(r["anchor_top10_mean"] + 0.16, y,
                f"$+{r['gap_anchor_minus_model']:.2f}$", fontsize=8.0,
                color=INK, va="center", ha="left", fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[d] for d in order], color=INK)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(3.2, 8.5)
    ax.set_xlabel("weighted score")
    # title spans the canvas, not the (narrower) axes, so it does not overflow
    fig.text(0.5, 0.875, "Anchor Separation Shrinks in Dense Fields",
             fontsize=9.5, color=INK, ha="center", va="bottom")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.15, lw=0.5)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=4.2, mfc=GREY, mec="white",
               mew=0.6, label="weak controls"),
        Line2D([], [], marker="o", ls="none", ms=5.0, mfc=ORANGE, mec="white",
               mew=0.7, label="model (Active)"),
        Line2D([], [], marker="o", ls="none", ms=5.0, mfc=INK, mec="white",
               mew=0.7, label="landmark papers"),
    ]
    # two columns: at the 8pt floor the three labels do not fit on one row
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, -0.60),
              ncol=2, frameon=False, fontsize=8.0, handletextpad=0.4,
              columnspacing=1.0, labelspacing=0.35, borderpad=0.0)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_domain_validity.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig_domain_validity.pdf/.png")
    for d in order:
        r = rows[d]
        print(f"  {d:<10} weak={r['weak_control_mean']}  model="
              f"{r['model_active_mean']:.2f}  anchor={r['anchor_top10_mean']:.2f}"
              f"  gap={r['gap_anchor_minus_model']:+.2f}")


if __name__ == "__main__":
    main()
