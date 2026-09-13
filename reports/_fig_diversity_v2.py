r"""Appendix diversity figure, redrawn (visual-overhaul round, 2026-08-03).

Replaces idea_diversity_single.png, whose dumbbell placed the two means on an
arbitrary track and so showed no distribution. This version plots the data: one
point per model (n=28 paired models) for each of the four within-cell diversity
metrics, on that metric's own delta axis, with the zero line and the mean.
Self-ROUGE-L is flipped to distinctness (1-overlap) so that right is more
diverse in every panel.

Data: reports/e28_idea_diversity.json (per_model deltas; the pooled cell-level
means and CIs quoted in the appendix text come from the same file). Read-only.

All points come from the same source (one paired model each), so they carry one
colour; the zero line, not the fill, marks direction. Canvas is 3.03in = ACL \columnwidth with
the axes in figure coordinates, so point sizes are 1:1 in the compiled paper
(include at width=\columnwidth).
Writes fig_diversity_v2.{pdf,png} to reports/figures/summary/.
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

E28 = json.loads((ROOT / "reports" / "e28_idea_diversity.json").read_text())

ORANGE, INK, MUTED, GREY = "#dd6b20", "#1f2937", "#475569", "#94a3b8"
rng = np.random.default_rng(20260803)

# ACLPUB legibility floor: nothing below 8pt. The panel titles are shortened to
# fit a half-column panel at that size; the full metric names are in the caption.
plt.rcParams.update({
    "font.size": 8.0, "axes.titlesize": 8.5, "xtick.labelsize": 8.0,
    "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# (json key, panel title, flip sign so that right = more diverse)
PANELS = [("vendi", "Vendi", False),
          ("cosdist", "Cosine dist.", False),
          ("rougeL", "1$-$ROUGE-L", True),
          ("distinct2", "Distinct-2", False)]


def main():
    per_model = E28["per_model"]
    models = sorted(per_model)

    fig = plt.figure(figsize=(3.03, 2.95))
    axes = [fig.add_axes([0.075 + 0.505 * c, 0.590 - 0.355 * r, 0.395, 0.185])
            for r in range(2) for c in range(2)]

    for ax, (key, title, flip) in zip(axes, PANELS):
        d = np.array([per_model[m][key] for m in models]) * (-1 if flip else 1)
        y = rng.uniform(-0.55, 0.55, len(d))
        pos = d > 0
        ax.scatter(d, y, s=9, facecolor=ORANGE, edgecolor="white",
                   lw=0.35, alpha=0.9, zorder=3)
        ax.axvline(0, color="#334155", lw=0.8, zorder=2)
        ax.axvline(d.mean(), color=INK, lw=1.1, ls=(0, (2.5, 1.5)), zorder=4)
        ax.text(0.985, 1.04, f"${d.mean():+.2f}$ ({int(pos.sum())}/{len(d)})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0,
                color=MUTED)
        ax.set_title(title, pad=11.0, loc="left", color=INK)
        ax.set_ylim(-1.0, 1.0)
        lim = max(abs(d).max() * 1.12, 0.02)
        ax.set_xlim(-lim * 0.55, lim)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=2, pad=1.5)
        ax.locator_params(axis="x", nbins=3)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", alpha=0.12, lw=0.5)
        ax.set_axisbelow(True)

    fig.text(0.5, 0.070, "Active $-$ Static, one point per model; right of the\n"
             "line $=$ more diverse. Each panel on its own scale.",
             ha="center", va="center", fontsize=8.0, color=MUTED,
             linespacing=1.4)
    fig.text(0.5, 0.955, "Active Ideas Are More Mutually Distinct",
             ha="center", va="center", fontsize=9.5, color=INK)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_diversity_v2.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote fig_diversity_v2.pdf/.png  (n_models={len(models)}, "
          f"n_pairs={E28['n_pairs']})")
    for key, title, flip in PANELS:
        d = np.array([per_model[m][key] for m in models]) * (-1 if flip else 1)
        print(f"  {key:<10} mean {d.mean():+.3f}  up {int((d > 0).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
