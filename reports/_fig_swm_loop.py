r"""Schematic of the closed-loop \swm{} interface (single-column paper figure).

Illustrates the mechanism described in Section "Scientific World Modeling"
(Method paragraph) and Appendix sec:swm: the Active agent gains a fourth
command, SIMULATE, which sends the current draft to a black-box world model;
the world model answers on two *separate* channels (novelty and feasibility)
that a referee assembles; the referee's output carries a protected "novel core"
which in turn constrains what either channel may edit. The next action either
sends the agent back to search/refine or accepts, and accepting passes through
a gate that requires at least one simulation.

This is a schematic: it encodes the protocol, not measurements, so it reads no
data file. The design commitment it makes visible is the one the appendix
argues a single pre-generation scaffold lacks -- novelty and feasibility are
not traded against each other because both are constrained by the pinned core.

The referee is the icon in reports/assets/referee-svgrepo-com.svg (SVG Repo),
converted to vector matplotlib paths by reports/_svg_icon.py so the whole
figure stays vector -- no SVG rasteriser is installed here, and a bitmap would
degrade in print. Structure is drawn in neutral ink/gray; the orange accent
marks the protected novel core, the constraint rail it feeds back into the two
channels, and the finalization gate, so the figure introduces no colour that
would read as the paper's Static/Active track semantics.
The loop-back label sits in the strip under the agent rather than in a left
margin, which lets the diagram body use nearly the full column width.
Canvas is 3.03in = ACL \columnwidth with the axes filling the figure, so text
sizes are 1:1 in the compiled paper (include at width=\columnwidth).
Writes fig_swm_loop.{pdf,png} to reports/figures/summary/.
Run with the base conda python.
"""
import sys
from pathlib import Path as _P

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = _P(__file__).parent.parent
sys.path.insert(0, str(ROOT / "reports"))
from _svg_icon import icon_patches  # noqa: E402

OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)
ICON = ROOT / "reports" / "assets" / "referee-svgrepo-com.svg"

ORANGE, ORANGE_SOFT, ORANGE_BG = "#dd6b20", "#f0a868", "#fdf1e6"
INK, MUTED = "#1f2937", "#475569"
LINE, FILL, SOFT = "#94a3b8", "#f1f5f9", "#cbd5e1"

# The canvas is taller than the earlier draft because every label now sits at
# the 8pt ACLPUB legibility floor (bold headers 8.5pt): the layout is written in
# axes fractions, so the extra height is what buys the leading those sizes need.
W_IN, H_IN = 3.03, 3.10
ASPECT = W_IN / H_IN          # x-units per y-unit, for drawing true shapes
FS, FS_HEAD = 8.0, 8.5

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def box(ax, x0, y0, x1, y1, fc=FILL, ec=LINE, lw=0.8, ls="solid", r=0.012,
        z=2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec,
        lw=lw, linestyle=ls, zorder=z, mutation_aspect=1 / ASPECT))


def arrow(ax, a, b, color=INK, lw=0.9, style="arc3,rad=0", z=4, ls="solid",
          scale=6.5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=scale,
                                 lw=lw, color=color, connectionstyle=style,
                                 shrinkA=0, shrinkB=0, zorder=z,
                                 linestyle=ls))


def main():
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    L, R = 0.095, 0.995
    MID = 0.55

    # --- the agent --------------------------------------------------------
    box(ax, L, 0.872, R, 0.995, fc="white", ec=INK, lw=1.0)
    ax.text((L + R) / 2, 0.960, "Active agent", fontsize=FS_HEAD, color=INK,
            ha="center", va="center", fontweight="bold", zorder=6)
    ax.text((L + R) / 2, 0.907, "search  $\\cdot$  fetch  $\\cdot$  draft",
            fontsize=FS, color=MUTED, ha="center", va="center", zorder=6)

    arrow(ax, (MID, 0.872), (MID, 0.806))
    ax.text(MID + 0.028, 0.839, "SIMULATE: draft", fontsize=FS,
            family="monospace", color=INK, ha="left", va="center", zorder=6)

    # --- the world model --------------------------------------------------
    box(ax, L, 0.285, R, 0.806, fc="white", ec=LINE, lw=0.9,
        ls=(0, (2.6, 1.8)))
    ax.text((L + R) / 2, 0.772, "Scientific World Model", fontsize=FS_HEAD,
            color=INK, ha="center", va="center", fontweight="bold", zorder=6)

    ch_l, ch_r = 0.155, 0.575
    y_nov, y_fea = 0.660, 0.527
    box(ax, ch_l, y_nov - 0.058, ch_r, y_nov + 0.058, fc=FILL, ec=SOFT, lw=0.7)
    ax.text(ch_l + 0.020, y_nov + 0.026, "novelty channel", fontsize=FS,
            color=INK, ha="left", va="center", zorder=6)
    ax.text(ch_l + 0.020, y_nov - 0.026, "what is actually new", fontsize=FS,
            color=MUTED, ha="left", va="center", zorder=6)
    box(ax, ch_l, y_fea - 0.058, ch_r, y_fea + 0.058, fc=FILL, ec=SOFT, lw=0.7)
    ax.text(ch_l + 0.020, y_fea + 0.026, "feasibility channel", fontsize=FS,
            color=INK, ha="left", va="center", zorder=6)
    ax.text(ch_l + 0.020, y_fea - 0.026, "weaknesses + a repair",
            fontsize=FS, color=MUTED, ha="left", va="center", zorder=6)

    # both channels feed the referee
    cx_r, cy_r = 0.780, 0.650
    for y in (y_nov, y_fea):
        ax.plot([ch_r, 0.660], [y, cy_r], color=LINE, lw=0.8, zorder=3,
                solid_capstyle="round")
    arrow(ax, (0.660, cy_r), (0.706, cy_r), color=LINE)
    for p in icon_patches(ICON, cx_r, cy_r, 0.170, ASPECT, zorder=5):
        ax.add_patch(p)
    ax.text(cx_r, 0.523, "Referee", fontsize=FS, color=INK, ha="center",
            va="center", zorder=6)

    # the referee's output carries the pinned core
    arrow(ax, (cx_r, 0.498), (cx_r, 0.432), color=LINE)

    # --- the pinned core, and the constraint it feeds back ----------------
    core_l, core_r = 0.155, 0.950
    box(ax, core_l, 0.318, core_r, 0.430, fc=ORANGE_BG, ec=ORANGE, lw=0.9)
    ax.text((core_l + core_r) / 2, 0.400, "novel core: pinned", fontsize=FS,
            color=ORANGE, ha="center", va="center", fontweight="bold",
            zorder=6)
    ax.text((core_l + core_r) / 2, 0.348, "no edit may weaken it",
            fontsize=FS, color=ORANGE, ha="center", va="center", zorder=6)

    rail = 0.126
    ax.plot([core_l, rail, rail], [0.374, 0.374, y_nov], color=ORANGE_SOFT,
            lw=0.8, ls=(0, (2.2, 1.6)), zorder=3, solid_capstyle="round")
    for y in (y_nov, y_fea):
        arrow(ax, (rail, y), (ch_l - 0.002, y), color=ORANGE_SOFT, lw=0.8,
              ls=(0, (2.2, 1.6)), scale=5.5)

    # --- next actions -----------------------------------------------------
    arrow(ax, (L, 0.545), (0.155, 0.872), style="arc3,rad=-0.45", lw=0.9)
    ax.text(0.180, 0.839, "Search / Refine", fontsize=FS, color=INK,
            ha="left", va="center", zorder=6)

    ax.plot([MID, MID], [0.285, 0.228], color=INK, lw=0.9, zorder=4)
    ax.text(MID + 0.028, 0.2565, "Accept", fontsize=FS, color=INK, ha="left",
            va="center", zorder=6)
    # the gate sits on the accept path: no finalizing without a simulation
    box(ax, 0.265, 0.155, 0.835, 0.228, fc="white", ec=ORANGE, lw=0.9, z=5)
    ax.text(MID, 0.1915, "gate: at least one simulation", fontsize=FS,
            color=ORANGE, ha="center", va="center", zorder=6)
    arrow(ax, (MID, 0.155), (MID, 0.100))

    box(ax, L, 0.010, R, 0.100, fc="white", ec=INK, lw=1.0)
    ax.text((L + R) / 2, 0.055, "final hypothesis", fontsize=FS_HEAD,
            color=INK, ha="center", va="center", fontweight="bold", zorder=6)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_swm_loop.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig_swm_loop.pdf/.png (schematic; referee icon vectorised "
          f"from {ICON.name})")


if __name__ == "__main__":
    main()
