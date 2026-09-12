"""Paper Figure 2, redesign: two-lane protocol diagram with a real Active loop.

Schematic only -- no data dependency. Every label restates a protocol fact
already stated in Sections 3-4 (subfield counts, track conditions, output
format, critic count, dimensions, weights).

What the redesign changes versus the round-1 box diagram:
  * The two tracks become lanes with different *shapes*, not two boxes with
    different text. Static is a straight line: given references, one pass.
    Active is a loop: the model issues SEARCH/FETCH against Semantic Scholar
    and reads what it chooses, up to a 10-call budget, before it writes. The
    silhouette carries the contrast, so the reader does not have to compare
    two paragraphs of small type to find it.
  * The scoring weights are drawn as bars instead of printed as a formula, so
    the dominance of Originality is visible rather than parsed.
  * What is held constant is stated once, on the rail between the lanes,
    instead of floating between two boxes.

Every string is budgeted against its own box: at 8pt this face runs about
4.4pt per character (7.6pt runs about 4.2), and 1.0 x-unit = 6.3in = 454pt, so
a box of width w holds roughly w*454/4.4 characters.
Canvas is 6.3in wide = ACL \\textwidth; include at width=\\textwidth with no
rescaling so the point sizes here are the point sizes in the paper.
Writes fig2_v2_flow.{pdf,png} to reports/figures/summary/ under a NEW name.
Run with /usr/bin/python3.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2b6cb0", "#dd6b20"
BLUE_BG, ORANGE_BG, GREY_BG = "#eaf1f9", "#fdefe3", "#f1f5f9"
EDGE, INK, MUTED, ARR = "#94a3b8", "#1f2937", "#475569", "#64748b"
PANEL_BG, PANEL_EDGE, RULE = "#fbfcfd", "#e2e8f0", "#cbd5e1"

FS, FS_TITLE, FS_PANEL, FS_TINY = 8.0, 8.5, 9.5, 7.6
plt.rcParams.update({"font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42})

# lane geometry, kept in one place so the arrows and the labels cannot drift
YS, YA, BH = 0.680, 0.255, 0.200          # lane centres, box height
TASK = (0.004, 0.136)                      # x, w
# the gap between L1 and L2 is deliberately wide: it is where the Active loop
# is drawn, and a narrow gap collapses the two arcs into a pair of ticks
L1 = (0.150, 0.130)                        # first box of each lane
L2 = (0.336, 0.130)                        # second box of each lane
HYP = (0.480, 0.132)
SCO = (0.622, 0.374)


def rbox(ax, x, y, w, h, face, edge, lw=1.0, z=2, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.010",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=z,
        linestyle=(0, (2.5, 2)) if dashed else "solid"))


def titled_box(ax, x, y_c, w, title, sub, edge, face, tcolor=None,
               dashed=False, title_fs=FS):
    rbox(ax, x, y_c - BH / 2, w, BH, face, edge, dashed=dashed)
    ax.text(x + w / 2, y_c + 0.038, title, fontsize=title_fs,
            fontweight="bold", color=tcolor or edge, ha="center", va="center",
            zorder=4, linespacing=1.25)
    ax.text(x + w / 2, y_c - 0.050, sub, fontsize=FS_TINY, color=MUTED,
            ha="center", va="center", zorder=4, linespacing=1.3)


def arrow(ax, p0, p1, color=ARR, rad=0.0, scale=8.0, lw=1.1, z=6, shrink=1.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color,
        shrinkA=shrink, shrinkB=shrink, zorder=z,
        connectionstyle=f"arc3,rad={rad}"))


def main():
    fig = plt.figure(figsize=(6.3, 2.62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------------------------------------------------------------- task
    tx, tw = TASK
    rbox(ax, tx, 0.250, tw, 0.440, GREY_BG, EDGE)
    ax.add_patch(Rectangle((tx + 0.0015, 0.264), 0.0075, 0.412,
                           facecolor=EDGE, edgecolor="none", zorder=3))
    ax.text(tx + tw / 2 + 0.006, 0.612, "Task", fontsize=FS_TITLE,
            fontweight="bold", color=INK, ha="center", va="center", zorder=4)
    ax.text(tx + tw / 2 + 0.006, 0.420,
            "100 subfields\n5 domains\n40 scored\n3 ideas / cell",
            fontsize=FS_TINY, color=MUTED, ha="center", va="center",
            linespacing=1.7, zorder=4)

    # ------------------------------------------------------- static lane
    ax.text(L1[0], YS + BH / 2 + 0.055, "Static (Track B)", fontsize=FS_TITLE,
            fontweight="bold", color=BLUE, ha="left", va="center", zorder=7)
    titled_box(ax, L1[0], YS, L1[1], "Curated refs", "given, fixed",
               BLUE, BLUE_BG)
    titled_box(ax, L2[0], YS, L2[1], "One pass", "no retrieval",
               BLUE, "#ffffff")
    arrow(ax, (tx + tw, 0.545), (L1[0], YS - 0.030), BLUE, rad=-0.24)
    arrow(ax, (L1[0] + L1[1], YS), (L2[0], YS), BLUE, shrink=0.5)

    # ------------------------------------------------------- active lane
    titled_box(ax, L1[0], YA, L1[1], "Model", "picks queries", ORANGE,
               ORANGE_BG)
    titled_box(ax, L2[0], YA, L2[1], "Semantic\nScholar", "live index",
               ORANGE, "#ffffff", dashed=True, title_fs=FS_TINY)
    arrow(ax, (tx + tw, 0.400), (L1[0], YA + 0.030), ORANGE, rad=0.24)
    # the loop is the point: out on search/fetch, back with abstracts
    arrow(ax, (L1[0] + L1[1] + 0.002, YA + 0.046), (L2[0] - 0.002, YA + 0.046),
          ORANGE, rad=-0.62, scale=8.5, shrink=0.5)
    arrow(ax, (L2[0] - 0.002, YA - 0.046), (L1[0] + L1[1] + 0.002, YA - 0.046),
          ORANGE, rad=-0.62, scale=8.5, shrink=0.5)
    mid = (L1[0] + L1[1] + L2[0]) / 2
    # loop labels live outside the boxes: the gap itself is only ~25pt wide
    ax.text(mid, YA + BH / 2 + 0.012, "search / fetch", fontsize=FS_TINY,
            color=ORANGE, ha="center", va="bottom", zorder=7)
    ax.text(mid, YA - BH / 2 - 0.028, "abstracts back  ·  $\\leq$10 calls",
            fontsize=FS_TINY, color=MUTED, ha="center", va="top", zorder=7)
    ax.text(L1[0], YA - BH / 2 - 0.088, "Active (Track C)", fontsize=FS_TITLE,
            fontweight="bold", color=ORANGE, ha="left", va="center", zorder=7)

    # ------------------------------------------- what is held constant
    ax.text((L1[0] + L2[0] + L2[1]) / 2, 0.503,
            "matched across tracks\nsubfields · format · length · critics",
            fontsize=FS_TINY, color=MUTED, ha="center", va="center", zorder=7,
            linespacing=1.45,
            bbox=dict(facecolor="white", alpha=0.96, edgecolor=PANEL_EDGE,
                      boxstyle="round,pad=0.30", linewidth=0.7))

    # ---------------------------------------------------------- hypothesis
    hx, hw = HYP
    rbox(ax, hx, YA - BH / 2, hw, (YS + BH / 2) - (YA - BH / 2), GREY_BG, EDGE)
    for y0, cbar in ((0.470, BLUE), (0.172, ORANGE)):
        ax.add_patch(Rectangle((hx + 0.0015, y0), 0.0075, 0.288,
                               facecolor=cbar, edgecolor="none", zorder=3))
    ax.text(hx + hw / 2 + 0.006, 0.610, "Hypothesis", fontsize=FS,
            fontweight="bold", color=INK, ha="center", va="center", zorder=4)
    ax.text(hx + hw / 2 + 0.006, 0.415, "80–150 words\nsame format\nboth tracks",
            fontsize=FS_TINY, color=MUTED, ha="center", va="center",
            linespacing=1.7, zorder=4)
    arrow(ax, (L2[0] + L2[1], YS), (hx, 0.600), BLUE, rad=0.16, shrink=0.5)
    arrow(ax, (L2[0] + L2[1], YA), (hx, 0.320), ORANGE, rad=-0.16, shrink=0.5)

    # ------------------------------------------------------------- scoring
    sx, sw = SCO
    rbox(ax, sx, 0.088, sw, 0.760, PANEL_BG, PANEL_EDGE, lw=0.7, z=0)
    ax.text(sx + 0.006, 0.882, "Literature-Verified Scoring",
            fontsize=FS_PANEL, fontweight="bold", color=INK, ha="left",
            va="bottom")
    ax.plot([sx, sx + sw], [0.868] * 2, color=RULE, lw=0.7, zorder=1)
    arrow(ax, (hx + hw, 0.560), (sx + 0.012, 0.700), ARR, rad=-0.18)

    ix, iw = sx + 0.012, sw - 0.024
    rbox(ax, ix, 0.612, iw, 0.176, "#ffffff", EDGE)
    ax.text(ix + iw / 2, 0.740, "1   Prior-art retrieval", fontsize=FS,
            fontweight="bold", color=INK, ha="center", va="center", zorder=4)
    ax.text(ix + iw / 2, 0.660, "date-filtered, queried from the idea",
            fontsize=FS_TINY, color=MUTED, ha="center", va="center", zorder=4)

    rbox(ax, ix, 0.404, iw, 0.176, "#ffffff", EDGE)
    ax.text(ix + iw / 2, 0.532, "2   Three critics", fontsize=FS,
            fontweight="bold", color=INK, ha="center", va="center", zorder=4)
    ax.text(ix + iw / 2, 0.446, "score against that evidence,\ndrop the most generous",
            fontsize=FS_TINY, color=MUTED, ha="center", va="center", zorder=4,
            linespacing=1.35)
    arrow(ax, (ix + iw / 2, 0.606), (ix + iw / 2, 0.586), ARR, scale=7.5,
          shrink=0.0)
    arrow(ax, (ix + iw / 2, 0.398), (ix + iw / 2, 0.378), ARR, scale=7.5,
          shrink=0.0)

    # stage 3: the weights, drawn rather than written as a formula
    rbox(ax, ix, 0.108, iw, 0.264, GREY_BG, EDGE)
    ax.text(ix + iw / 2, 0.330, "3   Weighted total", fontsize=FS,
            fontweight="bold", color=INK, ha="center", va="center", zorder=4)
    dims = [("Originality", 2.0), ("Impact", 1.5), ("Feasibility", 1.0),
            ("Clarity", 0.5), ("Specificity", 0.5)]
    bx = ix + 0.108
    bw = iw - 0.108 - 0.030            # leaves room for the value label
    for i, (name, w) in enumerate(dims):
        yb = 0.286 - i * 0.038
        shade = "#2b6cb0" if i == 0 else "#7f9ec9" if i == 1 else "#b9c8dc"
        ax.add_patch(Rectangle((bx, yb - 0.011), bw * w / 2.0, 0.021,
                               facecolor=shade, edgecolor="none", zorder=4))
        ax.text(bx - 0.006, yb, name, fontsize=FS_TINY, color=MUTED,
                ha="right", va="center", zorder=4)
        ax.text(bx + bw * w / 2.0 + 0.005, yb, f"{w:g}", fontsize=FS_TINY,
                color=MUTED, ha="left", va="center", zorder=4)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_v2_flow.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig2_v2_flow.pdf/.png")


if __name__ == "__main__":
    main()
