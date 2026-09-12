r"""Where the Active gain comes from (single-column paper figure, 2026-08-03).

Splits out the right half of the former two-panel fig4_mechanism. The total
gain C-B decomposes additively into a content term (R'-B) and a process term
(C-R'); both are positive, so the split is drawn as one stacked bar and each
term's share of the total is its length (the process term is 77%). A pie chart
is not used: two slices, one of them non-significant, and a second contrast on
the same scale all read better as lengths than as angles.

The recall control (B-R) is negative and is NOT a share of that total -- it is
a separate contrast -- so it is a second bar on the same axis, labelled by its
own y tick, not a slice of the stack.

Data (read-only, cell-level paired contrasts, matching Appendices
app:recall / app:replay):
  reports/e38_replay_refs.json   C-B, R'-B, C-R'  (n=279 shared cells)
  reports/e32_recall_only.json   B-R              (n=280 shared cells)

The figure carries data and labels only; definitions, units and significance
tests are stated in the caption and the body text, not inside the axes.
Bar encoding is the paper's: filled = significant, hollow = not significant
(no red/green); Static blue, Active orange, neutral gray. Canvas is 3.03in =
ACL \columnwidth with the axes in figure coordinates, so point sizes are 1:1 in
the compiled paper (include at width=\columnwidth).
Writes fig_gain_decomposition.{pdf,png} to reports/figures/summary/.
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

E32 = json.loads((ROOT / "reports" / "e32_recall_only.json").read_text())
E38 = json.loads((ROOT / "reports" / "e38_replay_refs.json").read_text())

BLUE, ORANGE, INK, GREY = "#2b6cb0", "#dd6b20", "#1f2937", "#64748b"

# ACLPUB legibility floor: 8pt for any text in the figure, 8.5pt axis labels,
# 9.5pt title. The canvas width equals the \columnwidth the figure is included
# at, so these are the point sizes in the compiled paper.
plt.rcParams.update({
    "font.size": 8.0, "axes.labelsize": 8.5, "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.5, "axes.titlesize": 9.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    c32, c38 = E32["cell_level"], E38["cell_level"]
    total = c38["C_minus_B"]["delta_mean"]
    cont, proc = c38["Rp_minus_B"]["delta_mean"], c38["C_minus_Rp"]["delta_mean"]
    ctrl = c32["B_minus_R"]["delta_mean"]
    # shares of the total; the process share is the complement so the two
    # printed percentages sum to 100 (the terms add to the total up to the
    # 4-dp rounding stored in the JSON)
    sh_c = cont / total
    sh_p = 1.0 - sh_c

    fig = plt.figure(figsize=(3.03, 1.72))
    ax = fig.add_axes([0.335, 0.265, 0.615, 0.470])
    ax.axvline(0, color="#94a3b8", lw=0.9, zorder=1)

    ax.barh(1, cont, height=0.50, left=0, facecolor="white", edgecolor=BLUE,
            lw=1.1, zorder=3)
    ax.barh(1, proc, height=0.50, left=cont, facecolor=ORANGE,
            edgecolor="white", lw=0.8, zorder=3)
    ax.text(cont + proc / 2, 1, f"{sh_p:.0%}", fontsize=8.0, color="white",
            ha="center", va="center", fontweight="bold", zorder=4)
    ax.text(total + 0.014, 1, f"${total:+.2f}$", fontsize=8.0, color=INK,
            ha="left", va="center", fontweight="bold")
    # direct labels instead of a legend: each segment names itself. At the 8pt
    # floor the content label is wider than the content segment, so it sits
    # above the bar on a leader line instead of inside it.
    ax.plot([cont / 2, cont / 2], [1.27, 1.53], color=BLUE, lw=0.7, zorder=2)
    ax.text(cont / 2, 1.58, f"content {sh_c:.0%}", fontsize=8.0, color=BLUE,
            ha="center", va="bottom")
    ax.text(cont + proc / 2, 1.44, "process", fontsize=8.0, color=ORANGE,
            ha="center", va="center")

    ax.barh(0, ctrl, height=0.38, facecolor="white", edgecolor=GREY, lw=1.1,
            zorder=3)
    ax.text(0.012, 0, f"${ctrl:+.2f}$", fontsize=8.0, color=GREY,
            ha="left", va="center")

    ax.set_yticks([1, 0])
    # the control label wraps so the left margin stays narrow enough for the
    # 8pt floor to fit inside a single column
    ax.set_yticklabels(["Active $-$ Static", "Static $-$\nrecall-only"],
                       color=INK)
    ax.set_ylim(-0.60, 2.05)
    ax.set_xlim(-0.20, 0.47)
    ax.set_xticks([0.0, 0.2, 0.4])
    ax.set_xlabel("paired difference (score points)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.15, lw=0.5)
    ax.set_axisbelow(True)
    # title spans the whole canvas, not just the axes, so it is not clipped
    fig.text(0.5, 0.912, "Three Quarters of the Gain Is the Process",
             fontsize=9.5, color=INK, ha="center", va="center")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_gain_decomposition.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig_gain_decomposition.pdf/.png")
    print(f"  total  C-B   {total:+.4f}  n={c38['C_minus_B']['n']}")
    print(f"  content R'-B {cont:+.4f}  share {sh_c:.1%}  "
          f"p={c38['Rp_minus_B']['wilcoxon_p']}")
    print(f"  process C-R' {proc:+.4f}  share {sh_p:.1%}  "
          f"p={c38['C_minus_Rp']['wilcoxon_p']}")
    print(f"  control B-R  {ctrl:+.4f}  n={c32['B_minus_R']['n']}  "
          f"p={c32['B_minus_R']['wilcoxon_p']}")


if __name__ == "__main__":
    main()
