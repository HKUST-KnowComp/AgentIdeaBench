"""Paper Figure 4: mechanism in two panels (visual-overhaul round, 2026-08-03).

Panel A -- where the Active gain lands: per-dimension Active-Static deltas with
paired bootstrap 95% CIs over the 28 open-weight paired models
(reports/e39_perdim_ci.json; the point estimates reproduce the published
+1.21/+0.63/+0.58/+0.23/-0.14 exactly, the intervals are the new part).
Panel B -- where the gain comes from. The total gain (C-B) splits additively
into a content term (R'-B) and a process term (C-R'); both are positive, so the
split is drawn as a stacked share bar and the process share (77%) is readable
directly from the bar. The recall control (B-R) is negative and is NOT a share
of that total -- it is a separate contrast, so it sits below a divider on the
same axis rather than inside the stack. All cell-level paired contrasts,
matching the numbers in Appendices app:recall / app:replay
(reports/e32_recall_only.json, e38_replay_refs.json).

Marker and bar encoding is the paper's: filled = significant, hollow = not
significant (no red/green). Colors follow the paper-wide semantics: Static blue #2b6cb0,
Active orange #dd6b20, neutral gray. Canvas is 6.3in = ACL \textwidth with the
axes laid out in figure coordinates, so point sizes are 1:1 in the compiled
paper (include at width=\textwidth).
Writes fig4_mechanism.{pdf,png} to reports/figures/summary/. Read-only.
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
E32 = json.loads((ROOT / "reports" / "e32_recall_only.json").read_text())
E38 = json.loads((ROOT / "reports" / "e38_replay_refs.json").read_text())

BLUE, ORANGE, INK, MUTED = "#2b6cb0", "#dd6b20", "#1f2937", "#475569"
GREY = "#64748b"

plt.rcParams.update({
    "font.size": 6.6, "axes.labelsize": 7.0, "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.4, "axes.titlesize": 8.0, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

DIM_LABEL = {"feasibility": "Feasibility", "clarity": "Clarity",
             "specificity": "Specificity", "impact": "Impact",
             "originality": "Originality"}


def style(ax):
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.15, lw=0.5)
    ax.set_axisbelow(True)


def forest(ax, rows, xlim, xlabel, title, notes=None):
    """rows: list of (label, est, lo, hi, color, significant)."""
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0, color="#94a3b8", lw=0.9, zorder=1)
    span = xlim[1] - xlim[0]
    for i, (y, (lab, est, lo, hi, col, sig)) in enumerate(zip(ys, rows)):
        ax.plot([lo, hi], [y, y], color=col, lw=1.5, solid_capstyle="round",
                alpha=0.9, zorder=2)
        ax.plot([est], [y], marker="o", ms=5.2, mfc=col if sig else "white",
                mec=col, mew=1.3, zorder=3)
        txt = ax.text(hi + span * 0.022, y, f"${est:+.2f}$", fontsize=6.6,
                      color=col, va="center", ha="left",
                      fontweight="bold" if sig else "normal")
        if notes:
            ax.annotate(notes[i], xycoords=txt, xy=(1, 0.5), xytext=(3, 0),
                        textcoords="offset points", fontsize=6.2,
                        color="#475569", va="center", ha="left")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_ylim(-0.75, len(rows) - 0.3)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=5)
    style(ax)


def main():
    fig = plt.figure(figsize=(6.3, 2.32))
    axA = fig.add_axes([0.093, 0.305, 0.330, 0.560])
    axB = fig.add_axes([0.590, 0.305, 0.325, 0.560])

    # --- panel A: per-dimension deltas -------------------------------------
    pd_ = E39["per_dimension"]
    order = sorted(pd_, key=lambda d: -pd_[d]["delta_mean"])
    rowsA = []
    for d in order:
        s = pd_[d]
        sig = s["wilcoxon_p"] < 0.05
        rowsA.append((DIM_LABEL[d], s["delta_mean"], s["delta_ci95"][0],
                      s["delta_ci95"][1], ORANGE if sig else GREY, sig))
    forest(axA, rowsA, (-0.55, 1.72), "Active $-$ Static (score points)",
           "Grounding Rises, Originality Stays Flat")
    axA.text(1.0, -0.30, f"unit: model ($n{{=}}{E39['n_models']}$)",
             transform=axA.transAxes, ha="right", va="top",
             fontsize=6.2, color=MUTED)

    # --- panel B: share of the gain, then the separate recall control ------
    c32, c38 = E32["cell_level"], E38["cell_level"]
    total = c38["C_minus_B"]["delta_mean"]
    cont, proc = c38["Rp_minus_B"], c38["C_minus_Rp"]
    ctrl = c32["B_minus_R"]
    # shares of the total; the process share is taken as the complement so the
    # two printed percentages sum to 100 (the two terms sum to the total up to
    # the 4-dp rounding in the JSON)
    sh_c = cont["delta_mean"] / total
    sh_p = 1.0 - sh_c

    Y_BAR, Y_CTRL = 1.05, -1.02
    axB.axvline(0, color="#94a3b8", lw=0.9, zorder=1)

    # the additive split, drawn as one stacked bar so the shares are lengths
    axB.barh(Y_BAR, cont["delta_mean"], height=0.46, left=0, facecolor="white",
             edgecolor=BLUE, lw=1.1, zorder=3)
    axB.barh(Y_BAR, proc["delta_mean"], height=0.46, left=cont["delta_mean"],
             facecolor=ORANGE, edgecolor="white", lw=0.8, zorder=3)

    # total spans the whole bar
    axB.plot([0, 0, total, total], [Y_BAR + 0.42, Y_BAR + 0.30,
             Y_BAR + 0.30, Y_BAR + 0.42], color="#94a3b8", lw=0.8, zorder=2)
    axB.text(total / 2, Y_BAR + 0.52, f"total Active $-$ Static ${total:+.2f}$",
             fontsize=6.4, color=INK, ha="center", va="bottom")

    # share labels sit under their own segment
    axB.text(cont["delta_mean"] / 2, Y_BAR - 0.36, f"{sh_c:.0%}", fontsize=6.8,
             color=BLUE, ha="center", va="center", fontweight="bold")
    axB.text(cont["delta_mean"] + proc["delta_mean"] / 2, Y_BAR - 0.36,
             f"{sh_p:.0%}", fontsize=6.8, color=ORANGE, ha="center",
             va="center", fontweight="bold")

    axB.text(-0.145, 0.26, f"content  replay $-$ Static   ${cont['delta_mean']:+.2f}$  (n.s.)",
             fontsize=6.1, color=BLUE, ha="left", va="center")
    axB.text(-0.145, -0.09, f"process  Active $-$ replay  ${proc['delta_mean']:+.2f}$  "
             f"($p{{<}}0.001$)", fontsize=6.1, color=ORANGE, ha="left",
             va="center", fontweight="bold")

    # the recall control is a different contrast, not a slice of the total
    axB.axhline(-0.52, color="#e2e8f0", lw=0.7, zorder=1)
    axB.text(-0.145, -0.70, "separate control, not part of the total:",
             fontsize=6.0, color=MUTED, ha="left", va="center")
    axB.barh(Y_CTRL, ctrl["delta_mean"], height=0.34, facecolor="white",
             edgecolor=GREY, lw=1.1, zorder=3)
    axB.text(0.022, Y_CTRL, f"Static $-$ recall-only  ${ctrl['delta_mean']:+.2f}$ (n.s.)",
             fontsize=6.0, color=MUTED, ha="left", va="center")

    axB.set_xlim(-0.15, 0.47)
    axB.set_ylim(-1.45, 2.05)
    axB.set_yticks([])
    axB.set_xticks([0.0, 0.2, 0.4])
    axB.set_xlabel("paired difference (score points)")
    axB.set_title("Three Quarters of the Gain Is the Process", pad=5)
    style(axB)
    axB.text(1.0, -0.30,
             f"unit: cell ($n{{=}}{c38['C_minus_B']['n']}$ replay, "
             f"${c32['B_minus_R']['n']}$ recall)",
             transform=axB.transAxes, ha="right", va="top",
             fontsize=6.2, color=MUTED)
    # spell out what the right panel shows (figure-level takeaway line)
    fig.text(0.5, 0.045, "Only the process term is significant: what works is "
             "the model gathering evidence itself, not the papers it finds.",
             fontsize=6.4, color="#334155", ha="center", va="center")
    rowsB = [("total", total, *c38["C_minus_B"]["delta_ci95"], INK, True),
             ("content", cont["delta_mean"], *cont["delta_ci95"], BLUE, False),
             ("process", proc["delta_mean"], *proc["delta_ci95"], ORANGE, True),
             ("recall control", ctrl["delta_mean"], *ctrl["delta_ci95"], GREY, False)]

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_mechanism.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig4_mechanism.pdf/.png")
    for lab, est, lo, hi, _, sig in rowsA + rowsB:
        print(f"  {lab:<26} {est:+.2f} [{lo:+.2f},{hi:+.2f}] sig={sig}")


if __name__ == "__main__":
    main()
