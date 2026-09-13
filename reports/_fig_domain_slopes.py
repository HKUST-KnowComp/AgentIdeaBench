r"""Per-discipline Static vs Active cutoff slopes (single-column paper figure).

Backs the main-text sentence "Active is steeper in all five disciplines"
(\S\ref{sec:scaling}), which until now carried no figure.

DATA SOURCE -- read carefully. The live reports/e23_ablations.json was
recomputed on 2026-08-30 when the E42 closed-gateway roster landed, and now
holds n=50/52 slopes that include closed-source models. The paper's headline
statistics stay on the 28 open-weight primary-roster models (n=24 Static /
26 Active with disclosed cutoffs), so this figure reads the pre-E42 snapshot
archived at archive/reports/pre_e42_mainstats_20260830/e23_ablations.json.
Cross-check: the five per-discipline slopes in that snapshot average
0.5436 (Static) and 1.1617 (Active), reproducing the published +0.54/+1.16
and the ~2.1x ratio exactly.

Canvas is 3.03in = ACL \columnwidth, so point sizes here are the point sizes
in the compiled paper (include at width=\columnwidth).
Writes fig_domain_slopes.{pdf,png} to reports/figures/summary/. Read-only.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

SRC = ROOT / "archive" / "reports" / "pre_e42_mainstats_20260830" / "e23_ablations.json"
if not SRC.exists():
    # archive/ is local-only, so the public checkout ships the same pinned
    # snapshot under reports/ instead. Same bytes, same provenance note above.
    SRC = ROOT / "reports" / "e23_ablations_pre_e42_20260830.json"
SLOPES = json.loads(SRC.read_text())["E_per_domain_slopes"]

ORANGE, INK, GREY, SLATE = "#dd6b20", "#1f2937", "#64748b", "#475569"

plt.rcParams.update({
    "font.size": 8.0, "axes.labelsize": 8.5, "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.5, "axes.titlesize": 9.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

LABEL = {"CS": "Computer sci.", "Physics": "Physics", "Biology": "Biology",
         "Chemistry": "Chemistry", "Medicine": "Medicine"}


def main():
    rows = [(d, SLOPES[d]["static"]["slope"], SLOPES[d]["active"]["slope"])
            for d in SLOPES]
    rows.sort(key=lambda r: r[2])          # ascending, so best sits on top

    fig = plt.figure(figsize=(3.03, 1.94))
    ax = fig.add_axes([0.292, 0.230, 0.588, 0.545])
    xlim = (0.30, 1.44)
    span = xlim[1] - xlim[0]

    ys = list(range(len(rows)))
    for y, (_, s, a) in zip(ys, rows):
        ax.plot([s, a], [y, y], color="#cbd5e1", lw=2.2,
                solid_capstyle="round", zorder=1)
        ax.plot([s], [y], marker="o", ms=4.8, mfc=SLATE, mec=SLATE, zorder=3)
        ax.plot([a], [y], marker="o", ms=5.4, mfc=ORANGE, mec=ORANGE, zorder=3)
        ax.text(a + span * 0.030, y, f"${a:+.2f}$", fontsize=8.0, color=ORANGE,
                va="center", ha="left", fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[r[0]] for r in rows], color=INK)
    ax.set_ylim(-0.62, len(rows) - 0.02)
    ax.set_xlim(*xlim)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xlabel("")
    fig.text(0.5, 0.040, "Points per year of knowledge cutoff",
             fontsize=8.5, color=INK, ha="center", va="bottom")

    fig.text(0.5, 0.930, "Active Scales Faster in Every Discipline",
             fontsize=9.5, color=INK, ha="center", va="bottom")

    # direct labels on the top row, which cost less space than a legend box
    # and cannot collide with the bottom row the way a corner legend did
    top_y, top_s, top_a = len(rows) - 1, rows[-1][1], rows[-1][2]
    ax.text(top_s, top_y + 0.46, "Static", fontsize=8.0, color=SLATE,
            ha="center", va="bottom")
    ax.text(top_a, top_y + 0.46, "Active", fontsize=8.0, color=ORANGE,
            ha="center", va="bottom", fontweight="bold")

    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.15, lw=0.5)
    ax.set_axisbelow(True)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_domain_slopes.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)

    print(f"wrote fig_domain_slopes.pdf/.png   source={SRC.name}")
    ms = sum(r[1] for r in rows) / len(rows)
    ma = sum(r[2] for r in rows) / len(rows)
    for d, s, a in sorted(rows, key=lambda r: -r[2]):
        n_s = SLOPES[d]["static"]["n"]
        n_a = SLOPES[d]["active"]["n"]
        print(f"  {d:<10} static {s:+.3f} (n={n_s})  active {a:+.3f} "
              f"(n={n_a})  gap {a-s:+.3f}  ratio {a/s:.2f}x")
    print(f"  mean       static {ms:+.4f}   active {ma:+.4f}   "
          f"ratio {ma/ms:.3f}x   [paper: +0.54 / +1.16 / ~2.1x]")


if __name__ == "__main__":
    main()
