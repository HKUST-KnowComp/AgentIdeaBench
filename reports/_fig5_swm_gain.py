r"""Paper Figure 5: SWM gains against a compute-matched baseline (2026-08-03).

Replaces the absolute-score table (tab:swm) with the quantity the section
actually argues about: the gain over the plain Active baseline, per backbone,
for the three closed-loop designs, drawn against the compute-matched
best-of-3 baseline that costs the same number of LLM calls.

Data (read-only, no API):
  reports/e27_swm.json          per-backbone delta vs. the Active baseline for
                                S2 (+multi-role), S4 (+dynamic panel),
                                S4b (+dynamic+hard); these are the same numbers
                                quoted in Section 6 (qwen-9b +0.61, 27b +0.36,
                                v4-flash +0.14, v4-pro -0.06 for S4b).
  reports/e31_matched_compute.json  per-backbone best-of-3 minus single-run
                                Active baseline (the compute-matched arm).
The 397b backbone is excluded: it has n=4 paired cells and no S4 arm, matching
the four-backbone roster of the section it replaces.

Design colors are a single-hue sequential ramp because the three designs are
ordered by scaffold complexity; the compute-matched baseline is neutral gray.
Canvas is 3.03in = ACL \columnwidth with the axes in figure coordinates, so
point sizes are 1:1 in the compiled paper (include at width=\columnwidth).
Writes fig5_swm_gain.{pdf,png} to reports/figures/summary/.
Run with the base conda python.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

E27 = json.loads((ROOT / "reports" / "e27_swm.json").read_text())
E31 = json.loads((ROOT / "reports" / "e31_matched_compute.json").read_text())

INK, MUTED = "#1f2937", "#475569"
RAMP = ["#f6a723", "#dd6b20", "#8c3a13"]      # light -> dark, scaffold complexity
BAR = "#e2e8f0"
BAR_EDGE = "#94a3b8"

plt.rcParams.update({
    "font.size": 6.6, "axes.labelsize": 7.0, "xtick.labelsize": 6.6,
    "ytick.labelsize": 7.0, "axes.titlesize": 7.4, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

BACKBONES = [("qwen/qwen3.5-9b", "qwen-9b"),
             ("qwen/qwen3.5-27b", "qwen-27b"),
             ("deepseek/deepseek-v4-flash", "v4-flash"),
             ("deepseek/deepseek-v4-pro", "v4-pro")]
DESIGNS = [("swm_S2", "$+$ multi-role"),
           ("swm_S4", "$+$ dynamic panel"),
           ("swm_S4b", "$+$ dynamic $+$ hard")]


def main():
    fig = plt.figure(figsize=(3.03, 2.15))
    ax = fig.add_axes([0.155, 0.300, 0.825, 0.545])

    bo3 = E31["bo3_minus_base_single"]["per_backbone"]
    xs = range(len(BACKBONES))
    w = 0.26

    for i, (key, short) in enumerate(BACKBONES):
        for j, (dk, _) in enumerate(DESIGNS):
            v = E27[dk]["per_model_delta"].get(key)
            if v is None:
                continue
            ax.bar(i + (j - 1) * w, v, width=w * 0.92, color=RAMP[j],
                   edgecolor="white", lw=0.5, zorder=2)
        b = bo3[key]["delta_mean"]
        ax.plot([i - 1.55 * w, i + 1.55 * w], [b, b], ls=(0, (2.5, 1.6)),
                color="#334155", lw=1.2, zorder=4)
        ax.text(i, b + 0.045, f"{b:+.2f}", fontsize=5.8, color="#334155",
                ha="center", va="bottom", zorder=4)

    ax.axhline(0, color="#334155", lw=0.8, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([s for _, s in BACKBONES], color=INK)
    ax.set_xlim(-0.62, len(BACKBONES) - 0.38)
    ax.set_ylim(-0.22, 1.16)
    ax.set_ylabel("gain over Active baseline")
    ax.set_title("Best-of-3 Matches or Beats Every Design", pad=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.15, lw=0.5)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=RAMP[j], edgecolor="white", lw=0.5, label=lab)
               for j, (_, lab) in enumerate(DESIGNS)]
    handles.append(Line2D([], [], ls=(0, (2.5, 1.6)), color="#334155", lw=1.2,
                          label="best-of-3 (same call budget)"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.545),
              ncol=2, frameon=False, fontsize=6.0, handletextpad=0.5,
              columnspacing=1.0, borderpad=0.0, labelspacing=0.35,
              handlelength=1.6)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5_swm_gain.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig5_swm_gain.pdf/.png")
    for key, short in BACKBONES:
        gains = {dk: E27[dk]["per_model_delta"].get(key) for dk, _ in DESIGNS}
        print(f"  {short:<10} " + "  ".join(f"{k[4:]}={v:+.2f}" for k, v in gains.items())
              + f"   best-of-3={bo3[key]['delta_mean']:+.2f}")


if __name__ == "__main__":
    main()
