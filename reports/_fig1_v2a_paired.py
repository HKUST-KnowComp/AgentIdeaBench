"""Paper Figure 1, redesign candidate A: paired arrows + gain panel.

Same data as reports/_fig1_two_panel.py (reports/e22_new_axis_stats.json,
e30_review_stats.json, e37_review_r2_stats.json) -- nothing is recomputed and
no number changes. What changes is the encoding.

Old encoding: two independent scatters (Static points, Active points) plus two
regression lines. The reader has to match a blue dot to an orange dot by eye to
see what Active did to a given model.
New encoding:
  Top panel  -- one arrow per model, drawn from its Static score to its Active
    score at that model's knowledge cutoff. Arrow direction *is* the finding:
    early cutoffs point down (Active hurts), late cutoffs point up and grow
    longer (Active helps, increasingly). The two regression lines stay, so the
    +0.54 vs +1.16 per-year slopes are still readable, and the frontier
    compression band and top-8 span annotation stay on the Static side.
  Bottom panel -- the same quantity the arrows encode, read directly:
    Active - Static per model against cutoff, as stems from a zero line, with
    the fitted trend. Sign is the story, so the zero line is the only reference
    the reader needs.

Writes fig1_v2a_paired.{pdf,png} to reports/figures/summary/ under a NEW name;
the round-1 figure is untouched. Run with /usr/bin/python3.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(20260828)

E22 = json.loads((ROOT / "reports" / "e22_new_axis_stats.json").read_text())
E30 = json.loads((ROOT / "reports" / "e30_review_stats.json").read_text())
E37 = json.loads((ROOT / "reports" / "e37_review_r2_stats.json").read_text())

BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#64748b"
INK, MUTED, RULE, GRID = "#1f2937", "#334155", "#cbd5e1", "#e6ebf1"
PLATE = dict(facecolor="white", alpha=0.85, edgecolor="none",
             boxstyle="round,pad=0.28")
BAND_X0 = 2024.7

plt.rcParams.update({
    "font.size": 8.0, "axes.labelsize": 8.5, "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0, "axes.titlesize": 9.5, "axes.linewidth": 0.7,
    "axes.edgecolor": RULE, "axes.labelcolor": INK,
    "xtick.color": RULE, "ytick.color": RULE,
    "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "xtick.minor.size": 1.6, "xtick.minor.width": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def paired_rows():
    """Open-weight models that have a cutoff and both track scores: the same
    set the paired statistics in the paper are computed over."""
    out = []
    for r in E22["per_model"]:
        if r["model"].startswith("google/gemini-"):
            continue
        if not r.get("cutoff") or r["static"] is None or r["active"] is None:
            continue
        out.append(r)
    return sorted(out, key=lambda r: r["cutoff"])


def jitter(xs):
    """Four models share cutoff 2025.28 and three share 2025.04; without a
    nudge their arrows would draw on top of each other. The nudge is +-0.035yr,
    far below the axis tick spacing, and is presentation-only."""
    xs = np.asarray(xs, float)
    out = xs.copy()
    for v in np.unique(xs):
        idx = np.where(xs == v)[0]
        if len(idx) > 1:
            off = np.linspace(-0.035, 0.035, len(idx))
            out[idx] = v + off
    return out


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))


def panel_arrows(ax, rows):
    x = jitter([r["cutoff"] for r in rows])
    st = np.array([r["static"] for r in rows])
    ac = np.array([r["active"] for r in rows])

    ax.axvspan(BAND_X0, 2026.22, color="#94a3b8", alpha=0.11, zorder=0)

    # one arrow per model: tail = Static, head = Active
    for xi, s, a in zip(x, st, ac):
        col = ORANGE if a >= s else BLUE
        ax.add_patch(FancyArrowPatch(
            (xi, s), (xi, a), arrowstyle="-|>", mutation_scale=5.5,
            lw=1.15, color=col, alpha=0.75, shrinkA=0, shrinkB=0, zorder=3))
        ax.plot([xi], [s], marker="o", ms=2.6, mfc="white", mec=col, mew=0.8,
                zorder=4)

    # the two fits stay: the slopes are quoted in the text and the caption
    sl = E22["F3_slopes_vs_cutoff"]
    xs = np.linspace(min(x), max(x), 100)
    for key, col, ls_ in (("static", BLUE, (0, (5, 2))), ("active", ORANGE, "-")):
        y = st if key == "static" else ac
        s, c = np.polyfit(np.array([r["cutoff"] for r in rows]), y, 1)
        ax.plot(xs, s * xs + c, color=col, lw=2.0, ls=ls_, zorder=5,
                solid_capstyle="round", alpha=0.95)

    # the two slope labels sit in the empty lower-left corner, colour-keyed to
    # their own line: on the right edge they collided with the spread rail, and
    # on the lines themselves they covered the arrows
    for i, (key, col) in enumerate((("active", ORANGE), ("static", BLUE))):
        ax.text(0.035, 0.145 - i * 0.085,
                f"{key.capitalize()} $+{sl[key]['slope_per_yr']:.2f}$/yr",
                transform=ax.transAxes, fontsize=8.0, fontweight="bold",
                color=col, ha="left", va="center", zorder=7)

    # Right rail: the same eight models (the top-8 by Static, e37) measured as
    # a range under each track. This is the compression finding as a picture --
    # Static squeezes them into 0.39, Active spreads them over 1.39 -- instead
    # of a sentence the reader has to take on faith.
    sp = E37["discrimination"]["point_spread_top8_by_static"]
    names = set(sp["models"])
    t8 = [r for r in rows if r["model"].split("/")[-1] in names]
    rail = {"static": (min(r["static"] for r in t8), max(r["static"] for r in t8)),
            "active": (min(r["active"] for r in t8), max(r["active"] for r in t8))}
    for key, xr, col, ha, dx in (("static", 2026.42, BLUE, "right", -3),
                                 ("active", 2026.70, ORANGE, "left", 3)):
        y0, y1 = rail[key]
        ax.plot([xr, xr], [y0, y1], color=col, lw=3.2, solid_capstyle="butt",
                alpha=0.85, zorder=6)
        for yy in (y0, y1):
            ax.plot([xr - 0.075, xr + 0.075], [yy, yy], color=col, lw=0.9,
                    zorder=6)
        ax.annotate(f"{sp[key]:.2f}", (xr, (y0 + y1) / 2), xytext=(dx * 2, 0),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=8.0, fontweight="bold", color=col, zorder=9,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                              boxstyle="round,pad=0.12"))
    ax.text(2026.56, max(rail["active"]) + 0.18, "top-8 spread", fontsize=8.0,
            color=MUTED, ha="center", va="bottom", zorder=7)

    ax.text(0.035, 0.965, "arrow = one model,\nStatic $\\rightarrow$ Active",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.0,
            color=MUTED, linespacing=1.35, bbox=PLATE, zorder=8)
    ax.set_ylabel("Weighted score")
    ax.set_title("Active Redraws the Frontier", pad=5, color=INK)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_ylim(2.9, 7.02)


def panel_gain(ax, rows):
    x = jitter([r["cutoff"] for r in rows])
    g = np.array([r["active"] - r["static"] for r in rows])
    cols = [ORANGE if v >= 0 else BLUE for v in g]

    ax.axhline(0, color="#94a3b8", lw=0.9, zorder=2)
    ax.vlines(x, 0, g, color=cols, lw=1.15, alpha=0.7, zorder=3)
    ax.scatter(x, g, c=cols, s=17, zorder=4, edgecolor="white", lw=0.4)

    xr = np.array([r["cutoff"] for r in rows])
    s, c = np.polyfit(xr, g, 1)
    xs = np.linspace(xr.min(), xr.max(), 100)
    ax.plot(xs, s * xs + c, color=GREY, lw=1.6, ls=(0, (4, 2)), zorder=5)

    it = E30["f3_interaction"]["model_level"]
    ci = it["interaction_ci95_cluster_bootstrap_models"]
    ax.text(0.035, 0.965,
            "track$\\times$cutoff interaction\n"
            f"$+{it['beta_interaction_per_yr']:.2f}$/yr, "
            f"95% CI $[{ci[0]:.2f}, {ci[1]:.2f}]$\n"
            "permutation $p<0.001$",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.0,
            color=MUTED, linespacing=1.35, bbox=PLATE, zorder=8)

    # name the two extremes so the sign flip has a concrete anchor at each end
    lo = int(np.argmin(g)); hi = int(np.argmax(g))
    for i, off, ha, va in ((lo, (0, -4), "center", "top"),
                           (hi, (5, 0), "left", "center")):
        short = rows[i]["model"].split("/")[-1]
        ax.annotate(short, (x[i], g[i]), xytext=off,
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=7.6, color=MUTED, zorder=9,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                              boxstyle="round,pad=0.15"))

    ax.set_ylabel("Active $-$ Static")
    ax.set_xlabel("Model knowledge cutoff (year)")
    ax.set_title("The Gain Flips Sign Over Time", pad=5, color=INK)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylim(-1.35, 1.75)


def main():
    rows = paired_rows()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.03, 4.5),
                                   constrained_layout=True, sharex=True)
    fig.get_layout_engine().set(h_pad=0.045, hspace=0.02)
    panel_arrows(ax1, rows)
    panel_gain(ax2, rows)
    for ax in (ax1, ax2):
        style(ax)
        ax.set_xticks([2024, 2025, 2026])
    ax2.set_xlim(2023.55, 2027.00)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_v2a_paired.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote fig1_v2a_paired.pdf/.png  (n={len(rows)} paired models)")


if __name__ == "__main__":
    main()
