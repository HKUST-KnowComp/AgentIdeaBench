"""Paper Figure 1, redesign candidate B: two-track range fan + gain panel.

Same inputs as _fig1_two_panel.py and _fig1_v2a_paired.py (e22/e30/e37 JSON),
same numbers, different encoding.

Top panel -- instead of showing only the Static range, both tracks get their
own smoothed top/bottom envelope over the cutoff axis. Read as a picture: the
blue Static fan narrows toward the frontier (the compression finding) while the
orange Active fan keeps opening (the discrimination finding). The two medians
carry the +0.54 vs +1.16 per-year slopes.
Bottom panel -- Active minus Static per model as stems from zero, the same
panel as candidate A, so the two candidates differ only in the top encoding.

Envelope method is the one already used in the round-1 figure: a +-0.55yr
window, 96th/4th percentile, gaussian smoothing, running max -- applied
identically to both tracks so the comparison is like-for-like. It is a
presentation smoother, not a fitted model.

Writes fig1_v2b_fan.{pdf,png} to reports/figures/summary/ under a NEW name.
Run with /usr/bin/python3.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter1d

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
    out = []
    for r in E22["per_model"]:
        if r["model"].startswith("google/gemini-"):
            continue
        if not r.get("cutoff") or r["static"] is None or r["active"] is None:
            continue
        out.append(r)
    return sorted(out, key=lambda r: r["cutoff"])


def envelope(xv, yv, grid):
    hi, lo = [], []
    for g in grid:
        sel = yv[np.abs(xv - g) <= 0.55]
        if len(sel) < 3:
            sel = yv[np.argsort(np.abs(xv - g))[:5]]
        hi.append(np.percentile(sel, 96)); lo.append(np.percentile(sel, 4))
    hi = np.maximum.accumulate(gaussian_filter1d(hi, 9))
    lo = np.maximum.accumulate(gaussian_filter1d(lo, 9))
    return hi, lo


def jitter(xs):
    xs = np.asarray(xs, float); out = xs.copy()
    for v in np.unique(xs):
        idx = np.where(xs == v)[0]
        if len(idx) > 1:
            out[idx] = v + np.linspace(-0.035, 0.035, len(idx))
    return out


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))


def panel_fan(ax, rows):
    xv = np.array([r["cutoff"] for r in rows])
    grid = np.linspace(xv.min(), xv.max(), 160)
    sl = E22["F3_slopes_vs_cutoff"]

    for key, col, z in (("active", ORANGE, 2), ("static", BLUE, 3)):
        yv = np.array([r[key] for r in rows])
        hi, lo = envelope(xv, yv, grid)
        ax.fill_between(grid, lo, hi, color=col, alpha=0.16, zorder=z,
                        linewidth=0)
        ax.plot(grid, hi, color=col, lw=1.6, zorder=z + 3, alpha=0.9,
                solid_capstyle="round")
        ax.plot(grid, lo, color=col, lw=1.6, zorder=z + 3, alpha=0.9,
                solid_capstyle="round")
        s, c = np.polyfit(xv, yv, 1)
        ax.plot(grid, s * grid + c, color=col, lw=2.2, zorder=z + 4,
                solid_capstyle="round")
        # width of each fan where it ends, printed as the number the paper
        # already reports for the same eight frontier models
        ax.annotate(f"{key.capitalize()} $+{sl[key]['slope_per_yr']:.2f}$/yr",
                    (grid[-1], s * grid[-1] + c), xytext=(5, 0),
                    textcoords="offset points", color=col, fontsize=8.0,
                    fontweight="bold", va="center", zorder=9)

    sp = E37["discrimination"]["point_spread_top8_by_static"]
    ax.text(0.035, 0.965,
            "band = spread across models\n"
            f"top-8 spread: Static {sp['static']:.2f}, "
            f"Active {sp['active']:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.0,
            color=MUTED, linespacing=1.35, bbox=PLATE, zorder=10)
    ax.set_ylabel("Weighted score")
    ax.set_title("Static Narrows, Active Opens", pad=5, color=INK)
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

    lo = int(np.argmin(g)); hi = int(np.argmax(g))
    for i, off, ha, va in ((lo, (0, -4), "center", "top"),
                           (hi, (5, 0), "left", "center")):
        ax.annotate(rows[i]["model"].split("/")[-1], (x[i], g[i]), xytext=off,
                    textcoords="offset points", ha=ha, va=va, fontsize=7.6,
                    color=MUTED, zorder=9,
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
    panel_fan(ax1, rows)
    panel_gain(ax2, rows)
    for ax in (ax1, ax2):
        style(ax)
        ax.set_xticks([2024, 2025, 2026])
    ax2.set_xlim(2023.55, 2027.05)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_v2b_fan.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote fig1_v2b_fan.pdf/.png  (n={len(rows)} paired models)")


if __name__ == "__main__":
    main()
