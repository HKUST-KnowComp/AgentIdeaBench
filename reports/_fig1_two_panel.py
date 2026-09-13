"""Paper Figure 1, two stacked panels.

Single-column, page-1 top-right placement: panels stacked vertically with a
shared cutoff axis and a Title-Case headline above each panel (titles sit above
the axes, never inside the plot area).
Top panel: static observation saturates -- per-model Static score vs knowledge
cutoff with a smoothed top/mid/bottom envelope and a light-gray band over the
late-2024..mid-2025 compression region, labelled inside the band itself.
Bottom panel: Static vs Active cutoff scaling -- per-track regression with
bootstrap CI bands, direct line-end slope labels, and a single significance
marker for the slope difference.

ROSTER (2026-09-06). Both panels are restricted to the 30 primary-roster models
listed in Table~\\ref{tab:models} of the paper, read out of that table so the
figure cannot drift from it. This matters: reports/e22_new_axis_stats.json was
refreshed on 2026-08-30 to carry all 67 scored models, including the closed
azure release ladder that the paper holds out, and an unfiltered read plots 45
models and fits slopes the paper never reports. On the primary roster the fitted
slopes are Static +0.5436/yr (n=24) and Active +1.1617/yr (n=26), which
reproduce the +0.54 and +1.16 quoted in \\S\\ref{sec:scaling} and the
0.5436 / 0.6182 decomposition in reports/e30_review_stats.json.

Reads reports/e22_new_axis_stats.json (per-model static/active/cutoff),
reports/e30_review_stats.json (track x cutoff interaction permutation p),
docs/paper/acl_latex.tex (the primary roster, from Table tab:models).
Read-only. Writes fig1_static_active_cutoff.{pdf,png} to
reports/figures/summary/.

Visual contract: ink hierarchy is curves > band > points; only horizontal
gridlines; left/bottom spines only; every floating annotation sits on a
translucent white plate so it never reads through a curve or a marker. Colors
follow the paper-wide semantics: Static blue #2b6cb0, Active orange #dd6b20,
neutral gray (pair CVD-validated). Canvas width equals the \\columnwidth the
figure is included at, so the point sizes here are the point sizes in the
compiled paper.
Run with /usr/bin/python3 (mpl 3.9, numpy 2.0, scipy 1.13).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(20260803)

E22 = json.loads((ROOT / "reports" / "e22_new_axis_stats.json").read_text())
E30 = json.loads((ROOT / "reports" / "e30_review_stats.json").read_text())

BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#64748b"
# points are the lightest ink in the panel: the eye should land on the envelope
# and the two regression lines first, and only then resolve the scatter. The
# top panel is Static-only, so its points carry the Static hue at low weight
# rather than a neutral gray, which keeps the colour semantics of both panels
# identical.
DOT = "#8fabce"
INK, MUTED, RULE, GRID = "#1f2937", "#334155", "#cbd5e1", "#e6ebf1"
# translucent plate behind every floating annotation
PLATE = dict(facecolor="white", alpha=0.82, edgecolor="none",
             boxstyle="round,pad=0.28")
# The compression band opens at the late-2024 cutoffs and runs to the right
# edge of the axis: the strongest Static models all fall inside it.
BAND_X0 = 2024.7
YLABEL = "Average Ideation Score"

# ACLPUB legibility floor: nothing in the figure is below 7.5pt, axis labels are
# 8.5pt and panel titles 9.5pt. The canvas width equals the \columnwidth the
# figure is included at, so these are the point sizes in the compiled paper.
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


def primary_roster():
    """The 30 model ids of Table tab:models, resolved against E22's keys.

    The paper's roster lives in the LaTeX table, so we read it from there
    instead of restating it here; a model added to or dropped from the paper
    then changes this figure without a second edit.
    """
    tex_path = ROOT / "docs" / "paper" / "acl_latex.tex"
    if not tex_path.exists():
        # Public checkouts do not ship the paper source, so they carry the same
        # 30 ids as reports/primary_roster.json, generated from that table.
        return set(json.loads(
            (ROOT / "reports" / "primary_roster.json").read_text())["models"])
    tex = tex_path.read_text().split("\n")
    start = next(i for i, l in enumerate(tex) if l.startswith("\\label{tab:models}"))
    body_start = next(i for i, l in enumerate(tex) if "Model (OpenRouter id)" in l)
    short = []
    for line in tex[body_start + 2:start]:
        if "&" not in line:
            continue
        cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
        for k in (0, 5):                       # the table is two blocks wide
            if k < len(cells) and cells[k] and not cells[k].startswith("\\"):
                short.append(cells[k])
    keys = list(E22["per_model"][0].keys()) and [r["model"] for r in E22["per_model"]]

    def resolve(sid):
        exact = [m for m in keys if m.split("/")[-1] == sid]
        if exact:
            return exact[0]
        pref = [m for m in keys if m.split("/")[-1].startswith(sid)]
        if len(pref) == 1:
            return pref[0]
        loose = [m for m in keys
                 if sid.replace("-2501", "").replace("-instruct", "")
                 in m.split("/")[-1]]
        return loose[0] if len(loose) == 1 else None

    resolved = {sid: resolve(sid) for sid in short}
    missing = [s for s, v in resolved.items() if v is None]
    if len(short) != 30 or missing:
        raise SystemExit(f"roster read failed: {len(short)} ids, missing {missing}")
    return {v for v in resolved.values()}


def roster_rows():
    keep = primary_roster()
    return [r for r in E22["per_model"] if r["model"] in keep]


def series(rows, key):
    pts = sorted((r["cutoff"], r[key]) for r in rows
                 if r.get("cutoff") and r.get(key) is not None)
    return (np.array([a for a, _ in pts]), np.array([b for _, b in pts]))


def boot_band(x, y, xs, n=3000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    preds = np.empty((n, len(xs)))
    for b in range(n):
        i = np.random.randint(0, len(x), len(x))
        s, c = np.polyfit(x[i], y[i], 1)
        preds[b] = s * xs + c
    return np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)


def style(ax):
    """Left/bottom rules only, horizontal gridlines only: the x axis is a date,
    so vertical rules add ink without adding a reading aid."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", direction="out")


def panel_a(ax, rows):
    xv, yv = series(rows, "static")
    grid = np.linspace(xv.min(), xv.max(), 160)
    hi, lo = [], []
    for g in grid:
        sel = yv[np.abs(xv - g) <= 0.55]
        if len(sel) < 3:
            sel = yv[np.argsort(np.abs(xv - g))[:5]]
        hi.append(np.percentile(sel, 96)); lo.append(np.percentile(sel, 4))
    hi = np.maximum.accumulate(gaussian_filter1d(hi, 9))
    lo = np.maximum.accumulate(gaussian_filter1d(lo, 9))
    mid = gaussian_filter1d((hi + lo) / 2, 4)

    ax.fill_between(grid, lo, hi, color=BLUE, alpha=0.09, zorder=1)
    ax.scatter(xv, yv, c=DOT, s=15, alpha=0.85, edgecolor="white", lw=0.35,
               zorder=3)
    ax.plot(grid, mid, color=GREY, lw=1.0, ls=(0, (4, 3)), alpha=0.65, zorder=4)
    ax.plot(grid, lo, color=BLUE, lw=1.1, alpha=0.45, solid_capstyle="round",
            zorder=5)
    ax.plot(grid, hi, color=BLUE, lw=2.4, solid_capstyle="round", zorder=6)
    # a terminal dot anchors the flattened end of the top envelope, which is
    # the feature the panel is about
    ax.plot([grid[-1]], [hi[-1]], marker="o", ms=3.4, color=BLUE, zorder=7,
            markeredgecolor="white", markeredgewidth=0.6)

    # direct line-end labels for the envelope (no legend)
    for label, curve, col, al in (("top", hi, BLUE, 1.0),
                                  ("middle", mid, GREY, 0.85),
                                  ("bottom", lo, BLUE, 0.6)):
        ax.annotate(label, (grid[-1], curve[-1]), xytext=(5, 0),
                    textcoords="offset points", color=col, alpha=al,
                    fontsize=8.0, va="center")

    ax.margins(y=0.14)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * 0.10)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel(YLABEL)
    ax.set_title("Static Observation Saturates", pad=5, color=INK)
    return len(xv)


def panel_b(ax, rows):
    counts, slopes = {}, {}
    for key, col, lab in (("static", BLUE, "Static"), ("active", ORANGE, "Active")):
        x, y = series(rows, key)
        counts[key], = (len(x),)
        ax.scatter(x, y, c=col, s=15, alpha=0.34, edgecolor="white", lw=0.35,
                   zorder=3)
        s, c = np.polyfit(x, y, 1)
        slopes[key] = s
        xs = np.linspace(x.min(), x.max(), 100)
        b_lo, b_hi = boot_band(x, y, xs)
        ax.fill_between(xs, b_lo, b_hi, color=col, alpha=0.10, zorder=1)
        ax.plot(xs, s * xs + c, color=col, lw=2.4, solid_capstyle="round",
                zorder=5)
        ax.plot([xs[-1]], [s * xs[-1] + c], marker="o", ms=3.4, color=col,
                zorder=6, markeredgecolor="white", markeredgewidth=0.6)
        # direct line-end label instead of a legend; the slope is refitted on
        # the plotted points, so the number cannot drift from the picture
        ax.annotate(f"{lab}\n$+{s:.2f}$/yr",
                    (xs[-1], s * xs[-1] + c), xytext=(6, 0),
                    textcoords="offset points", color=col, fontsize=8.0,
                    fontweight="bold", va="center", linespacing=1.2)

    # one significance marker, not the full interaction fit: the permutation
    # test behind it is reported in Appendix~\ref{app:stats}
    p = E30["f3_interaction"]["model_level"]["permutation_p_two_sided"]
    ptxt = "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"
    ax.text(0.035, 0.955, f"slope difference {ptxt}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.0, color=MUTED, bbox=PLATE, zorder=8)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * 0.09)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel(YLABEL)
    # the per-condition sample sizes differ (24 Static, 26 Active) and printing
    # both on the figure reads as an inconsistency rather than as information,
    # so the sample is stated once, in the caption
    ax.set_xlabel("Model knowledge cutoff (year)")
    ax.set_title("Active Exploration Keeps Scaling", pad=5, color=INK)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 0.72)  # room for end labels
    return counts, slopes


def main():
    rows = roster_rows()
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(3.03, 4.50), constrained_layout=True, sharex=True)
    fig.get_layout_engine().set(h_pad=0.045, hspace=0.02)
    n_a = panel_a(axA, rows)
    counts, slopes = panel_b(axB, rows)
    for ax in (axA, axB):
        style(ax)
        ax.set_xticks([2024, 2025, 2026])
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    # drawn after panel_b fixes the shared x-limits so the band can run all the
    # way to the right edge instead of stopping at a hard-coded year
    x1 = axA.get_xlim()[1]
    axA.axvspan(BAND_X0, x1, color="#94a3b8", alpha=0.13, zorder=0)
    axA.axvline(BAND_X0, color="#94a3b8", lw=0.8, ls=(0, (3, 3)), alpha=0.9,
                zorder=2)
    # the band carries its own label instead of being explained from the corner
    ya0, ya1 = axA.get_ylim()
    axA.text((BAND_X0 + x1) / 2, ya0 + (ya1 - ya0) * 0.035,
             "frontier compression", fontsize=8.0, color=GREY,
             ha="center", va="bottom", zorder=8, bbox=PLATE)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_static_active_cutoff.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig1_static_active_cutoff.pdf/.png")
    print(f"  primary roster rows          {len(rows)}")
    print(f"  panel A (Static vs cutoff)   n={n_a}")
    print(f"  panel B  Static n={counts['static']}  slope {slopes['static']:+.4f}/yr"
          f"   (paper +0.54)")
    print(f"  panel B  Active n={counts['active']}  slope {slopes['active']:+.4f}/yr"
          f"   (paper +1.16)")


if __name__ == "__main__":
    main()
