"""Generate one visualization per finding for the all-experiments summary.
Reads reports/e22_new_axis_stats.json + e23_ablations.json + the score DB.
Writes PNGs to reports/figures/summary/. Read-only. /usr/bin/python3.
"""
import json, sqlite3, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from experiments.e22_new_axis_stats import seed_scores, _load_dict, to_decimal_year

OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(7)  # reproducible jitter
E22 = json.loads((ROOT / "reports" / "e22_new_axis_stats.json").read_text())
E23 = json.loads((ROOT / "reports" / "e23_ablations.json").read_text())
cutoffs = _load_dict("reports/_make_cross_year_plot.py", "KNOWLEDGE_CUTOFFS")
BLUE, ORANGE, GREEN, RED, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"


def per_model():
    by = seed_scores()
    models = sorted({m for (m, tr) in by})
    rows = []
    for m in models:
        b = by.get((m, "B"), []); c = by.get((m, "C"), [])
        rows.append({"m": m, "cut": cutoffs.get(m),
                     "static": np.mean(b) if b else None,
                     "active": np.mean(c) if c else None,
                     "boost": (np.mean(c) - np.mean(b)) if (b and c) else None})
    return rows


def fig_f1(rows):
    from scipy.ndimage import gaussian_filter1d
    pts = sorted([(r["cut"], r["static"]) for r in rows if r["cut"] and r["static"] is not None])
    xv = np.array([a for a, _ in pts]); yv = np.array([b for _, b in pts])
    # smooth rising envelope: sliding-window high/low, gaussian-smoothed, then made monotone
    grid = np.linspace(xv.min(), xv.max(), 160)
    win = 0.55  # +/- years
    hi, lo = [], []
    for g in grid:
        sel = yv[np.abs(xv - g) <= win]
        if len(sel) < 3:  # widen near sparse edges
            sel = yv[np.argsort(np.abs(xv - g))[:5]]
        hi.append(np.percentile(sel, 96)); lo.append(np.percentile(sel, 4))
    hi = np.maximum.accumulate(gaussian_filter1d(hi, 9))
    lo = np.maximum.accumulate(gaussian_filter1d(lo, 9))
    mid = gaussian_filter1d((hi + lo) / 2, 4)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.fill_between(grid, lo, hi, color=BLUE, alpha=0.15, zorder=1)
    ax.plot(grid, hi, color=BLUE, lw=2.6, solid_capstyle="round", label="top of the range")
    ax.plot(grid, lo, color=GREEN, lw=2.6, solid_capstyle="round", label="bottom of the range")
    ax.plot(grid, mid, color=GREY, lw=1.4, ls="--", alpha=0.75, label="middle")
    ax.scatter(xv, yv, c="#475569", s=26, alpha=0.5, zorder=3, edgecolor="white", lw=0.5)
    # mark the mid-range plateau of the top edge (a hint of saturation among open models)
    pm = (grid >= 2024.7) & (grid <= 2025.35)
    if pm.any():
        ax.plot(grid[pm], hi[pm], color=RED, lw=3.2, alpha=0.85, solid_capstyle="round", zorder=4)
        gi = grid[pm][len(grid[pm]) // 2]
        ax.annotate("frontier flattens\n(a plateau)", xy=(gi, hi[pm].mean()),
                    xytext=(gi + 0.25, hi[pm].mean() - 0.55), fontsize=8.5, color=RED,
                    ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xlabel("Model knowledge cutoff (year)"); ax.set_ylabel("Static score (given references)")
    ax.set_title("Static quality rises, but the frontier plateaus")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, frameon=False); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "f1_static_cutoff.png", dpi=150); plt.close(fig)


def fig_f2(rows):
    pts = sorted([(r["static"], r["boost"]) for r in rows
                  if r["static"] is not None and r["boost"] is not None])
    x = np.array([a for a, _ in pts]); y = np.array([b for _, b in pts])
    b1 = E23["baseline_f2"]
    # split into capability quartiles; show mean gain per group as bars (no fit line)
    qedges = np.quantile(x, [0, 0.25, 0.5, 0.75, 1.0])
    labels = ["Weakest\n25%", "Lower-mid\n25%", "Upper-mid\n25%", "Strongest\n25%"]
    groups = [[] for _ in range(4)]
    for xv, yv in pts:
        gi = min(3, np.searchsorted(qedges[1:], xv, side="right"))
        groups[gi].append(yv)
    means = [np.mean(g) if g else 0 for g in groups]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axhline(0, color="black", lw=0.8)
    bars = ax.bar(range(4), means, width=0.6,
                  color=[RED if m < 0 else GREEN for m in means], alpha=0.85, zorder=2)
    # overlay individual models as dots
    for gi, g in enumerate(groups):
        ax.scatter([gi + np.random.uniform(-0.12, 0.12) for _ in g], g,
                   c=GREY, s=16, alpha=0.5, zorder=3)
    for bar, m in zip(bars, means):
        ax.annotate(f"{m:+.2f}", (bar.get_x() + bar.get_width() / 2, m),
                    textcoords="offset points", xytext=(0, 6 if m >= 0 else -13),
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Model capability group (by Static score)")
    ax.set_ylabel("Average Active gain  (Active − Static)")
    ax.set_title("Finding 2 — the gain climbs from weak to strong models")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "f2_boost_vs_capability.png", dpi=140); plt.close(fig)


def fig_f2_conv():
    d = E23["D_ideas_per_seed_convergence"]
    labels = ["1 idea", "2 ideas", "3 ideas"]
    r = [d["idx1"]["pearson_r"], d["idx12"]["pearson_r"], d["idx123"]["pearson_r"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(labels, r, "-o", color=BLUE, lw=2, ms=9)
    for i, v in enumerate(r): ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)
    ax.set_ylabel("Strength of Finding 2  (Pearson r)"); ax.set_xlabel("Ideas averaged per seed")
    ax.set_title("More samples per seed → the effect gets cleaner")
    ax.set_ylim(0.3, 0.8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "f2_convergence.png", dpi=140); plt.close(fig)


def fig_f3(rows):
    ps = [(r["cut"], r["static"]) for r in rows if r["cut"] and r["static"] is not None]
    pa = [(r["cut"], r["active"]) for r in rows if r["cut"] and r["active"] is not None]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for pts, col, lab, key in [(ps, BLUE, "Static", "static"), (pa, ORANGE, "Active", "active")]:
        x = np.array([a for a, _ in pts]); y = np.array([b for _, b in pts])
        ax.scatter(x, y, c=col, s=34, alpha=0.7, edgecolor="white", zorder=3)
        m, b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
        sl = E22["F3_slopes_vs_cutoff"][key]["slope_per_yr"]
        ax.plot(xs, m * xs + b, color=col, lw=2.4, label=f"{lab}  (+{sl:.2f}/yr)")
    ratio = E22["F3_slopes_vs_cutoff"]["active_over_static_slope_ratio"]
    ax.set_xlabel("Model knowledge cutoff (year)"); ax.set_ylabel("Score")
    ax.set_title(f"Finding 3 — Both rise with cutoff; Active is {ratio:.1f}× steeper")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "f3_slopes.png", dpi=140); plt.close(fig)


def fig_f0():
    f = E22["F0_partial_corr"]
    labels = ["Recency → score\n(controlling for size)", "Model size → score\n(controlling for recency)"]
    vals = [f["partial_r_cutoff_score_given_logparams"], f["partial_r_logparams_score_given_cutoff"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bars = ax.bar(labels, vals, color=[BLUE, ORANGE], alpha=0.9, width=0.5)
    for bar, v in zip(bars, vals): ax.annotate(f"{v:+.2f}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("Partial correlation")
    ax.set_ylim(0, 0.9)
    ax.set_title("Finding 4 — Recency and size each predict score, independently")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "f0_partial_corr.png", dpi=140); plt.close(fig)


def fig_f18_dims():
    b = E23["B_per_dim_boost"]
    order = ["feasibility", "clarity", "specificity", "impact", "originality"]
    names = ["Feasibility", "Clarity", "Specificity", "Impact", "Originality"]
    vals = [b[k]["mean_dim_boost"] for k in order]
    cols = [GREEN if v > 0 else RED for v in vals]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(names, vals, color=cols, alpha=0.9, width=0.62)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:+.2f}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 6 if v > 0 else -14), ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("Active gain on this dimension")
    ax.set_title("What Active improves: feasible/specific/clear — not original")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "f18_dimensions.png", dpi=140); plt.close(fig)


def fig_refs():
    sim4 = json.loads((ROOT / "reports" / "e11_ref_overlap" / "v3_refs_sim4.json").read_text())["sim4_static_refs_vs_active_refs"]
    labels = ["Static idea\nvs its refs", "Active idea\nvs static's refs",
              "Active idea\nvs its own refs", "Static refs\nvs Active refs"]
    vals = [0.504, 0.489, 0.487, sim4]
    cols = [BLUE, GREEN, ORANGE, "#6b46c1"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(labels, vals, color=cols, alpha=0.9, width=0.62)
    for bar, v in zip(bars, vals): ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Semantic similarity (cosine)")
    ax.set_title("Same topic, different papers")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, axis="y")
    fig.text(0.5, 0.01, "The two reference pools are ~0.87 alike in topic, yet share almost no actual papers (overlap 0.03).",
             ha="center", fontsize=8.2, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "refs_overlap.png", dpi=140); plt.close(fig)


def fig_anchor():
    doms = ["CS", "Physics", "Medicine", "Biology", "Chemistry"]
    anchor = [7.34, 6.10, 6.43, 6.15, 6.02]
    model = [5.00, 4.69, 5.14, 5.07, 5.11]  # 40-sub model Active means
    x = np.arange(len(doms)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(x - w / 2, anchor, w, label="Human landmark papers", color=GREEN, alpha=0.9)
    ax.bar(x + w / 2, model, w, label="Model ideas (Active avg)", color=BLUE, alpha=0.9)
    for i, (a, m) in enumerate(zip(anchor, model)):
        ax.annotate(f"+{a-m:.2f}", (i, max(a, m) + 0.1), ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(doms); ax.set_ylabel("Score")
    ax.set_ylim(0, 8.4); ax.set_title("Human landmark papers still clearly beat model ideas")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "anchor_gap.png", dpi=140); plt.close(fig)


def main():
    rows = per_model()
    fig_f1(rows); fig_f2(rows); fig_f2_conv(); fig_f3(rows); fig_f0()
    fig_f18_dims(); fig_refs(); fig_anchor()
    print("wrote figures to", OUT)
    for p in sorted(OUT.glob("*.png")): print(" ", p.name)


if __name__ == "__main__":
    main()
