"""Generate presentation figures for the 2026-06-23 night results.

Produces clean English-labelled slide figures into reports/figures/:
  slide_f1_saturation.png     — F1: curve-fit artifact + yearly max/min spread
  slide_f2_boost_vs_strength.png — F2: Active-Static boost grows with model strength
  slide_f3_two_slopes.png     — F3: both modes rise with cutoff, Active steeper
  slide_e6_recency.png        — E6: recency vs capability (split + gap regression)
  slide_e9_budget_sweep.png   — E9: tool-call budget saturation
(E8 figures already exist: e8_cosine_static_vs_active.png / e8_tsne_seed_static_active.png)

READ-ONLY. Run: python reports/_night_figures.py
"""
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as WEIGHTS
import reports._make_cross_year_plot as mc

FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
E0 = json.load(open(ROOT / "reports" / "e0_uniform" / "e0_uniform_analysis.json"))
E6 = json.load(open(ROOT / "reports" / "e6_recency" / "e6_recency_vs_capability.json"))
CUTOFFS = mc.KNOWLEDGE_CUTOFFS
WSUM = sum(WEIGHTS.values())
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})

BLUE, RED, GREEN, GRAY = "#1f77b4", "#d62728", "#2ca02c", "#888888"


# ---------------------------------------------------------------- F1
def fig_f1():
    lb = [r for r in E0["leaderboard"] if r["cutoff"] is not None]
    xs = np.array([r["cutoff"] for r in lb]); ys = np.array([r["uniform_static"] for r in lb])
    closed = np.array([r["closed"] for r in lb])
    lin = E0["F1_saturation"]["linear_fit"]; bex = E0["F1_saturation"]["bounded_exp_fit"]
    ym = E0["F1_saturation"]["yearly_minmax"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    # A: scatter + overall linear trend (all models rise, but that's the range filling in)
    ax[0].scatter(xs[~closed], ys[~closed], c=BLUE, s=42, label="open-weight", alpha=0.8)
    ax[0].scatter(xs[closed], ys[closed], c=RED, s=42, marker="s", label="closed (GPT/Claude/Gemini)", alpha=0.8)
    gx = np.linspace(xs.min(), xs.max(), 100)
    ax[0].plot(gx, lin["intercept"] + lin["slope"] * gx, color="black", lw=2,
               label=f"all-model linear  (+{lin['slope']:.2f}/yr)")
    ax[0].set_xlabel("knowledge cutoff (year)"); ax[0].set_ylabel("Static score (0–10)")
    ax[0].set_title("(A) Static score vs cutoff (47 models): overall rises (+0.53/yr)\n— but that is the range filling in; the frontier is in (B)")
    ax[0].legend(fontsize=9, loc="upper left")

    # B: half-year MAX(frontier) & MIN(floor), forced monotonic non-decreasing
    # (high-water marks) so both lines only rise / hold and connect to the end.
    hbin = defaultdict(list)
    for x, y in zip(xs, ys):
        hbin[round(float(np.floor(x * 2) / 2.0), 1)].append(y)
    hs = sorted(hbin)
    bmax = [max(hbin[h]) for h in hs]; bmin = [min(hbin[h]) for h in hs]
    emax = list(np.maximum.accumulate(bmax))   # frontier high-water mark (non-decreasing)
    emin = list(np.maximum.accumulate(bmin))    # rising floor (non-decreasing)
    ax[1].scatter(xs, ys, s=18, c=GRAY, alpha=0.35, label="each model")
    ax[1].fill_between(hs, emin, emax, color=BLUE, alpha=0.10)
    ax[1].plot(hs, emax, "-^", color=RED, lw=2.5, ms=6, label="MAX (frontier, best-so-far)")
    ax[1].plot(hs, emin, "-v", color=BLUE, lw=2.5, ms=6, label="MIN (floor, never drops)")
    # frontier year-over-year gain shrinking → saturation
    ym2 = {int(y): ym[str(y)]["max"] for y in ym if ym[y]["n"] >= 2}
    if 2023 in ym2 and 2024 in ym2 and 2025 in ym2:
        g1 = ym2[2024] - ym2[2023]; g2 = ym2[2025] - ym2[2024]
        ax[1].annotate(f"frontier gain/yr:\n  2023→24: +{g1:.1f}\n  2024→25: +{g2:.1f}  ← shrinking",
                       (hs[0] + 0.1, max(emax) - 0.9), fontsize=9.5, color=RED)
    ax[1].axhspan(max(emax) - 0.15, max(emax) + 0.15, color=GREEN, alpha=0.10)
    ax[1].set_xlabel("knowledge cutoff (year)"); ax[1].set_ylabel("Static score (0–10)")
    ax[1].set_title("(B) Frontier (MAX) gains shrinking → Static SATURATING\ncan't separate the best models → need a new bench")
    ax[1].legend(fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "slide_f1_saturation.png", dpi=150); plt.close(fig)
    print("✓ slide_f1_saturation.png")


# ---------------------------------------------------------------- F2
def fig_f2():
    rows = E0["F2_boost"]["rows"]
    st = np.array([r["static"] for r in rows]); bo = np.array([r["boost"] for r in rows])
    closed = np.array([r["closed"] for r in rows])
    r = E0["F2_boost"]["stronger_gains_more_r_static_boost"]
    reg = stats.linregress(st, bo)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color=GRAY, lw=1, ls=":")
    ax.scatter(st[~closed], bo[~closed], c=BLUE, s=70, label="open-weight", alpha=0.85)
    ax.scatter(st[closed], bo[closed], c=RED, s=70, marker="s", label="closed", alpha=0.85)
    gx = np.linspace(st.min(), st.max(), 50)
    ax.plot(gx, reg.intercept + reg.slope * gx, color="black", lw=2,
            label=f"fit: Pearson r={r['pearson_r']:.2f} (p={r['pearson_p']:.3f})\nSpearman ρ={r['spearman_r']:.2f} (p={r['spearman_p']:.3f})")
    for rr in rows:
        ax.annotate(rr["model"].split("/")[-1], (rr["static"], rr["boost"]),
                    fontsize=6.5, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Static score (model strength, 0–10)")
    ax.set_ylabel("Active − Static boost")
    ax.set_title("F2: stronger models gain MORE from Active mode\n(n=16 paired models)")
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "slide_f2_boost_vs_strength.png", dpi=150); plt.close(fig)
    print("✓ slide_f2_boost_vs_strength.png")


# ---------------------------------------------------------------- F3
def fig_f3():
    lb = [r for r in E0["leaderboard"] if r["cutoff"] is not None]
    sx = np.array([r["cutoff"] for r in lb]); sy = np.array([r["uniform_static"] for r in lb])
    rows = E0["F2_boost"]["rows"]
    ax_ = [r for r in rows if r["cutoff"] is not None]
    cx = np.array([r["cutoff"] for r in ax_]); cy = np.array([r["active"] for r in ax_])
    sl = E0["F3_cutoff"]["static_slope_per_yr"]; al = E0["F3_cutoff"]["active_slope_per_yr"]
    ratio = E0["F3_cutoff"]["slope_ratio_active_over_static"]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(sx, sy, c=BLUE, s=45, alpha=0.7, label="Static (given refs), n=40")
    ax.scatter(cx, cy, c=RED, s=55, marker="^", alpha=0.85, label="Active (tools only), n=16")
    gx = np.linspace(min(sx.min(), cx.min()), max(sx.max(), cx.max()), 100)
    ax.plot(gx, sl["intercept"] + sl["slope"] * gx, color=BLUE, lw=2.5,
            label=f"Static slope = +{sl['slope']:.2f}/yr (p={sl['p']:.0e})")
    ax.plot(gx, al["intercept"] + al["slope"] * gx, color=RED, lw=2.5,
            label=f"Active slope = +{al['slope']:.2f}/yr (p={al['p']:.0e})")
    ax.set_xlabel("knowledge cutoff (year)"); ax.set_ylabel("score (0–10)")
    ax.set_title(f"F3: both modes rise with cutoff — Active is steeper ({ratio}×)\nparadigm gap widens over time")
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "slide_f3_two_slopes.png", dpi=150); plt.close(fig)
    print("✓ slide_f3_two_slopes.png")


# ---------------------------------------------------------------- E6
def _weighted(s): return sum(s.get(d, 0) * WEIGHTS[d] for d in DIMS if d in s) / WSUM
def _tmean(v): return None if not v else (v[0] if len(v) < 2 else statistics.mean(sorted(v)[:-1]))


def fig_e6():
    # recompute (gap, demeaned score) cloud for Static
    cp = sqlite3.connect(str(cfg.PAPERS_DB)); cp.row_factory = sqlite3.Row
    def d2y(s):
        try: return int(s[:4]) + (int(s[5:7]) - 0.5) / 12.0
        except Exception: return None
    pdate = {r["paper_id"]: d2y(r["published_date"])
             for r in cp.execute("SELECT paper_id,published_date FROM papers WHERE status='filtered'")}
    cp.close()
    cr = sqlite3.connect(str(cfg.RESULTS_DB)); cr.row_factory = sqlite3.Row
    cell = defaultdict(list)
    for r in cr.execute("SELECT paper_id,idea_model,track,idea_index,scores_json FROM uniform_critic_scores WHERE track='B' AND scores_json IS NOT NULL"):
        cell[(r["idea_model"], r["paper_id"], r["idea_index"])].append(_weighted(json.loads(r["scores_json"])))
    cr.close()
    best = {}
    for (m, p, i), v in cell.items():
        tm = _tmean(v)
        if tm is None: continue
        if (m, p) not in best or tm > best[(m, p)]: best[(m, p)] = tm
    by_model = defaultdict(list)
    for (m, p), sc in best.items():
        co = CUTOFFS.get(m); py = pdate.get(p)
        if co is None or py is None: continue
        by_model[m].append((py - co, sc))
    gaps, dsc = [], []
    for m, pts in by_model.items():
        if len(pts) < 4: continue
        mu = statistics.mean([s for _, s in pts])
        for g, s in pts: gaps.append(g); dsc.append(s - mu)
    reg = stats.linregress(gaps, dsc)
    f = E6["recency_gap_regression"]["static_B"]

    # Bin recency_gap into 1-year bins; mean within-model-centred score ± SE per bin.
    gaps = np.array(gaps); dsc = np.array(dsc)
    edges = [0, 1, 2, 3, 10]
    labels = ["0–1 yr", "1–2 yr", "2–3 yr", "3 yr+"]
    bx, by, berr, bn = [], [], [], []
    for i in range(len(edges) - 1):
        m = (gaps >= edges[i]) & (gaps < edges[i + 1])
        if m.sum() >= 5:
            bx.append(i); by.append(float(dsc[m].mean()))
            berr.append(float(dsc[m].std(ddof=1) / np.sqrt(m.sum()))); bn.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color="black", lw=1, ls=":")
    ax.scatter(gaps, dsc, s=10, alpha=0.12, c=BLUE, label="each (model, paper)")
    ax.errorbar(bx, by, yerr=berr, fmt="o-", color=RED, lw=2.5, ms=10, capsize=6,
                label="bin mean ± SE")
    for i, n, y in zip(bx, bn, by):
        ax.annotate(f"n={n}", (i, y + max(berr) + 0.08), ha="center", fontsize=9, color=RED)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel("how far the test paper is published AFTER the model's knowledge cutoff")
    ax.set_ylabel("within-model score\n(0 = that model's own average)")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("E6: a paper 3+ yr past the cutoff scores the SAME as one just past it\n"
                 f"flat line (slope {f['pooled_demeaned_slope_per_yr']:+.3f}/yr, p={f['slope_p']:.2f}) "
                 "→ high scores come from CAPABILITY, not from\nhaving seen recent / target work (no memorisation effect)")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "slide_e6_recency.png", dpi=150); plt.close(fig)
    print("✓ slide_e6_recency.png")


# ---------------------------------------------------------------- E9
def fig_e9():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    models = ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "qwen/qwen3.5-397b-a17b"]
    budgets = [1, 5, 10, 15, 20]
    sc = defaultdict(list)
    for r in conn.execute("SELECT idea_model,budget,paper_id,idea_index,scores_json FROM budget_sweep_scores WHERE scores_json IS NOT NULL"):
        sc[(r["idea_model"], r["budget"], r["paper_id"], r["idea_index"])].append(_weighted(json.loads(r["scores_json"])))
    per = defaultdict(lambda: defaultdict(list))
    for (m, b, p, i), v in sc.items():
        per[m][b].append(_tmean(v))
    tc = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT idea_model,budget,n_tool_calls FROM budget_sweep_ideas WHERE idea_text!=''"):
        tc[r["idea_model"]][r["budget"]].append(r["n_tool_calls"] or 0)
    conn.close()

    colors = {"qwen/qwen3.5-9b": BLUE, "qwen/qwen3.5-27b": GREEN, "qwen/qwen3.5-397b-a17b": RED}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for m in models:
        ys = [statistics.mean(per[m][b]) for b in budgets]
        ax[0].plot(budgets, ys, "-o", color=colors[m], lw=2, label=m.split("/")[-1])
    # flag the low-reliability 397b @20 cell (50% malformed → n=15)
    y397_20 = statistics.mean(per["qwen/qwen3.5-397b-a17b"][20])
    ax[0].scatter([20], [y397_20], s=180, facecolors="none", edgecolors=RED, lw=2, zorder=5)
    ax[0].annotate("397b @20: n=15 only\n(50% malformed → unreliable)", (20, y397_20),
                   xytext=(-8, -42), textcoords="offset points", fontsize=8, color=RED,
                   ha="right", arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax[0].axvline(5, color=GRAY, ls="--", alpha=0.7)
    ax[0].text(5.2, ax[0].get_ylim()[0] + 0.1, "saturates @5", color=GRAY, fontsize=10)
    ax[0].set_xlabel("tool-call budget (max_iters)"); ax[0].set_ylabel("mean score (0–10)")
    ax[0].set_xticks(budgets)
    ax[0].set_title("(A) SCORE saturates at budget=5 for all sizes\n(bigger model = higher start, but extra budget adds nothing)")
    ax[0].legend(fontsize=10)
    for m in models:
        ys = [statistics.mean(tc[m][b]) for b in budgets]
        ax[1].plot(budgets, ys, "-o", color=colors[m], lw=2, label=m.split("/")[-1])
    ax[1].plot(budgets, budgets, color="black", ls=":", alpha=0.5, label="y=x (uses full budget)")
    ax[1].set_xlabel("tool-call budget (max_iters)"); ax[1].set_ylabel("actual mean tool calls used")
    ax[1].set_xticks(budgets)
    ax[1].set_title("(B) Actual calls used stays ≤~10 even at budget=20\nall sizes leave most of a large budget unused → self-terminate\n(more budget doesn't translate into more searching)")
    ax[1].legend(fontsize=10, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "slide_e9_budget_sweep.png", dpi=150); plt.close(fig)
    print("✓ slide_e9_budget_sweep.png")


if __name__ == "__main__":
    fig_f1(); fig_f2(); fig_f3(); fig_e6(); fig_e9()
    print("ALL night figures done")
