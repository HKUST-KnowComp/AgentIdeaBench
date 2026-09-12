r"""Score and tool-call usage against the Active interaction budget.

Style target is a budget-sweep panel: markers at each budget, vertical 95% CI
bars with caps, and a light band joining them, so a flat run of overlapping
bars reads as saturation rather than as noise.

Data: results.db.budget_sweep_{ideas,scores}_v3 (experiments/budget_sweep_v2.py
--suffix _v3). Every cell in _v3 was collected in the 2026-09-01/02 batch, so
budget is not confounded with collection date and the retained backbones share one
panel. The earlier _v2 tables mixed a 2026-05/06 Qwen batch with the new one
and needed two batch-separated panels; that version is kept at
archive/reports/_fig_turn_budget_twopanel_20260902.py.

kimi-k2.6 was collected in the same _v3 batch and its rows stay in the DB
untouched; it is excluded from this figure and from the numbers printed
here by a roster decision on 2026-09-07, not by any data-quality filter.
The panel is therefore five backbones. To put it back, restore its tuple
in MODELS and rerun -- nothing else is conditioned on it.

Rows generated before the Semantic Scholar retry fix (active_agent.SS_BACKOFFS)
live only in the unsuffixed base tables and are excluded here, because a
throttled SEARCH silently returned an empty result and the agent explored blind
while still spending a tool call.

Panel A aggregation nests critic -> idea -> paper -> model, so a paper with a
dropped idea cannot outweigh a complete one, and the CI is a 4000-draw
bootstrap over the 10 papers (the sampling unit). Panel B averages n_tool_calls
with the same nesting and the same bootstrap. Both panels keep exactly the
rollouts that produced idea text, including the ones flagged malformed_x3 that
still emitted a FINAL, so the two panels describe one population.

Cells retaining fewer than MIN_PAPERS papers are drawn hollow with a dashed
connector and must not be read as a drop.

Writes fig_turn_budget.{pdf,png} to reports/figures/summary/. Read-only on the
DB. Run with the base conda python.
"""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

SUFFIX = "_v3"
BUDGETS = [1, 2, 5, 10, 15, 20]

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())

# Warm = the same-vendor Qwen3.5 size ladder, cool = the cross-family arm.
# kimi-k2.6 is collected and kept in the DB but excluded from this panel:
# see FIG8_KIMI_EXCLUDED below.
MODELS = [
    ("qwen/qwen3.5-397b-a17b", "Qwen3.5-397B", "#7b341e", "o"),
    ("qwen/qwen3.5-27b",       "Qwen3.5-27B",  "#dd6b20", "s"),
    ("qwen/qwen3.5-9b",        "Qwen3.5-9B",   "#f6ad55", "^"),
    ("google/gemma-4-31b-it",  "Gemma-4-31B",  "#4c51bf", "v"),
    ("xiaomi/mimo-v2.5",       "MiMo-V2.5",    "#97266d", "P"),
]
FIG8_KIMI_EXCLUDED = "moonshotai/kimi-k2.6"

INK, MUTED = "#1f2937", "#64748b"
# the aggregate is the panel's subject, so it gets the darkest ink in it
AGG = "#1f2937"
MIN_PAPERS = 10
# budgets from here on are the ones tested for pairwise indistinguishability
NS_FROM = 5
N_BOOT = 4000

plt.rcParams.update({
    "font.size": 8.0, "axes.labelsize": 8.5, "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0, "axes.titlesize": 8.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def weighted(scores: dict) -> float:
    return sum(W[d] * float(scores.get(d, 0)) for d in DIMS) / WS


def _boot(cells: dict) -> dict:
    """(model, budget) -> per-paper values  ==>  mean / lo / hi / n_papers."""
    rng = np.random.default_rng(0)
    out = {}
    for key, vals in cells.items():
        arr = np.asarray(vals, dtype=float)
        boot = np.array([rng.choice(arr, arr.size, replace=True).mean()
                         for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[key] = {"mean": float(arr.mean()), "lo": float(lo),
                    "hi": float(hi), "n_papers": int(arr.size)}
    return out


def load():
    conn = sqlite3.connect(f"file:{ROOT / 'data' / 'results.db'}?mode=ro",
                           uri=True)
    conn.row_factory = sqlite3.Row

    per_idea = defaultdict(list)
    for r in conn.execute(
            f"SELECT idea_model, budget, paper_id, idea_index, scores_json "
            f"FROM budget_sweep_scores{SUFFIX} WHERE scores_json IS NOT NULL"):
        try:
            s = json.loads(r["scores_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(s, dict):
            continue
        per_idea[(r["idea_model"], r["budget"],
                  r["paper_id"], r["idea_index"])].append(weighted(s))

    calls = defaultdict(list)
    for r in conn.execute(
            f"SELECT idea_model, budget, paper_id, n_tool_calls "
            f"FROM budget_sweep_ideas{SUFFIX} "
            f"WHERE idea_text IS NOT NULL AND TRIM(idea_text) <> '' "
            f"AND n_tool_calls IS NOT NULL"):
        calls[(r["idea_model"], r["budget"], r["paper_id"])].append(
            float(r["n_tool_calls"]))
    conn.close()

    per_paper = defaultdict(list)
    for (m, b, p, _), vals in per_idea.items():
        per_paper[(m, b, p)].append(float(np.mean(vals)))

    score_cells, call_cells = defaultdict(list), defaultdict(list)
    for (m, b, _), vals in per_paper.items():
        score_cells[(m, b)].append(float(np.mean(vals)))
    for (m, b, _), vals in calls.items():
        call_cells[(m, b)].append(float(np.mean(vals)))

    return _boot(score_cells), _boot(call_cells)


def draw_aggregate(ax, stats, ylabel, title):
    """Left panel: the retained backbones as translucent points, their
    cross-model mean +- 1 SD as the figure's main line.

    The panel's claim is about the roster, not about any one backbone, so the
    per-model series are demoted to scatter and only the aggregate carries ink.
    The spread is the standard deviation ACROSS the backbone means at that
    budget, not a within-backbone interval: it says how differently the roster
    responds to the budget, which is the quantity the operating point is chosen
    against.
    """
    xs = np.arange(len(BUDGETS), dtype=float)
    ax.grid(axis="y", alpha=0.18, lw=0.5, ls="--")
    ax.set_axisbelow(True)

    per_budget = []
    for x, b in zip(xs, BUDGETS):
        vals = []
        for key, label, col, mk in MODELS:
            s = stats.get((key, b))
            if not s:
                continue
            vals.append(s["mean"])
            ax.plot([x], [s["mean"]], marker=mk, ms=3.2, color=col, alpha=0.32,
                    mec="none", ls="none", zorder=2)
        per_budget.append(vals)

    mu = np.array([np.mean(v) for v in per_budget])
    sd = np.array([np.std(v, ddof=1) for v in per_budget])
    ax.fill_between(xs, mu - sd, mu + sd, color=AGG, alpha=0.085, lw=0, zorder=3)
    ax.plot(xs, mu - sd, color=AGG, lw=0.6, alpha=0.30, zorder=3)
    ax.plot(xs, mu + sd, color=AGG, lw=0.6, alpha=0.30, zorder=3)
    ax.plot(xs, mu, color=AGG, lw=1.9, zorder=5, solid_capstyle="round")
    ax.plot(xs, mu, marker="o", ms=4.0, ls="none", color=AGG, mfc="white",
            mew=1.4, zorder=6)

    # the operating point the paper actually runs at
    x10 = float(BUDGETS.index(10))
    ax.axvline(x10, color=MUTED, lw=0.8, ls=(0, (3, 2)), alpha=0.85, zorder=1)
    ax.annotate("operating\nbudget", xy=(x10 - 0.10, ax.get_ylim()[0]),
                xytext=(-2, 3), textcoords="offset points", color=MUTED,
                fontsize=6.6, ha="right", va="bottom", linespacing=1.15,
                zorder=7)
    # The panel's actual claim is that nothing past a couple of turns is
    # distinguishable, so the smallest p over every pair of budgets >= NS_FROM is
    # computed here rather than asserted: if a later batch made some pair
    # significant, this annotation would say so instead of silently flattering
    # the operating point.
    idx = [i for i, b in enumerate(BUDGETS) if b >= NS_FROM]
    M = np.array([[stats[(k, b)]["mean"] for b in BUDGETS]
                  for k, _, _, _ in MODELS if (k, BUDGETS[0]) in stats])
    pmin = min(sps.ttest_rel(M[:, j], M[:, i]).pvalue
               for a, i in enumerate(idx) for j in idx[a + 1:])
    ax.annotate(f"mean $\\pm$ 1 SD across backbones\n"
                f"pairwise n.s. beyond budget {NS_FROM} ($p{{\\geq}}{pmin:.2f}$)",
                xy=(0.02, 0.965), xycoords="axes fraction", color=AGG,
                fontsize=6.6, ha="left", va="top", linespacing=1.25, zorder=7)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in BUDGETS], color=INK)
    ax.set_xlim(-0.32, len(BUDGETS) - 0.68)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, pad=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2.5, width=0.7)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_color(INK)
    return mu, sd


def draw(ax, stats, ylabel, title):
    xs = np.arange(len(BUDGETS), dtype=float)
    ax.grid(axis="y", alpha=0.18, lw=0.5, ls="--")
    ax.set_axisbelow(True)

    for key, label, col, mk in MODELS:
        pts = [stats.get((key, b)) for b in BUDGETS]
        full = [(x, s) for x, s in zip(xs, pts)
                if s and s["n_papers"] >= MIN_PAPERS]
        if not full:
            continue
        fx = np.array([x for x, _ in full])
        fy = np.array([s["mean"] for _, s in full])
        flo = np.array([s["lo"] for _, s in full])
        fhi = np.array([s["hi"] for _, s in full])

        ax.fill_between(fx, flo, fhi, color=col, alpha=0.055, lw=0, zorder=1)
        ax.plot(fx, fy, color=col, lw=1.2, zorder=3, label=label)
        ax.errorbar(fx, fy, yerr=[fy - flo, fhi - fy], fmt=mk, ms=3.6,
                    color=col, mfc=col, mec=col, mew=0.9, elinewidth=0.85,
                    capsize=1.8, capthick=0.85, zorder=4)

        for x, s in zip(xs, pts):
            if not s or s["n_papers"] >= MIN_PAPERS:
                continue
            ax.plot([fx[-1], x], [fy[-1], s["mean"]], color=col, lw=1.0,
                    ls=(0, (2, 2)), alpha=0.75, zorder=2)
            ax.errorbar([x], [s["mean"]],
                        yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],
                        fmt=mk, ms=3.6, color=col, mfc="white", mec=col,
                        mew=1.0, elinewidth=0.85, capsize=1.8, capthick=0.85,
                        alpha=0.8, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in BUDGETS], color=INK)
    ax.set_xlim(-0.32, len(BUDGETS) - 0.68)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, pad=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2.5, width=0.7)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_color(INK)


def main():
    scores, calls = load()

    fig = plt.figure(figsize=(6.3, 2.55))
    axL = fig.add_axes([0.070, 0.205, 0.395, 0.615])
    axR = fig.add_axes([0.585, 0.205, 0.395, 0.615])

    mu, sd = draw_aggregate(axL, scores, "Weighted score", "Idea quality")
    draw(axR, calls, "Tool calls used", "Budget actually spent")

    # The cap itself: a model that always exhausted its budget would sit here.
    axR.plot(np.arange(len(BUDGETS)), BUDGETS, color=MUTED, lw=0.9,
             ls=(0, (3, 2)), zorder=2)
    axR.annotate("cap", xy=(len(BUDGETS) - 1.06, BUDGETS[-1]), color=MUTED,
                 fontsize=7.0, ha="right", va="bottom")
    axR.set_yscale("log")
    axR.set_yticks([1, 2, 5, 10, 20])
    axR.set_yticklabels(["1", "2", "5", "10", "20"], color=INK)
    axR.minorticks_off()

    # One legend row across the top: every series in either panel would sit on
    # top of the data or the right panel's ylabel anywhere inside the axes.
    handles, labels = axR.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODELS),
               bbox_to_anchor=(0.525, 1.012), frameon=False, fontsize=7.2,
               handlelength=1.5, columnspacing=1.35, handletextpad=0.45,
               borderpad=0.0)

    fig.text(0.525, 0.035, "Active interaction budget (max tool calls)",
             fontsize=8.5, color=INK, ha="center", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_turn_budget.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)

    print(f"wrote fig_turn_budget.pdf/.png  (source: *{SUFFIX})")
    print("\nCross-backbone aggregate (left panel)")
    for b, m_, s_ in zip(BUDGETS, mu, sd):
        print(f"  budget {b:>2}  mean {m_:.3f}  sd {s_:.3f}")
    print(f"  delta mean 10 -> 20: {mu[BUDGETS.index(20)] - mu[BUDGETS.index(10)]:+.3f}"
          f"   (sd at 10 = {sd[BUDGETS.index(10)]:.3f})")
    print(f"\nPaired t over the {len(MODELS)} backbones, "
          f"every pair of budgets")
    Mm = np.array([[scores[(k, b)]["mean"] for b in BUDGETS]
                   for k, _, _, _ in MODELS])
    for a, bi in enumerate(BUDGETS):
        for c, bj in enumerate(BUDGETS):
            if c <= a:
                continue
            d = Mm[:, c] - Mm[:, a]
            r = sps.ttest_rel(Mm[:, c], Mm[:, a])
            print(f"  b{bi:>2} -> b{bj:>2}  d={d.mean():+.3f}  p={r.pvalue:.3f}")
    for name, stats, fmt in (("Weighted score", scores, "{:.3f}"),
                             ("Tool calls used", calls, "{:.2f}")):
        print(f"\n{name}")
        for key, label, _, _ in MODELS:
            row = []
            for b in BUDGETS:
                s = stats.get((key, b))
                if not s:
                    row.append(f"b{b}=MISSING")
                    continue
                mark = "" if s["n_papers"] >= MIN_PAPERS else "*"
                row.append(f"b{b}=" + fmt.format(s["mean"]) +
                           "[" + fmt.format(s["lo"]) + "," +
                           fmt.format(s["hi"]) + "]" + mark)
            print(f"  {label:<13} " + "  ".join(row))
    print(f"\n* = fewer than {MIN_PAPERS} papers retained; drawn hollow, "
          f"not joined to the band")
    print("Utilization at b=20 (mean tool calls / 20):")
    for key, label, _, _ in MODELS:
        s = calls.get((key, 20))
        if s:
            print(f"  {label:<13} {s['mean'] / 20:.1%}  "
                  f"({s['mean']:.2f}/20, n_papers={s['n_papers']})")


if __name__ == "__main__":
    main()
