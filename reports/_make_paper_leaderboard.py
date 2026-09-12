"""Cross-year Static leaderboard figure for paper (Finding 0: benchmark establishes a scaling).

28-model Static scores plotted vs (a) release date, (b) knowledge cutoff.
"""
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from reports._make_cross_year_plot import RELEASE_DATES, KNOWLEDGE_CUTOFFS, YEAR_GROUPS

PAPER = ROOT / "docs" / "paper"
FIG = PAPER / "figures"
TAB = PAPER / "tables"

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in DIMS) / WS


def load_static_v1():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    per_paper = defaultdict(lambda: defaultdict(list))
    for m, pid, sj in conn.execute("""
        SELECT idea_model, paper_id, scores_json FROM results
        WHERE prompt_version='v1_paper_refs' AND track='B' AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        per_paper[m][pid].append(weighted(s))
    conn.close()
    out = {}
    for m, pp in per_paper.items():
        vals = [statistics.mean(v) for v in pp.values()]
        if len(vals) >= 20:  # need ≥20 papers to count as full eval
            out[m] = (statistics.mean(vals), statistics.stdev(vals) / (len(vals)**0.5) if len(vals)>1 else 0, len(vals))
    return out


FAMILY_COLOR = {
    "qwen": "#7c3aed", "deepseek": "#0ea5e9", "google": "#10a37f",
    "moonshotai": "#f59e0b", "z-ai": "#dc2626", "mistralai": "#6b7280",
    "meta-llama": "#3b82f6", "openai": "#111827", "anthropic": "#ea580c",
    "xiaomi": "#a855f7",
}


def make_leaderboard():
    static = load_static_v1()
    # Map model to label
    label_for = {mid: lab for ys in YEAR_GROUPS.values() for mid, lab in ys}

    # Two-panel plot: top by release date, bottom by cutoff
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))

    # Top: by release date
    for m, (score, sem, n) in static.items():
        if m not in RELEASE_DATES:
            continue
        x = RELEASE_DATES[m]
        fam = m.split("/")[0]
        color = FAMILY_COLOR.get(fam, "#888")
        ax1.errorbar(x, score, yerr=sem, fmt='o', markersize=10, color=color,
                     markeredgecolor="#111", markeredgewidth=1.0, capsize=3,
                     ecolor="#888", elinewidth=0.8, zorder=3)
        ax1.annotate(label_for.get(m, m.split("/")[-1]), (x, score),
                     xytext=(5, 4), textcoords="offset points", fontsize=7)

    ax1.set_xlabel("Release date", fontsize=11)
    ax1.set_ylabel("Static (Track B) weighted score", fontsize=11)
    ax1.set_title(f"SciSynthBench Static leaderboard, {len(static)} models, "
                  f"25 papers × 5 domains = 125 evaluations per model",
                  fontsize=11, pad=10)
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))
    ax1.grid(axis="y", alpha=0.3, linestyle=":")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom: by knowledge cutoff
    for m, (score, sem, n) in static.items():
        if m not in KNOWLEDGE_CUTOFFS or KNOWLEDGE_CUTOFFS[m] is None:
            continue
        x = KNOWLEDGE_CUTOFFS[m]
        fam = m.split("/")[0]
        color = FAMILY_COLOR.get(fam, "#888")
        ax2.errorbar(x, score, yerr=sem, fmt='s', markersize=10, color=color,
                     markeredgecolor="#111", markeredgewidth=1.0, capsize=3,
                     ecolor="#888", elinewidth=0.8, zorder=3)
        ax2.annotate(label_for.get(m, m.split("/")[-1]), (x, score),
                     xytext=(5, 4), textcoords="offset points", fontsize=7)

    # OLS line vs cutoff
    xs = [KNOWLEDGE_CUTOFFS[m] for m in static if m in KNOWLEDGE_CUTOFFS and KNOWLEDGE_CUTOFFS[m] is not None]
    ys = [static[m][0] for m in static if m in KNOWLEDGE_CUTOFFS and KNOWLEDGE_CUTOFFS[m] is not None]
    if len(xs) >= 5:
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx2 = sum((x - mx) ** 2 for x in xs)
        slope = num / dx2 if dx2 > 0 else 0
        intercept = my - slope * mx
        dy2 = sum((y - my) ** 2 for y in ys)
        r = num / (dx2 * dy2) ** 0.5 if dx2 * dy2 > 0 else 0
        x_lo, x_hi = min(xs) - 0.1, max(xs) + 0.1
        ax2.plot([x_lo, x_hi], [slope * x_lo + intercept, slope * x_hi + intercept],
                 "k--", alpha=0.6, linewidth=1.5,
                 label=f"OLS: slope=$+${slope:.2f}/yr, r={r:.2f}")
        ax2.legend(loc="lower right", fontsize=9, frameon=False)

    ax2.set_xlabel("Knowledge cutoff date (decimal year)", fontsize=11)
    ax2.set_ylabel("Static (Track B) weighted score", fontsize=11)
    ax2.set_title(f"Same leaderboard, x-axis = knowledge cutoff (signal stronger here)",
                  fontsize=11, pad=10)
    ax2.xaxis.set_major_locator(MultipleLocator(1.0))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(round(x))}"))
    ax2.grid(axis="y", alpha=0.3, linestyle=":")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG / "leaderboard_release_vs_cutoff.pdf", bbox_inches="tight")
    plt.savefig(FIG / "leaderboard_release_vs_cutoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG / 'leaderboard_release_vs_cutoff.pdf'}")

    # Also print summary
    print(f"\nLeaderboard ({len(static)} models, sorted by score):")
    for m, (s, sem, n) in sorted(static.items(), key=lambda kv: -kv[1][0]):
        cutoff = KNOWLEDGE_CUTOFFS.get(m, "?")
        print(f"  {m:<45} score={s:.2f}±{sem:.2f} (n={n})  cutoff={cutoff}")


if __name__ == "__main__":
    make_leaderboard()
