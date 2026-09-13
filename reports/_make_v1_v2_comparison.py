"""Compare v1 (paper-specific refs, no title) vs v2 (topic refs, with title) Active boost.

3 models have both v1 and v2 data: qwen3.5-9b, mistral-7b-v0.1, deepseek-r1-0528.

Output: bar chart showing v1 boost vs v2 boost per model.
"""
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

PAPER = ROOT / "docs" / "paper"
FIG = PAPER / "figures"

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in DIMS) / WS


def load_paper_boost(prompt_version):
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    per_paper = defaultdict(lambda: defaultdict(list))
    for m, t, pid, sj in conn.execute(f"""
        SELECT idea_model, track, paper_id, scores_json FROM results
        WHERE prompt_version=? AND track IN ('B','C') AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
    """, (prompt_version,)):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        per_paper[(m, t)][pid].append(weighted(s))
    conn.close()
    # Per-paper mean across critics
    out = defaultdict(dict)
    for (m, t), pp in per_paper.items():
        for pid, vs in pp.items():
            out[(m, t)][pid] = statistics.mean(vs)
    return out


def main():
    v1 = load_paper_boost("v1_paper_refs")
    v2 = load_paper_boost("v2_topic_refs")

    common_models = sorted(
        {m for (m, t) in v1 if t == "C"} & {m for (m, t) in v2 if t == "C"}
    )
    rows = []
    for m in common_models:
        b1 = v1.get((m, "B"), {})
        c1 = v1.get((m, "C"), {})
        b2 = v2.get((m, "B"), {})
        c2 = v2.get((m, "C"), {})
        if not all([b1, c1, b2, c2]):
            continue
        b1_mean = statistics.mean(b1.values())
        c1_mean = statistics.mean(c1.values())
        b2_mean = statistics.mean(b2.values())
        c2_mean = statistics.mean(c2.values())
        rows.append({
            "model": m,
            "label": m.split("/")[-1],
            "v1_B": b1_mean, "v1_C": c1_mean, "v1_boost": c1_mean - b1_mean,
            "v2_B": b2_mean, "v2_C": c2_mean, "v2_boost": c2_mean - b2_mean,
        })

    if not rows:
        print("No common models. Aborting.")
        return

    # Print table
    print(f"{'Model':<40} {'v1_B':>6} {'v1_C':>6} {'v1Δ':>6} {'v2_B':>6} {'v2_C':>6} {'v2Δ':>6}")
    for r in rows:
        print(f"  {r['label']:<38} {r['v1_B']:>6.2f} {r['v1_C']:>6.2f} {r['v1_boost']:>+6.2f}  "
              f"{r['v2_B']:>6.2f} {r['v2_C']:>6.2f} {r['v2_boost']:>+6.2f}")

    # Bar plot: side-by-side v1 vs v2 boost
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(rows)
    xs = np.arange(n)
    width = 0.35
    v1_boosts = [r["v1_boost"] for r in rows]
    v2_boosts = [r["v2_boost"] for r in rows]
    ax.bar(xs - width/2, v1_boosts, width, label="v1 (no title, paper-specific refs)",
           color="#10a37f", edgecolor="#111", linewidth=0.8)
    ax.bar(xs + width/2, v2_boosts, width, label="v2 (with title, topic-search refs)",
           color="#f59e0b", edgecolor="#111", linewidth=0.8)
    for i, (v1b, v2b) in enumerate(zip(v1_boosts, v2_boosts)):
        ax.text(i - width/2, v1b + 0.03 if v1b > 0 else v1b - 0.08, f"{v1b:+.2f}",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width/2, v2b + 0.03 if v2b > 0 else v2b - 0.08, f"{v2b:+.2f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Active mode boost (Track C − Track B)", fontsize=12)
    ax.set_title("Finding 4: Prompt-design sensitivity — v1 vs v2 Active boost",
                 fontsize=12, pad=10)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / "finding4_v1_v2_boost.pdf", bbox_inches="tight")
    plt.savefig(FIG / "finding4_v1_v2_boost.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {FIG / 'finding4_v1_v2_boost.pdf'}")


if __name__ == "__main__":
    main()
