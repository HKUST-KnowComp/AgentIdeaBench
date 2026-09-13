"""Coherence Check / Boilerplate trigger rate in Static vs Active critic responses.

The critic prompt has 3 caps:
  - Coherence Check (keyword stuffing) → O cap 5, S cap 6
  - Factual Consistency Check → O cap 4, F cap 3, S cap 4 if INCORRECT
  - Boilerplate Check → O cap 5/6, I cap 6

We detect cap-trigger by regex on the raw critic response. A robust new finding:
**Active mode reduces cap-trigger rates substantially**, providing direct evidence
for the mechanism in Finding 2 (Originality bottleneck explained by Coherence Check).

Output: bar chart + table.
"""
import json
import re
import sqlite3
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
TAB = PAPER / "tables"

COH_PATTERNS = [
    r"KEYWORD STUFFING",
    r"keyword.stuff",
    r"cap.*[Oo]rig.*at 5",
    r"[Cc]oherence [Cc]heck.*applied",
    r"capped.*[Oo]rig.*5",
    r"BOILERPLATE",
]
trigger_re = re.compile("|".join(COH_PATTERNS))


def main():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    per_model = defaultdict(lambda: [0, 0])
    for m, t, raw in conn.execute("""
        SELECT idea_model, track, raw_response FROM results
        WHERE prompt_version='v1_paper_refs' AND track IN ('B','C') AND idea_index=1
          AND critic_model != '' AND raw_response IS NOT NULL
    """):
        if not raw:
            continue
        per_model[(m, t)][1] += 1
        if trigger_re.search(raw):
            per_model[(m, t)][0] += 1
    conn.close()

    # Compute pairs (model, B%, C%, ΔB-C)
    models_with_both = sorted({m for (m, t) in per_model if t == "C" and per_model[(m, t)][1] >= 20})
    rows = []
    for m in models_with_both:
        b_trig, b_n = per_model.get((m, "B"), [0, 0])
        c_trig, c_n = per_model.get((m, "C"), [0, 0])
        if b_n < 20 or c_n < 20:
            continue
        rows.append({
            "model": m, "b_pct": 100 * b_trig / b_n, "c_pct": 100 * c_trig / c_n,
            "delta": 100 * (b_trig / b_n - c_trig / c_n), "b_n": b_n, "c_n": c_n,
        })
    rows.sort(key=lambda r: -r["delta"])

    print(f"{'Model':<42} {'Static%':>8} {'Active%':>8} {'Δ(B-C)':>8}")
    for r in rows:
        print(f"  {r['model']:<40} {r['b_pct']:>7.1f}% {r['c_pct']:>7.1f}% {r['delta']:>+7.1f}%")

    # Plot: side-by-side bars
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n = len(rows)
    xs = np.arange(n)
    width = 0.38
    b_pcts = [r["b_pct"] for r in rows]
    c_pcts = [r["c_pct"] for r in rows]
    ax.bar(xs - width/2, b_pcts, width, label="Static (Track B)",
           color="#888", edgecolor="#111", linewidth=0.8)
    ax.bar(xs + width/2, c_pcts, width, label="Active (Track C)",
           color="#10a37f", edgecolor="#111", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["model"].split("/")[-1] for r in rows],
                       rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Coherence/Boilerplate cap-trigger rate (%)", fontsize=12)
    ax.set_title(f"Finding 5: Active mode reduces rubric-cap triggering "
                 f"({n} models with $\\geq$20 ratings per track)",
                 fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / "finding5_coh_trigger.pdf", bbox_inches="tight")
    plt.savefig(FIG / "finding5_coh_trigger.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {FIG / 'finding5_coh_trigger.pdf'}")

    # Aggregate stats
    mean_b = np.mean(b_pcts)
    mean_c = np.mean(c_pcts)
    mean_delta = np.mean([r["delta"] for r in rows])
    print(f"\nMean Static trigger rate: {mean_b:.1f}%")
    print(f"Mean Active trigger rate: {mean_c:.1f}%")
    print(f"Mean reduction (B-C):     {mean_delta:+.1f}%")
    n_decrease = sum(1 for r in rows if r["delta"] > 0)
    print(f"Models showing decrease:  {n_decrease}/{n}")


if __name__ == "__main__":
    main()
