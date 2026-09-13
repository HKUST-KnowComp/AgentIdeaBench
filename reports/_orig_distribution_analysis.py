"""Histogram of Originality scores per model × track.

Tests Finding 2 mechanism: if Coherence Check caps Originality at 5, the distribution
should show a pile-up at score=5 for keyword-stuffed responses.

If Active reduces the pile-up at 5, that's evidence that tools allow models to
articulate more coherent (cap-free) hypotheses.
"""
import json
import sqlite3
import sys
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

PAPER = ROOT / "docs" / "paper"
FIG = PAPER / "figures"


def main():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    per_model = defaultdict(list)  # (model, track) -> [O scores]
    for m, t, sj in conn.execute("""
        SELECT idea_model, track, scores_json FROM results
        WHERE prompt_version='v1_paper_refs' AND track IN ('B','C') AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        if "originality" in s:
            per_model[(m, t)].append(float(s["originality"]))
    conn.close()

    # Pick 4 representative models: 2 with high cap-trigger reduction, 2 with low
    TARGETS = [
        ("meta-llama/llama-3.1-8b-instruct", "Llama-3.1-8B (big -Δ)"),
        ("mistralai/mistral-7b-instruct-v0.1", "Mistral-7B-v0.1 (big -Δ)"),
        ("qwen/qwen3.5-9b", "Qwen3.5-9B (medium -Δ)"),
        ("google/gemma-4-31b-it", "Gemma-4-31B (small -Δ)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    axes = axes.flatten()
    for i, (m, label) in enumerate(TARGETS):
        ax = axes[i]
        b = per_model.get((m, "B"), [])
        c = per_model.get((m, "C"), [])
        if not b or not c:
            ax.set_title(f"{label}: no data")
            continue
        bins = np.arange(0.5, 11, 1)
        ax.hist(b, bins=bins, alpha=0.55, label=f"Static (n={len(b)})", color="#888", edgecolor="#111")
        ax.hist(c, bins=bins, alpha=0.55, label=f"Active (n={len(c)})", color="#10a37f", edgecolor="#111")
        # Mark cap=5 line
        ax.axvline(5, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
        ax.text(5.05, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1, "cap=5",
                color="red", fontsize=8, fontweight="bold")
        ax.set_xlabel("Originality score", fontsize=10)
        if i % 2 == 0:
            ax.set_ylabel("Count", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(1, 11))
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Originality score distribution: Static vs Active (red line = Coherence-Check cap)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / "originality_distribution.pdf", bbox_inches="tight")
    plt.savefig(FIG / "originality_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG / 'originality_distribution.pdf'}")

    # Print summary table: pile-up at 5
    print(f"\n{'Model':<42} {'Track':<7} {'mean O':>7} {'@5%':>6} {'@<=5%':>7} {'n':>4}")
    for m, _ in TARGETS:
        for t in ["B", "C"]:
            scores = per_model.get((m, t), [])
            if not scores:
                continue
            mean = sum(scores) / len(scores)
            pct5 = 100 * sum(1 for s in scores if s == 5) / len(scores)
            pct_le5 = 100 * sum(1 for s in scores if s <= 5) / len(scores)
            print(f"  {m:<40} {t:<7} {mean:>7.2f} {pct5:>5.1f}% {pct_le5:>6.1f}% {len(scores):>4}")


if __name__ == "__main__":
    main()
