"""Compare Track C Originality scores: WITH cap (original) vs WITHOUT cap (ablation).

If removing the Coherence Check + Boilerplate Check caps lifts O scores substantially,
the caps were binding in the original — direct evidence for F5 mechanism.
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


def main():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))

    # Original Track C O scores
    orig_o = defaultdict(list)  # model -> [O]
    for m, sj in conn.execute("""
        SELECT idea_model, scores_json FROM results
        WHERE prompt_version='v1_paper_refs' AND track='C' AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
          AND idea_model IN (
            'meta-llama/llama-3.1-8b-instruct',
            'mistralai/mistral-7b-instruct-v0.1',
            'qwen/qwen3.5-9b',
            'google/gemma-4-31b-it')
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        if "originality" in s:
            orig_o[m].append(float(s["originality"]))

    # Ablation Track C O scores
    abl_o = defaultdict(list)
    for m, sj in conn.execute("""
        SELECT idea_model, scores_json FROM cap_ablation_scores
        WHERE track='C' AND scores_json IS NOT NULL
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        if "originality" in s:
            abl_o[m].append(float(s["originality"]))

    models = sorted(set(orig_o) | set(abl_o))
    print(f"{'Model':<42} {'origO_mean':>11} {'abl_O_mean':>11} {'Δ':>6} {'%5_o':>6} {'%5_a':>6}")
    rows = []
    for m in models:
        o = orig_o.get(m, [])
        a = abl_o.get(m, [])
        if not o or not a:
            continue
        mo, ma = statistics.mean(o), statistics.mean(a)
        pct5_o = 100 * sum(1 for x in o if x == 5) / len(o)
        pct5_a = 100 * sum(1 for x in a if x == 5) / len(a)
        print(f"  {m:<40} {mo:>11.2f} {ma:>11.2f} {ma-mo:>+6.2f} {pct5_o:>5.1f}% {pct5_a:>5.1f}%")
        rows.append({"model": m, "orig": mo, "abl": ma, "delta": ma - mo,
                     "pct5_orig": pct5_o, "pct5_abl": pct5_a,
                     "n_o": len(o), "n_a": len(a)})

    # Plot: distribution shift
    fig, axes = plt.subplots(1, len(rows), figsize=(3.5 * len(rows), 4.5), sharey=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        bins = np.arange(0.5, 11, 1)
        o_vals = orig_o[r["model"]]
        a_vals = abl_o[r["model"]]
        ax.hist(o_vals, bins=bins, alpha=0.55, label=f"With cap (n={len(o_vals)})",
                color="#888", edgecolor="#111")
        ax.hist(a_vals, bins=bins, alpha=0.55, label=f"No cap (n={len(a_vals)})",
                color="#10a37f", edgecolor="#111")
        ax.axvline(5, color="red", linestyle="--", alpha=0.6, linewidth=1.2)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Originality")
        ax.set_title(r["model"].split("/")[-1], fontsize=10)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Count", fontsize=11)
    plt.suptitle(f"Cap-Removed Ablation: Track C Originality distributions",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / "finding5_cap_ablation.pdf", bbox_inches="tight")
    plt.savefig(FIG / "finding5_cap_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {FIG / 'finding5_cap_ablation.pdf'}")

    # Summary
    print("\nSummary:")
    deltas = [r["delta"] for r in rows]
    print(f"  Mean ΔO (no-cap − with-cap): {statistics.mean(deltas):+.2f}")
    print(f"  Mean pct5 drop: {statistics.mean(r['pct5_orig'] - r['pct5_abl'] for r in rows):+.1f}%")


if __name__ == "__main__":
    main()
