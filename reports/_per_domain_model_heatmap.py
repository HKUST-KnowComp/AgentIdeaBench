"""Per-model × per-domain Active boost heatmap.

Reveals which model families excel in which scientific domains.
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


def main():
    conn_p = sqlite3.connect(str(cfg.PAPERS_DB))
    pdom = dict(conn_p.execute("SELECT paper_id, domain FROM papers"))
    conn_p.close()

    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    per_paper = defaultdict(list)  # (model, track, paper) -> [weighted]
    for m, t, pid, sj in conn.execute("""
        SELECT idea_model, track, paper_id, scores_json FROM results
        WHERE prompt_version='v1_paper_refs' AND track IN ('B','C') AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        per_paper[(m, t, pid)].append(weighted(s))
    conn.close()
    # Mean across critics
    mscore = {k: statistics.mean(v) for k, v in per_paper.items()}

    # Boost per (model, domain)
    DOMAINS = ["Biology", "CS", "Chemistry", "Medicine", "Physics"]
    boost_md = defaultdict(list)  # (model, domain) -> [boost values per paper]
    models_with_c = set()
    for (m, t, pid), sc in mscore.items():
        if t == "C":
            models_with_c.add(m)
    for m in models_with_c:
        for pid, dom in pdom.items():
            b = mscore.get((m, "B", pid))
            c = mscore.get((m, "C", pid))
            if b is None or c is None:
                continue
            boost_md[(m, dom)].append(c - b)

    # Models to include (sufficient data)
    valid_models = {m for m, _ in boost_md.keys()
                    if sum(len(boost_md[(m, d)]) for d in DOMAINS) >= 15}
    valid_models = sorted(valid_models)

    # Build matrix
    matrix = []
    for m in valid_models:
        row = []
        for dom in DOMAINS:
            vals = boost_md.get((m, dom), [])
            row.append(statistics.mean(vals) if vals else float("nan"))
        matrix.append(row)
    matrix = np.array(matrix)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, max(4, len(valid_models) * 0.4)))
    vmin, vmax = -1.5, 2.5
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(len(valid_models)):
        for j in range(len(DOMAINS)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=9, color="#111" if abs(v) < 1.0 else "white")
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels(DOMAINS, fontsize=10)
    ax.set_yticks(range(len(valid_models)))
    ax.set_yticklabels([m.split("/")[-1] for m in valid_models], fontsize=9)
    ax.set_xlabel("Scientific domain", fontsize=11)
    ax.set_title("Per-model × per-domain Active boost (Track C $-$ Track B)",
                 fontsize=12, pad=10)
    plt.colorbar(im, ax=ax, label="Boost")
    plt.tight_layout()
    plt.savefig(FIG / "perdomain_model_heatmap.pdf", bbox_inches="tight")
    plt.savefig(FIG / "perdomain_model_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote heatmap with {len(valid_models)} models × {len(DOMAINS)} domains")

    # Print summary
    print(f"\n{'Model':<42} " + " ".join(f"{d:>12}" for d in DOMAINS))
    for i, m in enumerate(valid_models):
        row = matrix[i]
        cells = [f"{v:>+12.2f}" if not np.isnan(v) else "         n/a" for v in row]
        print(f"  {m:<40} " + " ".join(cells))


if __name__ == "__main__":
    main()
