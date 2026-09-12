"""Finding C: 12/13 paired models show positive Active boost (robustness).

Horizontal bar chart, models sorted by boost.
Green bars = positive boost; red = negative.
Static and Active means shown as dot markers on each row.
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

W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())
FIG = ROOT / "reports" / "figures"


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in W) / WS


def main():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    b_rows = conn.execute(
        "SELECT idea_model, paper_id, scores_json FROM results "
        "WHERE track='B' AND prompt_version='v1_paper_refs' "
        "  AND critic_model != '' AND scores_json IS NOT NULL"
    ).fetchall()
    c_rows = conn.execute(
        "SELECT idea_model, paper_id, scores_json FROM results "
        "WHERE track='C' AND prompt_version='v1_paper_refs' "
        "  AND critic_model != '' AND scores_json IS NOT NULL"
    ).fetchall()
    conn.close()

    def to_dict(rows):
        by_key = defaultdict(list)
        for m, p, sj in rows:
            try:
                by_key[(m, p)].append(weighted(json.loads(sj)))
            except Exception:
                pass
        return {k: statistics.mean(v) for k, v in by_key.items()}

    static = to_dict(b_rows)
    active = to_dict(c_rows)

    paired = []
    for m in sorted({m for (m, _) in active}):
        sm = [v for (mm, _), v in static.items() if mm == m]
        am = [v for (mm, _), v in active.items() if mm == m]
        if sm and am:
            paired.append({
                "model": m,
                "short": m.split("/")[-1].replace("-instruct", "").replace("-it", ""),
                "static": statistics.mean(sm),
                "active": statistics.mean(am),
                "boost": statistics.mean(am) - statistics.mean(sm),
            })

    paired.sort(key=lambda r: r["boost"])
    n_pos = sum(1 for r in paired if r["boost"] > 0)
    n_neg = sum(1 for r in paired if r["boost"] <= 0)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))
    y_pos = np.arange(len(paired))
    boosts = np.array([r["boost"] for r in paired])
    colors = ["#10a37f" if b > 0 else "#dc2626" for b in boosts]
    ax.barh(y_pos, boosts, color=colors, edgecolor="#1f2937", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    # Annotate each bar with Static / Active / boost — always to the right of the bar end
    # for negative bars, the right side is the zero axis. Put the label to the right of zero.
    for i, r in enumerate(paired):
        x = r["boost"]
        if x >= 0:
            label_x = x + 0.04
            ha = "left"
        else:
            label_x = 0.04  # always right of zero axis
            ha = "left"
        ax.text(label_x, i,
                f"{x:+.2f}   (Static {r['static']:.2f} → Active {r['active']:.2f})",
                va="center", ha=ha,
                fontsize=8.5, color="#1f2937")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["short"] for r in paired], fontsize=9.5)
    ax.set_xlabel("Active − Static boost  (weighted score points)", fontsize=11.5)
    ax.set_title(f"{n_pos}/{len(paired)} models gain from Active paradigm (positive boost)",
                 fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(min(boosts) - 0.3, max(boosts) + 1.5)

    plt.tight_layout()
    out_pdf = FIG / "active_boost_ranked.pdf"
    out_png = FIG / "active_boost_ranked.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close()

    print(f"n_positive = {n_pos} / {len(paired)}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
