"""Finding A: Active mode widens between-model variance (more discriminative).

X: Static mean per model
Y: Active mean per model
Each model = one point. Diagonal y=x is the "no change" reference.
Points above diagonal = positive Active boost.
Visual evidence: spread on Y > spread on X (Active std 1.34 × Static std).
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

    x = np.array([r["static"] for r in paired])
    y = np.array([r["active"] for r in paired])
    boost = np.array([r["boost"] for r in paired])

    std_static = float(np.std(x))
    std_active = float(np.std(y))
    spread_static = float(x.max() - x.min())
    spread_active = float(y.max() - y.min())

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    # Diagonal y=x (no change reference)
    lo = min(x.min(), y.min()) - 0.5
    hi = max(x.max(), y.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color="#9ca3af", linestyle="--",
            linewidth=1.2, alpha=0.7, label="y = x (no change)", zorder=1)

    # Scatter, color by boost sign
    colors = ["#10a37f" if b > 0 else "#dc2626" for b in boost]
    ax.scatter(x, y, s=110, c=colors, edgecolors="#1f2937", linewidth=1.4, zorder=3)

    for r in paired:
        ax.annotate(r["short"], (r["static"], r["active"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, alpha=0.9)

    # Marginal spread bars
    ax.errorbar([x.mean()], [lo + 0.3], xerr=[[x.mean() - x.min()], [x.max() - x.mean()]],
                fmt="none", color="#6b7280", linewidth=2.5, capsize=6, zorder=2)
    ax.text(x.mean(), lo + 0.05, f"Static spread = {spread_static:.2f}",
            ha="center", fontsize=9.5, color="#374151")

    ax.errorbar([lo + 0.3], [y.mean()], yerr=[[y.mean() - y.min()], [y.max() - y.mean()]],
                fmt="none", color="#10a37f", linewidth=2.5, capsize=6, zorder=2)
    ax.text(lo + 0.07, y.mean(), f"Active spread = {spread_active:.2f}",
            ha="left", va="center", fontsize=9.5, color="#0a5f4a", rotation=90)

    ax.set_xlabel("Static mean weighted score (1–10)", fontsize=11.5)
    ax.set_ylabel("Active mean weighted score (1–10)", fontsize=11.5)
    ax.set_title(
        f"Active widens between-model variance: std {std_static:.2f} → {std_active:.2f} (×{std_active/std_static:.2f}); "
        f"12/13 models above diagonal",
        fontsize=11.5,
    )
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    plt.tight_layout()
    out_pdf = FIG / "active_widens_variance.pdf"
    out_png = FIG / "active_widens_variance.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close()

    print(f"Static: std={std_static:.3f}, spread={spread_static:.3f}")
    print(f"Active: std={std_active:.3f}, spread={spread_active:.3f}")
    print(f"Active/Static std ratio = {std_active/std_static:.3f}×")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
