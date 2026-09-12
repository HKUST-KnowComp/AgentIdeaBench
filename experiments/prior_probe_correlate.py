"""Prior-coverage × Static-score correlation analysis.

Reads experiments/prior_probe.py output (table `prior_probe`) and Static
Mode model_scores (Track B), and tests:

    Is each model's mean Static score predictable from its mean prior
    coverage of the 25 test-set abstracts?

If correlation r is high (e.g. > 0.7), the Static benchmark is largely
measuring training-set leakage, not hypothesis-generation skill. If
correlation is near zero, that's evidence the bench measures something
beyond prior knowledge.

Outputs:
  - reports/prior_probe_correlate.png/pdf  (scatter + best-fit line)
  - reports/prior_probe_correlate.json     (per-model numbers + Pearson/Spearman)

Read-only on results.db; safe to re-run.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from reports._make_cross_year_plot import (  # noqa: E402
    YEAR_GROUPS, RELEASE_DATES, FAMILY_STYLE, family_for,
)

METRIC = "rouge_l_f1"  # primary coverage metric


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _spearman(xs, ys):
    def _rank(vs):
        sv = sorted((v, i) for i, v in enumerate(vs))
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1][0] == sv[i][0]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sv[k][1]] = avg
            i = j + 1
        return ranks
    return _pearson(_rank(xs), _rank(ys))


def load_data(db_results: Path):
    conn = sqlite3.connect(db_results)
    conn.row_factory = sqlite3.Row

    # Mean coverage per model
    cov_rows = conn.execute(f"""
        SELECT idea_model,
               AVG({METRIC}) AS mean_cov,
               AVG(self_reported_unknown) AS unknown_rate,
               AVG(length_ratio) AS mean_len_ratio,
               COUNT(*) AS n
        FROM prior_probe
        WHERE error IS NULL AND recitation_text IS NOT NULL
        GROUP BY idea_model
    """).fetchall()
    coverage = {r["idea_model"]: dict(r) for r in cov_rows}

    # Static score per model
    sc_rows = conn.execute("""
        SELECT idea_model, AVG(mean_absolute_score) AS static_score
        FROM model_scores WHERE track='B' GROUP BY idea_model
    """).fetchall()
    static = {r["idea_model"]: r["static_score"] for r in sc_rows
              if r["static_score"] is not None}
    conn.close()
    return coverage, static


def main():
    coverage, static = load_data(ROOT / "data" / "results.db")
    label_for = {mid: lab for ylist in YEAR_GROUPS.values() for mid, lab in ylist}

    rows = []
    for mid, cov in coverage.items():
        if mid not in static or mid not in label_for:
            continue
        rows.append({
            "model_id": mid,
            "label": label_for[mid],
            "release_date": RELEASE_DATES.get(mid),
            "family": family_for(mid),
            "mean_coverage": cov["mean_cov"],
            "unknown_rate": cov["unknown_rate"],
            "length_ratio": cov["mean_len_ratio"],
            "static_score": static[mid],
            "n_papers": cov["n"],
        })
    rows.sort(key=lambda r: r["mean_coverage"])

    if len(rows) < 3:
        print(f"Only {len(rows)} models with both prior_probe and Static — need >=3 for correlation. Aborting plot.")
        return

    xs = [r["mean_coverage"] for r in rows]
    ys = [r["static_score"] for r in rows]
    pearson = _pearson(xs, ys)
    spearman = _spearman(xs, ys)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(11, 6.8))
    drawn = set()
    for r in rows:
        disp, color = FAMILY_STYLE.get(r["family"], (r["family"], "#444"))
        ax.scatter(
            r["mean_coverage"], r["static_score"],
            s=80, color=color, edgecolor="white", linewidth=1.0, zorder=3,
            label=disp if r["family"] not in drawn else None,
        )
        drawn.add(r["family"])
        ax.annotate(
            r["label"], (r["mean_coverage"], r["static_score"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7.5, color="#222", zorder=4,
        )

    # Best-fit line (least-squares)
    if pearson is not None and len(rows) >= 3:
        n = len(rows)
        mx, my = sum(xs) / n, sum(ys) / n
        var_x = sum((x - mx) ** 2 for x in xs)
        if var_x > 0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var_x
            intercept = my - slope * mx
            x_line = [min(xs), max(xs)]
            y_line = [slope * x + intercept for x in x_line]
            ax.plot(x_line, y_line, color="gray", linestyle="--",
                    linewidth=1.2, alpha=0.7,
                    label=f"OLS fit (slope={slope:.2f})")

    ax.set_xlabel(f"Mean prior coverage ({METRIC} on {rows[0]['n_papers']} test abstracts)",
                  fontsize=12)
    ax.set_ylabel("Static-Mode weighted score (out of 10)", fontsize=12)
    interpretation = ""
    if pearson is not None:
        if abs(pearson) > 0.7:
            interpretation = " — strong (bench may be measuring prior)"
        elif abs(pearson) > 0.4:
            interpretation = " — moderate"
        else:
            interpretation = " — weak (bench independent of prior)"
    ax.set_title(
        f"Prior coverage vs Static score across {len(rows)} cross-year models\n"
        f"Pearson r = {pearson:.3f}  |  Spearman ρ = {spearman:.3f}{interpretation}",
        fontsize=12, pad=14,
    )
    ax.grid(axis="both", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)

    plt.tight_layout()
    out_pdf = ROOT / "reports" / "prior_probe_correlate.pdf"
    out_png = ROOT / "reports" / "prior_probe_correlate.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    # Console summary table
    print(f"\n{'model':45s} {'cov':>6s} {'unk%':>6s} {'static':>7s}")
    for r in rows:
        print(f"{r['label']:45s} {r['mean_coverage']:>6.3f} "
              f"{r['unknown_rate']*100:>5.1f}% {r['static_score']:>7.3f}")

    print(f"\nPearson r  = {pearson:.4f}")
    print(f"Spearman ρ = {spearman:.4f}")

    summary = {
        "n_models": len(rows),
        "metric": METRIC,
        "pearson": pearson,
        "spearman": spearman,
        "per_model": rows,
    }
    out_json = ROOT / "reports" / "prior_probe_correlate.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
