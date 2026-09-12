"""Per-dimension scatter plots over release date.

For each of the 5 critic dimensions (Originality, Feasibility, Clarity,
Impact, Specificity) plus the weighted total, draw a scatter where:
  - x = continuous decimal release date (OR `created` listing date)
  - y = mean dimension score across papers (Track B / Static Mode)
  - colors = model family (OpenAI / Anthropic / Qwen / Gemma / Llama / Mistral)

Aggregation, per (idea_model, dimension):
  1. For each (paper, idea_index): trimmed-mean(critic raw 0-10) per dim.
     Trim rule = drop highest critic when n_critics >= 3, else plain mean.
  2. best_idea_index for that (paper, model, track='B') = the one already
     chosen by model_scores (using weighted total). All 5 dims read from
     that SAME idea — so dimensions are commensurable.
  3. Mean across all papers for that idea_model.

Output: reports/per_dim_scatter.{pdf,png} + per_dim_scatter.json
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import sqlite3
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Reuse maps from the box/scatter plot script to avoid drift.
from reports._make_cross_year_plot import (  # noqa: E402
    YEAR_GROUPS, RELEASE_DATES, FAMILY_STYLE, family_for,
)

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
WEIGHTS = {
    "originality": 2.0,
    "feasibility": 1.0,
    "clarity": 0.5,
    "impact": 1.5,
    "specificity": 0.5,
}


def _trimmed_mean(vals):
    """Match analysis/compute_scores.py: drop highest when n>=3, else plain mean."""
    if not vals:
        return None
    if len(vals) <= 2:
        return sum(vals) / len(vals)
    return sum(sorted(vals)[:-1]) / (len(vals) - 1)


def aggregate_per_dim(db_results: Path) -> dict:
    """Return {idea_model: {dim: mean_across_papers}} for Track B."""
    conn = sqlite3.connect(db_results)
    conn.row_factory = sqlite3.Row

    # 1) best_idea_index map from model_scores (Track B)
    best_idx = {
        (r["paper_id"], r["idea_model"]): r["best_idea_index"]
        for r in conn.execute(
            "SELECT paper_id, idea_model, best_idea_index "
            "FROM model_scores WHERE track='B' AND best_idea_index IS NOT NULL"
        )
    }

    # 2) Load all Track B scored results, group by (paper, model, idx)
    groups = defaultdict(list)  # (pid, model, idx) -> [scores_json dicts]
    for r in conn.execute(
        "SELECT paper_id, idea_model, idea_index, scores_json "
        "FROM results WHERE track='B' AND scores_json IS NOT NULL "
        "AND critic_model != ''"
    ):
        try:
            s = json.loads(r["scores_json"])
        except Exception:
            continue
        if not all(d in s for d in DIMS):
            continue
        groups[(r["paper_id"], r["idea_model"], r["idea_index"])].append(s)
    conn.close()

    # 3) Per (paper, model): take only the best idea_index, trimmed-mean per dim
    per_paper_per_model = defaultdict(dict)  # model -> {pid: {dim: val}}
    for (pid, model, idx), critic_scores in groups.items():
        if best_idx.get((pid, model)) != idx:
            continue
        for d in DIMS:
            vals = [s[d] for s in critic_scores]
            tm = _trimmed_mean(vals)
            if tm is not None:
                per_paper_per_model[model].setdefault(pid, {})[d] = tm

    # 4) Mean across papers
    out = {}
    for model, paper_dict in per_paper_per_model.items():
        dim_means = {}
        for d in DIMS:
            vals = [p[d] for p in paper_dict.values() if d in p]
            if vals:
                dim_means[d] = sum(vals) / len(vals)
        # Weighted total computed from per-paper dim values (same rule as
        # compute_scores.py's "raw" weighted mean), then averaged over papers.
        paper_weighted = []
        for p in paper_dict.values():
            num = sum(p[d] * WEIGHTS[d] for d in DIMS if d in p)
            den = sum(WEIGHTS[d] for d in DIMS if d in p)
            if den > 0:
                paper_weighted.append(num / den)
        if paper_weighted:
            dim_means["weighted"] = sum(paper_weighted) / len(paper_weighted)
        dim_means["n_papers"] = len(paper_dict)
        out[model] = dim_means
    return out


DIM_TITLES = {
    "originality": "Originality (rubric weight ×2)",
    "feasibility": "Feasibility (rubric weight ×1)",
    "clarity":     "Clarity (rubric weight ×0.5)",
    "impact":      "Impact (rubric weight ×1.5)",
    "specificity": "Specificity (rubric weight ×0.5)",
    "weighted":    "Weighted total (out of 10)",
}


def make_scatter_for_dim(per_model: dict, dim: str, out_pdf: Path, out_png: Path):
    """One full-size scatter for a single dimension. Matches the look of
    reports/cross_year_scatter.png exactly (size, labels, envelope, legend)."""
    label_for = {mid: lab for ylist in YEAR_GROUPS.values() for mid, lab in ylist}

    points = []  # (x, y, mid, lab, fam)
    for mid, dims in per_model.items():
        if mid not in RELEASE_DATES or mid not in label_for:
            continue
        if dim not in dims:
            continue
        points.append((RELEASE_DATES[mid], dims[dim], mid, label_for[mid], family_for(mid)))
    points.sort(key=lambda p: p[0])

    fig, ax = plt.subplots(figsize=(12, 6.8))

    drawn = set()
    for x, y, mid, lab, fam in points:
        disp, color = FAMILY_STYLE.get(fam, (fam, "#444"))
        ax.scatter(
            x, y, s=70, color=color, edgecolor="white", linewidth=1.0,
            zorder=3, label=disp if fam not in drawn else None,
        )
        drawn.add(fam)
        ax.annotate(
            lab, (x, y), xytext=(5, 5), textcoords="offset points",
            fontsize=7.5, color="#222", zorder=4,
        )

    # Rolling min/max envelope (±0.4 yr)
    if points:
        xs = [p[0] for p in points]
        x_min, x_max = min(xs), max(xs)
        span = x_max - x_min
        sample_xs = [x_min + span * i / 80 for i in range(81)]
        roll_top, roll_bot = [], []
        for sx in sample_xs:
            vals = [p[1] for p in points if abs(p[0] - sx) <= 0.4]
            roll_top.append(max(vals) if vals else None)
            roll_bot.append(min(vals) if vals else None)

        def _draw(line_vals, color, label):
            seg_x, seg_y = [], []
            for sx, v in zip(sample_xs, line_vals):
                if v is None:
                    if seg_x:
                        ax.plot(seg_x, seg_y, color=color, linewidth=1.4,
                                alpha=0.5, linestyle="--",
                                label=label if label else None, zorder=1)
                        label = None
                        seg_x, seg_y = [], []
                else:
                    seg_x.append(sx); seg_y.append(v)
            if seg_x:
                ax.plot(seg_x, seg_y, color=color, linewidth=1.4, alpha=0.5,
                        linestyle="--", label=label, zorder=1)

        _draw(roll_top, "#16a34a", "Rolling max (±0.4 yr)")
        _draw(roll_bot, "#dc2626", "Rolling min (±0.4 yr)")

    # Year guide lines
    for yr in range(2023, 2027):
        ax.axvline(x=yr, color="gray", linestyle=":", linewidth=0.6, alpha=0.35)

    # Y-range: tight, but min 2.5 span and pad 25% so labels don't clip
    if points:
        ys = [p[1] for p in points]
        lo, hi = min(ys), max(ys)
        c = (lo + hi) / 2
        half = max((hi - lo) / 2 * 1.25, 1.25)
        ax.set_ylim(c - half, c + half)

    ax.set_xlabel("Model release date (continuous, OR listing date)", fontsize=12)
    ax.set_ylabel(f"{dim.capitalize()} score (mean across papers)", fontsize=12)
    ax.set_xlim(2023.0, 2026.6)
    ax.set_title(
        f"{DIM_TITLES[dim]} over time — Static Mode\n"
        f"Each dot = one of the {len(points)} cross-year idea models with per-dim data",
        fontsize=12, pad=14,
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


def make_plot(per_model: dict, out_pdf: Path, out_png: Path):
    """Backward-compat alias — emits the 6-panel combined view AND the per-dim
    individual plots beside it."""
    # Combined 2x3 (kept as a quick overview)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    label_for = {mid: lab for ylist in YEAR_GROUPS.values() for mid, lab in ylist}

    for ax, dim in zip(axes, list(DIM_TITLES.keys())):
        points = []
        for mid, dims in per_model.items():
            if mid not in RELEASE_DATES or mid not in label_for or dim not in dims:
                continue
            points.append((RELEASE_DATES[mid], dims[dim], family_for(mid)))
        points.sort(key=lambda p: p[0])
        drawn = set()
        for x, y, fam in points:
            disp, color = FAMILY_STYLE.get(fam, (fam, "#444"))
            ax.scatter(x, y, s=55, color=color, edgecolor="white", linewidth=0.8,
                       zorder=3, label=disp if (dim == "weighted" and fam not in drawn) else None)
            drawn.add(fam)
        for yr in range(2023, 2027):
            ax.axvline(x=yr, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)
        if points:
            ys = [p[1] for p in points]
            c = (min(ys) + max(ys)) / 2
            half = max((max(ys) - min(ys)) / 2 * 1.25, 1.0)
            ax.set_ylim(c - half, c + half)
        ax.set_title(DIM_TITLES[dim], fontsize=11, pad=6)
        ax.set_xlim(2023.0, 2026.6)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[3:]:
        ax.set_xlabel("Release date (decimal year)", fontsize=10)
    for ax in axes[::3]:
        ax.set_ylabel("Score", fontsize=10)
    axes[-1].legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Per-dimension Static-Mode performance over time (overview)",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


def write_summary(per_model: dict, out_path: Path):
    """Dump per-model per-dim numbers plus year-bucket min/max/spread."""
    label_for = {mid: lab for ylist in YEAR_GROUPS.values() for mid, lab in ylist}

    # Year buckets per dim
    year_summary = {}
    for year, model_list in YEAR_GROUPS.items():
        year_summary[year] = {}
        for dim in DIMS + ["weighted"]:
            vals = []
            for mid, _lab in model_list:
                v = per_model.get(mid, {}).get(dim)
                if v is not None:
                    vals.append(v)
            if vals:
                year_summary[year][dim] = {
                    "n": len(vals),
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                    "mean": round(sum(vals) / len(vals), 4),
                    "spread": round(max(vals) - min(vals), 4),
                }

    summary = {
        "per_model": {
            mid: {
                "label": label_for.get(mid, mid),
                "release_date_decimal": RELEASE_DATES.get(mid),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in dims.items()},
            }
            for mid, dims in per_model.items()
            if mid in label_for
        },
        "by_year": year_summary,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


def main():
    per_model = aggregate_per_dim(ROOT / "data" / "results.db")
    print(f"Aggregated {len(per_model)} models")
    out_dir = ROOT / "reports"

    # One full-size scatter per dimension (matches cross_year_scatter.png look)
    for dim in DIM_TITLES.keys():
        make_scatter_for_dim(
            per_model, dim,
            out_dir / f"per_dim_{dim}.pdf",
            out_dir / f"per_dim_{dim}.png",
        )

    # Keep the 2x3 overview too (handy for a single comparison slide)
    make_plot(per_model,
              out_dir / "per_dim_scatter.pdf",
              out_dir / "per_dim_scatter.png")

    write_summary(per_model, out_dir / "per_dim_scatter.json")


if __name__ == "__main__":
    main()
