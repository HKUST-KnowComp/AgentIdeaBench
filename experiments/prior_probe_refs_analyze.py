"""Analyze refs corpus prior_probe results.

For each cross-year idea_model, compute:
  - "claimed_known": fraction of the 96 refs the model did NOT mark
    UNKNOWN_PAPER (i.e. model claims to recognize)
  - "accuracy_when_claimed": mean ROUGE-L F1 across non-unknown calls
  - "composite": claimed_known × accuracy_when_claimed (overall prior signal)
  - "mean_rouge": mean ROUGE-L F1 across ALL 96 calls (no gating)
  - "ref_year_split": same metrics restricted to refs with ref_year <=
    estimated train-cutoff-year vs > cutoff

Plus correlation of (composite) with Static-Mode score, by family color.

Outputs:
  reports/prior_probe_refs.png/pdf  (scatter: composite vs Static)
  reports/prior_probe_refs_breadth.png/pdf  (release_date vs composite)
  reports/prior_probe_refs.json
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from reports._make_cross_year_plot import (  # noqa: E402
    YEAR_GROUPS, RELEASE_DATES, FAMILY_STYLE, family_for,
)


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


def load(db_results: Path):
    conn = sqlite3.connect(db_results)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT idea_model, paper_id, rouge_l_f1, self_reported_unknown,
               error, ref_year, ref_domain
        FROM prior_probe_refs
    """).fetchall()
    static = {r["idea_model"]: r["static_score"] for r in conn.execute(
        "SELECT idea_model, AVG(mean_absolute_score) AS static_score "
        "FROM model_scores WHERE track='B' GROUP BY idea_model"
    ) if r["static_score"] is not None}
    conn.close()
    return rows, static


def aggregate(rows, static):
    """Per-model aggregates."""
    label_for = {mid: lab for ylist in YEAR_GROUPS.values() for mid, lab in ylist}
    by_model: dict = {}
    for r in rows:
        m = r["idea_model"]
        if r["error"]:
            continue
        d = by_model.setdefault(m, {"all": [], "unknown": 0, "n": 0,
                                     "by_year": {}})
        d["n"] += 1
        if r["self_reported_unknown"]:
            d["unknown"] += 1
        else:
            d["all"].append(r["rouge_l_f1"] or 0.0)
        d["by_year"].setdefault(r["ref_year"] or 0, []).append({
            "rouge": r["rouge_l_f1"] or 0.0,
            "unknown": bool(r["self_reported_unknown"]),
        })

    out = []
    for m, d in by_model.items():
        if m not in label_for or m not in RELEASE_DATES or m not in static:
            continue
        n = d["n"]
        if n == 0:
            continue
        claimed_known = 1 - d["unknown"] / n
        acc = sum(d["all"]) / len(d["all"]) if d["all"] else 0.0
        composite = claimed_known * acc
        mean_rouge_all = sum((v["rouge"] for vs in d["by_year"].values() for v in vs)) / n
        # Pre/post-2023 split (rough proxy: training cutoff ~ release_date - 0.5y)
        # We do not have authoritative cutoffs; instead, just expose by_year buckets.
        buckets = {
            "le_2018": {"n": 0, "unknown": 0, "rouge_sum": 0.0, "rouge_n": 0},
            "2019_2021": {"n": 0, "unknown": 0, "rouge_sum": 0.0, "rouge_n": 0},
            "2022_2023": {"n": 0, "unknown": 0, "rouge_sum": 0.0, "rouge_n": 0},
            "2024_plus": {"n": 0, "unknown": 0, "rouge_sum": 0.0, "rouge_n": 0},
        }
        for y, vs in d["by_year"].items():
            if y <= 2018: key = "le_2018"
            elif y <= 2021: key = "2019_2021"
            elif y <= 2023: key = "2022_2023"
            else: key = "2024_plus"
            for v in vs:
                buckets[key]["n"] += 1
                if v["unknown"]:
                    buckets[key]["unknown"] += 1
                else:
                    buckets[key]["rouge_sum"] += v["rouge"]
                    buckets[key]["rouge_n"] += 1
        bucket_summary = {}
        for k, b in buckets.items():
            if b["n"] > 0:
                bucket_summary[k] = {
                    "n": b["n"],
                    "claimed_known": round(1 - b["unknown"] / b["n"], 4),
                    "acc_when_known": round(b["rouge_sum"] / b["rouge_n"], 4)
                                        if b["rouge_n"] else 0.0,
                }
        out.append({
            "model_id": m,
            "label": label_for[m],
            "family": family_for(m),
            "release_date": RELEASE_DATES[m],
            "n": n,
            "claimed_known": round(claimed_known, 4),
            "acc_when_claimed": round(acc, 4),
            "composite": round(composite, 4),
            "mean_rouge_all": round(mean_rouge_all, 4),
            "static_score": static[m],
            "by_year_bucket": bucket_summary,
        })
    out.sort(key=lambda r: r["composite"])
    return out


def make_scatter_composite_vs_static(rows, out_pdf, out_png):
    fig, ax = plt.subplots(figsize=(11, 6.8))
    drawn = set()
    xs = [r["composite"] for r in rows]
    ys = [r["static_score"] for r in rows]
    for r in rows:
        disp, color = FAMILY_STYLE.get(r["family"], (r["family"], "#444"))
        ax.scatter(r["composite"], r["static_score"], s=80, color=color,
                   edgecolor="white", linewidth=1.0, zorder=3,
                   label=disp if r["family"] not in drawn else None)
        drawn.add(r["family"])
        ax.annotate(r["label"], (r["composite"], r["static_score"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7.5, color="#222", zorder=4)
    pearson = _pearson(xs, ys) or 0
    spearman = _spearman(xs, ys) or 0

    if len(rows) >= 3:
        n = len(rows)
        mx, my = sum(xs) / n, sum(ys) / n
        var_x = sum((x - mx) ** 2 for x in xs)
        if var_x > 0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var_x
            intercept = my - slope * mx
            xl = [min(xs), max(xs)]
            yl = [slope * x + intercept for x in xl]
            ax.plot(xl, yl, color="gray", linestyle="--", linewidth=1.2,
                    alpha=0.7, label=f"OLS (slope={slope:.2f})")

    interp = ""
    if abs(pearson) > 0.7: interp = " — strong"
    elif abs(pearson) > 0.4: interp = " — moderate"
    else: interp = " — weak"

    ax.set_xlabel("Composite prior coverage on 96 refs  "
                  "(claimed_known × ROUGE-L when claimed)", fontsize=11)
    ax.set_ylabel("Static-Mode weighted score (out of 10)", fontsize=12)
    ax.set_title(
        f"Refs prior coverage vs Static score ({len(rows)} models)\n"
        f"Pearson r = {pearson:.3f} | Spearman ρ = {spearman:.3f}{interp}",
        fontsize=12, pad=14,
    )
    ax.grid(axis="both", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight"); plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    return pearson, spearman


def make_scatter_breadth_over_time(rows, out_pdf, out_png):
    """Release date (x) vs composite prior coverage (y). Tests user's
    hypothesis: do newer models systematically know more of the domain refs?"""
    fig, ax = plt.subplots(figsize=(12, 6.8))
    drawn = set()
    for r in rows:
        disp, color = FAMILY_STYLE.get(r["family"], (r["family"], "#444"))
        ax.scatter(r["release_date"], r["composite"], s=80, color=color,
                   edgecolor="white", linewidth=1.0, zorder=3,
                   label=disp if r["family"] not in drawn else None)
        drawn.add(r["family"])
        ax.annotate(r["label"], (r["release_date"], r["composite"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7.5, color="#222", zorder=4)
    # Year guide lines
    for yr in range(2023, 2027):
        ax.axvline(x=yr, color="gray", linestyle=":", linewidth=0.6, alpha=0.35)
    xs = [r["release_date"] for r in rows]
    ys = [r["composite"] for r in rows]
    pearson = _pearson(xs, ys) or 0
    spearman = _spearman(xs, ys) or 0
    ax.set_xlabel("Model release date (decimal year)", fontsize=12)
    ax.set_ylabel("Composite prior coverage on 96 refs", fontsize=12)
    ax.set_xlim(2023.0, 2026.6)
    ax.set_title(
        f"Does training breadth grow with release date? ({len(rows)} models)\n"
        f"Pearson r = {pearson:.3f} | Spearman ρ = {spearman:.3f}",
        fontsize=12, pad=14,
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight"); plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    return pearson, spearman


def main():
    rows_raw, static = load(ROOT / "data" / "results.db")
    rows = aggregate(rows_raw, static)
    print(f"\n{'model':45s} {'n':>3s} {'claim%':>7s} {'acc':>6s} {'comp':>6s} {'static':>7s}")
    for r in rows:
        print(f"{r['label']:45s} {r['n']:>3d} "
              f"{r['claimed_known']*100:>6.1f}% {r['acc_when_claimed']:>6.3f} "
              f"{r['composite']:>6.3f} {r['static_score']:>7.3f}")

    if len(rows) >= 3:
        p1, s1 = make_scatter_composite_vs_static(
            rows,
            ROOT / "reports" / "prior_probe_refs.pdf",
            ROOT / "reports" / "prior_probe_refs.png",
        )
        p2, s2 = make_scatter_breadth_over_time(
            rows,
            ROOT / "reports" / "prior_probe_refs_breadth.pdf",
            ROOT / "reports" / "prior_probe_refs_breadth.png",
        )
        print(f"\nComposite vs Static : Pearson {p1:.4f}, Spearman {s1:.4f}")
        print(f"Release date vs Composite: Pearson {p2:.4f}, Spearman {s2:.4f}")
        out_json = ROOT / "reports" / "prior_probe_refs.json"
        out_json.write_text(json.dumps({
            "n_models": len(rows),
            "composite_vs_static": {"pearson": p1, "spearman": s1},
            "release_vs_composite": {"pearson": p2, "spearman": s2},
            "per_model": rows,
        }, indent=2))
        print(f"Saved: {out_json}")
    else:
        print(f"Only {len(rows)} models — need >=3 for correlation/plot")


if __name__ == "__main__":
    main()
