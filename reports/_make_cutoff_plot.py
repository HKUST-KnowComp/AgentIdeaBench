"""Cross-year scatter + baseline correction by knowledge cutoff date.

Same data as _make_cross_year_plot.py, but x-axis = each model's documented
knowledge-cutoff date (instead of release date). Also computes:

  1. Baseline cutoff   = median of all known cutoffs (single number)
  2. Regression slope  = β from fitting (score ~ cutoff_year) across N models
  3. Corrected score   = raw_score − β × (cutoff − baseline)
     → "what would this model score if it had been trained with information up to
       the baseline date".

Outputs:
  reports/cross_year_cutoff_scatter.{png, pdf}
  reports/cross_year_cutoff_corrected.{md, json}

Models with cutoff = None (openai/gpt-5, openai/gpt-5.5) are EXCLUDED from
the plot and from the regression baseline.
"""
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

try:
    from adjustText import adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False

from reports._make_cross_year_plot import (
    YEAR_GROUPS, KNOWLEDGE_CUTOFFS, FAMILY_STYLE, family_for,
    to_decimal_year,
)


def _load_static_scores() -> dict:
    """Return {model_id: weighted_static_score} from model_scores aggregate."""
    db = ROOT / "data" / "results.db"
    conn = sqlite3.connect(str(db))
    out = {}
    cur = conn.execute("""
        SELECT idea_model, AVG(mean_absolute_score)
        FROM model_scores
        WHERE track='B' AND mean_absolute_score IS NOT NULL
        GROUP BY idea_model
    """)
    for mid, score in cur.fetchall():
        out[mid] = score
    conn.close()
    return out


def _load_active_scores() -> dict:
    """Return {model_id: weighted_active_score} from model_scores track='C'."""
    db = ROOT / "data" / "results.db"
    conn = sqlite3.connect(str(db))
    out = {}
    cur = conn.execute("""
        SELECT idea_model, AVG(mean_absolute_score)
        FROM model_scores
        WHERE track='C' AND mean_absolute_score IS NOT NULL
        GROUP BY idea_model
    """)
    for mid, score in cur.fetchall():
        out[mid] = score
    conn.close()
    return out


def _linear_fit(xs: list, ys: list) -> tuple:
    """Plain OLS slope + intercept (no numpy). Returns (slope, intercept, r2)."""
    n = len(xs)
    if n < 2:
        return (0.0, statistics.mean(ys) if ys else 0.0, 0.0)
    mx = statistics.mean(xs); my = statistics.mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((ys[i] - (slope * xs[i] + intercept)) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return (slope, intercept, r2)


def main():
    # Build per-model (cutoff, score) table — skip models with no cutoff or no score
    scores = _load_static_scores()
    active_scores = _load_active_scores()
    label_for = {mid: lab for year_list in YEAR_GROUPS.values() for mid, lab in year_list}

    table = []
    unknown_cutoff = []
    for mid, label in label_for.items():
        score = scores.get(mid)
        cutoff = KNOWLEDGE_CUTOFFS.get(mid)
        if score is None:
            continue
        if cutoff is None:
            unknown_cutoff.append((mid, label, score))
            continue
        active = active_scores.get(mid)
        table.append({
            "model": mid,
            "label": label,
            "cutoff": cutoff,
            "score": score,
            "active": active,
            "boost": (active - score) if active is not None else None,
        })

    table.sort(key=lambda r: r["cutoff"])

    # Baseline = median cutoff
    cutoffs = [r["cutoff"] for r in table]
    raw_scores = [r["score"] for r in table]
    baseline_cutoff = statistics.median(cutoffs)

    # OLS regression score ~ cutoff
    slope, intercept, r2 = _linear_fit(cutoffs, raw_scores)

    for r in table:
        r["corrected"] = r["score"] - slope * (r["cutoff"] - baseline_cutoff)

    # ── Bin-baseline correction (non-parametric alternative) ───────────────
    # Each model's bin = floor(cutoff). Baseline_bin = mean score of all models
    # in that bin. corrected_bin = raw_score - baseline_bin.
    # Bins with n<3 are flagged as low-confidence in the report.
    bin_membership = {}
    for r in table:
        b = int(r["cutoff"])  # 1-yr bin start
        bin_membership.setdefault(b, []).append(r["score"])

    bin_stats = {}
    for b, vals in bin_membership.items():
        bin_stats[b] = {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "median": statistics.median(vals),
        }

    for r in table:
        b = int(r["cutoff"])
        bs = bin_stats[b]
        r["bin"] = f"{b}–{b+1}"
        r["bin_n"] = bs["n"]
        r["bin_mean"] = bs["mean"]
        r["bin_median"] = bs["median"]
        r["corrected_bin_mean"] = r["score"] - bs["mean"]
        r["corrected_bin_median"] = r["score"] - bs["median"]

    # ── Scatter plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))

    # Compute regression-line + 95% CI band first (so it sits behind dots)
    import math
    x_min, x_max = min(cutoffs), max(cutoffs)
    # Sample many x points for smooth line
    xs_line = [x_min + (x_max - x_min) * i / 60 for i in range(61)]
    ys_line = [slope * x + intercept for x in xs_line]

    # 95% CI for the regression line at each x (standard formula)
    n = len(cutoffs)
    mx = sum(cutoffs) / n
    sxx = sum((x - mx) ** 2 for x in cutoffs)
    # residual standard error
    ss_res = sum((raw_scores[i] - (slope * cutoffs[i] + intercept)) ** 2
                 for i in range(n))
    se_resid = math.sqrt(ss_res / max(n - 2, 1))
    t_crit = 2.06  # df=25, 95% CI two-sided
    ci_lo = []
    ci_hi = []
    for x in xs_line:
        se_y = se_resid * math.sqrt(1.0 / n + (x - mx) ** 2 / sxx)
        ci_lo.append(slope * x + intercept - t_crit * se_y)
        ci_hi.append(slope * x + intercept + t_crit * se_y)
    ax.fill_between(xs_line, ci_lo, ci_hi, color="#10a37f", alpha=0.12,
                    zorder=1, label="OLS 95% CI")

    # Per-half-year bin means (light grey markers) — gives an "average score
    # at this cutoff window" visual cue independent of the line fit.
    bin_size = 0.5
    bin_edges_x = []
    bin_means_y = []
    bin_lows, bin_highs = [], []
    for bin_start in [2021.5 + i * bin_size for i in range(int((2026.5 - 2021.5) / bin_size) + 1)]:
        in_bin = [r["score"] for r in table if bin_start <= r["cutoff"] < bin_start + bin_size]
        if len(in_bin) >= 2:
            mn = sum(in_bin) / len(in_bin)
            bin_edges_x.append(bin_start + bin_size / 2)
            bin_means_y.append(mn)
            bin_lows.append(min(in_bin))
            bin_highs.append(max(in_bin))
    ax.scatter(bin_edges_x, bin_means_y, marker="_", color="#444", s=600,
               linewidth=2.5, zorder=2.5,
               label=f"per-{bin_size}yr-bin mean (n≥2)")

    # Bolder solid regression line
    ax.plot(xs_line, ys_line, color="#10a37f", linewidth=2.4, alpha=0.9,
            zorder=2,
            label=f"OLS β={slope:+.2f}/yr · R²={r2:.2f}")

    # Baseline vertical line
    ax.axvline(baseline_cutoff, color="#666", linestyle="--", linewidth=1.2,
               alpha=0.6,
               label=f"baseline (median cutoff = {baseline_cutoff:.2f})")

    # Lighter scatter (so trend is visible)
    drawn_fams = set()
    texts = []
    for r in table:
        fam = family_for(r["model"])
        disp, color = FAMILY_STYLE.get(fam, ("?", "#888"))
        ax.scatter(r["cutoff"], r["score"], s=85, color=color,
                   edgecolor="white", linewidth=1.0, zorder=3, alpha=0.85,
                   label=disp if fam not in drawn_fams else None)
        drawn_fams.add(fam)
        texts.append(ax.text(r["cutoff"], r["score"], r["label"],
                             fontsize=7.5, color="#333", zorder=4))

    # Active overlay: triangle marker + dashed line for the boost (C-B)
    active_drawn = False
    for r in table:
        if r["active"] is None:
            continue
        # Dashed line from static dot up to active triangle
        ax.plot([r["cutoff"], r["cutoff"]], [r["score"], r["active"]],
                color="#444", linestyle=":", linewidth=1.2, alpha=0.55,
                zorder=2.8)
        # Triangle marker at active score
        ax.scatter(r["cutoff"], r["active"], marker="^", s=110,
                   color="none", edgecolor="#111", linewidth=1.5,
                   zorder=4.5,
                   label="Active Mode score (▲)" if not active_drawn else None)
        # 加成 annotation
        ax.annotate(f"+{r['boost']:.2f}",
                    xy=(r["cutoff"], (r["score"] + r["active"]) / 2),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color="#444", zorder=5, alpha=0.85)
        active_drawn = True

    ax.set_xlabel("Model knowledge cutoff date", fontsize=12)
    ax.set_ylabel("Score (weighted, out of 10)  ●=Static  ▲=Active", fontsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(round(x))}"))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="x", which="minor", length=3)
    ax.set_xlim(min(cutoffs) - 0.5, max(cutoffs) + 0.5)
    # y-range must accommodate active scores too (always higher than static)
    score_range = [r["score"] for r in table] + [
        r["active"] for r in table if r["active"] is not None]
    ax.set_ylim(min(score_range) - 0.5, max(score_range) + 0.5)
    n_active = sum(1 for r in table if r["active"] is not None)
    ax.set_title(
        f"Score vs. knowledge cutoff date\n"
        f"{len(table)} models · ●=Static (all) · ▲=Active ({n_active} models) · "
        f"baseline = median cutoff",
        fontsize=12, pad=14,
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)

    if _HAS_ADJUST_TEXT and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.5, alpha=0.6),
            expand=(1.2, 1.4),
            force_text=(0.0, 0.6),
            force_pull=(1.0, 0.05),
            only_move={"text": "y", "static": "y", "explode": "y"},
        )

    out_png = ROOT / "reports" / "cross_year_cutoff_scatter.png"
    out_pdf = ROOT / "reports" / "cross_year_cutoff_scatter.pdf"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    # ── Markdown report ───────────────────────────────────────────────────
    md = []
    md.append("# Cross-year static leaderboard, corrected for knowledge cutoff")
    md.append("")
    md.append(f"- Baseline (median cutoff) = **{baseline_cutoff:.3f}**")
    md.append(f"- OLS regression: score ~ cutoff_year, β = {slope:+.3f}/yr, R² = {r2:.3f}")
    md.append(f"- Corrected score = raw_score − β × (cutoff − baseline_cutoff)")
    md.append("")
    md.append("## A. OLS regression correction (parametric, uses all 28 models)")
    md.append("")
    md.append("| Model | Cutoff | Static (B) | Active (C) | 加成 (C−B) | Corrected (OLS) | Δ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(table, key=lambda x: -x["corrected"]):
        delta = r["corrected"] - r["score"]
        active_str = f"{r['active']:.3f}" if r['active'] is not None else "—"
        boost_str = f"+{r['boost']:.3f}" if r['boost'] is not None else "—"
        md.append(f"| {r['label']} | {r['cutoff']:.3f} | {r['score']:.3f} | "
                  f"{active_str} | {boost_str} | {r['corrected']:.3f} | {delta:+.3f} |")
    md.append("")
    # Active-only summary section
    active_rows = [r for r in table if r["active"] is not None]
    if active_rows:
        md.append("### Active-mode 加成 summary (only models with Track C data)")
        md.append("")
        md.append("| Model | Cutoff | Static (B) | Active (C) | 加成 |")
        md.append("|---|---:|---:|---:|---:|")
        for r in sorted(active_rows, key=lambda x: -x["boost"]):
            md.append(f"| {r['label']} | {r['cutoff']:.3f} | {r['score']:.3f} | "
                      f"{r['active']:.3f} | **+{r['boost']:.3f}** |")
        boosts = [r["boost"] for r in active_rows]
        md.append("")
        md.append(f"- n = {len(active_rows)} models with Active data (out of 28 total)")
        md.append(f"- Mean 加成 = +{sum(boosts)/len(boosts):.3f}")
        md.append(f"- Range: +{min(boosts):.3f} to +{max(boosts):.3f}")
        md.append("")
    md.append("## B. Bin-relative correction (1-yr bins; baseline = bin mean)")
    md.append("")
    md.append("Baseline per bin (mean of all models with cutoff in that bin):")
    md.append("")
    md.append("| Bin | n | bin_mean | bin_median | confidence |")
    md.append("|---|---:|---:|---:|---|")
    for b in sorted(bin_stats):
        bs = bin_stats[b]
        conf = "✓ stable (n≥5)" if bs["n"] >= 5 else ("marginal" if bs["n"] >= 3 else "LOW (n<3)")
        md.append(f"| {b}–{b+1} | {bs['n']} | {bs['mean']:.3f} | {bs['median']:.3f} | {conf} |")
    md.append("")
    md.append("Per-model bin-relative score (= raw − bin_mean; positive = above the typical model with same-era knowledge):")
    md.append("")
    md.append("| Model | Cutoff | Raw | Bin | bin_mean | Corrected (bin) | Δ |")
    md.append("|---|---:|---:|---|---:|---:|---:|")
    for r in sorted(table, key=lambda x: -x["corrected_bin_mean"]):
        flag = " ⚠" if r["bin_n"] < 3 else ""
        md.append(f"| {r['label']} | {r['cutoff']:.3f} | {r['score']:.3f} | "
                  f"{r['bin']} (n={r['bin_n']}){flag} | {r['bin_mean']:.3f} | "
                  f"{r['corrected_bin_mean']:+.3f} | {r['corrected_bin_mean']:+.3f} |")
    md.append("")
    if unknown_cutoff:
        md.append("### Excluded (no documented cutoff)")
        md.append("")
        for mid, lab, sc in unknown_cutoff:
            md.append(f"- {lab} (raw score {sc:.3f}) — cutoff: Unknown")
        md.append("")
    md.append("### Column definitions")
    md.append("")
    md.append("- **Cutoff**: decimal-year representation of the model's documented "
              "knowledge cutoff (mid-month). Source: vendor docs or model cards.")
    md.append("- **Static (B)**: weighted Static-Mode score (O×2 + F + C×0.5 + I×1.5 + S×0.5), "
              "trimmed-mean over critics, best-of-3 per paper, mean across papers. From "
              "`model_scores` table track='B'.")
    md.append("- **Active (C)**: same weighted score but for Track C (Active mode with SS "
              "search tools). `—` = no Active-mode data available for this model.")
    md.append("- **加成 (C−B)**: Active − Static; positive = tool-use ideation outperforms "
              "static survey-refs ideation. Only computable for models with both tracks.")
    md.append("- **Corrected (OLS)**: Static_raw − β × (cutoff − baseline). β is the OLS "
              "slope of Static score vs. cutoff across all 28 models. Removes the linear "
              "effect of knowing more recent literature.")
    md.append("- **Δ (OLS)**: Corrected − Static; positive → under-performed for its "
              "cutoff date; negative → over-performed given how recent its training is.")
    md.append("")

    out_md = ROOT / "reports" / "cross_year_cutoff_corrected.md"
    out_md.write_text("\n".join(md))
    print(f"Saved: {out_md}")

    # JSON
    out_json = ROOT / "reports" / "cross_year_cutoff_corrected.json"
    out_json.write_text(json.dumps({
        "baseline_cutoff": baseline_cutoff,
        "regression": {"slope": slope, "intercept": intercept, "r2": r2},
        "table": table,
        "unknown_cutoff": [{"model": m, "label": l, "score": s} for m, l, s in unknown_cutoff],
    }, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
