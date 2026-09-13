"""Active-mode 加成 plot: (C − B) score boost vs knowledge cutoff date.

Plots one point per model that has BOTH Static (Track B) and Active (Track C)
aggregate scores in `model_scores`. Currently 6 models out of 28 cross-year.

n=6 is too small for regression; just scatter + labels + mean horizontal line.
"""
import sqlite3
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from adjustText import adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False

from reports._make_cross_year_plot import (
    YEAR_GROUPS, KNOWLEDGE_CUTOFFS, FAMILY_STYLE, family_for,
)


def main():
    db = ROOT / "data" / "results.db"
    conn = sqlite3.connect(str(db))
    static_scores = {}
    active_scores = {}
    for r in conn.execute("""
        SELECT idea_model, track, AVG(mean_absolute_score)
        FROM model_scores
        WHERE mean_absolute_score IS NOT NULL
        GROUP BY idea_model, track
    """):
        mid, track, score = r
        if track == "B":
            static_scores[mid] = score
        elif track == "C":
            active_scores[mid] = score
    conn.close()

    label_for = {mid: lab for year_list in YEAR_GROUPS.values() for mid, lab in year_list}

    rows = []
    for mid in set(static_scores) & set(active_scores):
        if mid not in KNOWLEDGE_CUTOFFS or KNOWLEDGE_CUTOFFS[mid] is None:
            continue
        rows.append({
            "model": mid,
            "label": label_for.get(mid, mid.split("/")[-1]),
            "cutoff": KNOWLEDGE_CUTOFFS[mid],
            "static": static_scores[mid],
            "active": active_scores[mid],
            "boost": active_scores[mid] - static_scores[mid],
        })
    rows.sort(key=lambda r: r["cutoff"])

    if not rows:
        print("No models have both Static + Active + cutoff. Aborting.")
        return

    boosts = [r["boost"] for r in rows]
    mean_boost = statistics.mean(boosts)

    fig, ax = plt.subplots(figsize=(10, 6))
    texts = []
    drawn_fams = set()
    for r in rows:
        fam = family_for(r["model"])
        disp, color = FAMILY_STYLE.get(fam, ("?", "#888"))
        ax.scatter(r["cutoff"], r["boost"], s=140, color=color,
                   edgecolor="#111", linewidth=1.2, zorder=3,
                   label=disp if fam not in drawn_fams else None)
        drawn_fams.add(fam)
        texts.append(ax.text(r["cutoff"], r["boost"], r["label"],
                             fontsize=9, color="#222", zorder=4))

    # Mean horizontal reference line
    ax.axhline(mean_boost, color="#10a37f", linestyle="--", linewidth=1.4,
               alpha=0.7, label=f"mean boost = +{mean_boost:.2f}")
    ax.axhline(0, color="#888", linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Model knowledge cutoff date", fontsize=12)
    ax.set_ylabel("Active boost = Active − Static (weighted score)", fontsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(round(x))}"))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="x", which="minor", length=3)
    x_range = [r["cutoff"] for r in rows]
    ax.set_xlim(min(x_range) - 0.3, max(x_range) + 0.3)
    ax.set_ylim(min(boosts) - 0.2, max(boosts) + 0.3)
    ax.set_title(
        f"Active-Mode boost (Active − Static) vs. knowledge cutoff\n"
        f"{len(rows)} models with both tracks (out of 28 cross-year)",
        fontsize=12, pad=12,
    )
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=2)

    if _HAS_ADJUST_TEXT and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.5, alpha=0.6),
            expand=(1.2, 1.4),
            force_text=(0.0, 0.6),
            force_pull=(1.0, 0.05),
            only_move={"text": "y", "static": "y", "explode": "y"},
        )

    out_png = ROOT / "reports" / "cross_year_active_boost.png"
    out_pdf = ROOT / "reports" / "cross_year_active_boost.pdf"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    # Print summary table
    print(f"\nActive 加成 per model (sorted by 加成 desc):")
    print(f"{'Model':<35} {'Cutoff':>7} {'Static':>7} {'Active':>7} {'Boost':>7}")
    for r in sorted(rows, key=lambda x: -x["boost"]):
        print(f"  {r['model']:<33} {r['cutoff']:>7.3f} {r['static']:>7.3f} "
              f"{r['active']:>7.3f} {r['boost']:>+7.3f}")
    print(f"\nmean 加成 = +{mean_boost:.3f}")
    print(f"range: +{min(boosts):.3f} to +{max(boosts):.3f}")


if __name__ == "__main__":
    main()
