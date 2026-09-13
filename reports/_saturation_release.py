"""Saturation analysis on X = release date (companion to _saturation_analysis.py which uses cutoff).

Same n=36 Static (open + closed), bounded saturation fit, outliers labeled.
Plus Active (paired n=13, open-weight) for context.
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
from reports._make_cross_year_plot import RELEASE_DATES

W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())

FIG = ROOT / "reports" / "figures"


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in W) / WS


def load_scores(track, prompt_version="v1_paper_refs"):
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    rows = conn.execute(
        "SELECT idea_model, paper_id, scores_json FROM results "
        "WHERE track=? AND prompt_version=? "
        "  AND critic_model != '' AND scores_json IS NOT NULL",
        (track, prompt_version),
    ).fetchall()
    conn.close()
    by_key = defaultdict(list)
    for m, pid, sj in rows:
        try:
            s = json.loads(sj)
        except Exception:
            continue
        by_key[(m, pid)].append(weighted(s))
    return {k: statistics.mean(v) for k, v in by_key.items() if v}


def short_model_name(m):
    short = m.split("/")[-1]
    repl = {"mistral-7b-instruct-v0.1": "mistral-7b",
            "llama-3.1-8b-instruct": "llama-3.1-8b",
            "deepseek-r1-0528": "deepseek-r1",
            "qwen-2.5-72b-instruct": "qwen2.5-72b",
            "qwen3-vl-8b-thinking": "qwen3-vl-8b",
            "gemma-3-27b-it": "gemma-3-27b",
            "qwen3-235b-a22b-thinking-2507": "qwen3-235b-thk",
            "qwen3.5-397b-a17b": "qwen3.5-397b",
            "kimi-k2.6": "kimi-k2.6",
            "gemma-4-31b-it": "gemma-4-31b"}
    return repl.get(short, short)


def r_squared(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def is_closed_weight(m):
    return m.startswith(("openai/", "anthropic/", "google/gemini"))


def main():
    static = load_scores("B")
    active = load_scores("C")
    models_paired = sorted({m for (m, _) in active})
    models_static_all = sorted({m for (m, _) in static})

    # Paired set (for Active panel)
    rows = []
    for m in models_paired:
        if m not in RELEASE_DATES:
            continue
        m_static = {p: w for (mm, p), w in static.items() if mm == m}
        m_active = {p: w for (mm, p), w in active.items() if mm == m}
        paired = [m_active[p] - m_static[p] for p in m_static if p in m_active]
        if not (m_static and m_active and paired):
            continue
        rows.append({
            "short": short_model_name(m),
            "model": m,
            "release": RELEASE_DATES[m],
            "static": statistics.mean(m_static.values()),
            "active": statistics.mean(m_active.values()),
        })
    rows.sort(key=lambda r: r["release"])

    # All Static (open + closed) with release date — drop gpt-5.5 (extreme outlier)
    EXCLUDE_FROM_FIT = {"openai/gpt-5.5"}
    static_rows = []
    for m in models_static_all:
        if m not in RELEASE_DATES:
            continue
        if m in EXCLUDE_FROM_FIT:
            continue
        m_static = {p: w for (mm, p), w in static.items() if mm == m}
        if not m_static:
            continue
        static_rows.append({
            "short": short_model_name(m),
            "model": m,
            "release": RELEASE_DATES[m],
            "static": statistics.mean(m_static.values()),
            "has_active": m in models_paired,
            "closed": is_closed_weight(m),
        })
    static_rows.sort(key=lambda r: r["release"])

    # Arrays
    x_s = np.array([r["release"] for r in static_rows])
    ys_ext = np.array([r["static"] for r in static_rows])
    cs = np.array([r["release"] for r in rows])
    ya = np.array([r["active"] for r in rows])

    # ─── Print stats ───
    print("=" * 70)
    print(f"ALL STATIC FIT (n={len(static_rows)}, X = release date)")
    print("=" * 70)
    c1 = np.polyfit(x_s, ys_ext, 1)
    r2_1 = r_squared(ys_ext, np.polyval(c1, x_s))
    c2 = np.polyfit(x_s, ys_ext, 2)
    r2_2 = r_squared(ys_ext, np.polyval(c2, x_s))
    print(f"  Linear     : slope = {c1[0]:+.3f}/yr     R² = {r2_1:.3f}")
    print(f"  Quadratic  : curvature = {c2[0]:+.3f}    R² = {r2_2:.3f}  (Δ over linear = {r2_2-r2_1:+.3f})")

    # Bounded saturation fit
    from scipy.optimize import curve_fit
    def sat(t, a, b, k, x0):
        return a - b * np.exp(-k * (t - x0))
    try:
        popt_all, _ = curve_fit(lambda t, a, b, k: sat(t, a, b, k, x_s.min()),
                                x_s, ys_ext,
                                p0=[6.5, 2.5, 1.0],
                                bounds=([5.5, 0.5, 0.4], [9, 5, 5]),
                                maxfev=20000)
        y_pred_all = sat(x_s, *popt_all, x_s.min())
        r2_all = r_squared(ys_ext, y_pred_all)
        resid_all = ys_ext - y_pred_all
        print(f"  Bounded saturation : asymptote={popt_all[0]:.3f}, b={popt_all[1]:.3f}, k={popt_all[2]:.3f}, R²={r2_all:.3f}")
        print(f"  Residual std       : {resid_all.std():.3f}")
    except Exception as e:
        print(f"Saturation fit failed: {e}")
        return

    thresh = 1.5 * resid_all.std()
    outlier_idx = np.where(np.abs(resid_all) > thresh)[0]
    print(f"\nOutliers (|resid| > {thresh:.2f}):")
    for i in outlier_idx:
        r = static_rows[i]
        flag = "[CLOSED]" if r["closed"] else "[OPEN]"
        sign = "above" if resid_all[i] > 0 else "below"
        print(f"  {flag} {r['model']:<44} actual={ys_ext[i]:.2f}  pred={y_pred_all[i]:.2f}  resid={resid_all[i]:+.2f}  ({sign} curve)")

    # Active fit (paired only)
    sa, ia = np.polyfit(cs, ya, 1)
    r2_act = r_squared(ya, sa * cs + ia)
    print(f"\nActive linear (paired n={len(rows)}): slope = {sa:+.3f}/yr   R² = {r2_act:.3f}")

    # ─── Plot ───
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6.5))

    open_rows = [r for r in static_rows if not r["closed"]]
    closed_rows = [r for r in static_rows if r["closed"]]
    x_open = np.array([r["release"] for r in open_rows])
    y_open = np.array([r["static"] for r in open_rows])
    x_clos = np.array([r["release"] for r in closed_rows])
    y_clos = np.array([r["static"] for r in closed_rows])
    open_paired_idx = [i for i, r in enumerate(rows) if not r["short"].startswith(("gpt", "claude", "gemini"))]
    cs_open = cs[open_paired_idx]
    ya_open = ya[open_paired_idx]
    pe_mask = np.array([not r["has_active"] for r in open_rows])

    ax1.scatter(x_open[~pe_mask], y_open[~pe_mask], s=80, c="#6b7280",
                edgecolors="#1f2937", linewidth=1.2,
                label=f"Open-weight Static (n={len(open_rows)})", zorder=3)
    ax1.scatter(x_open[pe_mask], y_open[pe_mask], s=80, c="#6b7280",
                edgecolors="#dc2626", linewidth=1.6, marker="s",
                label=f"  └ plateau-extension subset (n={pe_mask.sum()})", zorder=3)
    ax1.scatter(x_clos, y_clos, s=90, c="#f59e0b", edgecolors="#92400e",
                linewidth=1.2, marker="D",
                label=f"Closed-weight Static (n={len(closed_rows)})", zorder=3)

    xs_line = np.linspace(x_s.min() - 0.1, x_s.max() + 0.1, 80)
    s_all, i_all = np.polyfit(x_s, ys_ext, 1)
    ax1.plot(xs_line, s_all * xs_line + i_all, color="#6b7280",
             linestyle=":", linewidth=1.4, alpha=0.55, zorder=1,
             label=f"Static linear (slope {s_all:+.2f}/yr)")
    ax1.plot(xs_line, sat(xs_line, *popt_all, x_s.min()), color="#6b7280",
             linestyle="-", linewidth=2.4, alpha=0.85, zorder=2,
             label=f"Static saturation (asymptote {popt_all[0]:.2f}, R²={r2_all:.2f})")
    ax1.axhline(popt_all[0], color="#6b7280", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)

    for i in outlier_idx:
        r = static_rows[i]
        sign = "+" if resid_all[i] > 0 else ""
        ax1.scatter([r["release"]], [r["static"]], s=200, facecolors="none",
                    edgecolors="#ef4444", linewidth=2.0, zorder=4)
        ax1.annotate(f"{r['short']}\n(resid {sign}{resid_all[i]:.1f})",
                     (r["release"], r["static"]),
                     xytext=(8, 8 if resid_all[i] > 0 else -22),
                     textcoords="offset points",
                     fontsize=7.5, alpha=0.95, color="#991b1b",
                     fontweight="bold")

    ax1.set_xlabel("Release date (decimal year)", fontsize=11)
    ax1.set_ylabel("Mean weighted score (1–10)", fontsize=11)
    n_above = sum(1 for i in outlier_idx if resid_all[i] > 0)
    n_below = sum(1 for i in outlier_idx if resid_all[i] < 0)
    ax1.set_title(f"Static plateaus around {popt_all[0]:.2f} on release axis (Static-only, n={len(static_rows)}, "
                  f"R²={r2_all:.2f}; gpt-5.5 excluded; {n_above} above + {n_below} below outliers labeled)",
                  fontsize=11)
    ax1.legend(loc="lower right", fontsize=7.8, frameon=False, ncol=1)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xticks(np.arange(int(x_s.min()), int(x_s.max()) + 2))
    ax1.set_xlim(x_s.min() - 0.3, x_s.max() + 0.3)
    ax1.set_ylim(0, 10)

    plt.tight_layout()
    out_pdf = FIG / "paradigm_saturation_release.pdf"
    out_png = FIG / "paradigm_saturation_release.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
