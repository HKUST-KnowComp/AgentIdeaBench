"""Saturation analysis: is Static plateauing while Active keeps rising?

Compare linear vs quadratic vs log fits over the 13 Active-paradigm models.
Also split-half slope analysis (early-cutoff vs late-cutoff).
Outputs an updated figure with saturation curves overlaid.
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
from reports._make_cross_year_plot import KNOWLEDGE_CUTOFFS, to_decimal_year

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

def main():
    static = load_scores("B")
    active = load_scores("C")
    # Build separate sets:
    #   - rows         : models with PAIRED Static+Active data (for Advantage/boost panel)
    #   - static_rows  : ALL models with Static data + known cutoff (for Static-extended fit)
    models_paired   = sorted({m for (m, _) in active})
    models_static_all = sorted({m for (m, _) in static})

    rows = []
    for m in models_paired:
        if m not in KNOWLEDGE_CUTOFFS:
            continue
        c = KNOWLEDGE_CUTOFFS[m]
        m_static = {p: w for (mm, p), w in static.items() if mm == m}
        m_active = {p: w for (mm, p), w in active.items() if mm == m}
        paired = [m_active[p] - m_static[p] for p in m_static if p in m_active]
        if not (m_static and m_active and paired):
            continue
        rows.append({
            "short": short_model_name(m),
            "cutoff": c,
            "static": statistics.mean(m_static.values()),
            "active": statistics.mean(m_active.values()),
            "boost": statistics.mean(paired),
        })
    rows.sort(key=lambda r: r["cutoff"])

    # Static-only extended set (includes plateau-extension models)
    def is_closed_weight(m):
        return m.startswith(("openai/", "anthropic/", "google/gemini"))
    static_rows = []
    for m in models_static_all:
        if m not in KNOWLEDGE_CUTOFFS:
            continue
        m_static = {p: w for (mm, p), w in static.items() if mm == m}
        if not m_static:
            continue
        static_rows.append({
            "short": short_model_name(m),
            "model": m,
            "cutoff": KNOWLEDGE_CUTOFFS[m],
            "static": statistics.mean(m_static.values()),
            "n_papers": len(m_static),
            "has_active": m in models_paired,
            "closed": is_closed_weight(m),
        })
    static_rows.sort(key=lambda r: r["cutoff"])

    x = np.array([r["cutoff"] for r in rows])
    ys = np.array([r["static"] for r in rows])
    ya = np.array([r["active"] for r in rows])
    yb = np.array([r["boost"] for r in rows])

    # Static-extended arrays (includes plateau-extension models)
    x_s = np.array([r["cutoff"] for r in static_rows])
    ys_ext = np.array([r["static"] for r in static_rows])

    print("=" * 70)
    print(f"STATIC-EXTENDED FIT (n={len(static_rows)}, X = knowledge cutoff)")
    print(f"  ({sum(1 for r in static_rows if not r['has_active'])} Static-only added")
    print(f"   to {sum(1 for r in static_rows if r['has_active'])} paired models)")
    print("=" * 70)
    # Linear
    c1 = np.polyfit(x_s, ys_ext, 1)
    r2_1 = r_squared(ys_ext, np.polyval(c1, x_s))
    # Quadratic
    c2 = np.polyfit(x_s, ys_ext, 2)
    r2_2 = r_squared(ys_ext, np.polyval(c2, x_s))
    # Saturation
    try:
        from scipy.optimize import curve_fit
        def sat(t, a, b, k):
            return a - b * np.exp(-k * (t - x_s.min()))
        popt, _ = curve_fit(sat, x_s, ys_ext,
                            p0=[float(ys_ext.max()) + 0.5,
                                float(ys_ext.max() - ys_ext.min()) + 0.5, 0.6],
                            maxfev=10000)
        r2_sat = r_squared(ys_ext, sat(x_s, *popt))
        sat_params_ext = f"a={popt[0]:.2f}, b={popt[1]:.2f}, k={popt[2]:.2f}"
    except Exception as e:
        r2_sat = None
        sat_params_ext = f"(scipy unavailable: {e})"
    print(f"  Linear     : y = {c1[0]:+.3f}*x {c1[1]:+.2f}    R² = {r2_1:.3f}")
    print(f"  Quadratic  : R² = {r2_2:.3f}   (Δ over linear = {r2_2 - r2_1:+.3f})")
    if r2_sat is not None:
        print(f"  Saturation : {sat_params_ext}   R² = {r2_sat:.3f}")
    # Split-half on extended set
    med_ext = float(np.median(x_s))
    early_ext = x_s < med_ext
    late_ext  = x_s >= med_ext
    s_early_ext = np.polyfit(x_s[early_ext], ys_ext[early_ext], 1)[0]
    s_late_ext  = np.polyfit(x_s[late_ext],  ys_ext[late_ext],  1)[0]
    print(f"  Split-half (median={med_ext:.2f}): early n={early_ext.sum()}  late n={late_ext.sum()}")
    print(f"               early slope = {s_early_ext:+.3f}/yr   late slope = {s_late_ext:+.3f}/yr   "
          f"Δ = {s_late_ext - s_early_ext:+.3f}")
    print()

    # ─── KEY ANALYSIS: open-weight vs closed-weight separation ───
    print("=" * 70)
    print("OPEN-WEIGHT vs CLOSED-WEIGHT split (Static only, X=cutoff)")
    print("=" * 70)
    for tag, rows_sub in [("OPEN-weight", [r for r in static_rows if not r["closed"]]),
                          ("CLOSED-weight", [r for r in static_rows if r["closed"]])]:
        xv = np.array([r["cutoff"] for r in rows_sub])
        yv = np.array([r["static"] for r in rows_sub])
        c1 = np.polyfit(xv, yv, 1)
        r2_1 = r_squared(yv, np.polyval(c1, xv))
        c2 = np.polyfit(xv, yv, 2)
        r2_2 = r_squared(yv, np.polyval(c2, xv))
        med = float(np.median(xv))
        e_m = xv < med
        l_m = xv >= med
        s_e = np.polyfit(xv[e_m], yv[e_m], 1)[0]
        s_l = np.polyfit(xv[l_m], yv[l_m], 1)[0]
        print(f"  {tag} (n={len(rows_sub)}):")
        print(f"    Linear     : slope = {c1[0]:+.3f}/yr     R² = {r2_1:.3f}")
        print(f"    Quadratic  : curvature = {c2[0]:+.3f}    R² = {r2_2:.3f}    (Δ = {r2_2-r2_1:+.3f})")
        print(f"    Split-half (median={med:.2f}, early n={e_m.sum()}, late n={l_m.sum()}):")
        print(f"      early slope = {s_e:+.3f}/yr     late slope = {s_l:+.3f}/yr     Δ = {s_l-s_e:+.3f}")
    print()

    # ─── 1. Compare linear vs quadratic vs saturation ───
    print("=" * 70)
    print(f"PAIRED-SET FIT (n={len(rows)}, X = knowledge cutoff)")
    print("=" * 70)
    for name, y in [("Static", ys), ("Active", ya), ("Advantage", yb)]:
        # Linear
        c1 = np.polyfit(x, y, 1)
        r2_1 = r_squared(y, np.polyval(c1, x))
        # Quadratic
        c2 = np.polyfit(x, y, 2)
        r2_2 = r_squared(y, np.polyval(c2, x))
        # Negative-exponential saturation: y = a - b * exp(-k*(x-x0))
        # Use scipy if available, else log-linear approx
        try:
            from scipy.optimize import curve_fit
            def sat(t, a, b, k):
                return a - b * np.exp(-k * (t - x.min()))
            popt, _ = curve_fit(sat, x, y,
                                p0=[float(y.max()) + 0.5,
                                    float(y.max() - y.min()) + 0.5, 0.6],
                                maxfev=10000)
            y_sat = sat(x, *popt)
            r2_sat = r_squared(y, y_sat)
            sat_params = f"a={popt[0]:.2f}, b={popt[1]:.2f}, k={popt[2]:.2f}"
        except Exception as e:
            r2_sat = None
            sat_params = f"(scipy unavailable: {e})"

        print(f"\n{name}:")
        print(f"  Linear fit     : y = {c1[0]:+.3f}*x + {c1[1]:+.2f}   R² = {r2_1:.3f}")
        print(f"  Quadratic fit  : y = {c2[0]:+.3f}*x² + {c2[1]:+.3f}*x + {c2[2]:+.2f}   R² = {r2_2:.3f}")
        if r2_sat is not None:
            print(f"  Saturation fit : {sat_params}   R² = {r2_sat:.3f}")

    # ─── 2. Split-half slope ───
    median_cutoff = float(np.median(x))
    print(f"\n{'=' * 70}")
    print(f"SPLIT-HALF SLOPE (median cutoff = {median_cutoff:.2f})")
    print(f"{'=' * 70}")
    early = x < median_cutoff
    late  = x >= median_cutoff
    print(f"Early group (cutoff < {median_cutoff:.2f}): n={early.sum()} models")
    print(f"Late  group (cutoff ≥ {median_cutoff:.2f}): n={late.sum()} models")
    for name, y in [("Static", ys), ("Active", ya), ("Advantage", yb)]:
        s_early = np.polyfit(x[early], y[early], 1)[0]
        s_late  = np.polyfit(x[late],  y[late],  1)[0]
        print(f"  {name:<10} early slope = {s_early:+.3f}/yr   late slope = {s_late:+.3f}/yr   "
              f"Δ = {s_late - s_early:+.3f}")

    # ─── 3. Plot — Static plateau over ALL 36 (open + closed), with outliers labeled ───
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6.5))
    cs = x  # paired set cutoffs

    open_rows  = [r for r in static_rows if not r["closed"]]
    closed_rows= [r for r in static_rows if r["closed"]]
    x_open = np.array([r["cutoff"] for r in open_rows])
    y_open = np.array([r["static"] for r in open_rows])
    x_clos = np.array([r["cutoff"] for r in closed_rows])
    y_clos = np.array([r["static"] for r in closed_rows])
    open_paired_idx = [i for i, r in enumerate(rows) if not r["short"].startswith(("gpt", "claude", "gemini"))]
    cs_open = cs[open_paired_idx]
    ya_open = ya[open_paired_idx]
    pe_mask = np.array([not r["has_active"] for r in open_rows])

    # Saturation fit on ALL 36 Static models (bounded to avoid degeneracy)
    from scipy.optimize import curve_fit
    def sat(t, a, b, k, x0):
        return a - b * np.exp(-k * (t - x0))
    popt_all, _ = curve_fit(lambda t, a, b, k: sat(t, a, b, k, x_s.min()),
                            x_s, ys_ext,
                            p0=[6.5, 2.5, 1.0],
                            bounds=([5.5, 0.5, 0.4], [9, 5, 5]),
                            maxfev=20000)
    y_pred_all = sat(x_s, *popt_all, x_s.min())
    r2_all = r_squared(ys_ext, y_pred_all)
    resid_all = ys_ext - y_pred_all
    thresh = 1.5 * resid_all.std()

    # Identify outliers (both directions)
    outlier_idx = np.where(np.abs(resid_all) > thresh)[0]
    outlier_models = {static_rows[i]["model"] for i in outlier_idx}
    print(f"\nOutliers (|resid| > {thresh:.2f}):")
    for i in outlier_idx:
        r = static_rows[i]
        flag = "[CLOSED]" if r["closed"] else "[OPEN]"
        sign = "above" if resid_all[i] > 0 else "below"
        print(f"  {flag} {r['model']:<44} actual={ys_ext[i]:.2f}  pred={y_pred_all[i]:.2f}  resid={resid_all[i]:+.2f}  ({sign} curve)")

    # Plot scatters
    ax1.scatter(x_open[~pe_mask], y_open[~pe_mask], s=80, c="#6b7280",
                edgecolors="#1f2937", linewidth=1.2,
                label=f"Open-weight Static (n={len(open_rows)})", zorder=3)
    ax1.scatter(x_open[pe_mask], y_open[pe_mask], s=80, c="#6b7280",
                edgecolors="#dc2626", linewidth=1.6, marker="s",
                label=f"  └ plateau-extension subset (n={pe_mask.sum()})", zorder=3)
    ax1.scatter(x_clos, y_clos, s=90, c="#f59e0b", edgecolors="#92400e",
                linewidth=1.2, marker="D",
                label=f"Closed-weight Static (n={len(closed_rows)})", zorder=3)
    ax1.scatter(cs_open, ya_open, s=80, c="#10a37f", edgecolors="#0a5f4a", linewidth=1.2,
                label=f"Active (paired open-weight, n={len(cs_open)})", zorder=3, marker="^")

    # Linear + saturation fits on ALL Static n=36
    xs_line = np.linspace(x_s.min() - 0.1, x_s.max() + 0.1, 80)
    s_all, i_all = np.polyfit(x_s, ys_ext, 1)
    ax1.plot(xs_line, s_all * xs_line + i_all, color="#6b7280",
             linestyle=":", linewidth=1.4, alpha=0.55, zorder=1,
             label=f"Static linear, all (slope {s_all:+.2f}/yr)")
    sa, ia = np.polyfit(cs_open, ya_open, 1)
    ax1.plot(xs_line, sa * xs_line + ia, color="#10a37f",
             linestyle=":", linewidth=1.4, alpha=0.55, zorder=1,
             label=f"Active linear (slope {sa:+.2f}/yr)")
    # Saturation curve
    ax1.plot(xs_line, sat(xs_line, *popt_all, x_s.min()), color="#6b7280",
             linestyle="-", linewidth=2.4, alpha=0.85, zorder=2,
             label=f"Static saturation, all n=36 (asymptote {popt_all[0]:.2f}, R²={r2_all:.2f})")
    # Asymptote horizontal reference
    ax1.axhline(popt_all[0], color="#6b7280", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)

    # Highlight outliers with large red ring + label
    for i in outlier_idx:
        r = static_rows[i]
        sign = "+" if resid_all[i] > 0 else ""
        ax1.scatter([r["cutoff"]], [r["static"]], s=200, facecolors="none",
                    edgecolors="#ef4444", linewidth=2.0, zorder=4)
        ax1.annotate(f"{r['short']}\n(resid {sign}{resid_all[i]:.1f})",
                     (r["cutoff"], r["static"]),
                     xytext=(8, 8 if resid_all[i] > 0 else -22),
                     textcoords="offset points",
                     fontsize=7.5, alpha=0.95, color="#991b1b",
                     fontweight="bold")

    ax1.set_xlabel("Knowledge cutoff (decimal year)", fontsize=11)
    ax1.set_ylabel("Mean weighted score (1–10)", fontsize=11)
    n_above = sum(1 for i in outlier_idx if resid_all[i] > 0)
    n_below = sum(1 for i in outlier_idx if resid_all[i] < 0)
    ax1.set_title(f"Static plateaus around {popt_all[0]:.2f} (n=36, R²={r2_all:.2f}; "
                  f"{n_above} above-curve + {n_below} below-curve outliers labeled)",
                  fontsize=11.5)
    ax1.legend(loc="lower right", fontsize=7.8, frameon=False, ncol=1)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xticks(np.arange(int(x_s.min()), int(x_s.max()) + 2))
    ax1.set_xlim(x_s.min() - 0.3, x_s.max() + 0.3)
    ax1.set_ylim(0, 10)

    plt.tight_layout()
    out_pdf = FIG / "paradigm_saturation.pdf"
    out_png = FIG / "paradigm_saturation.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out_pdf}")
    print(f"Wrote {out_png}")

if __name__ == "__main__":
    main()
