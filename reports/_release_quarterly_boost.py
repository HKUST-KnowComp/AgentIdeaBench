"""Paradigm advantage vs RELEASE DATE (sibling of _cutoff_quarterly_boost.py).

Same 2-panel figure design but X-axis is release date instead of knowledge
cutoff. This is the "model release timeline" view; users without official
cutoffs may prefer this proxy.
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
from reports._make_cross_year_plot import RELEASE_DATES, to_decimal_year

# see log.md 2026-05-19 13:05 entry).
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def quarter_bucket(decimal_year: float) -> str:
    year = int(decimal_year)
    frac = decimal_year - year
    if frac < 0.25:
        q = "Q1"
    elif frac < 0.5:
        q = "Q2"
    elif frac < 0.75:
        q = "Q3"
    else:
        q = "Q4"
    return f"{year} {q}"

def quarter_to_decimal(qstr):
    year, q = qstr.split()
    offset = {"Q1": 0.125, "Q2": 0.375, "Q3": 0.625, "Q4": 0.875}[q]
    return int(year) + offset

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
    repl = {
        "mistral-7b-instruct-v0.1": "mistral-7b",
        "llama-3.1-8b-instruct": "llama-3.1-8b",
        "deepseek-r1-0528": "deepseek-r1",
        "qwen-2.5-72b-instruct": "qwen2.5-72b",
        "qwen3-vl-8b-thinking": "qwen3-vl-8b",
        "gemma-3-27b-it": "gemma-3-27b",
        "qwen3-235b-a22b-thinking-2507": "qwen3-235b-thk",
        "qwen3.5-397b-a17b": "qwen3.5-397b",
        "kimi-k2.6": "kimi-k2.6",
        "gemma-4-31b-it": "gemma-4-31b",
    }
    return repl.get(short, short)

def main():
    static = load_scores("B")
    active = load_scores("C")
    models_active = sorted({m for (m, _) in active})

    per_model = []
    for m in models_active:
        if m not in RELEASE_DATES:
            continue
        rd = RELEASE_DATES[m]
        m_static = {p: w for (mm, p), w in static.items() if mm == m}
        m_active = {p: w for (mm, p), w in active.items() if mm == m}
        if not m_static or not m_active:
            continue
        paired = [m_active[p] - m_static[p] for p in m_static if p in m_active]
        per_model.append({
            "model": m,
            "short": short_model_name(m),
            "release": rd,
            "quarter": quarter_bucket(rd),
            "mean_static": statistics.mean(m_static.values()),
            "mean_active": statistics.mean(m_active.values()),
            "boost": statistics.mean(paired) if paired else None,
            "n_pairs": len(paired),
        })
    per_model.sort(key=lambda r: r["release"])

    by_q = defaultdict(lambda: {
        "models": set(), "static_obs": [], "active_obs": [], "paired_boost": [],
    })
    for r in per_model:
        b = r["quarter"]
        by_q[b]["models"].add(r["short"])
        m = r["model"]
        m_static = {p: w for (mm, p), w in static.items() if mm == m}
        m_active = {p: w for (mm, p), w in active.items() if mm == m}
        by_q[b]["static_obs"] += list(m_static.values())
        by_q[b]["active_obs"] += list(m_active.values())
        by_q[b]["paired_boost"] += [m_active[p] - m_static[p]
                                     for p in m_static if p in m_active]

    quarter_order = sorted(by_q.keys(), key=quarter_to_decimal)

    print(f"{'Quarter':<10} {'n_mod':>5} {'meanStatic':>11} {'meanActive':>11} "
          f"{'boost(paired)':>16} {'n_pairs':>8}")
    print("-" * 70)
    quarterly = {}
    for q in quarter_order:
        d = by_q[q]
        mS = statistics.mean(d["static_obs"]) if d["static_obs"] else None
        mA = statistics.mean(d["active_obs"]) if d["active_obs"] else None
        if d["paired_boost"]:
            mB = statistics.mean(d["paired_boost"])
            sd = (statistics.stdev(d["paired_boost"])
                  if len(d["paired_boost"]) > 1 else 0.0)
            sem = sd / (len(d["paired_boost"]) ** 0.5)
            ci = 1.96 * sem
            boost_str = f"{mB:+.2f} (±{ci:.2f})"
        else:
            mB, ci, boost_str = None, None, "  n/a"
        n_pairs = len(d["paired_boost"])
        print(f"{q:<10} {len(d['models']):>5} {mS or 0:>11.2f} {mA or 0:>11.2f} "
              f"{boost_str:>16} {n_pairs:>8}")
        quarterly[q] = {
            "n_models": len(d["models"]),
            "models": sorted(d["models"]),
            "mean_static": mS, "mean_active": mA,
            "boost_mean": mB, "boost_ci95_halfwidth": ci,
            "n_pairs": n_pairs,
            "midpoint": quarter_to_decimal(q),
        }

    out_path = ROOT / "reports" / "release_quarterly_boost.json"
    with open(out_path, "w") as f:
        json.dump({"per_model": per_model, "quarterly": quarterly}, f, indent=2)
    print(f"\nWrote {out_path}")

    # ────────── Figure ──────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1]}
    )
    cs = [r["release"] for r in per_model]
    ss = [r["mean_static"] for r in per_model]
    aa = [r["mean_active"] for r in per_model]
    bb = [r["boost"] for r in per_model]
    names = [r["short"] for r in per_model]

    ax1.scatter(cs, ss, s=80, c="#6b7280", edgecolors="#1f2937", linewidth=1.2,
                label="Static (Track B)", zorder=3)
    ax1.scatter(cs, aa, s=80, c="#10a37f", edgecolors="#0a5f4a", linewidth=1.2,
                label="Active (Track C)", zorder=3, marker="^")
    for c, s, a in zip(cs, ss, aa):
        ax1.plot([c, c], [s, a], color="gray", alpha=0.35, linewidth=0.9, zorder=2)

    cs_arr = np.array(cs)
    for ys, color, label in [(ss, "#6b7280", "Static fit"),
                              (aa, "#10a37f", "Active fit")]:
        ys_arr = np.array(ys)
        slope, intercept = np.polyfit(cs_arr, ys_arr, 1)
        xs_line = np.linspace(cs_arr.min() - 0.1, cs_arr.max() + 0.1, 50)
        ax1.plot(xs_line, slope * xs_line + intercept, color=color,
                 linestyle="--", linewidth=1.5, alpha=0.7, zorder=1,
                 label=f"{label} (slope {slope:+.2f}/yr)")

    for q, d in quarterly.items():
        if d["mean_static"] is None:
            continue
        mid = d["midpoint"]
        ax1.scatter(mid, d["mean_static"], s=180, marker="s",
                    facecolors="none", edgecolors="#1f2937", linewidth=1.5, zorder=4)
        ax1.scatter(mid, d["mean_active"], s=180, marker="s",
                    facecolors="none", edgecolors="#0a5f4a", linewidth=1.5, zorder=4)

    for c, a, n in zip(cs, aa, names):
        ax1.annotate(n, (c, a), xytext=(4, 6), textcoords="offset points",
                     fontsize=7, alpha=0.75)

    ax1.set_ylabel("Mean weighted score (1–10)", fontsize=11)
    ax1.set_title("Hypothesis quality by paradigm vs. release date "
                  "(13 open-weight models, v1 protocol)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom: boost
    colors = ["#dc2626" if b < 0 else "#10a37f" for b in bb]
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax2.scatter(cs, bb, s=80, c=colors, edgecolors="#1f2937", linewidth=1.2, zorder=3)
    for c, b, n in zip(cs, bb, names):
        ax2.annotate(n, (c, b), xytext=(4, 5 if b >= 0 else -10),
                     textcoords="offset points", fontsize=7, alpha=0.75)

    bb_arr = np.array(bb)
    slope_b, intercept_b = np.polyfit(cs_arr, bb_arr, 1)
    xs_line = np.linspace(cs_arr.min() - 0.1, cs_arr.max() + 0.1, 50)
    ax2.plot(xs_line, slope_b * xs_line + intercept_b, color="#10a37f",
             linestyle="--", linewidth=1.5, alpha=0.7, zorder=1,
             label=f"Linear fit (slope {slope_b:+.2f}/yr)")

    for q, d in quarterly.items():
        if d["boost_mean"] is None:
            continue
        mid = d["midpoint"]
        ax2.errorbar(mid, d["boost_mean"],
                     yerr=d["boost_ci95_halfwidth"] or 0,
                     fmt="s", markersize=8, capsize=4,
                     markerfacecolor="none", markeredgecolor="#1f2937",
                     ecolor="#1f2937", linewidth=1.3, zorder=4,
                     label="Quarterly bucket mean ±95% CI" if q == quarter_order[0] else None)

    ax2.set_xlabel("Release date (decimal year)", fontsize=11)
    ax2.set_ylabel("Paradigm advantage  (Active − Static)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xticks(np.arange(int(cs_arr.min()), int(cs_arr.max()) + 2))
    ax2.set_xlim(cs_arr.min() - 0.3, cs_arr.max() + 0.3)

    plt.tight_layout()
    out_pdf = FIG / "paradigm_advantage_vs_release.pdf"
    out_png = FIG / "paradigm_advantage_vs_release.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    print(f"\nLinear fit (n=13 models, vs release date):")
    print(f"  Static    : slope = {np.polyfit(cs_arr, np.array(ss), 1)[0]:+.3f}/yr")
    print(f"  Active    : slope = {np.polyfit(cs_arr, np.array(aa), 1)[0]:+.3f}/yr")
    print(f"  Advantage : slope = {slope_b:+.3f}/yr")
    print(f"\nPearson r (vs release date):")
    print(f"  Static    : r = {np.corrcoef(cs_arr, np.array(ss))[0, 1]:+.3f}")
    print(f"  Active    : r = {np.corrcoef(cs_arr, np.array(aa))[0, 1]:+.3f}")
    print(f"  Advantage : r = {np.corrcoef(cs_arr, bb_arr)[0, 1]:+.3f}")

if __name__ == "__main__":
    main()
