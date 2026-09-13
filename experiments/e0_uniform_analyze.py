"""E0 — Uniform-pool analysis: recompute leaderboard + F1/F2/F3 on modern5 scores.

After `uniform_critic_rescore.py --all` re-scores every model's existing ideas
with the SAME modern5 critic pool (table `uniform_critic_scores`), this script:

  1. Builds the uniform Static (Track B) leaderboard for all models.
  2. Compares each model's uniform score vs its OLD production score
     (`results` critic rows, whatever pool it originally used) → delta.
     Highlights the 7 old-pool models that were the reason for the rescore.
  3. Recomputes the findings on the uniform, cross-comparable scores:
       F1  — Static saturation: linear vs bounded-exp fit R², early/late slope,
             yearly min/max, open vs closed split.
       F2  — Active boost: paired C−B per model; sign test (how many up);
             variance(B) vs variance(C) across models (F-test);
             r(Static, boost) Pearson + Spearman  [stronger-model-gains-more].
       F3  — Static slope vs Active slope along knowledge cutoff (1.49×?);
             Spearman(boost, cutoff).

Aggregation matches production (analysis/compute_scores.py):
  per critic row: weighted = Σ(score·W)/ΣW  (normalised to 0–10)
  per (paper,model,track,idx): trimmed mean over critics (drop highest)
  per (paper,model,track): best idx by score
  per (model,track): mean over papers

READ-ONLY on both DBs. Writes reports/e0_uniform/*.{json,md}.

Usage:
  python experiments/e0_uniform_analyze.py
"""
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as WEIGHTS
import reports._make_cross_year_plot as mc

WSUM = sum(WEIGHTS.values())
CUTOFFS = mc.KNOWLEDGE_CUTOFFS
CLOSED_PREFIXES = ("openai/", "anthropic/", "google/")
OLD_POOL_MODELS = [
    "anthropic/claude-3.7-sonnet", "mistralai/mistral-small-2603",
    "moonshotai/kimi-k2.6", "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-32b", "qwen/qwen3-vl-8b-thinking", "z-ai/glm-5.1",
]


def is_closed(m):
    return m.startswith(CLOSED_PREFIXES)


def weighted_norm(scores: dict) -> float:
    w = sum(scores.get(d, 0) * WEIGHTS[d] for d in DIMS if d in scores)
    t = sum(WEIGHTS[d] for d in DIMS if d in scores)
    return w / t if t else 0.0


def trimmed_mean(vs):
    if not vs:
        return None
    if len(vs) < 2:
        return vs[0]
    return statistics.mean(sorted(vs)[:-1])   # drop highest, matches production


def aggregate(rows):
    """rows: iterable of dict(paper_id, idea_model, track, idea_index, scores_json).
    Returns per (model, track) -> {mean, std, n_papers, papers:{pid:score}}."""
    # per (model,track,paper,idx) -> [weighted per critic]
    cell = defaultdict(list)
    for r in rows:
        sj = r["scores_json"]
        if not sj:
            continue
        try:
            s = json.loads(sj) if isinstance(sj, str) else sj
        except Exception:
            continue
        cell[(r["idea_model"], r["track"], r["paper_id"], r["idea_index"])].append(weighted_norm(s))
    # trimmed mean over critics
    idx_score = {k: trimmed_mean(v) for k, v in cell.items() if trimmed_mean(v) is not None}
    # best idx per (model,track,paper)
    best = defaultdict(dict)
    for (m, t, p, i), sc in idx_score.items():
        d = best[(m, t, p)]
        if "s" not in d or sc > d["s"]:
            d["s"] = sc
    out = {}
    per_mt = defaultdict(dict)
    for (m, t, p), d in best.items():
        per_mt[(m, t)][p] = d["s"]
    for (m, t), papers in per_mt.items():
        vals = list(papers.values())
        out[(m, t)] = {
            "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_papers": len(vals), "papers": papers,
        }
    return out


def load_uniform():
    c = sqlite3.connect(str(cfg.RESULTS_DB)); c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT paper_id, idea_model, track, idea_index, scores_json "
        "FROM uniform_critic_scores WHERE scores_json IS NOT NULL")]
    c.close()
    return rows


def load_old_production():
    c = sqlite3.connect(str(cfg.RESULTS_DB)); c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT paper_id, idea_model, track, idea_index, scores_json "
        "FROM results WHERE prompt_version='v1_paper_refs' AND critic_model!='' "
        "  AND scores_json IS NOT NULL")]
    c.close()
    return rows


def linfit(xs, ys):
    if len(xs) < 3:
        return None
    r = stats.linregress(xs, ys)
    return {"slope": float(r.slope), "intercept": float(r.intercept),
            "r2": float(r.rvalue ** 2), "p": float(r.pvalue), "n": len(xs)}


def bounded_exp_r2(xs, ys):
    """Fit y = a - b*exp(-k*(x-x0)), k>=0 free; return R²."""
    from scipy.optimize import curve_fit
    xs = np.array(xs, float); ys = np.array(ys, float)
    x0 = xs.min()
    def f(x, a, b, k):
        return a - b * np.exp(-k * (x - x0))
    try:
        p0 = [ys.max(), ys.max() - ys.min(), 0.5]
        popt, _ = curve_fit(f, xs, ys, p0=p0, maxfev=20000,
                            bounds=([-50, -50, 0], [50, 50, 50]))
        resid = ys - f(xs, *popt)
        ss_res = float(np.sum(resid ** 2)); ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        return {"r2": 1 - ss_res / ss_tot if ss_tot else None, "k": float(popt[2]),
                "asymptote_a": float(popt[0])}
    except Exception as e:
        return {"error": str(e)[:120]}


def main():
    uni = aggregate(load_uniform())
    old = aggregate(load_old_production())

    models_B = sorted({m for (m, t) in uni if t == "B"},
                      key=lambda m: -uni[(m, "B")]["mean"])

    # ---- (1)+(2) leaderboard + old vs uniform ----
    leaderboard = []
    for m in models_B:
        u = uni[(m, "B")]
        o = old.get((m, "B"))
        leaderboard.append({
            "model": m, "cutoff": CUTOFFS.get(m),
            "closed": is_closed(m),
            "uniform_static": round(u["mean"], 4), "uniform_std": round(u["std"], 3),
            "n_papers": u["n_papers"],
            "old_static": round(o["mean"], 4) if o else None,
            "delta": round(u["mean"] - o["mean"], 4) if o else None,
            "old_pool_model": m in OLD_POOL_MODELS,
        })

    # ---- F2: boost = C - B paired by model ----
    f2_rows = []
    for m in sorted({mm for (mm, t) in uni if t == "C"}):
        if (m, "B") in uni and (m, "C") in uni:
            b = uni[(m, "B")]["mean"]; c = uni[(m, "C")]["mean"]
            f2_rows.append({"model": m, "cutoff": CUTOFFS.get(m),
                            "static": round(b, 4), "active": round(c, 4),
                            "boost": round(c - b, 4), "closed": is_closed(m)})
    boosts = [r["boost"] for r in f2_rows]
    statics = [r["static"] for r in f2_rows]
    actives = [r["active"] for r in f2_rows]
    up = sum(1 for x in boosts if x > 0); down = sum(1 for x in boosts if x < 0)
    sign_p = (2 * stats.binom.cdf(min(up, down), up + down, 0.5)) if (up + down) else None
    # variance widen: F-test var(C) vs var(B)
    varB = statistics.variance(statics) if len(statics) > 1 else None
    varC = statistics.variance(actives) if len(actives) > 1 else None
    if varB and varC:
        F = varC / varB
        dfn = dfd = len(statics) - 1
        f_p = 2 * min(stats.f.cdf(F, dfn, dfd), 1 - stats.f.cdf(F, dfn, dfd))
    else:
        F = f_p = None
    # stronger-model-gains-more: r(static, boost)
    if len(statics) > 2:
        pear = stats.pearsonr(statics, boosts); spear = stats.spearmanr(statics, boosts)
        r_sb = {"pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
                "spearman_r": float(spear[0]), "spearman_p": float(spear[1])}
    else:
        r_sb = None

    f2 = {
        "n_paired": len(f2_rows),
        "n_up": up, "n_down": down, "sign_test_p": float(sign_p) if sign_p is not None else None,
        "mean_boost": round(float(np.mean(boosts)), 4) if boosts else None,
        "var_static": round(varB, 4) if varB else None,
        "var_active": round(varC, 4) if varC else None,
        "variance_ratio_C_over_B": round(F, 4) if F else None,
        "variance_F_p": round(f_p, 4) if f_p else None,
        "stronger_gains_more_r_static_boost": r_sb,
        "rows": f2_rows,
    }

    # ---- F3: Static slope vs Active slope along cutoff ----
    def with_cutoff(track):
        xs, ys = [], []
        for (m, t), d in uni.items():
            if t == track and CUTOFFS.get(m) is not None:
                xs.append(CUTOFFS[m]); ys.append(d["mean"])
        return xs, ys
    xB, yB = with_cutoff("B"); xC, yC = with_cutoff("C")
    slopeB = linfit(xB, yB); slopeC = linfit(xC, yC)
    # boost vs cutoff (paired models only)
    bc_x = [r["cutoff"] for r in f2_rows if r["cutoff"] is not None]
    bc_y = [r["boost"] for r in f2_rows if r["cutoff"] is not None]
    if len(bc_x) > 2:
        sp = stats.spearmanr(bc_x, bc_y); pe = stats.pearsonr(bc_x, bc_y)
        boost_cutoff = {"spearman_r": float(sp[0]), "spearman_p": float(sp[1]),
                        "pearson_r": float(pe[0]), "pearson_p": float(pe[1]), "n": len(bc_x)}
    else:
        boost_cutoff = None
    f3 = {
        "static_slope_per_yr": slopeB, "active_slope_per_yr": slopeC,
        "slope_ratio_active_over_static": (round(slopeC["slope"] / slopeB["slope"], 3)
                                           if slopeB and slopeC and slopeB["slope"] else None),
        "boost_vs_cutoff": boost_cutoff,
    }

    # ---- F1: Static saturation on uniform ----
    xs1, ys1 = with_cutoff("B")
    lin = linfit(xs1, ys1)
    bexp = bounded_exp_r2(xs1, ys1)
    # open vs closed split, early vs late slope (split at median cutoff)
    open_pts = [(CUTOFFS[m], uni[(m, "B")]["mean"]) for (m, t) in uni
                if t == "B" and CUTOFFS.get(m) is not None and not is_closed(m)]
    closed_pts = [(CUTOFFS[m], uni[(m, "B")]["mean"]) for (m, t) in uni
                  if t == "B" and CUTOFFS.get(m) is not None and is_closed(m)]
    def split_slopes(pts):
        if len(pts) < 6:
            return None
        pts = sorted(pts); med = statistics.median([p[0] for p in pts])
        early = [(x, y) for x, y in pts if x <= med]; late = [(x, y) for x, y in pts if x > med]
        return {"early": linfit([p[0] for p in early], [p[1] for p in early]),
                "late": linfit([p[0] for p in late], [p[1] for p in late])}
    # yearly min/max by cutoff year
    yearly = defaultdict(list)
    for (m, t), d in uni.items():
        if t == "B" and CUTOFFS.get(m) is not None:
            yearly[int(CUTOFFS[m])].append(d["mean"])
    yearly_minmax = {str(y): {"n": len(v), "min": round(min(v), 3), "max": round(max(v), 3),
                              "spread": round(max(v) - min(v), 3), "mean": round(float(np.mean(v)), 3)}
                     for y, v in sorted(yearly.items())}
    f1 = {
        "linear_fit": lin, "bounded_exp_fit": bexp,
        "linear_beats_bounded": (lin and bexp and bexp.get("r2") is not None
                                 and lin["r2"] >= bexp["r2"]),
        "open_split_slopes": split_slopes(open_pts),
        "closed_split_slopes": split_slopes(closed_pts),
        "n_open": len(open_pts), "n_closed": len(closed_pts),
        "yearly_minmax": yearly_minmax,
    }

    result = {"leaderboard": leaderboard, "F1_saturation": f1, "F2_boost": f2, "F3_cutoff": f3,
              "n_models_B": len(models_B), "critic_pool": "modern5 (uniform_critic_scores)"}

    out = ROOT / "reports" / "e0_uniform"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e0_uniform_analysis.json").write_text(json.dumps(result, indent=2))

    # console summary
    print(f"\n=== E0 uniform leaderboard (Static, modern5) — {len(models_B)} models ===")
    print(f"{'model':42s} {'cutoff':>8} {'uni':>6} {'old':>6} {'Δ':>6}")
    for r in leaderboard:
        print(f"{r['model']:42s} {str(r['cutoff'] or '')[:7]:>8} "
              f"{r['uniform_static']:>6.2f} "
              f"{(r['old_static'] if r['old_static'] is not None else float('nan')):>6.2f} "
              f"{(r['delta'] if r['delta'] is not None else float('nan')):>+6.2f}"
              f"{'  <old-pool' if r['old_pool_model'] else ''}")
    print(f"\n=== F2 (n={f2['n_paired']}): up={f2['n_up']} down={f2['n_down']} "
          f"sign_p={f2['sign_test_p']} mean_boost={f2['mean_boost']}")
    print(f"  var B={f2['var_static']} var C={f2['var_active']} ratio={f2['variance_ratio_C_over_B']} F_p={f2['variance_F_p']}")
    print(f"  r(static,boost)={f2['stronger_gains_more_r_static_boost']}")
    print(f"\n=== F3: Static slope={f3['static_slope_per_yr']}")
    print(f"        Active slope={f3['active_slope_per_yr']}")
    print(f"        ratio active/static={f3['slope_ratio_active_over_static']}")
    print(f"        boost vs cutoff={f3['boost_vs_cutoff']}")
    print(f"\n=== F1: linear={f1['linear_fit']}")
    print(f"        bounded_exp={f1['bounded_exp_fit']}  linear_beats_bounded={f1['linear_beats_bounded']}")
    print(f"        yearly_minmax={json.dumps(f1['yearly_minmax'])}")
    print(f"\n✓ wrote {out}/e0_uniform_analysis.json")


if __name__ == "__main__":
    main()
