"""E41 analysis — where the 2026 frontier closed models land, and what they do
to the capability gate (F2).

Read-only. Reuses e21's `_load()` so the per-(model, track, subdomain) weighted
scores are byte-identical to the pipeline's own aggregation, and e36's JSON for
the model-level totals. Writes reports/e41_frontier_analysis.json.

First-order quantities only; every number here is computed from
lit8d_scores_3seed or reports/e36_leaderboard_subscores.json.

  /opt/homebrew/Caskroom/miniforge/base/bin/python experiments/e41_frontier_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import experiments.e21_pilot20_3seed as e21  # noqa: E402

E36 = ROOT / "reports" / "e36_leaderboard_subscores.json"
OUT = ROOT / "reports" / "e41_frontier_analysis.json"
DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]

FRONTIER = {"azure/openai/gpt-5.6-sol", "azure/openai/gpt-5.6-terra",
            "azure/openai/gpt-5.6-luna", "azure/anthropic/claude-opus-5",
            "azure/anthropic/claude-sonnet-5"}
GEMINI = lambda m: m.startswith("google/gemini")

# Critic-ceiling reference points, both from earlier experiments in this repo:
# CORE-7 award-paper anchors under the same lit8d rubric (F14) and the highest
# weighted score any anchor reached under any rubric variant (F11).
CORE7_MEAN = 7.71
ANCHOR_MAX_WEIGHTED = 7.32


def group_of(m, agg):
    if m in FRONTIER:
        return "frontier2026"
    if GEMINI(m):
        return "gemini"
    return "open_weight"


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {"n": int(len(x)), "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp)}


def main():
    agg = json.load(open(E36))
    groups = {m: group_of(m, agg) for m in agg}

    # ---- model-level gate (F2) on nested rosters -------------------------
    def subset(pred):
        ms = [m for m in agg if pred(groups[m])]
        return ms, corr([agg[m]["static_total"] for m in ms],
                        [agg[m]["gain"] for m in ms])

    ow, r_ow = subset(lambda g: g == "open_weight")
    og, r_og = subset(lambda g: g in ("open_weight", "gemini"))
    allm, r_all = subset(lambda g: True)
    fr, r_fr = subset(lambda g: g == "frontier2026")

    gate = {
        "open_weight_28": r_ow,
        "plus_gemini_33": r_og,
        "plus_frontier_38": r_all,
        "frontier_only_5": r_fr,
    }

    # ---- per-model paired test over subdomains (cell level) --------------
    seed_w, n_ideas, meta = e21._load()
    by = {}
    for (m, tr, sub), w in seed_w.items():
        by.setdefault(m, {"B": {}, "C": {}})[tr][sub] = w
    per_model = {}
    for m in sorted(agg):
        b, c = by[m]["B"], by[m]["C"]
        shared = sorted(set(b) & set(c))
        d = [c[s] - b[s] for s in shared]
        w_p = float(stats.wilcoxon(d).pvalue) if len(d) >= 6 else None
        per_model[m] = {
            "short": agg[m]["short"], "group": groups[m],
            "n_shared_subdomains": len(shared),
            "static": float(np.mean([b[s] for s in shared])),
            "active": float(np.mean([c[s] for s in shared])),
            "gain": float(np.mean(d)),
            "wilcoxon_p": w_p,
            "frac_subdomains_improved": float(np.mean([x > 0 for x in d])),
        }

    # ---- per-dimension Active-minus-Static, frontier group ---------------
    # e36 stores Active dims only, so recompute both tracks here from the same
    # trimmed-mean pipeline used by e36 (score rows -> idx -> subdomain -> model).
    import sqlite3
    from collections import defaultdict
    import config as cfg
    from utils.constants import SCORING_DIMS as SD
    from experiments.e10_idea_anchor_calibration import trimmed_mean
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    idea = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT idea_model,track,subdomain,idea_index,scores_json "
                          "FROM lit8d_scores_3seed WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"])
        k = (r["idea_model"], r["track"], r["subdomain"], r["idea_index"])
        for d in SD:
            idea[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()

    def tm(vs):
        return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)

    sub_dim = defaultdict(lambda: defaultdict(list))
    for (m, tr, sub, idx) in idea:
        for d in SD:
            sub_dim[(m, tr, sub)][d].append(tm(idea[(m, tr, sub, idx)][d]))
    mt = defaultdict(lambda: defaultdict(list))
    for (m, tr, sub) in sub_dim:
        for d in SD:
            mt[(m, tr)][d].append(float(np.mean(sub_dim[(m, tr, sub)][d])))

    def dim_gain(models):
        out = {}
        for d in SD:
            vals = [float(np.mean(mt[(m, "C")][d]) - np.mean(mt[(m, "B")][d]))
                    for m in models]
            out[d] = {"mean_gain": float(np.mean(vals)),
                      "n_models": len(vals),
                      "n_positive": int(sum(v > 0 for v in vals))}
        return out

    per_dim = {"frontier2026": dim_gain(fr),
               "open_weight_28": dim_gain(ow),
               "gemini_5": dim_gain([m for m in agg if groups[m] == "gemini"])}

    # ---- ceiling headroom -------------------------------------------------
    best_static = max(agg[m]["static_total"] for m in fr)
    best_active = max(agg[m]["active_total"] for m in fr)
    ceiling = {
        "core7_anchor_mean_lit8d": CORE7_MEAN,
        "max_anchor_weighted_any_variant": ANCHOR_MAX_WEIGHTED,
        "frontier_best_static": float(best_static),
        "frontier_best_active": float(best_active),
        "headroom_best_active_to_core7": float(CORE7_MEAN - best_active),
        "best_open_weight_active": float(max(
            agg[m]["active_total"] for m in ow)),
        "best_open_weight_static": float(max(
            agg[m]["static_total"] for m in ow)),
    }

    res = {
        "roster": {"total_paired": len(agg), "open_weight": len(ow),
                   "gemini_heldout": len(og) - len(ow), "frontier2026": len(fr)},
        "capability_gate_F2": gate,
        "per_model": per_model,
        "per_dimension_gain": per_dim,
        "ceiling": ceiling,
        "sources": {
            "model_totals": "reports/e36_leaderboard_subscores.json",
            "cell_scores": "results.db :: lit8d_scores_3seed",
            "aggregation": "e21._load() (trimmed mean = drop-highest of 3 critics)",
        },
    }
    OUT.write_text(json.dumps(res, indent=1))

    # ---- console summary --------------------------------------------------
    print("=== F2 capability gate: r(Static, Active gain) on nested rosters ===")
    for k, v in gate.items():
        print(f"  {k:<20} n={v['n']:<3} Pearson r={v['pearson_r']:+.3f} "
              f"(p={v['pearson_p']:.2g})  Spearman rho={v['spearman_rho']:+.3f} "
              f"(p={v['spearman_p']:.2g})")

    print("\n=== frontier five: paired over shared subdomains ===")
    print(f"{'model':<18}{'n':>4}{'Static':>8}{'Active':>8}{'Gain':>8}{'p':>10}{'%subs+':>8}")
    for m in sorted(fr, key=lambda m: -per_model[m]["active"]):
        r = per_model[m]
        print(f"{r['short']:<18}{r['n_shared_subdomains']:>4}{r['static']:>8.2f}"
              f"{r['active']:>8.2f}{r['gain']:>+8.2f}{r['wilcoxon_p']:>10.3g}"
              f"{100*r['frac_subdomains_improved']:>7.0f}%")

    print("\n=== per-dimension Active-minus-Static ===")
    print(f"{'dim':<14}{'frontier5':>12}{'open28':>10}{'gemini5':>10}")
    for d in DIMS:
        print(f"{d:<14}{per_dim['frontier2026'][d]['mean_gain']:>+12.2f}"
              f"{per_dim['open_weight_28'][d]['mean_gain']:>+10.2f}"
              f"{per_dim['gemini_5'][d]['mean_gain']:>+10.2f}")

    print("\n=== ceiling ===")
    for k, v in ceiling.items():
        print(f"  {k:<38} {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
