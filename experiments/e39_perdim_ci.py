#!/usr/bin/env python3
"""
E39 -- per-dimension Active-Static deltas with uncertainty (for paper Figure 4A).

The paper reports the five per-dimension deltas as point estimates only
(feasibility +1.21, clarity +0.63, specificity +0.58, impact +0.23,
originality -0.14 n.s.). A forest plot needs an interval per dimension, so this
script recomputes the per-model per-dimension Static and Active means with the
*same* aggregation as e22/e36 (trimmed mean over critics = drop-highest-of-three
-> per-(model,track,subdomain,idx) dim value -> subdomain mean -> mean over
subdomains), then treats the model as the sampling unit (n=28 open-weight paired)
and adds a paired bootstrap CI, a Wilcoxon signed-rank test and a sign test.

Correctness check: the per-model mean delta must reproduce the published point
estimates to two decimals, and originality's improver count must be 15/28.

Read-only (SELECT only). Writes reports/e39_perdim_ci.json (new file).
Nothing runs on import.
"""
import sys
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from utils.constants import SCORING_DIMS as DIMS  # noqa: E402
from experiments.e10_idea_anchor_calibration import trimmed_mean  # noqa: E402

OUT = ROOT / "reports" / "e39_perdim_ci.json"
SEED, NBOOT = 42, 5000

# same roster rules as e36: Active-only reasoning models have no Static path,
# Gemini is the held-out closed family and is excluded from headline stats.
ACTIVE_ONLY = {"qwen/qwen3-235b-a22b-thinking-2507", "qwen/qwen3-vl-8b-thinking"}
PUBLISHED = {"feasibility": 1.21, "clarity": 0.63, "specificity": 0.58,
             "impact": 0.23, "originality": -0.14}


def per_model_dims():
    """{(model, track): {dim: mean over subdomains}} from lit8d_scores_3seed."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    conn.row_factory = sqlite3.Row
    idea = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT idea_model,track,subdomain,idea_index,scores_json "
                          "FROM lit8d_scores_3seed WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"])
        k = (r["idea_model"], r["track"], r["subdomain"], r["idea_index"])
        for d in DIMS:
            idea[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()

    def tm(vs):
        return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)

    idx_dim = {k: {d: tm(idea[k][d]) for d in DIMS} for k in idea}
    sub = defaultdict(lambda: defaultdict(list))          # (m,tr,subd) -> dim -> vals
    for (m, tr, subd, idx) in idea:
        for d in DIMS:
            sub[(m, tr, subd)][d].append(idx_dim[(m, tr, subd, idx)][d])
    mt = defaultdict(lambda: defaultdict(list))           # (m,tr) -> dim -> subd means
    for (m, tr, subd) in sub:
        for d in DIMS:
            mt[(m, tr)][d].append(float(np.mean(sub[(m, tr, subd)][d])))
    return {k: {d: float(np.mean(v[d])) for d in DIMS} for k, v in mt.items()}


def main():
    mt = per_model_dims()
    models = sorted({m for (m, _) in mt})
    paired = [m for m in models
              if (m, "B") in mt and (m, "C") in mt
              and m not in ACTIVE_ONLY and not m.startswith("google/gemini")]

    rng = np.random.default_rng(SEED)
    per_dim, per_model = {}, {}
    for m in paired:
        per_model[m] = {d: round(mt[(m, "C")][d] - mt[(m, "B")][d], 4) for d in DIMS}

    for d in DIMS:
        delta = np.array([mt[(m, "C")][d] - mt[(m, "B")][d] for m in paired])
        boots = np.array([rng.choice(delta, len(delta), replace=True).mean()
                          for _ in range(NBOOT)])
        n_pos = int((delta > 0).sum())
        per_dim[d] = {
            "n_models": len(delta),
            "static_mean": round(float(np.mean([mt[(m, "B")][d] for m in paired])), 4),
            "active_mean": round(float(np.mean([mt[(m, "C")][d] for m in paired])), 4),
            "delta_mean": round(float(delta.mean()), 4),
            "delta_median": round(float(np.median(delta)), 4),
            "delta_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                           round(float(np.percentile(boots, 97.5)), 4)],
            "wilcoxon_p": round(float(stats.wilcoxon(delta).pvalue), 4),
            "n_improved": n_pos,
            "sign_test_p": round(float(stats.binomtest(n_pos, len(delta), 0.5).pvalue), 4),
        }

    checks = {d: {"published": PUBLISHED[d],
                  "recomputed": per_dim[d]["delta_mean"],
                  "abs_diff": round(abs(PUBLISHED[d] - per_dim[d]["delta_mean"]), 4),
                  "matches_2dp": abs(PUBLISHED[d] - per_dim[d]["delta_mean"]) < 0.005}
              for d in DIMS}

    out = {
        "generated_by": "experiments/e39_perdim_ci.py",
        "note": ("Per-dimension Active-Static deltas. Unit = model (n=28 open-weight "
                 "paired); aggregation identical to e22/e36. CI = paired bootstrap over "
                 "models, 5000 resamples, seed 42. Read-only recomputation; adds "
                 "uncertainty to the point estimates already reported in the paper."),
        "n_models": len(paired),
        "models": paired,
        "per_dimension": per_dim,
        "per_model_delta": per_model,
        "reproduction_check_vs_paper": checks,
    }
    OUT.write_text(json.dumps(out, indent=1))

    for d in DIMS:
        s = per_dim[d]
        print(f"{d:<13} {s['delta_mean']:+.2f}  CI[{s['delta_ci95'][0]:+.2f},"
              f"{s['delta_ci95'][1]:+.2f}]  p={s['wilcoxon_p']:.3f}  "
              f"improved {s['n_improved']}/{s['n_models']}  "
              f"sign p={s['sign_test_p']:.2f}  (paper {PUBLISHED[d]:+.2f}, "
              f"match={checks[d]['matches_2dp']})")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
