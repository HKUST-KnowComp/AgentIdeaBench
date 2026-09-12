"""F0 partial correlations: (Static score, cutoff, log10 params) on 28 models.

Computes:
  partial r(cutoff, score | log_params)
  partial r(log_params, score | cutoff)
  r(cutoff, log_params)        — to verify independence

Uses corrected KNOWLEDGE_CUTOFFS dict (post 2026-05-19 user correction).
"""
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from reports._make_cross_year_plot import KNOWLEDGE_CUTOFFS

# Param sizes (billions, dense or active-experts; see _stats_analysis.py)
PARAMS = {
    "openai/gpt-4": 1760, "openai/gpt-4o": 200, "openai/gpt-4o-mini": 8,
    "openai/o1": 200, "openai/gpt-4.1": 200, "openai/gpt-5": 1700,
    "openai/gpt-5.4-mini": 8, "openai/gpt-5.4-nano": 3, "openai/gpt-5.5": 1700,
    "anthropic/claude-3.7-sonnet": 175, "anthropic/claude-sonnet-4": 200,
    "anthropic/claude-sonnet-4.6": 200,
    "google/gemini-2.5-pro": 200, "google/gemma-2-27b-it": 27,
    "google/gemma-3-27b-it": 27, "google/gemma-4-31b-it": 31,
    "meta-llama/llama-3.1-8b-instruct": 8, "meta-llama/llama-4-maverick": 400,
    "mistralai/mistral-7b-instruct-v0.1": 7,
    "mistralai/mistral-small-24b-instruct-2501": 24,
    "mistralai/mistral-small-2603": 24,
    "qwen/qwen-2.5-72b-instruct": 72, "qwen/qwen-2.5-7b-instruct": 7,
    "qwen/qwen3-8b": 8, "qwen/qwen3-32b": 32, "qwen/qwen3-vl-8b-thinking": 8,
    "qwen/qwen3-235b-a22b-thinking-2507": 235,
    "qwen/qwen3.5-9b": 9, "qwen/qwen3.5-27b": 27, "qwen/qwen3.5-397b-a17b": 397,
    "deepseek/deepseek-r1-0528": 671, "deepseek/deepseek-v4-pro": 671,
    "moonshotai/kimi-k2.5": 1000, "moonshotai/kimi-k2.6": 1000,
    "z-ai/glm-5.1": 110, "xiaomi/mimo-v2.5": 7,
    # 2026-05-19 plateau-extension batch — only models with OFFICIALLY DISCLOSED params
    "qwen/qwen3-30b-a3b-instruct-2507": 30,     # 30B-A3B per Qwen official
    "z-ai/glm-4.5-air": 12,                     # GLM-4.5-Air 12B per Z.AI official
    # NOTE: mistral-medium-3.1 / devstral-medium excluded — Mistral has not
    #       disclosed param counts officially; using estimates would weaken
    #       regression honesty. Their Static mean scores are still in
    #       saturation_analysis (no param required).
}

W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in W) / WS


def load_static_per_model():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    per = defaultdict(list)
    for m, sj in conn.execute("""
        SELECT idea_model, scores_json FROM results
        WHERE track='B' AND prompt_version='v1_paper_refs'
          AND critic_model != '' AND scores_json IS NOT NULL
          AND idea_model NOT LIKE 'baseline/%'
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        per[m].append(weighted(s))
    conn.close()
    return {m: statistics.mean(v) for m, v in per.items() if len(v) >= 5}


def pearson(x, y):
    x = np.array(x); y = np.array(y)
    return float(np.corrcoef(x, y)[0, 1])


def partial_corr(x, y, z):
    """partial r(x, y | z)"""
    r_xy = pearson(x, y)
    r_xz = pearson(x, z)
    r_yz = pearson(y, z)
    denom = math.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom == 0:
        return None
    return (r_xy - r_xz * r_yz) / denom


def main():
    static_scores = load_static_per_model()
    rows = []
    for m, score in static_scores.items():
        if m not in KNOWLEDGE_CUTOFFS or m not in PARAMS:
            continue
        rows.append({
            "model": m,
            "score": score,
            "cutoff": KNOWLEDGE_CUTOFFS[m],
            "log_params": math.log10(PARAMS[m]),
        })
    rows.sort(key=lambda r: r["cutoff"])

    print(f"F0 partial correlations  (n={len(rows)} models)\n")
    print(f"{'Model':<46} {'score':>6} {'cutoff':>7} {'log10P':>7}")
    print("-" * 70)
    for r in rows:
        print(f"  {r['model']:<44} {r['score']:>6.2f} {r['cutoff']:>7.2f} {r['log_params']:>7.2f}")

    xs = [r["cutoff"] for r in rows]
    ys = [r["score"] for r in rows]
    zs = [r["log_params"] for r in rows]

    print(f"\n{'='*70}\nCorrelations:\n{'='*70}")
    print(f"  r(cutoff, score)         = {pearson(xs, ys):+.3f}")
    print(f"  r(log_params, score)     = {pearson(zs, ys):+.3f}")
    print(f"  r(cutoff, log_params)    = {pearson(xs, zs):+.3f}")
    print()
    print(f"  partial r(cutoff, score | log_params)    = {partial_corr(xs, ys, zs):+.3f}")
    print(f"  partial r(log_params, score | cutoff)    = {partial_corr(zs, ys, xs):+.3f}")

    # Linear fit too
    print(f"\nLinear fits:")
    sl, ic = np.polyfit(xs, ys, 1)
    print(f"  score = {sl:+.3f} * cutoff + {ic:+.2f}    (slope per yr)")
    sl, ic = np.polyfit(zs, ys, 1)
    print(f"  score = {sl:+.3f} * log10(params) + {ic:+.2f}    (slope per decade)")

    out_path = ROOT / "reports" / "f0_partial_correlation.json"
    json.dump({
        "n_models": len(rows),
        "models": [r["model"] for r in rows],
        "r_cutoff_score": pearson(xs, ys),
        "r_logparams_score": pearson(zs, ys),
        "r_cutoff_logparams": pearson(xs, zs),
        "partial_r_cutoff_score_given_logparams": partial_corr(xs, ys, zs),
        "partial_r_logparams_score_given_cutoff": partial_corr(zs, ys, xs),
        "rows": rows,
    }, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
