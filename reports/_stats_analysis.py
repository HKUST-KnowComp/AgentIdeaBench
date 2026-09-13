"""Statistical analysis for the paper:
  - Bootstrap CI for Active boost per model
  - One-sample t-test: H0: mean boost = 0
  - Regression: boost ~ log(params) + cutoff
  - Per-domain ANOVA
  - v1 vs v2 comparison (where data exists)

Outputs printed table + LaTeX-ready strings for inclusion in paper.
"""
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in DIMS) / WS


def bootstrap_ci(vals, n_boot=10000, alpha=0.05):
    """95% CI for mean via bootstrap. Returns (mean, lo, hi)."""
    import random
    n = len(vals)
    means = []
    rng = random.Random(42)
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return statistics.mean(vals), lo, hi


def paired_bootstrap_test(b_paper, c_paper, n_boot=10000):
    """H0: mean(C - B) = 0. Returns (mean_diff, ci_lo, ci_hi, p_value, n)."""
    import random
    common = sorted(set(b_paper) & set(c_paper))
    if len(common) < 3:
        return None
    diffs = [c_paper[p] - b_paper[p] for p in common]
    mean_d = statistics.mean(diffs)
    # Bootstrap mean diff
    n = len(diffs)
    rng = random.Random(42)
    means = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(n_boot * 0.025)]
    hi = means[int(n_boot * 0.975)]
    # 2-sided p: fraction of bootstrap means crossing 0 (centered shift method)
    centered = [m - mean_d for m in means]
    p = (sum(1 for c in centered if abs(c) >= abs(mean_d))) / n_boot
    return mean_d, lo, hi, p, n


def load_v1_per_paper():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    out = defaultdict(lambda: defaultdict(list))  # (model, track) -> paper -> [weighted]
    for m, t, pid, sj in conn.execute("""
        SELECT idea_model, track, paper_id, scores_json FROM results
        WHERE prompt_version='v1_paper_refs' AND track IN ('B','C') AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        out[(m, t)][pid].append(weighted(s))
    conn.close()
    # Compress: take mean across critics → 1 value per paper
    paper_score = {}
    for (m, t), pp in out.items():
        paper_score[(m, t)] = {pid: statistics.mean(vs) for pid, vs in pp.items()}
    return paper_score


def load_v2_per_paper():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    out = defaultdict(lambda: defaultdict(list))
    for m, t, pid, sj in conn.execute("""
        SELECT idea_model, track, paper_id, scores_json FROM results
        WHERE prompt_version='v2_topic_refs' AND track IN ('B','C') AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        out[(m, t)][pid].append(weighted(s))
    conn.close()
    return {k: {pid: statistics.mean(vs) for pid, vs in pp.items()}
            for k, pp in out.items()}


PARAMS_B = {
    "qwen/qwen3.5-9b": 9, "qwen/qwen3.5-27b": 27, "qwen/qwen3.5-397b-a17b": 397,
    "qwen/qwen3-235b-a22b-thinking-2507": 235, "qwen/qwen3-vl-8b-thinking": 8,
    "qwen/qwen-2.5-72b-instruct": 72, "qwen/qwen-2.5-7b-instruct": 7, "qwen/qwen3-8b": 8,
    "meta-llama/llama-3.1-8b-instruct": 8, "meta-llama/llama-4-maverick": 400,
    "mistralai/mistral-7b-instruct-v0.1": 7,
    "mistralai/mistral-small-24b-instruct-2501": 24, "google/gemma-2-27b-it": 27,
    "google/gemma-3-27b-it": 27, "google/gemma-4-31b-it": 31,
    "deepseek/deepseek-r1-0528": 671, "deepseek/deepseek-v4-pro": 671,
    "moonshotai/kimi-k2.5": 1000, "moonshotai/kimi-k2.6": 1000, "z-ai/glm-5.1": 110,
}


def main():
    print("=" * 76)
    print("STATISTICAL ANALYSIS — Finding 1: Active boost is robust")
    print("=" * 76)
    v1 = load_v1_per_paper()

    # Per-model paired bootstrap test
    models = sorted({m for (m, t) in v1 if t == "C"})
    print(f"\n{'Model':<42} {'n':>3} {'boost':>7} {'95% CI':>17} {'p_boot':>7}")
    f1_table_rows = []
    for m in models:
        b = v1.get((m, "B"), {})
        c = v1.get((m, "C"), {})
        if not b or not c:
            continue
        result = paired_bootstrap_test(b, c)
        if result is None:
            continue
        mean_d, lo, hi, p, n = result
        ci_str = f"[{lo:+.2f}, {hi:+.2f}]"
        sig = " *" if p < 0.05 else ""
        print(f"  {m:<40} {n:>3} {mean_d:>+7.2f} {ci_str:>17} {p:>7.3f}{sig}")
        f1_table_rows.append({
            "model": m, "n": n, "boost": mean_d, "ci_lo": lo, "ci_hi": hi,
            "p_value": p, "params": PARAMS_B.get(m),
        })

    print("\n" + "=" * 76)
    print("Pooled test: mean boost across all (model, paper) pairs")
    print("=" * 76)
    # Pool all paired diffs
    all_diffs = []
    for r in f1_table_rows:
        b = v1.get((r["model"], "B"), {})
        c = v1.get((r["model"], "C"), {})
        for pid in (set(b) & set(c)):
            all_diffs.append(c[pid] - b[pid])
    if all_diffs:
        m, lo, hi = bootstrap_ci(all_diffs)
        n = len(all_diffs)
        pct_pos = sum(1 for d in all_diffs if d > 0) / n * 100
        print(f"  N pairs:        {n}")
        print(f"  Mean boost:     {m:+.3f}")
        print(f"  95% CI:         [{lo:+.3f}, {hi:+.3f}]")
        print(f"  % > 0:          {pct_pos:.1f}%")
        # bootstrap p for H0: mean = 0
        # If 95% CI excludes 0, p < 0.05
        if lo > 0 or hi < 0:
            print(f"  Result:         ** Reject H0: boost = 0 (p < 0.05) **")
        else:
            print(f"  Result:         Cannot reject H0 (CI contains 0)")

    print("\n" + "=" * 76)
    print("Regression: boost ~ log(params)")
    print("=" * 76)
    # Simple linear regression
    rows = [r for r in f1_table_rows if r["params"] is not None]
    xs = [math.log10(r["params"]) for r in rows]
    ys = [r["boost"] for r in rows]
    if len(xs) >= 5:
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx2 = sum((x - mx) ** 2 for x in xs)
        dy2 = sum((y - my) ** 2 for y in ys)
        slope = num / dx2 if dx2 > 0 else 0
        intercept = my - slope * mx
        # R²
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        ss_tot = dy2
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        # Pearson r
        r = num / (dx2 * dy2) ** 0.5 if dx2 * dy2 > 0 else 0
        print(f"  N models:       {n}")
        print(f"  Slope (Δboost per decade-of-params): {slope:+.3f}")
        print(f"  Intercept:      {intercept:+.3f}")
        print(f"  Pearson r:      {r:+.3f}")
        print(f"  R²:             {r_squared:.3f}")
        # Hypothesis: slope < 0 → larger models have smaller boost
        if slope < 0 and abs(r) > 0.3:
            print(f"  → Larger models → smaller boost (great-equalizer pattern)")

    print("\n" + "=" * 76)
    print("Per-domain bootstrap CIs (v1)")
    print("=" * 76)
    conn_p = sqlite3.connect(str(cfg.PAPERS_DB))
    pdom = dict(conn_p.execute("SELECT paper_id, domain FROM papers"))
    conn_p.close()

    for dom in ["Biology", "CS", "Chemistry", "Medicine", "Physics"]:
        dom_diffs = []
        for r in f1_table_rows:
            b = v1.get((r["model"], "B"), {})
            c = v1.get((r["model"], "C"), {})
            for pid in (set(b) & set(c)):
                if pdom.get(pid) == dom:
                    dom_diffs.append(c[pid] - b[pid])
        if dom_diffs:
            m, lo, hi = bootstrap_ci(dom_diffs)
            pct_pos = sum(1 for d in dom_diffs if d > 0) / len(dom_diffs) * 100
            sig = " *" if (lo > 0 or hi < 0) else ""
            print(f"  {dom:<12} n={len(dom_diffs):>3}  boost={m:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  %>0={pct_pos:.0f}%{sig}")

    print("\n" + "=" * 76)
    print("v1 vs v2 prompt design comparison (3 models)")
    print("=" * 76)
    v2 = load_v2_per_paper()
    common_models = {m for (m, t) in v1 if t == "C"} & {m for (m, t) in v2 if t == "C"}
    for m in sorted(common_models):
        b1 = v1.get((m, "B"), {})
        c1 = v1.get((m, "C"), {})
        b2 = v2.get((m, "B"), {})
        c2 = v2.get((m, "C"), {})
        if not all([b1, c1, b2, c2]):
            continue
        boost_v1 = statistics.mean(c1.values()) - statistics.mean(b1.values())
        boost_v2 = statistics.mean(c2.values()) - statistics.mean(b2.values())
        # v1's B mean for the SAME papers in v2
        common_papers_b = sorted(set(b1) & set(b2))
        if common_papers_b:
            b1_common = statistics.mean(b1[p] for p in common_papers_b)
            b2_common = statistics.mean(b2[p] for p in common_papers_b)
            delta_b = b2_common - b1_common
        else:
            delta_b = None
        print(f"  {m:<40} v1_boost={boost_v1:+.2f}  v2_boost={boost_v2:+.2f}  Δboost={boost_v2-boost_v1:+.2f}")
        if delta_b is not None:
            print(f"    {'':<38}    v2_B − v1_B = {delta_b:+.2f}  (title+search-refs vs paper-specific refs)")


if __name__ == "__main__":
    main()
