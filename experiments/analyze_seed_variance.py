"""
Analyze seed-variance results.

Key ratio: |397b_mean - 9b_mean| / within_model_sigma (per paper).
  > 2  -> signal clearly exceeds generation noise, scaling papers helps
  1-2  -> marginal, needs more seeds to tighten SE
  < 1  -> within-model seed variance dominates, rubric/critic changes needed

Also reports weighted score W = O*2 + F + C*0.5 + I*1.5 + S*0.5 (matches prod).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "experiments" / "seed_variance_results.json"

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
WEIGHTS = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
           "impact": 1.5, "specificity": 0.5}


def weighted(scores: dict) -> float:
    return sum(WEIGHTS[d] * scores[d]["score"] for d in DIMS)


def simple_mean(scores: dict) -> float:
    return sum(scores[d]["score"] for d in DIMS) / len(DIMS)


def stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    with open(RESULTS) as f:
        data = json.load(f)
    runs = data["runs"]

    # Per-run weighted + mean + per-dim scores
    rows = []
    for r in runs:
        if not r.get("scores"):
            continue
        s = r["scores"]
        rows.append({
            "model": r["model"],
            "paper_id": r["paper_id"],
            "domain": r["domain"],
            "seed": r["seed"],
            "W": weighted(s),
            "mean": simple_mean(s),
            "O": s["originality"]["score"],
            "F": s["feasibility"]["score"],
            "C": s["clarity"]["score"],
            "I": s["impact"]["score"],
            "S": s["specificity"]["score"],
            "words": len(r["hypothesis"].split()),
        })

    print(f"\n=== Raw runs ({len(rows)}) ===")
    print(f"{'Model':<30} {'Paper/Domain':<12} {'Seed':>5}  {'W':>6}  {'Mean':>5}  {'O F C I S':<12}  {'Wds':>4}")
    for r in sorted(rows, key=lambda x: (x["model"], x["domain"], x["seed"])):
        tag = r["model"].split("/")[-1]
        print(f"{tag:<30} {r['domain']:<12} {r['seed']:>5}  {r['W']:>6.2f}  {r['mean']:>5.2f}  "
              f"{r['O']} {r['F']} {r['C']} {r['I']} {r['S']}     {r['words']:>4}")

    # Aggregate per (model, paper)
    per = defaultdict(list)
    for r in rows:
        per[(r["model"], r["paper_id"], r["domain"])].append(r["W"])

    print(f"\n=== Per (model, paper): mean ± sigma across seeds ===")
    print(f"{'Model':<30} {'Domain':<12}  n  mean_W   sigma")
    for (m, pid, dom), vals in sorted(per.items()):
        mu = sum(vals) / len(vals)
        sd = stddev(vals)
        tag = m.split("/")[-1]
        print(f"{tag:<30} {dom:<12}  {len(vals)}  {mu:>6.2f}  {sd:>6.2f}")

    # Ratio per paper
    print(f"\n=== Signal vs noise (per paper) ===")
    print(f"{'Domain':<12}  gap=|397b-9b|   pooled_sigma   ratio")
    models = sorted({m for (m, _, _) in per.keys()})
    assert len(models) == 2, "expected exactly 2 models"
    big, small = sorted(models, key=lambda m: 0 if "397" in m else 1)

    pooled_gaps = []
    pooled_sigmas = []
    for (_, pid, dom), _ in list(per.items())[:]:
        pass  # iterate papers

    papers_seen = set()
    for (_, pid, dom) in per.keys():
        if pid in papers_seen:
            continue
        papers_seen.add(pid)
        big_vals = per.get((big, pid, dom), [])
        small_vals = per.get((small, pid, dom), [])
        if not big_vals or not small_vals:
            continue
        gap = abs(sum(big_vals)/len(big_vals) - sum(small_vals)/len(small_vals))
        sd_big = stddev(big_vals)
        sd_small = stddev(small_vals)
        pooled = math.sqrt((sd_big**2 + sd_small**2) / 2) if (sd_big or sd_small) else 0.0
        ratio = gap / pooled if pooled > 1e-9 else float("inf")
        pooled_gaps.append(gap)
        pooled_sigmas.append(pooled)
        print(f"{dom:<12}  {gap:>6.2f}          {pooled:>6.2f}         {ratio:>5.2f}")

    if pooled_gaps:
        mean_gap = sum(pooled_gaps) / len(pooled_gaps)
        mean_sd = sum(pooled_sigmas) / len(pooled_sigmas)
        overall = mean_gap / mean_sd if mean_sd > 1e-9 else float("inf")
        print(f"\nMean gap = {mean_gap:.2f}, mean sigma = {mean_sd:.2f}, overall ratio = {overall:.2f}")
        print("Interpretation:")
        if overall > 2:
            print(f"  ratio > 2 : signal > 2*noise. Scaling papers tightens SE, "
                  "model differences are real.")
        elif overall > 1:
            print(f"  1 < ratio <= 2: marginal. More seeds or papers needed.")
        else:
            print(f"  ratio <= 1: within-model seed noise dominates. "
                  "Critic/rubric must change before scaling.")


if __name__ == "__main__":
    main()
