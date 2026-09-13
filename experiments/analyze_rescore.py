"""
Compare three conditions on the 18 seed-variance hypotheses:
  A = gemini-2.5-flash + old rubric   (from seed_variance_results.json)
  B = gemini-2.5-flash + new rubric   (from seed_variance_rescored.json)
  C = openai/gpt-5     + new rubric   (from seed_variance_rescored.json)

For each, compute per (model, domain) mean W, within-model sigma,
between-model gap, ratio. Then inspect the 9b Physics seed=1337 outlier.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
ORIG = ROOT / "experiments" / "seed_variance_results.json"
RESC = ROOT / "experiments" / "seed_variance_rescored.json"

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5, "impact": 1.5, "specificity": 0.5}


def weighted(s): return sum(W[d] * s[d]["score"] for d in DIMS)


def stddev(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def flatten(runs, tag):
    """Return list of dicts with (tag, model, domain, paper_id, seed, W, scores)."""
    out = []
    for r in runs:
        if not r.get("scores"):
            continue
        s = r["scores"]
        out.append({
            "tag": tag,
            "model": r["model"],
            "domain": r["domain"],
            "paper_id": r["paper_id"],
            "seed": r["seed"],
            "W": weighted(s),
            "scores": s,
            "factual_check": r.get("factual_check"),
        })
    return out


def agg(rows):
    """Return (model, domain) -> (mean, sigma, n)."""
    per = defaultdict(list)
    for r in rows:
        per[(r["model"], r["domain"])].append(r["W"])
    return {k: (sum(v)/len(v), stddev(v), len(v)) for k, v in per.items()}


def signal_noise(rows):
    per = agg(rows)
    models = sorted({m for (m, _) in per.keys()})
    if len(models) != 2:
        return None
    big, small = sorted(models, key=lambda m: 0 if "397" in m else 1)
    domains = sorted({d for (_, d) in per.keys()})
    rows_out = []
    for dom in domains:
        if (big, dom) not in per or (small, dom) not in per:
            continue
        bmu, bsd, _ = per[(big, dom)]
        smu, ssd, _ = per[(small, dom)]
        gap = bmu - smu  # positive = 397b > 9b (as expected)
        pooled = math.sqrt((bsd**2 + ssd**2) / 2) if (bsd or ssd) else 0.0
        ratio = abs(gap) / pooled if pooled > 1e-9 else float("inf")
        rows_out.append({"domain": dom, "big_mu": bmu, "small_mu": smu,
                         "gap": gap, "pooled_sigma": pooled, "ratio": ratio})
    return rows_out


def main():
    orig = json.load(open(ORIG))
    resc = json.load(open(RESC))

    A = flatten(orig["runs"], "A")
    B = flatten([r for r in resc["runs"] if r["condition"] == "B_flash_newrubric"], "B")
    C = flatten([r for r in resc["runs"] if r["condition"] == "C_gpt5_newrubric"], "C")

    print(f"A (gemini-flash + old rubric): {len(A)} rows")
    print(f"B (gemini-flash + new rubric): {len(B)} rows")
    print(f"C (gpt-5 + new rubric):        {len(C)} rows")

    for tag, rows in [("A", A), ("B", B), ("C", C)]:
        print(f"\n=== Condition {tag}: signal vs noise ===")
        sn = signal_noise(rows)
        if not sn:
            print("  (insufficient data)")
            continue
        print(f"  {'Domain':<10} {'397b_W':>7} {'9b_W':>7} {'gap':>6} {'pooled_σ':>9} {'ratio':>6}")
        for r in sn:
            print(f"  {r['domain']:<10} {r['big_mu']:>7.2f} {r['small_mu']:>7.2f} "
                  f"{r['gap']:>+6.2f} {r['pooled_sigma']:>9.2f} {r['ratio']:>6.2f}")
        if sn:
            mean_gap = sum(abs(r["gap"]) for r in sn) / len(sn)
            mean_sig = sum(r["pooled_sigma"] for r in sn) / len(sn)
            overall = mean_gap / mean_sig if mean_sig > 1e-9 else float("inf")
            print(f"  overall: mean |gap|={mean_gap:.2f}, mean σ={mean_sig:.2f}, ratio={overall:.2f}")

    # Outlier inspection: 9b Physics seed=1337
    print(f"\n=== Outlier: 9b Physics seed=1337 under each condition ===")
    for tag, rows in [("A", A), ("B", B), ("C", C)]:
        for r in rows:
            if "9b" in r["model"] and r["domain"] == "Physics" and r["seed"] == 1337:
                s = r["scores"]
                print(f"  [{tag}] O={s['originality']['score']} F={s['feasibility']['score']} "
                      f"C={s['clarity']['score']} I={s['impact']['score']} "
                      f"S={s['specificity']['score']}  W={r['W']:.1f}")
                if r.get("factual_check"):
                    print(f"       factual_check: {json.dumps(r['factual_check'])[:250]}")

    # Direction correctness: per-paper, does 397b > 9b?
    print(f"\n=== Direction: 397b > 9b per paper? ===")
    for tag, rows in [("A", A), ("B", B), ("C", C)]:
        by_paper = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by_paper[r["domain"]][r["model"]].append(r["W"])
        print(f"  [{tag}]")
        for dom in sorted(by_paper.keys()):
            vals = by_paper[dom]
            keys = sorted(vals.keys())
            if len(keys) < 2:
                continue
            big = next((k for k in keys if "397" in k), None)
            small = next((k for k in keys if "9b" in k), None)
            if not big or not small:
                continue
            bmu = sum(vals[big])/len(vals[big])
            smu = sum(vals[small])/len(vals[small])
            arrow = "✓" if bmu > smu else "✗"
            print(f"    {dom:<10} 397b={bmu:.2f}  9b={smu:.2f}  {arrow}")


if __name__ == "__main__":
    main()
