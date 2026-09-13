#!/usr/bin/env python3
"""
E40 -- dump the per-discipline critic-validity numbers to JSON (for the appendix
figure).

The domain-validity paragraph of the paper (Appendix "Critic design and
calibration") reports top-10 anchor minus model gaps per discipline. Those
numbers are printed by `e18_multidomain_anchors.py --analyze` but were never
persisted, so a figure would have to hardcode them. This script reuses e18's own
aggregation helpers and writes the same table to JSON, plus a reproduction check
against the values quoted in the paper.

Read-only (SELECT only, no API). Writes reports/e40_domain_validity.json.
Nothing runs on import.
"""
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e18_multidomain_anchors import (  # noqa: E402
    _wit, DOMAINS, CS_CANDS,
)

OUT = ROOT / "reports" / "e40_domain_validity.json"
TOPN = 10
PUBLISHED_GAP = {"CS": 1.50, "Physics": 0.88, "Medicine": 0.43,
                 "Chemistry": 0.38, "Biology": 0.28}


def main():
    wit, grp, dom = _wit()
    pools = {"CS": sorted([wit[i] for i in CS_CANDS if i in wit], reverse=True)[:TOPN]}
    for d in DOMAINS:
        pools[d] = sorted([wit[i] for i in wit if grp[i] == f"domanchor:{d}"],
                          reverse=True)[:TOPN]

    rows = {}
    for d, pool in pools.items():
        model = [wit[i] for i in wit
                 if grp[i].startswith("deepseek-v4") and grp[i].endswith("-C")
                 and dom[i] == d]
        weak = [wit[i] for i in wit if grp[i].startswith("weakprobe") and dom[i] == d]
        rows[d] = {
            "n_anchors": len(pool),
            "anchor_top10_mean": round(float(np.mean(pool)), 4) if pool else None,
            "n_model_items": len(model),
            "model_active_mean": round(float(np.mean(model)), 4) if model else None,
            "model_active_max": round(float(np.max(model)), 4) if model else None,
            "n_weak_items": len(weak),
            "weak_control_mean": round(float(np.mean(weak)), 4) if weak else None,
            "gap_anchor_minus_model": (round(float(np.mean(pool) - np.mean(model)), 4)
                                       if pool and model else None),
        }

    checks = {d: {"published": PUBLISHED_GAP[d],
                  "recomputed": rows[d]["gap_anchor_minus_model"],
                  # <= because a value such as 0.875 rounds to the published 0.88
                  "matches_2dp": abs(PUBLISHED_GAP[d]
                                     - rows[d]["gap_anchor_minus_model"]) <= 0.005 + 1e-9}
              for d in PUBLISHED_GAP}

    out = {
        "generated_by": "experiments/e40_domain_validity_json.py",
        "note": ("Per-discipline lit8d validity. Anchors = top-10 rewritten landmark "
                 "papers of that discipline; model = deepseek-v4 Active items scored in "
                 "the same e12_gap_scores pass; weak = weak-probe control generators. "
                 "Weighted total under the production weighting. Aggregation reuses "
                 "e18_multidomain_anchors._wit (drop-highest trimmed mean over critics). "
                 "Chemistry has no weak-control items."),
        "topn": TOPN,
        "per_discipline": rows,
        "reproduction_check_vs_paper": checks,
    }
    OUT.write_text(json.dumps(out, indent=1))

    for d in ["CS", "Physics", "Medicine", "Chemistry", "Biology"]:
        r = rows[d]
        print(f"{d:<10} anchor {r['anchor_top10_mean']:.2f}  model "
              f"{r['model_active_mean']:.2f}  weak "
              f"{r['weak_control_mean'] if r['weak_control_mean'] is None else round(r['weak_control_mean'], 2)}"
              f"  gap {r['gap_anchor_minus_model']:+.2f}  "
              f"(paper {PUBLISHED_GAP[d]:+.2f}, match={checks[d]['matches_2dp']})")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
