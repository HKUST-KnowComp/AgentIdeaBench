"""E41 claim check — re-test each headline claim of the paper on the extended
38-model roster, against the same claim computed on the paper's own 28.

Read-only. Source = reports/e36_leaderboard_subscores.json (model-level totals,
produced by the untouched e36 aggregation). Writes reports/e41_claim_check.json.

Claims tested (paper wording in the `claim` field of each block):
  C1 capability gate      r(Static, gain)
  C2 gap widens w/ capability   quartile-mean gain by Static quartile
  C3 Active restores discrimination   var(Active)/var(Static) across models
  C4 Static compresses the top end    top-8 spread, Static vs Active
  C5 mechanism            per-dimension gain ordering

  /opt/homebrew/Caskroom/miniforge/base/bin/python experiments/e41_claim_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

E36 = ROOT / "reports" / "e36_leaderboard_subscores.json"
E41 = ROOT / "reports" / "e41_frontier_analysis.json"
OUT = ROOT / "reports" / "e41_claim_check.json"

FRONTIER = {"azure/openai/gpt-5.6-sol", "azure/openai/gpt-5.6-terra",
            "azure/openai/gpt-5.6-luna", "azure/anthropic/claude-opus-5",
            "azure/anthropic/claude-sonnet-5"}


def rosters(agg):
    """Roster keys carry their OWN size, derived from the data.

    They used to be hard-coded as open28/plus_gemini33/all38. Once E42 landed
    26 more closed routes those labels silently described 28/59/64 models, so
    every C2/C3/C4 number printed under "all38" was really an all-64 number.
    Sizes are now computed, never asserted.
    """
    ow = [m for m in agg if not agg[m]["closed"]]
    closed = [m for m in agg if agg[m]["closed"] and m not in FRONTIER]
    fr = [m for m in agg if m in FRONTIER]
    return {f"open{len(ow)}": ow,
            f"plus_closed{len(ow) + len(closed)}": ow + closed,
            f"all{len(ow) + len(closed) + len(fr)}": ow + closed + fr}


def gate_r(agg, ms):
    """C1 computed live: Pearson r between Static ability and Active gain."""
    b = [agg[m]["static_total"] for m in ms]
    g = [agg[m]["gain"] for m in ms]
    r, p = stats.pearsonr(b, g)
    rho, rp = stats.spearmanr(b, g)
    return {"n": len(ms), "pearson_r": float(r), "pearson_p": float(p),
            "spearman_rho": float(rho), "spearman_p": float(rp)}


def quartile_gains(agg, ms):
    """Mean gain within each quartile of Static ability (paper's gate figure)."""
    rows = sorted(ms, key=lambda m: agg[m]["static_total"])
    q = np.array_split(rows, 4)
    return [{"n": len(part),
             "static_range": [float(agg[part[0]]["static_total"]),
                              float(agg[part[-1]]["static_total"])],
             "mean_gain": float(np.mean([agg[m]["gain"] for m in part]))}
            for part in q]


def top_spread(agg, ms, k=8):
    """Spread (max-min) of the top-k models under each track, ranked within that
    track — the paper's 'Static compresses the frontier' measurement."""
    out = {}
    for key in ("static_total", "active_total"):
        vals = sorted((agg[m][key] for m in ms), reverse=True)[:k]
        out[key] = {"top_k": k, "max": float(vals[0]), "min": float(vals[-1]),
                    "spread": float(vals[0] - vals[-1])}
    return out


def main():
    agg = json.load(open(E36))
    e41 = json.load(open(E41))
    R = rosters(agg)
    res = {"roster_sizes": {k: len(v) for k, v in R.items()}}

    # ---- C1 capability gate ------------------------------------------------
    res["C1_capability_gate"] = {
        "claim": "The Active gain is strongly predicted by Static ability "
                 "(r=+0.69); the strongest models gain more than a full point.",
        # Live recompute on the CURRENT rosters. e41["capability_gate_F2"] is
        # kept beside it as the frozen 28/33/38 trajectory it was measured on.
        "by_roster": {k: gate_r(agg, v) for k, v in R.items()},
        "historical_e41_28_33_38": e41["capability_gate_F2"],
        "strongest_models_actual_gain": {
            agg[m]["short"]: round(agg[m]["gain"], 3)
            for m in sorted(agg, key=lambda m: -agg[m]["static_total"])[:5]},
    }

    # ---- C2 gap widens with capability ------------------------------------
    res["C2_gap_widens"] = {
        "claim": "The Active-Static gap widens as models become more capable.",
        "quartile_mean_gain": {k: quartile_gains(agg, v) for k, v in R.items()},
    }

    # ---- C3 discrimination -------------------------------------------------
    disc = {}
    for k, ms in R.items():
        b = np.array([agg[m]["static_total"] for m in ms])
        c = np.array([agg[m]["active_total"] for m in ms])
        disc[k] = {"var_static": float(b.var(ddof=1)),
                   "var_active": float(c.var(ddof=1)),
                   "variance_ratio_C_over_B": float(c.var(ddof=1) / b.var(ddof=1)),
                   "sd_static": float(b.std(ddof=1)),
                   "sd_active": float(c.std(ddof=1))}
    res["C3_discrimination"] = {
        "claim": "Active evaluation restores top-end separation; "
                 "var(Active)/var(Static) = 4.4x on the paper roster.",
        "by_roster": disc,
    }

    # ---- C4 top-end compression -------------------------------------------
    res["C4_top_end_compression"] = {
        "claim": "Under Static scoring the top of the field compresses "
                 "(top-8 spread only 0.39).",
        "by_roster": {k: top_spread(agg, v) for k, v in R.items()},
    }

    # ---- C5 mechanism ------------------------------------------------------
    pd = e41["per_dimension_gain"]
    def order(g):
        return sorted(pd[g], key=lambda d: -pd[g][d]["mean_gain"])
    res["C5_mechanism"] = {
        "claim": "Agent-controlled retrieval raises feasibility, clarity and "
                 "specificity; measured originality stays flat.",
        "scope_note": "Per-dimension gains come from e41_frontier_analysis.json "
                      "and cover only the groups measured there (frontier2026 / "
                      "open_weight_28 / gemini_5). The E42 closed roster is NOT "
                      "included here: e36 stores Active per-dimension scores but "
                      "no Static ones, so a per-dimension gain cannot be derived "
                      "from it. not computed for E42.",
        "per_dimension_gain": pd,
        "rank_order": {g: order(g) for g in pd},
        "originality_flat_everywhere": {
            g: pd[g]["originality"]["mean_gain"] for g in pd},
    }

    # ---- headroom ----------------------------------------------------------
    res["ceiling"] = e41["ceiling"]

    OUT.write_text(json.dumps(res, indent=1))

    # ---- console -----------------------------------------------------------
    print("=== C1 capability gate: r(Static, gain) — live on current rosters ===")
    for k, v in res["C1_capability_gate"]["by_roster"].items():
        print(f"  {k:<20} n={v['n']:<3} r={v['pearson_r']:+.3f} p={v['pearson_p']:.2g}"
              f"   rho={v['spearman_rho']:+.3f} p={v['spearman_p']:.2g}")
    print("  --- frozen E41 trajectory (28/33/38, for comparison) ---")
    for k, v in e41["capability_gate_F2"].items():
        print(f"  {k:<20} n={v['n']:<3} r={v['pearson_r']:+.3f} p={v['pearson_p']:.2g}")
    print("  actual gain of the 5 strongest-Static models:",
          res["C1_capability_gate"]["strongest_models_actual_gain"])

    print("\n=== C2 mean gain by Static quartile ===")
    for k in R:
        qs = res["C2_gap_widens"]["quartile_mean_gain"][k]
        print(f"  {k:<14} " + "  ".join(f"Q{i+1}:{q['mean_gain']:+.2f}"
                                        for i, q in enumerate(qs)))

    print("\n=== C3 var(Active)/var(Static) across models ===")
    for k, v in disc.items():
        print(f"  {k:<14} ratio={v['variance_ratio_C_over_B']:.2f}  "
              f"sd_B={v['sd_static']:.2f} sd_C={v['sd_active']:.2f}")

    print("\n=== C4 top-8 spread within each track ===")
    for k in R:
        t = res["C4_top_end_compression"]["by_roster"][k]
        print(f"  {k:<14} Static spread={t['static_total']['spread']:.2f} "
              f"(max {t['static_total']['max']:.2f})   "
              f"Active spread={t['active_total']['spread']:.2f} "
              f"(max {t['active_total']['max']:.2f})")

    print("\n=== C5 per-dimension gain ===")
    print(f"  {'group':<16}" + "".join(f"{d[:5]:>9}" for d in
          ["originality", "feasibility", "clarity", "impact", "specificity"]))
    for g in pd:
        print(f"  {g:<16}" + "".join(f"{pd[g][d]['mean_gain']:>+9.2f}" for d in
              ["originality", "feasibility", "clarity", "impact", "specificity"]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
