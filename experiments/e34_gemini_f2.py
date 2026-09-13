"""E34 — Gemini closed-frontier-family validation of F2 (capability gate).

Pure recompute on lit8d_scores_3seed (READ-ONLY, no DB writes, no SS, no API).
Reuses E23's exact recipe (load_raw -> seed_weighted(WCUR) -> model_means ->
f2_stats) so numbers are directly comparable to the paper's main F2.

Splits the per-model (static, boost) rows into:
  - open   : the 28 open-weight roster models (paper's main F2, expect r~0.69)
  - gemini : the 5 cutoff-safe Gemini models (google/gemini-3.6-flash EXCLUDED:
             leakage, cutoff 2026-03 > benchmark; it also has no Track C so no boost)
  - all    : open + gemini (n=33 robustness)

Reports each Gemini model's static/boost and whether the gate holds out-of-roster.

  /usr/bin/python3 experiments/e34_gemini_f2.py
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.e23_ablations import load_raw, seed_weighted, model_means, WCUR

OUT = Path(__file__).resolve().parents[1] / "reports" / "e34_gemini_f2.json"
# gemini-3.6-flash: leakage-excluded (cutoff 2026-03 > benchmark 2025-04..2026-01)
LEAKAGE = "google/gemini-3.6-flash"


def is_gemini(m):
    return m.startswith("google/gemini") and m != LEAKAGE


def f2_on(rows, keys):
    """Pearson/Spearman r(static, boost) over the given model keys."""
    pairs = [(rows[m]["static"], rows[m]["boost"]) for m in keys
             if rows.get(m) and rows[m]["static"] is not None and rows[m]["boost"] is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson_r": None, "note": "n<3, correlation not computed"}
    st = np.array([p[0] for p in pairs]); bo = np.array([p[1] for p in pairs])
    pear = stats.pearsonr(st, bo); spear = stats.spearmanr(st, bo)
    out = {"n": len(pairs), "pearson_r": round(float(pear[0]), 4),
           "pearson_p": float(f"{pear[1]:.2e}"),
           "spearman_rho": round(float(spear[0]), 4),
           "mean_boost": round(float(bo.mean()), 4),
           "static_range": [round(float(st.min()), 3), round(float(st.max()), 3)]}
    # bootstrap CI on pearson r
    rng = np.random.default_rng(20260725)
    if len(pairs) >= 5:
        bs = []
        for _ in range(5000):
            idx = rng.integers(0, len(pairs), len(pairs))
            if len(set(st[idx])) > 1:
                bs.append(stats.pearsonr(st[idx], bo[idx])[0])
        if bs:
            out["pearson_ci95"] = [round(float(np.percentile(bs, 2.5)), 4),
                                   round(float(np.percentile(bs, 97.5)), 4)]
    return out


def main():
    raw = load_raw()
    sw, _meta = seed_weighted(raw, WCUR)
    rows = model_means(sw)

    gem = sorted([m for m in rows if is_gemini(m)])
    have_leak = LEAKAGE in rows
    openw = sorted([m for m in rows if not m.startswith("google/gemini")
                    and rows[m]["boost"] is not None])
    allk = openw + [m for m in gem if rows[m]["boost"] is not None]

    per_gemini = []
    for m in gem:
        r = rows[m]
        per_gemini.append({"model": m, "static": round(r["static"], 4) if r["static"] is not None else None,
                           "active": round(r["active"], 4) if r["active"] is not None else None,
                           "boost": round(r["boost"], 4) if r["boost"] is not None else None})

    out = {
        "note": "F2 = Pearson r(static ability, Active-Static boost) across models, "
                "E23 recipe (WCUR weights, lit8d_scores_3seed). Gemini = held-out "
                "closed-frontier family; google/gemini-3.6-flash leakage-excluded.",
        "open_weight_f2": f2_on(rows, openw),
        "gemini_f2": f2_on(rows, gem),
        "combined_f2": f2_on(rows, allk),
        "per_gemini_model": per_gemini,
        "leakage_model_present_but_excluded": have_leak,
        "n_open": len(openw), "n_gemini": len([m for m in gem if rows[m]["boost"] is not None]),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("open_weight_f2", "gemini_f2", "combined_f2")}, indent=2))
    print("\nper-Gemini (static / boost):")
    for g in per_gemini:
        print(f"  {g['model']:34s} static={g['static']}  boost={g['boost']}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
