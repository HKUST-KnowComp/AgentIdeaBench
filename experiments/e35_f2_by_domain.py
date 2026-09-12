"""E35 — F2 capability-gate by discipline subset (READ-ONLY analysis, no writes to results.db).

Motivation (ARR review A3): the published F2 gate r=+0.6945 (static ability vs Active-Static
boost, non-gemini n=28, full 40 subfields) pools "reliable" (CS/Physics) and "soft"
(Bio/Chem/Med) disciplines. A reviewer will ask whether the headline survives on the
reliable-scorer regime. This probe re-computes r restricted to each discipline grouping,
reusing the EXACT aggregation from experiments/e23_ablations.py (trimmed-mean drop-highest
over 3 critics, weights O2:I1.5:F1:C0.5:S0.5 / 5.5) so numbers are on the same footing.
It also draws random same-size subfield subsets to separate a genuine discipline effect
from sample-size attenuation.

Output: reports/e35_f2_by_domain.json . Deterministic (seed 42). /usr/bin/python3.
"""
import sys, json, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.e23_ablations import load_raw, seed_weighted, model_means, f2_stats
from utils.constants import SCORING_WEIGHTS as WCUR

random.seed(42)
raw = load_raw()
ng = {k: v for k, v in raw.items() if "gemini" not in k[0]}   # non-gemini main analysis
all_subs = sorted({k[2] for k in ng})


def paired_n(sub):
    mt = {}
    for (m, tr, s, d, i) in sub:
        mt.setdefault(m, set()).add(tr)
    return sum(1 for trs in mt.values() if {"B", "C"} <= trs)


def r_stats(subset_keys):
    sw, _ = seed_weighted(subset_keys, WCUR)
    st = f2_stats(model_means(sw))
    return {"pearson_r": round(st["pearson_r"], 4), "pearson_p": st["pearson_p"],
            "spearman_r": round(st.get("spearman_r", float("nan")), 4),
            "n_paired": paired_n(subset_keys)}


def by_domains(doms):
    return {k: v for k, v in ng.items() if k[3] in doms}


groups = {
    "FULL_40_nongemini_SANITY": ng,
    "CS+Physics_reliable": by_domains({"CS", "Physics"}),
    "BioChemMed_soft": by_domains({"Biology", "Chemistry", "Medicine"}),
    "CS": by_domains({"CS"}), "Physics": by_domains({"Physics"}),
    "Biology": by_domains({"Biology"}), "Chemistry": by_domains({"Chemistry"}),
    "Medicine": by_domains({"Medicine"}),
}
result = {g: r_stats(sub) for g, sub in groups.items()}

# sample-size null: random subfield subsets matched to CS+Physics size (16)
k = len(set(kk[2] for kk in by_domains({"CS", "Physics"})))
rand_r = sorted(r_stats({kk: vv for kk, vv in ng.items()
                         if kk[2] in set(random.sample(all_subs, k))})["pearson_r"]
                for _ in range(200))
csp_r = result["CS+Physics_reliable"]["pearson_r"]
result["_random_subset_null"] = {
    "subset_size_subfields": k, "n_draws": 200,
    "mean_r": round(statistics.mean(rand_r), 4), "median_r": round(statistics.median(rand_r), 4),
    "pct5_r": round(rand_r[10], 4), "pct95_r": round(rand_r[190], 4),
    "observed_CSPhysics_r": csp_r,
    "frac_random_le_observed": round(sum(1 for r in rand_r if r <= csp_r) / len(rand_r), 3),
    "note": "CS+Physics r sits at this percentile of same-size random subfield subsets; "
            "a low fraction means the reliable-subset attenuation exceeds pure sample-size noise.",
}

out = ROOT / "reports" / "e35_f2_by_domain.json"
out.write_text(json.dumps(result, indent=2))
print(f"wrote {out}\n")
for g, s in result.items():
    if g.startswith("_"):
        continue
    print(f"  {g:<26} r={s['pearson_r']:+.3f} p={s['pearson_p']:.2e} n={s['n_paired']}")
print("\n  random-16 null:", json.dumps(result["_random_subset_null"]))
