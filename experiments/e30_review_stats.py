"""E30 — offline statistics redo for the ARR review (review points 2 & 4).

Four analyses, all on EXISTING data (read-only DB + cached embeddings, zero API):

  (a) swm_direct    : DIRECT paired S4b-vs-S4 test on shared (backbone, subdomain)
                      cells (the published p=.040/.065 are each arm vs active_base,
                      not S4b vs S4), pooled + per-backbone, with bootstrap CI and
                      matched-pairs rank-biserial effect size; plus Holm correction
                      over the three vs-base designs {S2, S4, S4b}.
  (b) f3_interaction: formal track x cutoff interaction (score ~ cutoff * track)
                      at model level, cluster-bootstrap CI (resample models) +
                      within-model track-permutation p; cell-level two-way cluster
                      bootstrap (models x subdomains) as robustness.
  (c) f2_family     : r(static, boost) robustness — family cluster bootstrap CI
                      (9 families) + leave-one-family-out range + concentration.
  (d) e28_recluster : diversity headline re-tested at model level (n~30) and
                      subfield level (n=100) with cluster-bootstrap CIs, from the
                      cached embeddings (reports/e28_idea_emb.npz), no re-embed.

Writes reports/e30_review_stats.json. Every block records test, unit, n, effect
size, CI, and correction so the paper's "Statistical reporting" appendix can
quote it verbatim.

Usage: /usr/bin/python3 experiments/e30_review_stats.py
"""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as W
from experiments.e10_idea_anchor_calibration import trimmed_mean
from experiments.e22_new_axis_stats import seed_scores, _load_dict

WSUM = sum(W.values())
OUT = ROOT / "reports" / "e30_review_stats.json"
EMB_CACHE = ROOT / "reports" / "e28_idea_emb.npz"
N_BOOT = 5000
RNG_SEED = 20260723


def tm(vs):
    return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)


def boot_ci(vals, rng, n=N_BOOT, stat=np.mean):
    vals = np.asarray(vals)
    bs = [float(stat(vals[rng.integers(0, len(vals), len(vals))])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def rank_biserial(d):
    """Matched-pairs rank-biserial from signed diffs (zeros dropped)."""
    d = np.asarray([x for x in d if x != 0])
    if len(d) == 0:
        return None
    r = stats.rankdata(np.abs(d))
    tp = r[d > 0].sum()
    tn = r[d < 0].sum()
    return round(float((tp - tn) / (tp + tn)), 4)


def paired_block(delta, rng, unit):
    delta = np.asarray(delta, dtype=float)
    out = {"unit": unit, "n": int(len(delta)),
           "delta_mean": round(float(delta.mean()), 4),
           "win_rate": round(float((delta > 0).mean()), 4),
           "rank_biserial": rank_biserial(delta)}
    if len(delta) >= 6:
        try:
            out["wilcoxon_p"] = round(float(stats.wilcoxon(delta).pvalue), 4)
        except ValueError:
            out["wilcoxon_p"] = None
        out["delta_ci95_bootstrap"] = boot_ci(delta, rng)
    return out


def holm(pvals):
    """Holm step-down adjusted p-values, same order as input."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = round(float(min(1.0, running)), 4)
    return adj


# ------------------------------------------------------------------ (a) SWM
def swm_cells():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    cell = defaultdict(lambda: defaultdict(list))
    for m, sub, c, sj in conn.execute(
            "SELECT gen_model, subdomain, condition, scores_json FROM swm_scores "
            "WHERE scores_json IS NOT NULL"):
        s = json.loads(sj)
        for d in DIMS:
            v = s[d]["score"] if isinstance(s[d], dict) else s[d]
            cell[(m, sub, c)][d].append(v)
    conn.close()
    w = {}
    for k, dv in cell.items():
        dims = {d: tm(dv[d]) for d in DIMS}
        if all(dims[d] is not None for d in DIMS):
            w[k] = sum(dims[d] * W[d] for d in DIMS) / WSUM
    return w


def analysis_swm(rng):
    w = swm_cells()
    by_ms = defaultdict(dict)
    for (m, sub, c), v in w.items():
        by_ms[(m, sub)][c] = v

    # direct S4b - S4 on shared cells
    per_backbone, pooled = {}, []
    for (m, sub), conds in by_ms.items():
        if "swm_S4" in conds and "swm_S4b" in conds:
            per_backbone.setdefault(m, []).append(conds["swm_S4b"] - conds["swm_S4"])
    for m, dl in sorted(per_backbone.items()):
        pooled += dl
    direct = {"test": "paired Wilcoxon signed-rank on per-(backbone, subdomain) "
                      "S4b minus S4 weighted lit8d cell scores (identical panels, "
                      "different aggregators); cells where both conditions scored",
              "pooled": paired_block(pooled, rng, "backbone x subdomain cell"),
              "per_backbone": {m: paired_block(dl, rng, "subdomain cell")
                               for m, dl in sorted(per_backbone.items())}}

    # vs-base pooled tests for {S2, S4, S4b} + Holm. Two poolings: "all" replicates
    # e27 (every cell present, incl. the 397b 4-5-cell pilot); "4bb" restricts to
    # the four full backbones (the tab:swm layout).
    four_bb = sorted({m for m, dl in per_backbone.items()})

    def vs_base_tests(restrict):
        tests = {}
        for cond in ["swm_S2", "swm_S4", "swm_S4b"]:
            dl = [conds[cond] - conds["active_base"] for (m, _), conds in by_ms.items()
                  if cond in conds and "active_base" in conds
                  and (restrict is None or m in restrict)]
            tests[cond] = paired_block(dl, rng, "backbone x subdomain cell")
        adj = holm([tests[c]["wilcoxon_p"] for c in ["swm_S2", "swm_S4", "swm_S4b"]])
        for i, c in enumerate(["swm_S2", "swm_S4", "swm_S4b"]):
            tests[c]["holm_p"] = adj[i]
        return tests

    return {"note": "review point 2: published p=.040 (S4b) / .065 (S4) are each "
                    "arm vs active_base; this block adds the DIRECT S4b-S4 paired "
                    "test and Holm correction over the three vs-base designs",
            "direct_S4b_minus_S4": direct,
            "vs_base_holm": {"family": ["swm_S2", "swm_S4", "swm_S4b"],
                             "correction": "Holm step-down over the 3 pooled "
                                           "vs-base Wilcoxon tests",
                             "tests": vs_base_tests(None),
                             "tests_four_backbones_only": vs_base_tests(set(four_bb)),
                             "four_backbones": four_bb}}


# ------------------------------------------------------------------ (b) F3
def fit_interaction(rows):
    """rows: (cutoff, track01, score). Returns (b0, b_cutoff, b_track, b_inter)."""
    A = np.array([[1.0, c, t, c * t] for c, t, _ in rows])
    y = np.array([s for _, _, s in rows])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def analysis_f3(rng):
    cutoffs = _load_dict("reports/_make_cross_year_plot.py", "KNOWLEDGE_CUTOFFS")
    by_mt = seed_scores()  # (model, track) -> list of per-subdomain seed scores
    models = sorted({m for (m, _) in by_mt if cutoffs.get(m)})
    c_mean = float(np.mean([cutoffs[m] for m in models]))
    # model-level long table (cutoff centered at roster mean so beta_track is
    # the track effect at the mean cutoff, not at year 0)
    long_rows, model_of_row = [], []
    for m in models:
        for tr, t01 in (("B", 0.0), ("C", 1.0)):
            v = by_mt.get((m, tr))
            if v:
                long_rows.append((cutoffs[m] - c_mean, t01, float(np.mean(v))))
                model_of_row.append(m)
    beta = fit_interaction(long_rows)
    both = [m for m in models if by_mt.get((m, "B")) and by_mt.get((m, "C"))]

    # cluster bootstrap: resample models with replacement
    rows_by_model = defaultdict(list)
    for r, m in zip(long_rows, model_of_row):
        rows_by_model[m].append(r)
    boots = []
    for _ in range(N_BOOT):
        pick = [models[i] for i in rng.integers(0, len(models), len(models))]
        sample = [r for m in pick for r in rows_by_model[m]]
        ts = {r[1] for r in sample}
        cs = {r[0] for r in sample}
        if len(ts) < 2 or len(cs) < 2:
            continue
        try:
            boots.append(float(fit_interaction(sample)[3]))
        except np.linalg.LinAlgError:
            continue
    ci = [round(float(np.percentile(boots, 2.5)), 4),
          round(float(np.percentile(boots, 97.5)), 4)]
    p_boot = 2 * min((np.array(boots) <= 0).mean(), (np.array(boots) >= 0).mean())

    # permutation: swap B/C labels within each both-track model
    obs = float(beta[3])
    perm = []
    for _ in range(N_BOOT):
        rows_p = []
        for m in models:
            flip = bool(rng.integers(0, 2)) if m in both else False
            for tr, t01 in (("B", 0.0), ("C", 1.0)):
                v = by_mt.get((m, tr))
                if v:
                    t = (1.0 - t01) if flip else t01
                    rows_p.append((cutoffs[m] - c_mean, t, float(np.mean(v))))
        perm.append(float(fit_interaction(rows_p)[3]))
    p_perm = float((np.abs(perm) >= abs(obs)).mean())

    # cell-level robustness: (model, subdomain, track) seed scores, two-way cluster
    cell_rows = []
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    per_cell = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT idea_model, track, subdomain, idea_index, scores_json "
                          "FROM lit8d_scores_3seed WHERE scores_json IS NOT NULL"):
        m, tr, sub, idx, sj = r
        if not cutoffs.get(m):
            continue
        s = json.loads(sj)
        per_cell[(m, tr, sub, idx)]["_"] = per_cell[(m, tr, sub, idx)].get("_", [])
        for d in DIMS:
            v = s[d]["score"] if isinstance(s[d], dict) else s[d]
            per_cell[(m, tr, sub, idx)][d].append(v)
    conn.close()
    idea_w = {}
    for k, dv in per_cell.items():
        dims = {d: tm(dv[d]) for d in DIMS}
        if all(dims[d] is not None for d in DIMS):
            idea_w[k] = sum(dims[d] * W[d] for d in DIMS) / WSUM
    seed_cell = defaultdict(list)
    for (m, tr, sub, _), v in idea_w.items():
        seed_cell[(m, tr, sub)].append(v)
    cell_data = [(m, sub, cutoffs[m] - c_mean, 0.0 if tr == "B" else 1.0,
                  float(np.mean(v))) for (m, tr, sub), v in seed_cell.items()]
    cmodels = sorted({m for m, *_ in cell_data})
    csubs = sorted({s for _, s, *_ in cell_data})
    beta_cell = fit_interaction([(c, t, s) for _, _, c, t, s in cell_data])
    by_pair = defaultdict(list)
    for m, sub, c, t, s in cell_data:
        by_pair[(m, sub)].append((c, t, s))
    boots_cell = []
    for _ in range(2000):
        pm = set(np.array(cmodels)[rng.integers(0, len(cmodels), len(cmodels))])
        psub = set(np.array(csubs)[rng.integers(0, len(csubs), len(csubs))])
        sample = [r for (m, sub), rows in by_pair.items()
                  if m in pm and sub in psub for r in rows]
        if len({r[1] for r in sample}) < 2:
            continue
        try:
            boots_cell.append(float(fit_interaction(sample)[3]))
        except np.linalg.LinAlgError:
            continue
    ci_cell = [round(float(np.percentile(boots_cell, 2.5)), 4),
               round(float(np.percentile(boots_cell, 97.5)), 4)]

    return {"note": "review point 4: direct track x cutoff interaction instead of "
                    "comparing two separately fitted slopes",
            "model_level": {
                "model": "score ~ cutoff_c + track + cutoff_c:track (OLS), track "
                         "C=1, cutoff centered at roster mean",
                "cutoff_center": round(c_mean, 4),
                "unit": "model x track mean over 40 subdomains",
                "n_rows": len(long_rows), "n_models": len(models),
                "n_models_both_tracks": len(both),
                "beta_cutoff_static_slope": round(float(beta[1]), 4),
                "beta_track_at_mean_cutoff": round(float(beta[2]), 4),
                "beta_interaction_per_yr": round(obs, 4),
                "interaction_ci95_cluster_bootstrap_models": ci,
                "bootstrap_p_two_sided": round(float(p_boot), 4),
                "permutation_p_two_sided": round(p_perm, 4),
                "permutation": "track labels flipped within each both-track model, "
                               f"{N_BOOT} draws"},
            "cell_level_robustness": {
                "unit": "model x subdomain x track (mean of 3 ideas)",
                "n_rows": len(cell_data), "n_models": len(cmodels),
                "n_subdomains": len(csubs),
                "beta_interaction_per_yr": round(float(beta_cell[3]), 4),
                "interaction_ci95_twoway_cluster_bootstrap": ci_cell,
                "bootstrap": "2000 draws resampling models and subdomains "
                             "independently"}}


# ------------------------------------------------------------------ (c) F2
def analysis_f2(rng):
    by_mt = seed_scores()
    models = sorted({m for (m, _) in by_mt})
    rows = {}
    for m in models:
        b = by_mt.get((m, "B")); c = by_mt.get((m, "C"))
        if b and c:
            rows[m] = (float(np.mean(b)), float(np.mean(c)) - float(np.mean(b)))
    fams = sorted({m.split("/")[0] for m in rows})
    fam_models = {f: [m for m in rows if m.split("/")[0] == f] for f in fams}
    st = np.array([rows[m][0] for m in sorted(rows)])
    bo = np.array([rows[m][1] for m in sorted(rows)])
    obs_r, obs_p = stats.pearsonr(st, bo)

    boots = []
    for _ in range(N_BOOT):
        pick = [fams[i] for i in rng.integers(0, len(fams), len(fams))]
        ms = [m for f in pick for m in fam_models[f]]
        x = np.array([rows[m][0] for m in ms]); y = np.array([rows[m][1] for m in ms])
        if len(ms) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            boots.append(float(stats.pearsonr(x, y)[0]))
    lofo = {}
    for f in fams:
        ms = [m for m in rows if m.split("/")[0] != f]
        x = np.array([rows[m][0] for m in ms]); y = np.array([rows[m][1] for m in ms])
        lofo[f] = {"n_dropped": len(fam_models[f]), "n_left": len(ms),
                   "pearson_r": round(float(stats.pearsonr(x, y)[0]), 4)}
    lvals = [v["pearson_r"] for v in lofo.values()]
    return {"note": "review point 4: family-cluster robustness for "
                    "F2 r(static, boost)",
            "unit": "model (n=28 with both tracks); cluster = family (id prefix)",
            "n_models": len(rows), "n_families": len(fams),
            "family_sizes": {f: len(fam_models[f]) for f in fams},
            "pearson_r": round(float(obs_r), 4), "pearson_p": round(float(obs_p), 6),
            "r_ci95_family_cluster_bootstrap": [
                round(float(np.percentile(boots, 2.5)), 4),
                round(float(np.percentile(boots, 97.5)), 4)],
            "leave_one_family_out": lofo,
            "lofo_r_range": [round(min(lvals), 4), round(max(lvals), 4)]}


# ------------------------------------------------------------------ (d) E28
def analysis_e28(rng):
    z = np.load(EMB_CACHE, allow_pickle=True)
    emb = np.asarray(z["emb"], dtype=np.float32)
    keys = [tuple(x) for x in z["key"]]  # (model, subdomain, track, idx_str)
    rows_by_cell = defaultdict(dict)
    for row, (m, s, t, i) in enumerate(keys):
        rows_by_cell[(m, s, t)][int(i)] = row

    def vendi(E):
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        lam = np.clip(np.linalg.eigvalsh(E @ E.T / E.shape[0]), 0, None)
        ssum = lam.sum()
        if ssum <= 0:
            return 1.0
        lam = lam / ssum
        nz = lam[lam > 1e-12]
        return float(np.exp(-(nz * np.log(nz)).sum()))

    per_cell = {k: vendi(np.stack([emb[d[i]] for i in sorted(d)[:3]]))
                for k, d in rows_by_cell.items() if len(d) >= 3}
    deltas = {}
    for (m, s, t) in per_cell:
        if t == "B" and (m, s, "C") in per_cell:
            deltas[(m, s)] = per_cell[(m, s, "C")] - per_cell[(m, s, "B")]

    d_all = np.array(list(deltas.values()))
    by_model = defaultdict(list)
    by_sub = defaultdict(list)
    for (m, s), d in deltas.items():
        by_model[m].append(d); by_sub[s].append(d)
    model_means = np.array([np.mean(v) for v in by_model.values()])
    sub_means = np.array([np.mean(v) for v in by_sub.values()])

    def cluster_ci(groups):
        vals = list(groups.values())
        bs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(vals), len(vals))
            bs.append(float(np.mean([np.mean(vals[i]) for i in idx])))
        return [round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4)]

    return {"note": "review point 4: diversity headline re-tested with model / "
                    "subfield clustering instead of treating 2793 correlated "
                    "cells as independent",
            "metric": "within-cell Vendi score of the 3 ideas (cached MiniLM "
                      "embeddings, reports/e28_idea_emb.npz)",
            "cell_level_reference": paired_block(d_all, rng,
                                                 "model x subdomain pair (correlated)"),
            "model_level_headline": {
                **paired_block(model_means, rng, "model (mean delta over its subdomains)"),
                "test": "Wilcoxon signed-rank over per-model mean deltas"},
            "subfield_level": {
                **paired_block(sub_means, rng, "subdomain (mean delta over models)"),
                "test": "Wilcoxon signed-rank over per-subdomain mean deltas"},
            "overall_delta_ci95_cluster_bootstrap_models": cluster_ci(by_model),
            "overall_delta_ci95_cluster_bootstrap_subdomains": cluster_ci(by_sub)}


def main():
    rng = np.random.default_rng(RNG_SEED)
    out = {"generated_by": "experiments/e30_review_stats.py",
           "rng_seed": RNG_SEED, "n_bootstrap": N_BOOT}
    print("== (a) SWM direct S4b-S4 + Holm ==", flush=True)
    out["swm_direct"] = analysis_swm(rng)
    print(json.dumps(out["swm_direct"]["direct_S4b_minus_S4"]["pooled"], indent=2))
    print(json.dumps({c: {k: v for k, v in t.items() if k in ("delta_mean", "wilcoxon_p", "holm_p")}
                      for c, t in out["swm_direct"]["vs_base_holm"]["tests"].items()}, indent=2))
    print("== (b) F3 interaction ==", flush=True)
    out["f3_interaction"] = analysis_f3(rng)
    print(json.dumps(out["f3_interaction"]["model_level"], indent=2))
    print("== (c) F2 family robustness ==", flush=True)
    out["f2_family"] = analysis_f2(rng)
    print(json.dumps({k: out["f2_family"][k] for k in
                      ("pearson_r", "r_ci95_family_cluster_bootstrap", "lofo_r_range")}, indent=2))
    print("== (d) E28 recluster ==", flush=True)
    out["e28_recluster"] = analysis_e28(rng)
    print(json.dumps(out["e28_recluster"]["model_level_headline"], indent=2))
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
