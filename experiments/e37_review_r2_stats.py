"""E37 — ARR review round-2 offline statistics (read-only, zero API).

Answers two reviewer asks with the exact production aggregation:

  #3  Capability gate, split-half validated. The gate r(Static, Active-Static
      gain)=+0.69 could be inflated because Static appears on both axes (the
      gain C-B shares B's sampling noise with the x-axis). We break that
      coupling: Static ability is measured on one random half of the 40
      subfields, the gain on the DISJOINT other half, so the two carry
      independent noise. r is recomputed over many random 20/20 splits.
      Independent-capability cross-checks: gain vs knowledge cutoff, gain vs
      log-parameters (proxies that share no scoring noise with the gain).

  #4  Discrimination, beyond a hand-picked top-8 range. Bootstrap over the 40
      subfields gives a per-model, per-track standard error, from which we
      report (a) full-roster and top-8 score spread with CIs and a paired
      bootstrap on their difference, (b) the variance ratio Var(Active)/
      Var(Static) with CI, (c) the count of statistically distinguishable
      model pairs (|delta| > 2*pooled SE) under each track, full roster and
      top-half.

Aggregation is identical to the pipeline (experiments/e32_recall_only.py):
per (model, track, subfield, idea_index) take the trimmed mean over 3 critics
per dimension (drop the single highest), weight O:2/I:1.5/F:1/C:0.5/S:0.5
normalized by 5.5 -> per-idea weighted score; mean over the 3 ideas -> cell;
mean over subfields -> model score. Validated against reports/
e22_new_axis_stats.json per_model before any statistic is computed.

Roster: 28 open-weight both-track models (both-track 33 minus 5 google/gemini
held-out). Read-only; writes only reports/e37_review_r2_stats.json.

Usage:  /usr/bin/python3 experiments/e37_review_r2_stats.py
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

WSUM = sum(W.values())
OUT_JSON = ROOT / "reports" / "e37_review_r2_stats.json"
E22 = ROOT / "reports" / "e22_new_axis_stats.json"
N_SPLIT = 2000
N_BOOT = 5000
SEED = 20260802


def _tm(vs):
    return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)


def build_cells(rows):
    """rows (model, track, sub, idx, scores_json) -> cell[(model,track,sub)] =
    mean over the 3 ideas of the per-idea weighted (critic-trimmed) score."""
    per_idea = defaultdict(lambda: defaultdict(list))
    for m, tr, sub, idx, sj in rows:
        s = json.loads(sj)
        for d in DIMS:
            per_idea[(m, tr, sub, idx)][d].append(
                s[d]["score"] if isinstance(s[d], dict) else s[d])
    idea_w = {}
    for k, dv in per_idea.items():
        dims = {d: _tm(dv[d]) for d in DIMS}
        if all(dims[d] is not None for d in DIMS):
            idea_w[k] = sum(dims[d] * W[d] for d in DIMS) / WSUM
    cells = defaultdict(list)
    for (m, tr, sub, _), w in idea_w.items():
        cells[(m, tr, sub)].append(w)
    return {k: float(np.mean(v)) for k, v in cells.items()}


def roster(conn):
    b = {r[0] for r in conn.execute(
        "SELECT DISTINCT idea_model FROM lit8d_scores_3seed WHERE track='B'")}
    c = {r[0] for r in conn.execute(
        "SELECT DISTINCT idea_model FROM lit8d_scores_3seed WHERE track='C'")}
    both = b & c
    return sorted(m for m in both if not m.startswith("google/gemini-"))


def model_means(cells, models, subs, track):
    """point per-model mean over available subs for a track."""
    out = {}
    for m in models:
        vs = [cells[(m, track, s)] for s in subs if (m, track, s) in cells]
        if vs:
            out[m] = float(np.mean(vs))
    return out


def ci95(a):
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def main():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    models = roster(conn)
    rows = [(m, tr, sub, idx, sj) for m, tr, sub, idx, sj in conn.execute(
        "SELECT idea_model, track, subdomain, idea_index, scores_json "
        "FROM lit8d_scores_3seed WHERE scores_json IS NOT NULL AND track IN ('B','C')")
        if m in set(models)]
    conn.close()
    cells = build_cells(rows)
    subs = sorted({s for (m, tr, s) in cells})
    n_sub = len(subs)

    B = model_means(cells, models, subs, "B")
    C = model_means(cells, models, subs, "C")
    models = [m for m in models if m in B and m in C]  # keep fully-paired
    gain = {m: C[m] - B[m] for m in models}
    short = lambda m: m.split("/")[-1]

    # ---- validate aggregation against e22 per_model ----------------------
    e22 = json.load(open(E22))
    e22pm = {r["model"]: r for r in e22["per_model"]}
    val = {"max_abs_static": 0.0, "max_abs_active": 0.0, "checked": 0, "worst": None}
    for m in models:
        if m in e22pm:
            ds = abs(B[m] - e22pm[m]["static"])
            da = abs(C[m] - e22pm[m]["active"])
            if ds > val["max_abs_static"]:
                val["max_abs_static"] = ds
            if max(ds, da) > (val.get("_w") or -1):
                val["_w"] = max(ds, da); val["worst"] = short(m)
            val["max_abs_active"] = max(val["max_abs_active"], da)
            val["checked"] += 1
    val.pop("_w", None)
    val = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in val.items()}
    print(f"[validate vs e22] models checked={val['checked']} "
          f"max|dstatic|={val['max_abs_static']} max|dactive|={val['max_abs_active']} "
          f"(worst {val['worst']})")

    rng = np.random.default_rng(SEED)
    mlist = models
    b_arr = np.array([B[m] for m in mlist])
    c_arr = np.array([C[m] for m in mlist])
    g_arr = np.array([gain[m] for m in mlist])

    # ---- #3 capability gate: naive + split-half --------------------------
    naive_r = float(stats.pearsonr(b_arr, g_arr)[0])
    naive_rho = float(stats.spearmanr(b_arr, g_arr)[0])

    split_rs, split_rhos, nA, nB_ = [], [], [], []
    subs_arr = np.array(subs, dtype=object)
    for _ in range(N_SPLIT):
        perm = rng.permutation(n_sub)
        half = n_sub // 2
        A = set(subs_arr[perm[:half]]); Bset = set(subs_arr[perm[half:]])
        abil, gn = [], []
        for m in mlist:
            av = [cells[(m, "B", s)] for s in A if (m, "B", s) in cells]
            gv = [cells[(m, "C", s)] - cells[(m, "B", s)] for s in Bset
                  if (m, "C", s) in cells and (m, "B", s) in cells]
            if av and gv:
                abil.append(np.mean(av)); gn.append(np.mean(gv))
        if len(abil) >= 10:
            split_rs.append(float(stats.pearsonr(abil, gn)[0]))
            split_rhos.append(float(stats.spearmanr(abil, gn)[0]))
            nA.append(len(A)); nB_.append(len(Bset))
    split_rs = np.array(split_rs)

    # independent-capability proxies (share no scoring noise with the gain)
    proxies = {}
    for key in ("cutoff", "log_params"):
        xs, ys = [], []
        for m in mlist:
            if m in e22pm and e22pm[m].get(key) is not None:
                xs.append(e22pm[m][key]); ys.append(gain[m])
        if len(xs) >= 5:
            proxies[key] = {
                "n": len(xs),
                "pearson_r": round(float(stats.pearsonr(xs, ys)[0]), 4),
                "pearson_p": round(float(stats.pearsonr(xs, ys)[1]), 4),
                "spearman_rho": round(float(stats.spearmanr(xs, ys)[0]), 4),
                "spearman_p": round(float(stats.spearmanr(xs, ys)[1]), 4)}

    gate = {
        "n_models": len(mlist),
        "naive_pearson_r": round(naive_r, 4),
        "naive_spearman_rho": round(naive_rho, 4),
        "split_half": {
            "n_splits": len(split_rs),
            "subfields_per_half": [int(np.median(nA)), int(np.median(nB_))],
            "pearson_r_mean": round(float(split_rs.mean()), 4),
            "pearson_r_median": round(float(np.median(split_rs)), 4),
            "pearson_r_ci95": ci95(split_rs),
            "pearson_r_frac_positive": round(float((split_rs > 0).mean()), 4),
            "spearman_rho_mean": round(float(np.mean(split_rhos)), 4),
            "note": "Static ability from one random 20-subfield half; Active-"
                    "Static gain from the disjoint other half. Independent noise "
                    "across the two halves, so shared-B coupling cannot inflate r."},
        "independent_capability_proxies": proxies,
    }

    # ---- #4 discrimination: bootstrap over subfields ---------------------
    idx_by_sub = {s: i for i, s in enumerate(subs)}
    # dense matrices [model, sub] with nan for missing
    matB = np.full((len(mlist), n_sub), np.nan)
    matC = np.full((len(mlist), n_sub), np.nan)
    for i, m in enumerate(mlist):
        for s in subs:
            if (m, "B", s) in cells:
                matB[i, idx_by_sub[s]] = cells[(m, "B", s)]
            if (m, "C", s) in cells:
                matC[i, idx_by_sub[s]] = cells[(m, "C", s)]

    top8 = [i for i, _ in sorted(enumerate(b_arr), key=lambda t: -t[1])[:8]]  # top-8 by Static
    order_b = np.argsort(-b_arr); order_c = np.argsort(-c_arr)
    tophalf_b = set(order_b[:len(mlist) // 2]); tophalf_c = set(order_c[:len(mlist) // 2])

    def _means(mat, cols):
        sub = mat[:, cols]
        return np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in sub])

    spread_full_B, spread_full_C = [], []
    spread_top8_B, spread_top8_C = [], []
    var_ratio = []
    bootB = np.zeros((N_BOOT, len(mlist))); bootC = np.zeros((N_BOOT, len(mlist)))
    for b in range(N_BOOT):
        cols = rng.integers(0, n_sub, n_sub)
        mb = _means(matB, cols); mc = _means(matC, cols)
        bootB[b] = mb; bootC[b] = mc
        spread_full_B.append(np.nanmax(mb) - np.nanmin(mb))
        spread_full_C.append(np.nanmax(mc) - np.nanmin(mc))
        spread_top8_B.append(np.nanmax(mb[top8]) - np.nanmin(mb[top8]))
        spread_top8_C.append(np.nanmax(mc[top8]) - np.nanmin(mc[top8]))
        var_ratio.append(np.nanvar(mc) / np.nanvar(mb))
    spread_full_B = np.array(spread_full_B); spread_full_C = np.array(spread_full_C)
    spread_top8_B = np.array(spread_top8_B); spread_top8_C = np.array(spread_top8_C)
    d_full = spread_full_C - spread_full_B
    d_top8 = spread_top8_C - spread_top8_B

    se_B = bootB.std(axis=0); se_C = bootC.std(axis=0)

    def distinguishable(pt, se, subset):
        idxs = list(subset)
        cnt = tot = 0
        for a in range(len(idxs)):
            for c_ in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[c_]
                tot += 1
                if abs(pt[i] - pt[j]) > 2 * np.sqrt(se[i] ** 2 + se[j] ** 2):
                    cnt += 1
        return cnt, tot

    dpB_full = distinguishable(b_arr, se_B, range(len(mlist)))
    dpC_full = distinguishable(c_arr, se_C, range(len(mlist)))
    dpB_top = distinguishable(b_arr, se_B, tophalf_b)
    dpC_top = distinguishable(c_arr, se_C, tophalf_c)

    discrimination = {
        "n_models": len(mlist),
        "point_spread_full": {"static": round(float(b_arr.max() - b_arr.min()), 4),
                              "active": round(float(c_arr.max() - c_arr.min()), 4)},
        "point_spread_top8_by_static": {
            "models": [short(mlist[i]) for i in top8],
            "static": round(float(b_arr[top8].max() - b_arr[top8].min()), 4),
            "active": round(float(c_arr[top8].max() - c_arr[top8].min()), 4)},
        "bootstrap_spread_full": {
            "static_ci95": ci95(spread_full_B), "active_ci95": ci95(spread_full_C),
            "diff_active_minus_static": round(float(d_full.mean()), 4),
            "diff_ci95": ci95(d_full),
            "diff_boot_p_one_sided": round(float((d_full <= 0).mean()), 4)},
        "bootstrap_spread_top8_by_static": {
            "static_ci95": ci95(spread_top8_B), "active_ci95": ci95(spread_top8_C),
            "diff_active_minus_static": round(float(d_top8.mean()), 4),
            "diff_ci95": ci95(d_top8),
            "diff_boot_p_one_sided": round(float((d_top8 <= 0).mean()), 4)},
        "variance_ratio_active_over_static": {
            "point": round(float(np.var(c_arr) / np.var(b_arr)), 4),
            "ci95": ci95(np.array(var_ratio))},
        "distinguishable_pairs": {
            "criterion": "|mean_i - mean_j| > 2*sqrt(SE_i^2+SE_j^2); SE = bootstrap-over-subfield std",
            "full_roster": {
                "static": {"distinguishable": dpB_full[0], "total": dpB_full[1],
                           "frac": round(dpB_full[0] / dpB_full[1], 4)},
                "active": {"distinguishable": dpC_full[0], "total": dpC_full[1],
                           "frac": round(dpC_full[0] / dpC_full[1], 4)}},
            "top_half": {
                "static": {"distinguishable": dpB_top[0], "total": dpB_top[1],
                           "frac": round(dpB_top[0] / max(dpB_top[1], 1), 4)},
                "active": {"distinguishable": dpC_top[0], "total": dpC_top[1],
                           "frac": round(dpC_top[0] / max(dpC_top[1], 1), 4)}}},
        "median_per_model_SE": {"static": round(float(np.median(se_B)), 4),
                                "active": round(float(np.median(se_C)), 4)},
    }

    out = {
        "generated_by": "experiments/e37_review_r2_stats.py",
        "rng_seed": SEED, "n_bootstrap": N_BOOT, "n_split": N_SPLIT,
        "n_subfields": n_sub, "roster": [short(m) for m in mlist],
        "aggregation_validation_vs_e22": val,
        "per_model": {short(m): {"static": round(B[m], 4), "active": round(C[m], 4),
                                 "gain": round(gain[m], 4)} for m in mlist},
        "gate_split_half": gate,
        "discrimination": discrimination,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ---- console summary --------------------------------------------------
    print("\n=== #3 GATE ===")
    print(f"  naive r={gate['naive_pearson_r']} (rho={gate['naive_spearman_rho']}), n={gate['n_models']}")
    sh = gate["split_half"]
    print(f"  split-half r: mean={sh['pearson_r_mean']} median={sh['pearson_r_median']} "
          f"CI95={sh['pearson_r_ci95']} frac_pos={sh['pearson_r_frac_positive']} "
          f"({sh['n_splits']} splits, {sh['subfields_per_half']} subs/half)")
    for k, v in proxies.items():
        print(f"  gain vs {k}: r={v['pearson_r']} p={v['pearson_p']} (n={v['n']})")
    print("\n=== #4 DISCRIMINATION ===")
    d = discrimination
    print(f"  spread full: static {d['point_spread_full']['static']} vs active "
          f"{d['point_spread_full']['active']}")
    print(f"  spread top8(by static): static {d['point_spread_top8_by_static']['static']} "
          f"[{d['bootstrap_spread_top8_by_static']['static_ci95']}] vs active "
          f"{d['point_spread_top8_by_static']['active']} "
          f"[{d['bootstrap_spread_top8_by_static']['active_ci95']}]; "
          f"diff={d['bootstrap_spread_top8_by_static']['diff_active_minus_static']} "
          f"CI={d['bootstrap_spread_top8_by_static']['diff_ci95']} "
          f"p={d['bootstrap_spread_top8_by_static']['diff_boot_p_one_sided']}")
    print(f"  var ratio C/B: {d['variance_ratio_active_over_static']['point']} "
          f"CI={d['variance_ratio_active_over_static']['ci95']}")
    dp = d["distinguishable_pairs"]
    print(f"  distinguishable pairs full: static {dp['full_roster']['static']['frac']} "
          f"vs active {dp['full_roster']['active']['frac']}")
    print(f"  distinguishable pairs top-half: static {dp['top_half']['static']['frac']} "
          f"vs active {dp['top_half']['active']['frac']}")
    print("\nwrote", OUT_JSON)


if __name__ == "__main__":
    main()
