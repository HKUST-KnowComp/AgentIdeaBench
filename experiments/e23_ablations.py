"""E23 — pure-recompute ablation suite on lit8d_scores_3seed.

All read-only recomputes from the existing 3-seed lit8d scores (no API, no DB
writes). Tests robustness of F2 (Active boost, stronger-gets-more) and F16/F17
under: (A) weighting schemes, (B) per-dimension boost decomposition,
(C) critic jackknife, (D) ideas/seed convergence, (E) per-domain cutoff slopes,
(F) reasoning vs non-reasoning split.

Writes reports/e23_ablations.json. Run:
  /usr/bin/python3 experiments/e23_ablations.py
"""
import ast, json, math, sqlite3, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as W
from experiments.e10_idea_anchor_calibration import trimmed_mean
from experiments.e22_new_axis_stats import to_decimal_year, _load_dict

WCUR = dict(W)
WSUM = sum(W.values())
# reasoning/thinking models (explicit reasoning traces in this roster)
REASONING = {"deepseek/deepseek-r1-0528", "qwen/qwen3-235b-a22b-thinking-2507",
             "qwen/qwen3-vl-8b-thinking"}


def load_raw():
    """rows keyed (model,track,subdomain,domain,idx) -> {dim: [critic scores]},
    plus per-row critic map for jackknife."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    # cell[(m,tr,sub,dom,idx)][dim] = list; also critic-tagged
    by_critic = defaultdict(lambda: defaultdict(dict))  # [(m,tr,sub,dom,idx)][critic][dim]=score
    for r in conn.execute("SELECT idea_model,track,subdomain,domain,idea_index,critic_model,scores_json "
                          "FROM lit8d_scores_3seed WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"])
        k = (r["idea_model"], r["track"], r["subdomain"], r["domain"], r["idea_index"])
        d = {dim: (s[dim]["score"] if isinstance(s[dim], dict) else s[dim]) for dim in DIMS}
        by_critic[k][r["critic_model"]] = d
    conn.close()
    return by_critic


def tm(vs):
    return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)


def weighted(dimvals, weights):
    ws = sum(weights.values())
    return sum(dimvals[d] * weights[d] for d in DIMS) / ws


def seed_weighted(by_critic, weights, critics=None, idxset=None):
    """seed_w[(model,track)] -> list of seed weighted means; also per-(m,tr,sub).
    critics: restrict to a subset of critic models. idxset: restrict idea_index."""
    # per idea: trimmed-mean over critics per dim, then weight
    idea_w = {}
    for k, cmap in by_critic.items():
        m, tr, sub, dom, idx = k
        if idxset is not None and idx not in idxset:
            continue
        cs = cmap if critics is None else {c: v for c, v in cmap.items() if c in critics}
        if not cs:
            continue
        dimvals = {}
        for dim in DIMS:
            vals = [v[dim] for v in cs.values()]
            dimvals[dim] = tm(vals)
        idea_w[k] = weighted(dimvals, weights)
    # collapse idx -> seed
    seed_vals = defaultdict(list); meta = {}
    for (m, tr, sub, dom, idx), w in idea_w.items():
        seed_vals[(m, tr, sub)].append(w); meta[(m, tr, sub)] = dom
    seed_w = {k: float(np.mean(v)) for k, v in seed_vals.items()}
    return seed_w, meta


def model_means(seed_w):
    by_mt = defaultdict(list)
    for (m, tr, sub), w in seed_w.items():
        by_mt[(m, tr)].append(w)
    models = sorted({m for (m, tr) in by_mt})
    rows = {}
    for m in models:
        b = by_mt.get((m, "B"), []); c = by_mt.get((m, "C"), [])
        rows[m] = {"static": np.mean(b) if b else None, "active": np.mean(c) if c else None,
                   "boost": (np.mean(c) - np.mean(b)) if (b and c) else None}
    return rows


def f2_stats(rows):
    pairs = [(r["static"], r["boost"]) for r in rows.values()
             if r["static"] is not None and r["boost"] is not None]
    st = [a for a, _ in pairs]; bo = [b for _, b in pairs]
    if len(pairs) < 3:
        return None
    npos = sum(1 for b in bo if b > 0)
    pear = stats.pearsonr(st, bo); spear = stats.spearmanr(st, bo)
    return {"n": len(pairs), "mean_boost": float(np.mean(bo)),
            "pos": npos, "sign_p": float(stats.binomtest(npos, len(pairs)).pvalue),
            "pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
            "spearman_r": float(spear[0]), "spearman_p": float(spear[1])}


def f2_extra(rows):
    """Bootstrap CI on the F2 Pearson r + top10/bottom10 mean boost (same recipe as
    f2_stats). Used for the paper's F2 sentence; keeps CI & tails consistent with r."""
    pairs = [(r["static"], r["boost"]) for r in rows.values()
             if r["static"] is not None and r["boost"] is not None]
    if len(pairs) < 12:
        return None
    st = np.array([a for a, _ in pairs]); bo = np.array([b for _, b in pairs])
    rng = np.random.default_rng(42); rs = []
    for _ in range(5000):
        idx = rng.integers(0, len(st), len(st))
        if np.std(st[idx]) > 0:
            rs.append(stats.pearsonr(st[idx], bo[idx])[0])
    order = np.argsort(st)
    return {"n": len(pairs),
            "r_bootstrap_ci95": [round(float(np.percentile(rs, 2.5)), 3),
                                 round(float(np.percentile(rs, 97.5)), 3)],
            "top10_mean_boost": round(float(bo[order[-10:]].mean()), 4),
            "bottom10_mean_boost": round(float(bo[order[:10]].mean()), 4)}


def rank_active(rows):
    return {m: r["active"] for m, r in rows.items() if r["active"] is not None}


def spearman_rank(a, b):
    keys = [k for k in a if k in b]
    return float(stats.spearmanr([a[k] for k in keys], [b[k] for k in keys])[0])


def main():
    raw = load_raw()
    critics = sorted({c for cmap in raw.values() for c in cmap})
    out = {"critics": critics}

    # baseline (current weights, all critics, all idx)
    base_sw, meta = seed_weighted(raw, WCUR)
    base_rows = model_means(base_sw)
    out["baseline_f2"] = f2_stats(base_rows)
    out["baseline_f2_extra"] = f2_extra(base_rows)
    base_rank = rank_active(base_rows)

    # ---- A: weight sensitivity ----
    schemes = {
        "current_O2F1C.5I1.5S.5": WCUR,
        "equal": {d: 1.0 for d in DIMS},
        "originality_only": {d: (1.0 if d == "originality" else 0.0) for d in DIMS},
        "drop_clarity_specificity": {"originality": 2.0, "feasibility": 1.0, "clarity": 0.0,
                                     "impact": 1.5, "specificity": 0.0},
    }
    A = {}
    for name, wsch in schemes.items():
        sw, _ = seed_weighted(raw, wsch)
        rows = model_means(sw)
        f2 = f2_stats(rows)
        A[name] = {"f2": f2, "rank_spearman_vs_current": spearman_rank(rank_active(rows), base_rank)}
    out["A_weight_sensitivity"] = A

    # ---- B: per-dimension boost decomposition ----
    # per model, per dim: active mean - static mean (trimmed over critics, mean over seeds)
    dim_seed = defaultdict(lambda: defaultdict(list))  # [(m,tr)][dim]=seed means
    perseed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # [(m,tr,sub)][dim]=idea trimmed
    for k, cmap in raw.items():
        m, tr, sub, dom, idx = k
        for dim in DIMS:
            perseed[(m, tr, sub)][dim][idx].append(tm([v[dim] for v in cmap.values()]))
    seedmean = defaultdict(lambda: defaultdict(dict))
    for (m, tr, sub), dd in perseed.items():
        for dim in DIMS:
            allidx = [np.mean(v) for v in dd[dim].values()]
            seedmean[(m, tr)].setdefault(dim, []).append(np.mean(allidx))
    B = {}
    for dim in DIMS:
        diffs = []
        st_static_all = []
        for m in {mm for (mm, tr) in seedmean}:
            b = seedmean.get((m, "B"), {}).get(dim); c = seedmean.get((m, "C"), {}).get(dim)
            if b and c:
                diffs.append((base_rows[m]["static"], np.mean(c) - np.mean(b)))
        if diffs:
            st = [a for a, _ in diffs]; db = [b for _, b in diffs]
            B[dim] = {"mean_dim_boost": float(np.mean(db)),
                      "r_static_dimboost": float(stats.pearsonr(st, db)[0]) if len(db) >= 3 else None}
    out["B_per_dim_boost"] = B

    # ---- C: critic jackknife ----
    C = {}
    for c in critics:
        sw, _ = seed_weighted(raw, WCUR, critics={c})
        C[f"only::{c.split('/')[-1]}"] = {"f2": f2_stats(model_means(sw)),
                                          "rank_spearman_vs_all3": spearman_rank(rank_active(model_means(sw)), base_rank)}
    for c in critics:  # leave-one-out
        keep = [x for x in critics if x != c]
        sw, _ = seed_weighted(raw, WCUR, critics=set(keep))
        C[f"drop::{c.split('/')[-1]}"] = {"f2": f2_stats(model_means(sw))}
    # cross-critic ranking agreement
    ranks = {c: rank_active(model_means(seed_weighted(raw, WCUR, critics={c})[0])) for c in critics}
    agree = {}
    for i in range(len(critics)):
        for j in range(i + 1, len(critics)):
            agree[f"{critics[i].split('/')[-1]}~{critics[j].split('/')[-1]}"] = spearman_rank(ranks[critics[i]], ranks[critics[j]])
    out["C_critic_jackknife"] = {"variants": C, "cross_critic_active_rank_spearman": agree}

    # ---- D: ideas/seed convergence ----
    D = {}
    for label, idxset in (("idx1", {1}), ("idx12", {1, 2}), ("idx123", {1, 2, 3})):
        sw, _ = seed_weighted(raw, WCUR, idxset=idxset)
        D[label] = f2_stats(model_means(sw))
    out["D_ideas_per_seed_convergence"] = D

    # ---- E: per-domain cutoff slopes ----
    cutoffs = _load_dict("reports/_make_cross_year_plot.py", "KNOWLEDGE_CUTOFFS")
    # per (model, domain) means
    dom_mt = defaultdict(lambda: defaultdict(list))  # [(dom,track)][model]=seed means
    for (m, tr, sub), w in base_sw.items():
        dom = meta[(m, tr, sub)]
        dom_mt[(dom, tr)].setdefault(m, []).append(w)
    E = {}
    for dom in sorted({d for (d, _) in dom_mt}):
        res = {}
        for tr, nm in (("B", "static"), ("C", "active")):
            pts = []
            for m, ws in dom_mt.get((dom, tr), {}).items():
                if m in cutoffs:
                    pts.append((cutoffs[m], np.mean(ws)))
            if len(pts) >= 3:
                sl, ic, rr, pp, _ = stats.linregress([a for a, _ in pts], [b for _, b in pts])
                res[nm] = {"n": len(pts), "slope": float(sl), "r2": float(rr**2), "p": float(pp)}
        out.setdefault("E_per_domain_slopes", {})[dom] = res

    # ---- F: reasoning vs non-reasoning ----
    grp = {"reasoning": [], "non_reasoning": []}
    for m, r in base_rows.items():
        if r["boost"] is None:
            continue
        (grp["reasoning"] if m in REASONING else grp["non_reasoning"]).append(r)
    F = {}
    for g, rs in grp.items():
        if rs:
            F[g] = {"n": len(rs), "mean_static": float(np.mean([x["static"] for x in rs])),
                    "mean_active": float(np.mean([x["active"] for x in rs])),
                    "mean_boost": float(np.mean([x["boost"] for x in rs]))}
    out["F_reasoning_split"] = F

    # ---- G: exclude self-evaluation (critic == idea model) ----
    # kimi-k2.6 / glm-5.1 are both critics AND generators → drop self-grades, recompute F2.
    raw_ns = {}
    for k, cmap in raw.items():
        m = k[0]
        filtered = {c: v for c, v in cmap.items() if c != m}
        if filtered:
            raw_ns[k] = filtered
    sw_ns, _ = seed_weighted(raw_ns, WCUR)
    rows_ns = model_means(sw_ns)
    out["G_exclude_self_eval"] = {
        "f2": f2_stats(rows_ns),
        "rank_spearman_vs_baseline": spearman_rank(rank_active(rows_ns), base_rank),
        "note": "drop critic rows where critic_model == idea_model (kimi-k2.6, glm-5.1 self-grades)",
    }

    outpath = ROOT / "reports" / "e23_ablations.json"
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # console summary
    b = out["baseline_f2"]
    print(f"baseline F2: n={b['n']} boost={b['mean_boost']:+.2f} pos={b['pos']} "
          f"pearson r={b['pearson_r']:+.2f} p={b['pearson_p']:.3g}")
    print("\n[A weight] pearson r(static,boost) & rank-corr vs current:")
    for k, v in out["A_weight_sensitivity"].items():
        f = v["f2"]; print(f"  {k:<28} r={f['pearson_r']:+.2f} p={f['pearson_p']:.3g} "
                            f"boost={f['mean_boost']:+.2f} rankρ={v['rank_spearman_vs_current']:.3f}")
    print("\n[B per-dim boost] mean dim-boost | r(static,dimboost):")
    for dim, v in out["B_per_dim_boost"].items():
        print(f"  {dim:<12} boost={v['mean_dim_boost']:+.2f} r={v['r_static_dimboost']}")
    print("\n[C critic] single-critic F2 pearson r & cross-critic active-rank ρ:")
    for k, v in out["C_critic_jackknife"]["variants"].items():
        if k.startswith("only"):
            print(f"  {k:<26} r={v['f2']['pearson_r']:+.2f} rankρvs3={v['rank_spearman_vs_all3']:.3f}")
    print("  cross-critic:", {k: round(v, 2) for k, v in out["C_critic_jackknife"]["cross_critic_active_rank_spearman"].items()})
    print("\n[D ideas/seed] pearson r(static,boost):")
    for k, v in out["D_ideas_per_seed_convergence"].items():
        print(f"  {k:<8} r={v['pearson_r']:+.2f} p={v['pearson_p']:.3g} boost={v['mean_boost']:+.2f} pos={v['pos']}/{v['n']}")
    print("\n[E per-domain slopes] static | active (slope):")
    for dom, v in out.get("E_per_domain_slopes", {}).items():
        s = v.get("static", {}); a = v.get("active", {})
        print(f"  {dom:<10} static={s.get('slope', float('nan')):+.3f}(n{s.get('n','?')}) "
              f"active={a.get('slope', float('nan')):+.3f}(n{a.get('n','?')})")
    print("\n[F reasoning split]:", {g: {kk: round(vv, 2) for kk, vv in v.items() if kk != 'n'} | {'n': v['n']}
                                     for g, v in out["F_reasoning_split"].items()})
    g = out["G_exclude_self_eval"]
    print(f"\n[G exclude self-eval] F2 pearson r={g['f2']['pearson_r']:+.2f} p={g['f2']['pearson_p']:.3g} "
          f"boost={g['f2']['mean_boost']:+.2f} | rankρ vs baseline={g['rank_spearman_vs_baseline']:.3f}")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
