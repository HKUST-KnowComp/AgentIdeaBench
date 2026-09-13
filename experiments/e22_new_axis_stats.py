"""E22 — recompute the old cutoff-axis statistics (F0/F1/F3) on the NEW paradigm:
v3 subdomain ideas (3 ideas/seed) scored by lit8d (table lit8d_scores_3seed).

The old F1/F3/F0 were computed on the OLD `results` table (smoke25 papers,
v1/v2 refs) scored by the modern5 uniform critic + old rubric. This script keeps
the SAME statistical recipe but swaps in:
  - idea data: per-model Static (track B) / Active (track C) mean over the
    pilot-20 subdomains, seed = mean of its 3 ideas (matches E21).
  - critic:    lit8d (literature-verified), weighted O2/F1/C0.5/I1.5/S0.5.
  - x-axis:    knowledge cutoff (decimal year), reused from
               reports/_make_cross_year_plot.py.
  - params:    reused from reports/_f0_partial_correlation.py.

Outputs primary numbers to reports/e22_new_axis_stats.json. Read-only on the DB.

  /usr/bin/python3 experiments/e22_new_axis_stats.py
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

WSUM = sum(W.values())


def to_decimal_year(y, m, d):
    from datetime import date
    start = date(y, 1, 1).toordinal()
    return y + (date(y, m, d).toordinal() - start) / 365.25


def _load_dict(src_file, name):
    """Exec only the target assignment from a report script, with
    to_decimal_year available, to grab a module-level dict literal."""
    src = (ROOT / src_file).read_text()
    tree = ast.parse(src)
    ns = {"to_decimal_year": to_decimal_year, "math": math}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    exec(compile(ast.Module([node], []), "<x>", "exec"), ns)
                    return ns[name]
    raise KeyError(name)


def seed_scores():
    """seed_w[(model, track)] -> list of per-seed weighted scores (3-idea mean)."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    idea = defaultdict(lambda: defaultdict(list)); imeta = {}
    for r in conn.execute("SELECT idea_model,track,subdomain,idea_index,scores_json "
                          "FROM lit8d_scores_3seed WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"]); k = (r["idea_model"], r["track"], r["subdomain"], r["idea_index"])
        imeta[k] = (r["idea_model"], r["track"])
        for d in DIMS:
            idea[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()

    def tm(vs): return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)
    idea_w = {k: sum(tm(idea[k][d]) * W[d] for d in DIMS) / WSUM for k in idea}
    seed_vals = defaultdict(list)
    for (m, tr, sub, idx), w in idea_w.items():
        seed_vals[(m, tr, sub)].append(w)
    # collapse idx -> seed mean, then bucket by (model, track)
    by_mt = defaultdict(list)
    for (m, tr, sub), ws in seed_vals.items():
        by_mt[(m, tr)].append(float(np.mean(ws)))
    return by_mt


def partial_corr(x, y, z):
    """partial r(x, y | z)."""
    x, y, z = map(np.asarray, (x, y, z))
    def resid(a):
        b = np.polyfit(z, a, 1)
        return a - (b[0] * z + b[1])
    rx, ry = resid(x), resid(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    cutoffs = _load_dict("reports/_make_cross_year_plot.py", "KNOWLEDGE_CUTOFFS")
    params = _load_dict("reports/_f0_partial_correlation.py", "PARAMS")
    by_mt = seed_scores()

    # per-model Static / Active means
    models = sorted({m for (m, tr) in by_mt})
    rows = []
    for m in models:
        b = by_mt.get((m, "B"), []); c = by_mt.get((m, "C"), [])
        rows.append({
            "model": m,
            "cutoff": cutoffs.get(m),
            "log_params": math.log10(params[m]) if m in params else None,
            "static": float(np.mean(b)) if b else None,
            "active": float(np.mean(c)) if c else None,
            "nB": len(b), "nC": len(c),
        })

    out = {"note": "lit8d 3-seed (lit8d_scores_3seed), all scored subdomains (40 per model, nB=nC=40); same recipe as old F1/F3/F0 but new data+critic",
           "per_model": rows}

    # ---- F1: Static vs cutoff (linear) ----
    f1 = [(r["cutoff"], r["static"]) for r in rows if r["cutoff"] and r["static"] is not None]
    xs = [a for a, _ in f1]; ys = [b for _, b in f1]
    sl, ic, rr, pp, se = stats.linregress(xs, ys)
    out["F1_static_vs_cutoff"] = {
        "n": len(f1), "slope_per_yr": sl, "intercept": ic, "r": rr, "r2": rr**2, "p": pp,
    }
    # yearly spread (by cutoff calendar year)
    yr_bucket = defaultdict(list)
    for r in rows:
        if r["cutoff"] and r["static"] is not None:
            yr_bucket[int(r["cutoff"])].append(r["static"])
    out["F1_yearly_spread"] = {str(y): {"n": len(v), "mean": float(np.mean(v)),
                                        "min": float(np.min(v)), "max": float(np.max(v)),
                                        "spread": float(np.max(v) - np.min(v))}
                               for y, v in sorted(yr_bucket.items())}

    # ---- F3: Static & Active slopes vs cutoff + ratio + boost~cutoff ----
    f3s = [(r["cutoff"], r["static"]) for r in rows if r["cutoff"] and r["static"] is not None]
    f3a = [(r["cutoff"], r["active"]) for r in rows if r["cutoff"] and r["active"] is not None]
    ssl, sic, sr, sp, _ = stats.linregress([a for a, _ in f3s], [b for _, b in f3s])
    asl, aic, ar, ap, _ = stats.linregress([a for a, _ in f3a], [b for _, b in f3a])
    # boost vs cutoff (matched models with both tracks + cutoff)
    bc = [(r["cutoff"], r["active"] - r["static"]) for r in rows
          if r["cutoff"] and r["static"] is not None and r["active"] is not None]
    bsp_r, bsp_p = stats.spearmanr([a for a, _ in bc], [b for _, b in bc])
    out["F3_slopes_vs_cutoff"] = {
        "static": {"n": len(f3s), "slope_per_yr": ssl, "r2": sr**2, "p": sp},
        "active": {"n": len(f3a), "slope_per_yr": asl, "r2": ar**2, "p": ap},
        "active_over_static_slope_ratio": asl / ssl if ssl else None,
        "boost_vs_cutoff_spearman": {"n": len(bc), "rho": bsp_r, "p": bsp_p},
    }

    # ---- F0: partial correlations (Static, cutoff, log_params) ----
    f0 = [r for r in rows if r["cutoff"] and r["log_params"] is not None and r["static"] is not None]
    X = [r["cutoff"] for r in f0]; Y = [r["static"] for r in f0]; Z = [r["log_params"] for r in f0]
    out["F0_partial_corr"] = {
        "n": len(f0),
        "r_cutoff_score": float(np.corrcoef(X, Y)[0, 1]),
        "r_logparams_score": float(np.corrcoef(Z, Y)[0, 1]),
        "r_cutoff_logparams": float(np.corrcoef(X, Z)[0, 1]),
        "partial_r_cutoff_score_given_logparams": partial_corr(X, Y, Z),
        "partial_r_logparams_score_given_cutoff": partial_corr(Z, Y, X),
    }

    outpath = ROOT / "reports" / "e22_new_axis_stats.json"
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # console summary
    print(f"models with lit8d 3-seed scores: {len(rows)}")
    f = out["F1_static_vs_cutoff"]
    print(f"\n[F1] Static ~ cutoff: n={f['n']} slope={f['slope_per_yr']:+.3f}/yr "
          f"R2={f['r2']:.3f} p={f['p']:.2g}")
    print("     yearly spread:", {y: f"{v['spread']:.2f}(n{v['n']})" for y, v in out["F1_yearly_spread"].items()})
    f3 = out["F3_slopes_vs_cutoff"]
    print(f"[F3] Static slope={f3['static']['slope_per_yr']:+.3f} (R2={f3['static']['r2']:.3f} p={f3['static']['p']:.2g}) | "
          f"Active slope={f3['active']['slope_per_yr']:+.3f} (R2={f3['active']['r2']:.3f} p={f3['active']['p']:.2g}) | "
          f"ratio={f3['active_over_static_slope_ratio']:.2f}x")
    print(f"     boost~cutoff Spearman rho={f3['boost_vs_cutoff_spearman']['rho']:+.3f} "
          f"p={f3['boost_vs_cutoff_spearman']['p']:.3g} (n={f3['boost_vs_cutoff_spearman']['n']})")
    f0d = out["F0_partial_corr"]
    print(f"[F0] n={f0d['n']} | partial r(cutoff,score|params)={f0d['partial_r_cutoff_score_given_logparams']:+.3f} | "
          f"partial r(params,score|cutoff)={f0d['partial_r_logparams_score_given_cutoff']:+.3f} | "
          f"r(cutoff,params)={f0d['r_cutoff_logparams']:+.3f}")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
