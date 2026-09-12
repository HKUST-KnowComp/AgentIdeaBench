"""E6 — Recency vs capability: what drives the cutoff–score correlation (F3)?

F3 confound: newer models are both "more recent" (later knowledge cutoff) and
"more capable". Does the cutoff–score climb come from (a) memorising / having
seen content close to the target, or (b) general capability?

KEY STRUCTURAL FACT (verified from papers.db): every filtered test paper is
published 2025-2026, i.e. AFTER the knowledge cutoff of essentially every model
in the roster (anti-leakage design). So:
  - before/after-cutoff split is (by construction) almost entirely "after" —
    no model has seen the *target* papers. Direct memorisation of the target is
    ruled out by design. We report the split counts to make this explicit.
  - The remaining question is whether models do worse on papers published
    FURTHER beyond their cutoff (knowledge-recency) or whether paper recency is
    irrelevant once you control for the model (pure capability). We test this
    with a within-model regression of score on recency_gap = paper_year - cutoff.

Analysis (uniform modern5 scores; READ-ONLY):
  1. Split counts: per model, #papers before vs after its cutoff.
  2. recency_gap regression: pool all (model, paper) scored cells, demean score
     within each model (model fixed effect), regress demeaned_score on
     recency_gap. Slope ~ 0 => capability, not target-recency, drives F3.
     Done separately for Static (B) and Active (C).
  3. Partial correlation with an EXTERNAL capability proxy (Arena Elo / MMLU):
     marked not computed — no reliable public capability score exists for these
     specific model versions; using params is disallowed (plan E6). Documented.

Writes reports/e6_recency/e6_recency_vs_capability.json (+ console).

Usage:
  python experiments/e6_recency_vs_capability.py
"""
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as WEIGHTS
import reports._make_cross_year_plot as mc

CUTOFFS = mc.KNOWLEDGE_CUTOFFS


def weighted_norm(s):
    w = sum(s.get(d, 0) * WEIGHTS[d] for d in DIMS if d in s)
    t = sum(WEIGHTS[d] for d in DIMS if d in s)
    return w / t if t else 0.0


def trimmed_mean(vs):
    if not vs:
        return None
    if len(vs) < 2:
        return vs[0]
    return statistics.mean(sorted(vs)[:-1])


def date_to_decimal_year(s):
    if not s or len(s) < 7:
        return None
    try:
        y = int(s[:4]); m = int(s[5:7])
        return y + (m - 0.5) / 12.0
    except Exception:
        return None


def load():
    cp = sqlite3.connect(str(cfg.PAPERS_DB)); cp.row_factory = sqlite3.Row
    pdate = {r["paper_id"]: date_to_decimal_year(r["published_date"])
             for r in cp.execute("SELECT paper_id, published_date FROM papers WHERE status='filtered'")}
    cp.close()
    cr = sqlite3.connect(str(cfg.RESULTS_DB)); cr.row_factory = sqlite3.Row
    rows = [dict(r) for r in cr.execute(
        "SELECT paper_id, idea_model, track, idea_index, scores_json "
        "FROM uniform_critic_scores WHERE scores_json IS NOT NULL")]
    cr.close()
    return pdate, rows


def cell_scores(rows):
    """(model,track,paper,idx)->trimmed weighted; then best idx per (model,track,paper)."""
    cell = defaultdict(list)
    for r in rows:
        try:
            s = json.loads(r["scores_json"])
        except Exception:
            continue
        cell[(r["idea_model"], r["track"], r["paper_id"], r["idea_index"])].append(weighted_norm(s))
    idx = {k: trimmed_mean(v) for k, v in cell.items() if trimmed_mean(v) is not None}
    best = {}
    for (m, t, p, i), sc in idx.items():
        key = (m, t, p)
        if key not in best or sc > best[key]:
            best[key] = sc
    return best   # (model,track,paper) -> score


def recency_regression(best, pdate, track):
    """Within-model demeaned score vs recency_gap (paper_year - cutoff)."""
    by_model = defaultdict(list)   # model -> [(gap, score)]
    for (m, t, p), sc in best.items():
        if t != track:
            continue
        co = CUTOFFS.get(m); py = pdate.get(p)
        if co is None or py is None:
            continue
        by_model[m].append((py - co, sc))
    gaps, dscores = [], []
    per_model_slopes = []
    for m, pts in by_model.items():
        if len(pts) < 4:
            continue
        mu = statistics.mean([s for _, s in pts])
        for g, s in pts:
            gaps.append(g); dscores.append(s - mu)
        gx = [g for g, _ in pts]; gy = [s for _, s in pts]
        if len(set(gx)) > 2:
            r = stats.linregress(gx, gy)
            per_model_slopes.append({"model": m, "slope": float(r.slope),
                                     "r2": float(r.rvalue ** 2), "n": len(pts)})
    if len(gaps) < 5:
        return {"n": len(gaps), "note": "insufficient"}
    reg = stats.linregress(gaps, dscores)
    return {
        "n_cells": len(gaps), "n_models": len(by_model),
        "pooled_demeaned_slope_per_yr": float(reg.slope),
        "slope_p": float(reg.pvalue), "r2": float(reg.rvalue ** 2),
        "gap_range": [round(min(gaps), 2), round(max(gaps), 2)],
        "per_model_slopes": sorted(per_model_slopes, key=lambda x: x["slope"]),
        "mean_per_model_slope": (round(float(np.mean([s["slope"] for s in per_model_slopes])), 4)
                                 if per_model_slopes else None),
    }


def main():
    pdate, rows = load()
    best = cell_scores(rows)

    # ---- (1) before/after-cutoff split counts ----
    split = {}
    models = sorted({m for (m, t, p) in best})
    n_after_total = n_before_total = 0
    for m in models:
        co = CUTOFFS.get(m)
        if co is None:
            continue
        before = after = 0
        for (mm, t, p), sc in best.items():
            if mm != m or t != "B":
                continue
            py = pdate.get(p)
            if py is None:
                continue
            if py < co:
                before += 1
            else:
                after += 1
        split[m] = {"cutoff": round(co, 3), "papers_before_cutoff": before,
                    "papers_after_cutoff": after}
        n_before_total += before; n_after_total += after

    # ---- (2) recency_gap regression ----
    f_static = recency_regression(best, pdate, "B")
    f_active = recency_regression(best, pdate, "C")

    result = {
        "design_fact": {
            "all_test_papers_published": "2025-2026",
            "total_static_cells_before_cutoff": n_before_total,
            "total_static_cells_after_cutoff": n_after_total,
            "interpretation": ("Nearly all (model,paper) pairs are post-cutoff: the "
                               "target papers are unseen by construction, so direct "
                               "memorisation of the target is ruled out by the "
                               "anti-leakage design. before/after split is degenerate."),
        },
        "recency_gap_regression": {
            "static_B": f_static, "active_C": f_active,
            "reading": ("pooled_demeaned_slope ~ 0 (and per-model slopes centred on 0) "
                        "=> paper recency beyond a model's cutoff does NOT lower its "
                        "score once the model is controlled => F3 cutoff climb reflects "
                        "capability / general knowledge, not target-recency memorisation."),
        },
        "partial_correlation_external_proxy": {
            "status": "not computed",
            "reason": ("No reliable public capability score (Arena Elo / MMLU-Pro) exists "
                       "for these specific model versions (e.g. gpt-5.4, qwen3.5, glm-5.1); "
                       "params are disallowed as a proxy per plan E6. Would need a curated "
                       "external proxy table to compute r(cutoff,score|cap) vs r(cap,score|cutoff)."),
        },
        "split_per_model": split,
    }

    out = ROOT / "reports" / "e6_recency"; out.mkdir(parents=True, exist_ok=True)
    (out / "e6_recency_vs_capability.json").write_text(json.dumps(result, indent=2))

    print("=== E6 before/after-cutoff split (Static cells) ===")
    print(f"  before-cutoff total: {n_before_total} | after-cutoff total: {n_after_total}")
    print("  => target papers unseen by design; memorisation-of-target ruled out.")
    print("\n=== Recency-gap regression (score vs years-past-cutoff, model-demeaned) ===")
    for tag, f in [("Static B", f_static), ("Active C", f_active)]:
        if "pooled_demeaned_slope_per_yr" in f:
            print(f"  {tag}: slope={f['pooled_demeaned_slope_per_yr']:+.4f}/yr "
                  f"p={f['slope_p']:.3f} r2={f['r2']:.3f} "
                  f"(n_cells={f['n_cells']}, n_models={f['n_models']}, "
                  f"mean per-model slope={f['mean_per_model_slope']})")
        else:
            print(f"  {tag}: {f}")
    print(f"\n✓ wrote {out}/e6_recency_vs_capability.json")


if __name__ == "__main__":
    main()
