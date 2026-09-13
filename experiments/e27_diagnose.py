"""E27 diagnosis — why is S4b not optimal on the deepseek backbones?

Read-only on data/results.db. Answers, per backbone:
  1. Ceiling check: absolute weighted lit8d score for active_base vs S2/S4/S4b
     (if base is already near the critic ceiling, no SWM design can add).
  2. SWM engagement: mean n_sims / n_tool_calls per condition (did the agent
     actually call SIMULATE and act on it?).
  3. Is S4>S4b on deepseek a real effect or noise? paired S4b-S4 per cell,
     mean + sign + Wilcoxon; same for S4b-base.
  4. Where does S4b lose vs S4 on deepseek? per-dimension S4b-S4.

  /usr/bin/python3 experiments/e27_diagnose.py
"""
import sqlite3, json, sys
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
MODELS = ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b",
          "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"]
CONDS = ["active_base", "swm_S2", "swm_S4", "swm_S4b"]


def _tm(vs):
    return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)


def load_cells():
    """(_dim_cell, _w_cell): weighted + per-dim trimmed-mean score per (model,sub,cond)."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT gen_model,subdomain,condition,scores_json "
                          "FROM swm_scores WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"]); k = (r["gen_model"], r["subdomain"], r["condition"])
        for d in DIMS:
            cell[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()
    dim_cell = {k: {d: _tm(cell[k][d]) for d in DIMS} for k in cell}
    w_cell = {k: sum(v[d] * W[d] for d in DIMS) / WSUM for k, v in dim_cell.items()}
    return dim_cell, w_cell


def load_engagement():
    """mean n_sims / n_tool_calls per (model,cond)."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    eng = defaultdict(lambda: {"n_sims": [], "n_tool_calls": []})
    for r in conn.execute("SELECT gen_model,condition,aux_json FROM swm_ideas WHERE TRIM(idea_text)!=''"):
        aux = json.loads(r["aux_json"] or "{}")
        k = (r["gen_model"], r["condition"])
        if aux.get("n_sims") is not None:
            eng[k]["n_sims"].append(aux["n_sims"])
        if aux.get("n_tool_calls") is not None:
            eng[k]["n_tool_calls"].append(aux["n_tool_calls"])
    conn.close()
    return eng


def main():
    dim_cell, w_cell = load_cells()
    eng = load_engagement()
    by_ms = defaultdict(dict); dby_ms = defaultdict(dict)
    for (m, sub, c), w in w_cell.items():
        by_ms[(m, sub)][c] = w; dby_ms[(m, sub)][c] = dim_cell[(m, sub, c)]

    print("=" * 78)
    print("1. CEILING CHECK — absolute weighted lit8d score per model/condition")
    print("=" * 78)
    print(f"{'backbone':26s} {'base':>7s} {'S2':>7s} {'S4':>7s} {'S4b':>7s}")
    for m in MODELS:
        vals = {}
        for c in CONDS:
            xs = [by_ms[k][c] for k in by_ms if k[0] == m and c in by_ms[k]]
            vals[c] = np.mean(xs) if xs else None
        row = " ".join(f"{vals[c]:7.2f}" if vals[c] is not None else f"{'--':>7s}" for c in CONDS)
        print(f"{m:26s} {row}")

    print()
    print("=" * 78)
    print("2. SWM ENGAGEMENT — mean n_sims / n_tool_calls per model/condition")
    print("=" * 78)
    print(f"{'backbone':26s} {'cond':10s} {'n_sims':>7s} {'n_tool':>7s}")
    for m in MODELS:
        for c in CONDS:
            e = eng.get((m, c), {"n_sims": [], "n_tool_calls": []})
            ns = np.mean(e["n_sims"]) if e["n_sims"] else float("nan")
            nt = np.mean(e["n_tool_calls"]) if e["n_tool_calls"] else float("nan")
            print(f"{m:26s} {c:10s} {ns:7.2f} {nt:7.2f}")

    print()
    print("=" * 78)
    print("3. IS S4>S4b ON DEEPSEEK REAL OR NOISE? paired per-cell contrasts")
    print("=" * 78)

    def paired(m, ca, cb):
        """ca - cb per matched subdomain cell."""
        d = [by_ms[k][ca] - by_ms[k][cb] for k in by_ms
             if k[0] == m and ca in by_ms[k] and cb in by_ms[k]]
        return np.array(d)

    for m in MODELS:
        for (ca, cb) in [("swm_S4b", "active_base"), ("swm_S4b", "swm_S4"), ("swm_S4", "active_base")]:
            d = paired(m, ca, cb)
            if len(d) < 2:
                continue
            p = stats.wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
            print(f"{m:26s} {ca.split('_')[-1]:4s}-{cb.split('_')[-1]:6s} "
                  f"n={len(d):2d} mean={d.mean():+.3f} median={np.median(d):+.3f} "
                  f"win={(d>0).mean():.2f} wilcoxon_p={p:.3f}")
        print()

    print("=" * 78)
    print("4. WHERE S4b LOSES vs S4 ON DEEPSEEK — per-dimension S4b - S4")
    print("=" * 78)
    print(f"{'backbone':26s} " + " ".join(f"{d[:5]:>7s}" for d in DIMS))
    for m in MODELS:
        row = []
        for d in DIMS:
            diffs = [dby_ms[k]["swm_S4b"][d] - dby_ms[k]["swm_S4"][d] for k in dby_ms
                     if k[0] == m and "swm_S4b" in dby_ms[k] and "swm_S4" in dby_ms[k]
                     and dby_ms[k]["swm_S4b"][d] is not None and dby_ms[k]["swm_S4"][d] is not None]
            row.append(np.mean(diffs) if diffs else float("nan"))
        print(f"{m:26s} " + " ".join(f"{v:+7.2f}" for v in row))


if __name__ == "__main__":
    main()
