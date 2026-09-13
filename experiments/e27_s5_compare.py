"""E27 S5 comparison — does the gated variant (S5) remove S4b's over-editing penalty
on the saturated deepseek backbones?

Read-only. For each deepseek backbone, on subdomains that have an S5 cell, compute paired
deltas vs active_base for S5 / S4b / S4 (matched subdomains only), plus the head-to-head
S5-S4b and S5-S4. Directional pilot (S5 gen is throttled), so report n and Wilcoxon p.

  /usr/bin/python3 experiments/e27_s5_compare.py
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
DEEP = ["deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"]


def _tm(v): return trimmed_mean(v) if len(v) >= 2 else (v[0] if v else None)


def main():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT gen_model,subdomain,condition,scores_json "
                          "FROM swm_scores WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"]); k = (r["gen_model"], r["subdomain"], r["condition"])
        for d in DIMS:
            cell[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()
    w = {k: sum(_tm(cell[k][d]) * W[d] for d in DIMS) / WSUM for k in cell}
    byms = defaultdict(dict)
    for (m, sub, c), v in w.items():
        byms[(m, sub)][c] = v

    def paired(m, ca, cb):
        d = [byms[k][ca] - byms[k][cb] for k in byms
             if k[0] == m and ca in byms[k] and cb in byms[k]]
        return np.array(d)

    def line(m, ca, cb):
        d = paired(m, ca, cb)
        if len(d) < 2:
            return f"    {ca.split('_')[-1]:4s}-{cb.split('_')[-1]:5s} n={len(d):2d}  (too few)"
        p = stats.wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
        return (f"    {ca.split('_')[-1]:4s}-{cb.split('_')[-1]:5s} n={len(d):2d} "
                f"mean={d.mean():+.3f} median={np.median(d):+.3f} win={(d>0).mean():.2f} p={p:.3f}")

    print("S5 (gated S4b) pilot comparison on deepseek — matched subdomains only\n")
    for m in DEEP:
        n5 = sum(1 for k in byms if k[0] == m and "swm_S5" in byms[k])
        print(f"{m}  (S5 cells scored: {n5})")
        # deltas vs base (on S5-having subdomains, for apples-to-apples restrict to S5 subs)
        s5subs = {k[1] for k in byms if k[0] == m and "swm_S5" in byms[k]}
        def paired_on(ca, cb, subs):
            d = [byms[(m, s)][ca] - byms[(m, s)][cb] for s in subs
                 if (m, s) in [(m, x) for x in [s]] and ca in byms[(m, s)] and cb in byms[(m, s)]]
            return np.array(d)
        for ca in ("swm_S5", "swm_S4b", "swm_S4"):
            d = np.array([byms[(m, s)][ca] - byms[(m, s)]["active_base"] for s in s5subs
                          if ca in byms[(m, s)] and "active_base" in byms[(m, s)]])
            if len(d):
                p = stats.wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
                print(f"    {ca.split('_')[-1]:4s}-base  n={len(d):2d} mean={d.mean():+.3f} "
                      f"win={(d>0).mean():.2f} p={p:.3f}   (on S5 subdomains)")
        print(line(m, "swm_S5", "swm_S4b"))
        print(line(m, "swm_S5", "swm_S4"))
        # per-dim S5 - base
        pd = {}
        for dim in DIMS:
            vs = []
            for s in s5subs:
                a = cell.get((m, s, "swm_S5")); b = cell.get((m, s, "active_base"))
                if a and b and a.get(dim) and b.get(dim):
                    vs.append(_tm(a[dim]) - _tm(b[dim]))
            pd[dim] = np.mean(vs) if vs else float("nan")
        print("    per-dim S5-base: " + " ".join(f"{d[:4]}{pd[d]:+.2f}" for d in DIMS))
        print()


if __name__ == "__main__":
    main()
