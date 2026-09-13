"""E15 — Goodhart check: were the v2 anchors selected by overfitting to the 3
critics used in E13/E14 scoring?

The v2 anchor recommendation was ranked by lit8c scores from CRITIC_MODELS[:3]
(qwen3.6-plus, kimi-k2.6, glm-5.1). If that ranking is real (not critic-pool
overfitting), HELD-OUT critics not used in selection should also rank the v2
anchors well above v4 ideas.

Held-out critics = CRITIC_MODELS[3:] = minimax-m2.7, deepseek-v4-flash. We score
anchors + candidates + deepseek-v4-pro Active ideas under lit8c with these two.
(v4-flash Active ideas are EXCLUDED as a scored group here to avoid v4-flash
self-grading; v4-pro is not in the critic pool, so it is clean.)

Writes variant tag 'lit8c_ho2' into e12_gap_scores (INSERT-only). Reuses the
existing e13_evidence for each item.

Usage:
  /usr/bin/python3 experiments/e15_goodhart_check.py --score
  /usr/bin/python3 experiments/e15_goodhart_check.py --analyze
"""
import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as W
from experiments.e10_idea_anchor_calibration import score_one, trimmed_mean
from experiments.e13_litverify_rubric8 import (
    build_items, format_evidence_block, LIT8C_SYSTEM,
)
from experiments.e14_anchor_recruit import load_candidates

VARIANT = "lit8c_ho2"
HELDOUT = list(cfg.CRITIC_MODELS)[3:]      # minimax-m2.7, deepseek-v4-flash
WSUM = sum(W.values())
V2NEW = ["cand_05", "cand_02", "cand_07", "cand_06", "cand_04", "cand_08"]
V2OLD = ["anchor_11", "anchor_07", "anchor_01", "anchor_15"]


def _evidence_refs(conn):
    refs = {}
    for r in conn.execute("SELECT item_id, evidence_json, cutoff_date FROM e13_evidence"):
        refs[r[0]] = format_evidence_block(json.loads(r[1] or "[]"), r[2])
    return refs


def score(workers=6):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    refs = _evidence_refs(conn)

    # item set: anchors + controls + v4-pro-C (25) from build_items, plus candidates
    items = []
    for iid, grp, model, tr, dom, txt in build_items(per_group=25, seed=0):
        if grp.startswith("deepseek-v4-flash"):
            continue   # skip: v4-flash is a held-out critic -> would self-grade
        items.append((iid, grp, dom, txt))
    for c in load_candidates():
        items.append((c["id"], "candidate", c["subdomain"].split("/")[0].strip(),
                      c["abstract"]))

    tasks = []
    for iid, grp, dom, txt in items:
        if iid not in refs:
            continue
        for cr in HELDOUT:
            if not conn.execute(
                    "SELECT 1 FROM e12_gap_scores WHERE item_id=? AND variant=? AND critic_model=?",
                    (iid, VARIANT, cr)).fetchone():
                tasks.append((iid, grp, dom, txt, cr))
    print(f"held-out critics={HELDOUT} items={len(items)} tasks={len(tasks)}")
    if not tasks:
        conn.close()
        return

    def _w(t):
        iid, grp, dom, txt, cr = t
        s, _ = score_one(txt, cr, dom, LIT8C_SYSTEM, refs[iid])
        return (iid, grp, dom, cr, s)

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            iid, grp, dom, cr, s = f.result()
            conn.execute("INSERT OR IGNORE INTO e12_gap_scores VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (iid, grp, grp, "-", dom, VARIANT, cr,
                          json.dumps(s) if s else None,
                          None if s else "parse_fail", ts))
            conn.commit()
            ok += 1 if s else 0
            err += 0 if s else 1
            if i % 20 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close()
    print(f"done: ok={ok} err={err}")


def _load(variant, critics_filter=None):
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(lambda: defaultdict(list)); grp = {}
    q = "SELECT item_id,grp,critic_model,scores_json FROM e12_gap_scores WHERE variant=? AND scores_json IS NOT NULL"
    for r in conn.execute(q, (variant,)):
        if critics_filter and r["critic_model"] not in critics_filter:
            continue
        s = json.loads(r["scores_json"]); grp[r["item_id"]] = r["grp"]
        for d in DIMS:
            cell[r["item_id"]][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()

    def tm(vs):
        return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)
    wit = {}
    for i, dd in cell.items():
        dt = {d: tm(dd[d]) for d in DIMS}
        wit[i] = sum(dt[d] * W[d] for d in DIMS) / WSUM
    return wit, grp


def analyze():
    ho, grp = _load(VARIANT)                       # held-out 2 critics
    sel, _ = _load("lit8c")                          # selection 3 critics

    v2 = [i for i in V2NEW + V2OLD if i in ho]
    v4 = [i for i in ho if grp[i] == "deepseek-v4-pro-C"]
    print(f"\n=== held-out critics ({', '.join(c.split('/')[-1] for c in HELDOUT)}), lit8c ===")
    print(f"v2 anchors(10): mean={np.mean([ho[i] for i in v2]):.2f}  "
          f"min={min(ho[i] for i in v2):.2f}")
    print(f"v4-pro-C(25):   mean={np.mean([ho[i] for i in v4]):.2f}  "
          f"max={max(ho[i] for i in v4):.2f}")
    print(f"gap(v2 mean − v4 max) = {np.mean([ho[i] for i in v2]) - max(ho[i] for i in v4):+.2f}")
    print(f"frac v4 >= v2 mean: {sum(1 for i in v4 if ho[i] >= np.mean([ho[j] for j in v2]))/len(v4):.0%}")

    # rank stability: candidates+anchors scored by both critic sets
    common = [i for i in ho if i in sel and (grp[i] in ("anchor", "candidate"))]
    xs = [sel[i] for i in common]; ys = [ho[i] for i in common]
    rho = stats.spearmanr(xs, ys).correlation
    print(f"\nrank stability on {len(common)} anchor+candidate items:")
    print(f"  Spearman(selection-3-critics, held-out-2-critics) = {rho:.3f}")
    print("  per-candidate (sel3 | ho2):")
    for i in sorted([c for c in common if c.startswith("cand")],
                    key=lambda x: -sel[x]):
        print(f"    {i}: {sel[i]:.2f} | {ho[i]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.score:
        score(a.workers)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
