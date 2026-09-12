"""E17 — bottom-end discrimination check for lit8d.

Key risk in applying lit8d to ALL 27 models: its ceiling-unlock (encourages using
9-10) might also inflate mediocre/weak-model ideas. If lit8d is a valid general
scorer, genuinely weak-model ideas must land LOW (well below the CORE-7 anchors,
near the controls), not get pulled up.

Sample the two weakest static-idea models (by idea length proxy) + one mid model,
run the full lit8d pipeline (SS evidence + 3 critics), and check where they land.

Idempotent; writes variant='lit8d', grp='weakprobe:<model>' into e12_gap_scores.

Usage:
  /usr/bin/python3 experiments/e17_weak_discrim.py --run
"""
import argparse, json, sqlite3, statistics, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as W
from experiments.e10_idea_anchor_calibration import score_one, trimmed_mean
from experiments.e13_litverify_rubric8 import (
    ensure_tables, build_evidence_for_item, format_evidence_block,
    LIT8D_SYSTEM, V4_IDEA_CUTOFF)

WSUM = sum(W.values())
PROBE = {
    "qwen/qwen-2.5-7b-instruct": 8,     # weakest
    "google/gemma-2-27b-it": 8,         # weak
    "qwen/qwen3-32b": 6,                # mid
}
CORE7 = ["cand_05","cand_02","cand_06","cand_04","cand_07","cand_13","cand_10"]


def run(workers=6):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tables(conn)
    critics = list(cfg.CRITIC_MODELS)[:3]
    items = []
    for m, n in PROBE.items():
        short = m.split("/")[-1]
        for r in conn.execute("SELECT subdomain, domain, idea_text FROM subdomain_ideas "
                              "WHERE idea_model=? AND track='B' AND TRIM(idea_text)!='' "
                              "ORDER BY subdomain LIMIT ?", (m, n)):
            iid = f"weak:{short}|{r['subdomain'][:34]}"
            items.append((iid, f"weakprobe:{short}", r["domain"], r["idea_text"]))
    # evidence
    ts = datetime.now(timezone.utc).isoformat()
    need = [it for it in items if not conn.execute("SELECT 1 FROM e13_evidence WHERE item_id=?", (it[0],)).fetchone()]
    print(f"{len(items)} weak ideas, {len(need)} need evidence", flush=True)
    def _ev(it):
        iid, grp, dom, txt = it
        q, ev, ns = build_evidence_for_item(iid, grp, txt, V4_IDEA_CUTOFF, None, None)
        return iid, grp, q, ev
    with ThreadPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(_ev, it) for it in need]):
            iid, grp, q, ev = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                         (iid, grp, V4_IDEA_CUTOFF, json.dumps(q), json.dumps(ev), len(ev), 0, ts))
            conn.commit()
    refs = {r[0]: format_evidence_block(json.loads(r[1] or "[]"), r[2])
            for r in conn.execute("SELECT item_id,evidence_json,cutoff_date FROM e13_evidence WHERE item_id LIKE 'weak:%'")}
    # score
    tasks = [(iid, grp, dom, txt, cr) for iid, grp, dom, txt in items for cr in critics
             if not conn.execute("SELECT 1 FROM e12_gap_scores WHERE item_id=? AND variant='lit8d' AND critic_model=?", (iid, cr)).fetchone()]
    print(f"scoring tasks: {len(tasks)}", flush=True)
    def _sc(t):
        iid, grp, dom, txt, cr = t
        s, _ = score_one(txt, cr, dom, LIT8D_SYSTEM, refs.get(iid, ""))
        return iid, grp, dom, cr, s
    ok=err=0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_sc, t) for t in tasks]), 1):
            iid, grp, dom, cr, s = f.result()
            conn.execute("INSERT OR IGNORE INTO e12_gap_scores VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (iid, grp, grp, "B", dom, "lit8d", cr,
                          json.dumps(s) if s else None, None if s else "parse_fail", ts))
            conn.commit(); ok+=1 if s else 0; err+=0 if s else 1
    print(f"scored ok={ok} err={err}", flush=True)

    # analyze
    cell=defaultdict(lambda: defaultdict(list)); grp={}
    for r in conn.execute("SELECT item_id,grp,scores_json FROM e12_gap_scores WHERE variant='lit8d' AND scores_json IS NOT NULL"):
        s=json.loads(r["scores_json"]); grp[r["item_id"]]=r["grp"]
        for d in DIMS: cell[r["item_id"]][d].append(s[d]["score"] if isinstance(s[d],dict) else s[d])
    def tm(vs): return trimmed_mean(vs) if len(vs)>=2 else (vs[0] if vs else None)
    wit={i:sum(tm(cell[i][d])*W[d] for d in DIMS)/WSUM for i in cell}
    conn.close()
    print("\n=== weak-model probe under lit8d ===")
    for m in PROBE:
        short=m.split("/")[-1]; g=f"weakprobe:{short}"
        xs=[wit[i] for i in wit if grp[i]==g]
        if xs: print(f"  {short:<26} n={len(xs)} mean={np.mean(xs):.2f} min={min(xs):.2f} max={max(xs):.2f}")
    core=[wit[i] for i in CORE7 if i in wit]
    v4a=[wit[i] for i in wit if grp[i].startswith('deepseek-v4') and grp[i].endswith('-C')]
    ctrl=[wit[i] for i in wit if grp[i]=='control']
    print(f"  {'CORE-7 anchors':<26} mean={np.mean(core):.2f}")
    print(f"  {'v4-active(200)':<26} mean={np.mean(v4a):.2f}")
    print(f"  {'controls(3)':<26} mean={np.mean(ctrl):.2f}")


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--run",action="store_true"); ap.add_argument("--workers",type=int,default=6)
    a=ap.parse_args()
    if a.run: run(a.workers)
