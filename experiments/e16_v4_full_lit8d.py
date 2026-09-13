"""E16 — score the FULL v4 idea set (deepseek-v4-flash + deepseek-v4-pro, all
100 static + 100 active each = 400 ideas) under the final lit8d critic.

E12-E15 scored only a 25/group sample. The user wants every v4 static+active
idea graded by the final standard so the anchor-vs-v4 comparison uses the whole
distribution, not a sample.

Pipeline (reuses E13 evidence + lit8d; INSERT-only, idempotent — the existing 25
sampled items per group are reused, only the remaining ~75/group are added):
  --prep  : SS prior-art evidence for every v4 idea (cutoff 2026-05-31),
            into e13_evidence
  --score : lit8d scoring (3 critics: qwen3.6-plus/kimi-k2.6/glm-5.1; v4-flash
            is CRITIC_MODELS[4], NOT in this slice, so no self-grading),
            into e12_gap_scores (variant=lit8d)
  --analyze : full-distribution stats vs the CORE-7 anchors

Usage:
  /usr/bin/python3 experiments/e16_v4_full_lit8d.py --prep
  /usr/bin/python3 experiments/e16_v4_full_lit8d.py --score
  /usr/bin/python3 experiments/e16_v4_full_lit8d.py --analyze
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as W
from experiments.e10_idea_anchor_calibration import score_one, trimmed_mean
from experiments.e13_litverify_rubric8 import (
    ensure_tables, build_evidence_for_item, format_evidence_block,
    LIT8D_SYSTEM, V4_IDEA_CUTOFF,
)

VARIANT = "lit8d"
V4_MODELS = ["deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"]
WSUM = sum(W.values())
CORE7 = ["cand_05", "cand_02", "cand_06", "cand_04", "cand_07", "cand_13", "cand_10"]


def all_v4_items():
    """Every v4 idea, keyed exactly like e12.build_items so the existing 25
    sampled items per group are reused (idempotent)."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    items = []
    for m in V4_MODELS:
        short = m.split("/")[-1]
        for tr in ("B", "C"):
            for r in conn.execute(
                    "SELECT subdomain, domain, idea_text FROM subdomain_ideas "
                    "WHERE idea_model=? AND track=? AND TRIM(idea_text)!='' "
                    "ORDER BY subdomain", (m, tr)):
                iid = f"{short}|{tr}|{r['subdomain'][:40]}"
                items.append((iid, f"{short}-{tr}", m, tr, r["domain"], r["idea_text"]))
    conn.close()
    # de-dup on item_id (subdomain[:40] collisions), keep first
    seen, uniq = set(), []
    for it in items:
        if it[0] in seen:
            continue
        seen.add(it[0]); uniq.append(it)
    return uniq


def prep(workers=4):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tables(conn)
    items = all_v4_items()
    todo = [it for it in items if not conn.execute(
        "SELECT 1 FROM e13_evidence WHERE item_id=?", (it[0],)).fetchone()]
    print(f"evidence: {len(items)} v4 ideas, {len(todo)} need prep", flush=True)
    ts = datetime.now(timezone.utc).isoformat()
    # SS is the bottleneck and build_evidence_for_item is internally throttled;
    # keep workers modest so the global SS lock isn't thrashed.
    def _w(it):
        iid, grp, m, tr, dom, txt = it
        q, ev, ns = build_evidence_for_item(iid, grp, txt, V4_IDEA_CUTOFF, None, None)
        return iid, grp, q, ev, ns
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, it) for it in todo]
        for f in as_completed(futs):
            iid, grp, q, ev, ns = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                         (iid, grp, V4_IDEA_CUTOFF, json.dumps(q), json.dumps(ev),
                          len(ev), ns, ts))
            conn.commit(); done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}]", flush=True)
    conn.close()
    print("prep done")


def score(n_critics=3, workers=6):
    critics = list(cfg.CRITIC_MODELS)[:n_critics]
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tables(conn)
    items = all_v4_items()
    refs = {}
    for iid, *_ in items:
        r = conn.execute("SELECT evidence_json, cutoff_date FROM e13_evidence WHERE item_id=?",
                         (iid,)).fetchone()
        if not r:
            print(f"ABORT: {iid} lacks evidence (run --prep)"); conn.close(); return
        refs[iid] = format_evidence_block(json.loads(r[0] or "[]"), r[1])

    tasks = []
    for iid, grp, m, tr, dom, txt in items:
        for cr in critics:
            if not conn.execute(
                    "SELECT 1 FROM e12_gap_scores WHERE item_id=? AND variant=? AND critic_model=?",
                    (iid, VARIANT, cr)).fetchone():
                tasks.append((iid, grp, m, tr, dom, txt, cr))
    print(f"score: {len(items)} items, {len(tasks)} tasks", flush=True)
    if not tasks:
        conn.close(); return

    def _w(t):
        iid, grp, m, tr, dom, txt, cr = t
        s, _ = score_one(txt, cr, dom, LIT8D_SYSTEM, refs[iid])
        return (iid, grp, m, tr, dom, cr, s)

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            iid, grp, m, tr, dom, cr, s = f.result()
            conn.execute("INSERT OR IGNORE INTO e12_gap_scores VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (iid, grp, m, tr, dom, VARIANT, cr,
                          json.dumps(s) if s else None,
                          None if s else "parse_fail", ts))
            conn.commit()
            ok += 1 if s else 0; err += 0 if s else 1
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close()
    print(f"done: ok={ok} err={err}")


def analyze():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(lambda: defaultdict(list)); grp = {}
    for r in conn.execute("SELECT item_id,grp,scores_json FROM e12_gap_scores "
                          "WHERE variant=? AND scores_json IS NOT NULL", (VARIANT,)):
        s = json.loads(r["scores_json"]); grp[r["item_id"]] = r["grp"]
        for d in DIMS:
            cell[r["item_id"]][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()

    def tm(vs):
        return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)
    wit = {i: sum(tm(dd[d]) * W[d] for d in DIMS) / WSUM for i, dd in cell.items()}

    def summ(xs):
        a = np.array(xs)
        return (f"n={len(xs)} mean={a.mean():.2f} median={np.median(a):.2f} "
                f"std={a.std(ddof=1):.2f} min={a.min():.2f} max={a.max():.2f} "
                f"p90={np.percentile(a,90):.2f}")

    print("=== FULL v4 distribution under lit8d ===")
    for g in ("deepseek-v4-flash-B", "deepseek-v4-flash-C",
              "deepseek-v4-pro-B", "deepseek-v4-pro-C"):
        xs = [wit[i] for i in wit if grp[i] == g]
        print(f"  {g:<24} {summ(xs)}")
    core = [wit[i] for i in CORE7 if i in wit]
    print(f"\n  CORE-7 anchors           {summ(core)}")

    allv4 = [wit[i] for i in wit if grp[i].startswith("deepseek-v4") and grp[i][-1] in "BC"]
    act = [wit[i] for i in wit if grp[i].startswith("deepseek-v4") and grp[i].endswith("-C")]
    print(f"\n  all v4 (400):            {summ(allv4)}")
    print(f"  all v4 ACTIVE (200):     {summ(act)}")
    cm = np.mean(core)
    print(f"\n  CORE-7 mean {cm:.2f}  −  best single v4 idea {max(allv4):.2f}  = {cm-max(allv4):+.2f}")
    print(f"  frac of 400 v4 ideas >= CORE-7 mean: "
          f"{sum(1 for x in allv4 if x>=cm)/len(allv4):.1%}")
    print(f"  frac of 400 v4 ideas >= 7.0: {sum(1 for x in allv4 if x>=7.0)/len(allv4):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.prep:
        prep(workers=min(a.workers, 4))
    if a.score:
        score(workers=a.workers)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
