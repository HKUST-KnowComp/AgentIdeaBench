"""E14 — recruit new anchors from NeurIPS 2025 award/oral papers, screened with
the E13 lit-verify pipeline (lit8b rubric).

Candidates live in experiments/anchor_candidates_2026.json: idea abstracts
rewritten to state the increment claim explicitly ("prevailing assumption X ->
this idea claims beyond-X"), per E13 lesson (a). cand_09 (Gated Attention) is a
deliberate high containment-risk probe — the screener should crush it.

Pipeline (reuses E13 tables/functions; INSERT-only):
  --register : write candidate meta into e13_anchor_meta (cutoff = own pub date)
  --prep     : evidence prep into e13_evidence (grp='candidate')
  --score    : score under lit8b into e12_gap_scores (grp='candidate')
  --analyze  : per-candidate dims + weighted vs existing anchor/v4 lit8b groups

Usage:
  /usr/bin/python3 experiments/e14_anchor_recruit.py --register --prep
  /usr/bin/python3 experiments/e14_anchor_recruit.py --score
  /usr/bin/python3 experiments/e14_anchor_recruit.py --analyze
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
    E13_VARIANTS,
)

CAND_FILE = ROOT / "experiments" / "anchor_candidates_2026.json"
VARIANT = "lit8b"      # overridden by --variant
WSUM = sum(W.values())


def load_candidates():
    return json.loads(CAND_FILE.read_text())["candidates"]


def register():
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_tables(conn)
    ts = datetime.now(timezone.utc).isoformat()
    for c in load_candidates():
        conn.execute("INSERT OR IGNORE INTO e13_anchor_meta VALUES (?,?,?,?,?,?,?)",
                     (c["id"], c["title"], c["ss_paper_id"],
                      c["publication_date"], c["publication_date"], 1, ts))
        print(f"  {c['id']} cutoff={c['publication_date']} {c['title'][:60]}")
    conn.commit()
    conn.close()


def prep():
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_tables(conn)
    ts = datetime.now(timezone.utc).isoformat()
    for c in load_candidates():
        if conn.execute("SELECT 1 FROM e13_evidence WHERE item_id=?",
                        (c["id"],)).fetchone():
            continue
        queries, evidence, n_self = build_evidence_for_item(
            c["id"], "candidate", c["abstract"],
            c["publication_date"], c["ss_paper_id"], c["title"])
        conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                     (c["id"], "candidate", c["publication_date"],
                      json.dumps(queries), json.dumps(evidence),
                      len(evidence), n_self, ts))
        conn.commit()
        print(f"  {c['id']} cutoff={c['publication_date']} hits={len(evidence)} "
              f"self_excluded={n_self}", flush=True)
    conn.close()


def score(n_critics=3, workers=6):
    critics = list(cfg.CRITIC_MODELS)[:n_critics]
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    cands = load_candidates()
    refs = {}
    for c in cands:
        r = conn.execute("SELECT evidence_json, cutoff_date FROM e13_evidence "
                         "WHERE item_id=?", (c["id"],)).fetchone()
        if not r:
            print(f"ABORT: {c['id']} lacks evidence (run --prep)")
            conn.close()
            return
        refs[c["id"]] = format_evidence_block(json.loads(r[0] or "[]"), r[1])

    tasks = []
    for c in cands:
        dom = c["subdomain"].split("/")[0].strip()
        for cr in critics:
            if not conn.execute(
                    "SELECT 1 FROM e12_gap_scores WHERE item_id=? AND variant=? AND critic_model=?",
                    (c["id"], VARIANT, cr)).fetchone():
                tasks.append((c["id"], dom, c["abstract"], cr))
    print(f"tasks={len(tasks)}")
    if not tasks:
        conn.close()
        return

    system = E13_VARIANTS[VARIANT]

    def _w(t):
        iid, dom, txt, cr = t
        s, _ = score_one(txt, cr, dom, system, refs[iid])
        return (iid, dom, cr, s)

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            iid, dom, cr, s = f.result()
            conn.execute("INSERT OR IGNORE INTO e12_gap_scores VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (iid, "candidate", "candidate", "-", dom, VARIANT, cr,
                          json.dumps(s) if s else None,
                          None if s else "parse_fail", ts))
            conn.commit()
            ok += 1 if s else 0
            err += 0 if s else 1
            if i % 9 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close()
    print(f"done: ok={ok} err={err}")


def analyze():
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.row_factory = sqlite3.Row
    titles = {c["id"]: c["title"] for c in load_candidates()}

    cell = defaultdict(lambda: defaultdict(list))
    grp_of = {}
    for r in conn.execute("SELECT item_id, grp, scores_json FROM e12_gap_scores "
                          "WHERE variant=? AND scores_json IS NOT NULL", (VARIANT,)):
        s = json.loads(r["scores_json"])
        grp_of[r["item_id"]] = r["grp"]
        for d in DIMS:
            v = s[d]["score"] if isinstance(s[d], dict) else s[d]
            cell[r["item_id"]][d].append(v)
    conn.close()

    witem, dims_item = {}, {}
    for iid, dd in cell.items():
        dt = {d: trimmed_mean(dd[d]) for d in DIMS}
        dims_item[iid] = dt
        witem[iid] = sum(dt[d] * W[d] for d in DIMS) / WSUM

    print(f"{'candidate':<12}{'weighted':>9}" + "".join(f"{d[:4]:>7}" for d in DIMS)
          + "  title")
    for iid in sorted(titles):
        if iid not in witem:
            print(f"{iid:<12}  (no scores)")
            continue
        dt = dims_item[iid]
        print(f"{iid:<12}{witem[iid]:>9.2f}" + "".join(f"{dt[d]:>7.1f}" for d in DIMS)
              + f"  {titles[iid][:52]}")

    for g, label in (("anchor", "old anchors (15)"),
                     ("candidate", "new candidates"),):
        vs = [witem[i] for i in witem if grp_of[i] == g]
        if vs:
            print(f"\n{label}: mean={np.mean(vs):.3f} max={max(vs):.2f} min={min(vs):.2f}")
    v4c = [witem[i] for i in witem if grp_of[i] == "deepseek-v4-pro-C"]
    if v4c:
        print(f"deepseek-v4-pro-C: mean={np.mean(v4c):.3f} top3="
              f"{sorted([round(v,2) for v in v4c], reverse=True)[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--variant", default="lit8b", choices=list(E13_VARIANTS))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    global VARIANT
    VARIANT = a.variant
    if a.register:
        register()
    if a.prep:
        prep()
    if a.score:
        score(workers=a.workers)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
