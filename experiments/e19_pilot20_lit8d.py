"""E19 — 20% pilot of the full experiment scored by the final lit8d critic.

Every model, 1/5 of its ideas: a fixed sample of 20 subdomains (4 per domain,
evenly spaced) is scored for ALL models, both tracks (Static B + Active C), so
Static-vs-Active and cross-model comparisons are on the same subdomains.

Scope: ~30 models x 20 subdomains x 2 tracks ~= 1157 ideas present x 3 critics.
Reuses lit8d + the E13 evidence pipeline (v4 evidence already cached). New table
lit8d_scores (INSERT-only, idempotent). Critics = CRITIC_MODELS[:3] (open-weight,
default key). NOTE: lit8d is domain-general only for CS/Physics (see F15 /
docs/lit8d_domain_validity_report); Bio/Chem/Med scores are soft rankings.

Usage (staged, each idempotent):
  /usr/bin/python3 experiments/e19_pilot20_lit8d.py --prep    # SS evidence (slow)
  /usr/bin/python3 experiments/e19_pilot20_lit8d.py --score
  /usr/bin/python3 experiments/e19_pilot20_lit8d.py --analyze
  /usr/bin/python3 experiments/e19_pilot20_lit8d.py --audit
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
DOMAINS = ["Biology", "CS", "Chemistry", "Medicine", "Physics"]
SUB_IDX = (0, 5, 10, 15)   # evenly-spaced subdomain picks per domain


def pick_subdomains(conn):
    picks = []
    for dom in DOMAINS:
        subs = [r[0] for r in conn.execute(
            "SELECT DISTINCT subdomain FROM subdomain_ideas WHERE domain=? ORDER BY subdomain", (dom,))]
        picks += [(dom, subs[i]) for i in SUB_IDX if i < len(subs)]
    return picks


def sample_items(conn):
    """(item_id, model, track, subdomain, domain, idea_text) for the 20-subdomain
    sample across all models/tracks. item_id matches e16 naming for evidence reuse."""
    picks = pick_subdomains(conn)
    models = [r[0] for r in conn.execute(
        "SELECT idea_model FROM subdomain_ideas GROUP BY idea_model HAVING COUNT(*)>50")]
    items = []
    for dom, sub in picks:
        for m in models:
            short = m.split("/")[-1]
            for tr in ("B", "C"):
                r = conn.execute("SELECT idea_text, domain FROM subdomain_ideas WHERE idea_model=? "
                                 "AND track=? AND subdomain=? AND TRIM(idea_text)!='' LIMIT 1",
                                 (m, tr, sub)).fetchone()
                if r:
                    iid = f"{short}|{tr}|{sub[:40]}"
                    items.append((iid, m, tr, sub, r[1] or dom, r[0]))
    # de-dup on item_id (a subdomain[:40] collision would merge; keep first)
    seen, uniq = set(), []
    for it in items:
        if it[0] in seen:
            continue
        seen.add(it[0]); uniq.append(it)
    return uniq


def ensure_score_tbl(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS lit8d_scores (
        item_id TEXT, idea_model TEXT, track TEXT, subdomain TEXT, domain TEXT,
        critic_model TEXT, scores_json TEXT, error TEXT, created_at TEXT,
        PRIMARY KEY (idea_model, track, subdomain, critic_model));""")
    conn.commit()


def prep(workers=4):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60); conn.row_factory = sqlite3.Row
    ensure_tables(conn)
    items = sample_items(conn)
    todo = [it for it in items if not conn.execute("SELECT 1 FROM e13_evidence WHERE item_id=?", (it[0],)).fetchone()]
    print(f"{len(items)} sampled ideas; {len(todo)} need evidence", flush=True)
    ts = datetime.now(timezone.utc).isoformat()
    def _ev(it):
        iid, m, tr, sub, dom, txt = it
        q, ev, ns = build_evidence_for_item(iid, "pilot20", txt, V4_IDEA_CUTOFF, None, None)
        return iid, q, ev, ns
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_ev, it) for it in todo]):
            iid, q, ev, ns = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                (iid, "pilot20", V4_IDEA_CUTOFF, json.dumps(q), json.dumps(ev), len(ev), ns, ts))
            conn.commit(); done += 1
            if done % 25 == 0 or done == len(todo): print(f"  [{done}/{len(todo)}]", flush=True)
    conn.close(); print("prep done")


def score(n_critics=3, workers=6):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60); conn.row_factory = sqlite3.Row
    ensure_score_tbl(conn)
    critics = list(cfg.CRITIC_MODELS)[:n_critics]
    items = sample_items(conn)
    refs = {}
    miss = 0
    for iid, *_ in items:
        r = conn.execute("SELECT evidence_json, cutoff_date FROM e13_evidence WHERE item_id=?", (iid,)).fetchone()
        if not r: miss += 1; continue
        refs[iid] = format_evidence_block(json.loads(r[0] or "[]"), r[1])
    if miss:
        print(f"WARN: {miss} items lack evidence (run --prep); scoring the rest")
    tasks = [(iid, m, tr, sub, dom, txt, cr) for iid, m, tr, sub, dom, txt in items if iid in refs
             for cr in critics
             if not conn.execute("SELECT 1 FROM lit8d_scores WHERE idea_model=? AND track=? AND subdomain=? AND critic_model=?",
                                 (m, tr, sub, cr)).fetchone()]
    print(f"score tasks: {len(tasks)}", flush=True)
    def _sc(t):
        iid, m, tr, sub, dom, txt, cr = t
        s, _ = score_one(txt, cr, dom, LIT8D_SYSTEM, refs[iid])
        return iid, m, tr, sub, dom, cr, s
    ts = datetime.now(timezone.utc).isoformat(); ok=err=0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_sc, t) for t in tasks]), 1):
            iid, m, tr, sub, dom, cr, s = f.result()
            conn.execute("INSERT OR IGNORE INTO lit8d_scores VALUES (?,?,?,?,?,?,?,?,?)",
                (iid, m, tr, sub, dom, cr, json.dumps(s) if s else None, None if s else "parse_fail", ts))
            conn.commit(); ok+=1 if s else 0; err+=0 if s else 1
            if i%100==0 or i==len(tasks): print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close(); print(f"done ok={ok} err={err}")


def _load():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(lambda: defaultdict(list)); meta = {}
    for r in conn.execute("SELECT idea_model,track,subdomain,domain,scores_json FROM lit8d_scores WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"]); k = (r["idea_model"], r["track"], r["subdomain"])
        meta[k] = (r["idea_model"], r["track"], r["domain"])
        for d in DIMS: cell[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()
    def tm(vs): return trimmed_mean(vs) if len(vs)>=2 else (vs[0] if vs else None)
    wit = {k: sum(tm(cell[k][d])*W[d] for d in DIMS)/WSUM for k in cell}
    return wit, meta


def analyze():
    wit, meta = _load()
    # per model: static/active means
    bym = defaultdict(lambda: {"B": [], "C": []})
    for k, w in wit.items():
        m, tr, dom = meta[k]; bym[m][tr].append(w)
    print(f"{'model':<40}{'nB':>4}{'static':>8}{'nC':>4}{'active':>8}{'boost':>8}")
    rows = []
    for m in sorted(bym):
        b = bym[m]["B"]; c = bym[m]["C"]
        sb = np.mean(b) if b else float('nan'); sc = np.mean(c) if c else float('nan')
        boost = (sc - sb) if (b and c) else float('nan')
        rows.append((m, len(b), sb, len(c), sc, boost))
    for m, nb, sb, nc, sc, boost in sorted(rows, key=lambda x: -(x[4] if not np.isnan(x[4]) else -9)):
        print(f"{m.split('/')[-1]:<40}{nb:>4}{sb:>8.2f}{nc:>4}{sc:>8.2f}{boost:>+8.2f}")
    allB = [w for k, w in wit.items() if meta[k][1]=="B"]
    allC = [w for k, w in wit.items() if meta[k][1]=="C"]
    print(f"\noverall Static n={len(allB)} mean={np.mean(allB):.2f} | Active n={len(allC)} mean={np.mean(allC):.2f} | boost={np.mean(allC)-np.mean(allB):+.2f}")
    boosts = [r[5] for r in rows if not np.isnan(r[5])]
    print(f"per-model boost: {sum(1 for x in boosts if x>0)}/{len(boosts)} positive, mean={np.mean(boosts):+.2f}")


def audit():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    ensure_score_tbl(conn)
    items = sample_items(conn)
    scored = conn.execute("SELECT COUNT(DISTINCT idea_model||track||subdomain) FROM lit8d_scores WHERE scores_json IS NOT NULL").fetchone()[0]
    ev = conn.execute("SELECT COUNT(*) FROM e13_evidence WHERE grp='pilot20'").fetchone()[0]
    print(f"sampled ideas: {len(items)} | evidence(pilot20): {ev} | scored idea-units: {scored}")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    for f in ("prep","score","analyze","audit"): ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.prep: prep(min(a.workers,4))
    if a.score: score(workers=a.workers)
    if a.analyze: analyze()
    if a.audit: audit()


if __name__ == "__main__":
    main()
