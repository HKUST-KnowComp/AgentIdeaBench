"""E18 — build per-domain anchor sets and verify lit8d separates anchors from
model ideas in EVERY domain (not just CS/ML), so lit8d can be trusted to score
all 27 models across all 5 domains.

Strategy (the user's "pick the top-N per domain" generalized to all domains):
  For each of the 4 non-CS domains (Biology / Chemistry / Physics / Medicine),
  pull candidate landmark papers by searching Semantic Scholar for each of the
  domain's subdomains and keeping the highest-cited recent (>=2023) papers.
  An LLM rewrites each real abstract into an idea-abstract stating the increment
  claim (same style as the hand-written CS anchors). Score all candidates under
  lit8d, keep the TOP-N per domain as that domain's anchor set. CS reuses the
  existing 13 candidates (top-N by lit8d).

Verification: for each domain, anchor-top-N vs that domain's v4-active ideas
(already scored under lit8d in e16) + weak-model ideas. "Clear gap in every
domain" => lit8d is domain-general => safe to run on all 27 models.

Tables (INSERT-only): domain_anchor_pool (raw+rewritten+meta), evidence in
e13_evidence, scores in e12_gap_scores (variant=lit8d, grp='domanchor:<domain>').

Usage (staged; each idempotent):
  /usr/bin/python3 experiments/e18_multidomain_anchors.py --collect
  /usr/bin/python3 experiments/e18_multidomain_anchors.py --rewrite
  /usr/bin/python3 experiments/e18_multidomain_anchors.py --prep
  /usr/bin/python3 experiments/e18_multidomain_anchors.py --score
  /usr/bin/python3 experiments/e18_multidomain_anchors.py --analyze --topn 10
"""
import argparse, json, re, sqlite3, statistics, sys, time
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
    ss_search, ensure_tables, build_evidence_for_item, format_evidence_block,
    LIT8D_SYSTEM, _title_overlap)

WSUM = sum(W.values())
DOMAINS = ["Biology", "Chemistry", "Physics", "Medicine"]  # CS reuses cand_*
CS_CANDS = ["cand_05","cand_02","cand_06","cand_04","cand_07","cand_13","cand_10",
            "cand_08","cand_12","cand_09","cand_11","cand_03","cand_01"]
PER_SUBDOMAIN = 2       # top-cited papers kept per subdomain search
MIN_YEAR = 2023
MIN_CIT = 20

REWRITE_SYS = "You rewrite a published paper abstract into a first-person research IDEA proposal. Return ONLY the proposal text."
REWRITE_TMPL = """Rewrite the abstract below into a 90-150 word first-person research IDEA \
proposal, as if proposing the work before doing it. Requirements:
- State the prevailing assumption or gap, then the central NEW claim/mechanism the \
idea asserts beyond it ("Prevailing view is X; I hypothesize Y").
- State the decisive test that would confirm/refute it.
- Keep it conceptual: no results, no "we found/showed". Present tense hypothesis + \
future-tense test. Do NOT invent numbers not in the abstract.

Title: {title}
Abstract: {abstract}

Return ONLY the proposal (no preamble)."""


def _tbl(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS domain_anchor_pool (
        anchor_id TEXT PRIMARY KEY, domain TEXT, subdomain TEXT, title TEXT,
        ss_paper_id TEXT, publication_date TEXT, citation_count INTEGER,
        real_abstract TEXT, idea_abstract TEXT, created_at TEXT);""")
    conn.commit()


def collect():
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60); conn.row_factory = sqlite3.Row
    _tbl(conn); ts = datetime.now(timezone.utc).isoformat()
    for dom in DOMAINS:
        subs = [r[0] for r in conn.execute(
            "SELECT DISTINCT subdomain FROM subdomain_ideas WHERE domain=? ORDER BY subdomain", (dom,))]
        kept = 0
        for sub in subs:
            hits = ss_search(sub, before_date=None, limit=12)
            hits = [h for h in hits if (h.get("year") or 0) >= MIN_YEAR
                    and (h.get("citationCount") or 0) >= MIN_CIT
                    and h.get("abstract") and h.get("publicationDate")]
            hits.sort(key=lambda h: -(h.get("citationCount") or 0))
            for h in hits[:PER_SUBDOMAIN]:
                aid = f"dom_{dom[:3].lower()}_{h['paperId'][:8]}"
                if conn.execute("SELECT 1 FROM domain_anchor_pool WHERE anchor_id=?", (aid,)).fetchone():
                    continue
                conn.execute("INSERT OR IGNORE INTO domain_anchor_pool VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (aid, dom, sub, h["title"], h["paperId"], h["publicationDate"],
                     h.get("citationCount") or 0, (h.get("abstract") or "")[:2500], None, ts))
                kept += 1
            conn.commit()
        print(f"{dom}: collected {kept} candidate papers", flush=True)
    conn.close()


def rewrite(workers=4):
    from utils.LLM import CriticLLM
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT anchor_id,title,real_abstract FROM domain_anchor_pool WHERE idea_abstract IS NULL").fetchall()
    print(f"rewrite: {len(rows)} papers", flush=True)
    llm = CriticLLM(model_name=cfg.CRITIC_MODELS[0])
    def _rw(r):
        for _ in range(3):
            try:
                out = llm.completion(REWRITE_TMPL.format(title=r["title"], abstract=r["real_abstract"][:2200]),
                                     system_prompt=REWRITE_SYS)
                if isinstance(out, tuple): out = out[0]
                out = (out or "").strip()
                if len(out) > 120: return r["anchor_id"], out[:1600]
            except Exception: time.sleep(2)
        return r["anchor_id"], None
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_rw, r) for r in rows]):
            aid, idea = f.result()
            if idea:
                conn.execute("UPDATE domain_anchor_pool SET idea_abstract=? WHERE anchor_id=?", (idea, aid))
                conn.commit()
            done += 1
            if done % 15 == 0 or done == len(rows): print(f"  [{done}/{len(rows)}]", flush=True)
    conn.close()


def prep(workers=4):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60); conn.row_factory = sqlite3.Row
    ensure_tables(conn); ts = datetime.now(timezone.utc).isoformat()
    rows = conn.execute("SELECT anchor_id,title,ss_paper_id,publication_date,idea_abstract "
                        "FROM domain_anchor_pool WHERE idea_abstract IS NOT NULL").fetchall()
    todo = [r for r in rows if not conn.execute("SELECT 1 FROM e13_evidence WHERE item_id=?", (r["anchor_id"],)).fetchone()]
    print(f"prep evidence: {len(todo)}", flush=True)
    def _ev(r):
        q, ev, ns = build_evidence_for_item(r["anchor_id"], "domanchor", r["idea_abstract"],
                                            r["publication_date"], r["ss_paper_id"], r["title"])
        return r["anchor_id"], r["publication_date"], q, ev, ns
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_ev, r) for r in todo]):
            aid, cutoff, q, ev, ns = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                (aid, "domanchor", cutoff, json.dumps(q), json.dumps(ev), len(ev), ns, ts))
            conn.commit(); done += 1
            if done % 15 == 0 or done == len(todo): print(f"  [{done}/{len(todo)}]", flush=True)
    conn.close()


def score(n_critics=3, workers=6):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60); conn.row_factory = sqlite3.Row
    critics = list(cfg.CRITIC_MODELS)[:n_critics]
    rows = conn.execute("SELECT p.anchor_id,p.domain,p.idea_abstract,e.evidence_json,e.cutoff_date "
        "FROM domain_anchor_pool p JOIN e13_evidence e ON e.item_id=p.anchor_id "
        "WHERE p.idea_abstract IS NOT NULL").fetchall()
    refs = {r["anchor_id"]: format_evidence_block(json.loads(r["evidence_json"] or "[]"), r["cutoff_date"]) for r in rows}
    tasks = [(r["anchor_id"], r["domain"], r["idea_abstract"], cr) for r in rows for cr in critics
             if not conn.execute("SELECT 1 FROM e12_gap_scores WHERE item_id=? AND variant='lit8d' AND critic_model=?", (r["anchor_id"], cr)).fetchone()]
    print(f"score tasks: {len(tasks)}", flush=True)
    def _sc(t):
        aid, dom, txt, cr = t
        s, _ = score_one(txt, cr, dom, LIT8D_SYSTEM, refs[aid]); return aid, dom, cr, s
    ts = datetime.now(timezone.utc).isoformat(); ok=err=0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_sc, t) for t in tasks]), 1):
            aid, dom, cr, s = f.result()
            conn.execute("INSERT OR IGNORE INTO e12_gap_scores VALUES (?,?,?,?,?,?,?,?,?,?)",
                (aid, f"domanchor:{dom}", "domanchor", "-", dom, "lit8d", cr,
                 json.dumps(s) if s else None, None if s else "parse_fail", ts))
            conn.commit(); ok+=1 if s else 0; err+=0 if s else 1
            if i%50==0 or i==len(tasks): print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close(); print(f"done ok={ok} err={err}")


def _wit(variant="lit8d"):
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell=defaultdict(lambda: defaultdict(list)); grp={}; dom={}
    for r in conn.execute("SELECT item_id,grp,domain,scores_json FROM e12_gap_scores WHERE variant=? AND scores_json IS NOT NULL",(variant,)):
        s=json.loads(r["scores_json"]); grp[r["item_id"]]=r["grp"]; dom[r["item_id"]]=r["domain"]
        for d in DIMS: cell[r["item_id"]][d].append(s[d]["score"] if isinstance(s[d],dict) else s[d])
    conn.close()
    def tm(vs): return trimmed_mean(vs) if len(vs)>=2 else (vs[0] if vs else None)
    wit={i:sum(tm(cell[i][d])*W[d] for d in DIMS)/WSUM for i in cell}
    return wit, grp, dom


def analyze(topn=10):
    wit, grp, dom = _wit()
    print(f"=== per-domain: top-{topn} anchors vs v4-active vs weak, under lit8d ===\n")
    # CS anchors = CS_CANDS
    cs = sorted([wit[i] for i in CS_CANDS if i in wit], reverse=True)[:topn]
    rows = [("CS", cs)]
    for d in DOMAINS:
        pool = sorted([wit[i] for i in wit if grp[i]==f"domanchor:{d}"], reverse=True)[:topn]
        rows.append((d, pool))
    print(f"{'domain':<10}{'anchor_topN':>12}{'v4act_mean':>12}{'v4act_max':>11}{'weak_mean':>11}{'gap(A-v4mean)':>15}")
    allgood = True
    for d, pool in rows:
        v4 = [wit[i] for i in wit if grp[i].startswith("deepseek-v4") and grp[i].endswith("-C") and dom[i]==d]
        weak = [wit[i] for i in wit if grp[i].startswith("weakprobe") and dom[i]==d]
        am = np.mean(pool) if pool else float('nan')
        v4m = np.mean(v4) if v4 else float('nan'); v4x=max(v4) if v4 else float('nan')
        wm = np.mean(weak) if weak else float('nan')
        gap = am - v4m
        flag = "" if gap>=1.0 else "  <-- WEAK GAP"
        print(f"{d:<10}{am:>12.2f}{v4m:>12.2f}{v4x:>11.2f}{wm:>11.2f}{gap:>15.2f}{flag}")
        if not (gap>=1.0): allgood=False
    print(f"\n{'ALL DOMAINS show clear (>=1.0) anchor-vs-v4 gap -> lit8d domain-general, safe to run all' if allgood else 'SOME domain has weak gap -> investigate before full run'}")


def main():
    ap=argparse.ArgumentParser()
    for f in ("collect","rewrite","prep","score","analyze"): ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--topn", type=int, default=10); ap.add_argument("--workers", type=int, default=6)
    a=ap.parse_args()
    if a.collect: collect()
    if a.rewrite: rewrite(min(a.workers,4))
    if a.prep: prep(min(a.workers,4))
    if a.score: score(workers=a.workers)
    if a.analyze: analyze(a.topn)


if __name__ == "__main__":
    main()
