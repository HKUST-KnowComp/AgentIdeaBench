"""E29 — score extra NLP subdomains with lit8d, for the NLP-focused human eval.

The human-eval annotators are NLP researchers, so the 100 pairs should sit in
NLP subfields. Only 1 of the 8 lit8d-scored CS subfields is clearly NLP
("...reasoning chain of thought"); the other NLP subfields have generated ideas
(subdomain_ideas) but no lit8d scores. This scores those extra NLP subdomains
into a SEPARATE table (e29_nlp_scores) using the exact lit8d pipeline, so the
benchmark's 40-subdomain scored scope (lit8d_scores_3seed) is untouched.

Scores idea_index 1 only (58 ideas/subdomain: ~28 Static + 30 Active), all 30
models, 3 open-weight critics, live SS prior-art. New table + evidence grp
'e29nlp'. Idempotent, resume-safe. All open-weight (default OPENROUTER_API_KEY).

Usage:
  /usr/bin/python3 experiments/e29_score_nlp.py --smoke
  /usr/bin/python3 experiments/e29_score_nlp.py --prep [--workers 3]
  /usr/bin/python3 experiments/e29_score_nlp.py --score [--workers 6]
  /usr/bin/python3 experiments/e29_score_nlp.py --status
"""
import argparse, json, sqlite3, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from experiments.e10_idea_anchor_calibration import score_one
from experiments.e13_litverify_rubric8 import (
    ensure_tables, build_evidence_for_item, format_evidence_block,
    LIT8D_SYSTEM, V4_IDEA_CUTOFF)

# 5 NLP subfields an NLP researcher can judge well (LLM CoT reasoning is already
# in lit8d_scores_3seed and is reused directly by the selector).
NLP_SUBS = [
    "LLM agent planning tool use autonomy",
    "code generation large language model program synthesis",
    "retrieval augmented generation knowledge grounding",
    "transformer efficiency long context attention",
    "multimodal vision language model alignment",
]
IDEA_IDX = (1,)
DOMAIN = "CS"


def _short(m):
    return m.split("/")[-1]


def _iid(short, tr, idx, sub):
    return f"e29nlp|{short}|{tr}|{idx}|{sub[:40]}"


def ensure_tbls(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e29_nlp_scores (
        item_id TEXT, idea_model TEXT, subdomain TEXT, domain TEXT, track TEXT,
        idea_index INTEGER, critic_model TEXT, scores_json TEXT, error TEXT, created_at TEXT,
        PRIMARY KEY (idea_model, subdomain, track, idea_index, critic_model));""")
    conn.commit()


def _ideas(conn, subs, idxs):
    rows = []
    for sub in subs:
        for m, tr, idx, txt in conn.execute(
                "SELECT idea_model, track, idea_index, idea_text FROM subdomain_ideas "
                "WHERE subdomain=? AND track IN ('B','C') AND TRIM(idea_text)!='' "
                "AND idea_index IN (%s)" % ",".join("?" * len(idxs)),
                (sub, *idxs)):
            rows.append((m, sub, tr, idx, txt))
    return rows


def prep(subs, idxs, workers=3):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tables(conn); ensure_tbls(conn)
    items = _ideas(conn, subs, idxs)
    todo = [it for it in items
            if not conn.execute("SELECT 1 FROM e13_evidence WHERE item_id=?",
                                (_iid(_short(it[0]), it[2], it[3], it[1]),)).fetchone()]
    print(f"{len(items)} NLP ideas; {len(todo)} need evidence", flush=True)
    ts = datetime.now(timezone.utc).isoformat()

    def _ev(it):
        m, sub, tr, idx, txt = it
        iid = _iid(_short(m), tr, idx, sub)
        q, ev, ns = build_evidence_for_item(iid, "e29nlp", txt, V4_IDEA_CUTOFF, None, None)
        return iid, q, ev, ns

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_ev, it) for it in todo]):
            iid, q, ev, ns = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                         (iid, "e29nlp", V4_IDEA_CUTOFF, json.dumps(q), json.dumps(ev), len(ev), ns, ts))
            conn.commit(); done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}]", flush=True)
    conn.close(); print("prep done")


def score(subs, idxs, workers=6):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tbls(conn)
    critics = list(cfg.CRITIC_MODELS)[:3]
    items = _ideas(conn, subs, idxs)
    refs = {}
    for m, sub, tr, idx, txt in items:
        iid = _iid(_short(m), tr, idx, sub)
        r = conn.execute("SELECT evidence_json, cutoff_date FROM e13_evidence WHERE item_id=?",
                         (iid,)).fetchone()
        if r:
            refs[iid] = format_evidence_block(json.loads(r[0] or "[]"), r[1])
    tasks = [(m, sub, tr, idx, txt, cr)
             for m, sub, tr, idx, txt in items if _iid(_short(m), tr, idx, sub) in refs
             for cr in critics
             if not conn.execute("SELECT 1 FROM e29_nlp_scores WHERE idea_model=? AND subdomain=? "
                                 "AND track=? AND idea_index=? AND critic_model=? AND error IS NULL "
                                 "AND scores_json IS NOT NULL", (m, sub, tr, idx, cr)).fetchone()]
    print(f"score tasks: {len(tasks)} (evidence-ready {len(refs)}/{len(items)})", flush=True)

    def _sc(t):
        m, sub, tr, idx, txt, cr = t
        iid = _iid(_short(m), tr, idx, sub)
        s, _ = score_one(txt, cr, DOMAIN, LIT8D_SYSTEM, refs[iid])
        return iid, m, sub, tr, idx, cr, s

    ts = datetime.now(timezone.utc).isoformat(); ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_sc, t) for t in tasks]), 1):
            iid, m, sub, tr, idx, cr, s = f.result()
            conn.execute("INSERT OR REPLACE INTO e29_nlp_scores VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (iid, m, sub, DOMAIN, tr, idx, cr, json.dumps(s) if s else None,
                          None if s else "parse_fail", ts))
            conn.commit(); ok += 1 if s else 0; err += 0 if s else 1
            if i % 25 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close(); print(f"score done ok={ok} err={err}")


def status(subs, idxs):
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    for sub in subs:
        n = conn.execute("SELECT COUNT(DISTINCT idea_model||track||idea_index) FROM e29_nlp_scores "
                         "WHERE subdomain=? AND scores_json IS NOT NULL", (sub,)).fetchone()[0]
        crit = conn.execute("SELECT COUNT(*) FROM e29_nlp_scores WHERE subdomain=? "
                            "AND scores_json IS NOT NULL", (sub,)).fetchone()[0]
        full3 = conn.execute(
            "SELECT COUNT(*) FROM (SELECT idea_model,track,idea_index FROM e29_nlp_scores "
            "WHERE subdomain=? AND scores_json IS NOT NULL GROUP BY idea_model,track,idea_index "
            "HAVING COUNT(DISTINCT critic_model)>=3)", (sub,)).fetchone()[0]
        print(f"{sub[:48]:48} ideas_scored={n:3} crit_rows={crit:4} full3crit={full3}")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 subdomain x first 2 ideas")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.smoke:
        subs = NLP_SUBS[:1]
        print("SMOKE:", subs)
        prep(subs, IDEA_IDX, workers=2)
        # score only a couple ideas
        conn = sqlite3.connect(str(cfg.RESULTS_DB)); ensure_tbls(conn); conn.close()
        score(subs, IDEA_IDX, workers=3)
        status(subs, IDEA_IDX)
        return
    if a.status:
        status(NLP_SUBS, IDEA_IDX); return
    if a.prep:
        prep(NLP_SUBS, IDEA_IDX, workers=min(a.workers, 3))
    if a.score:
        score(NLP_SUBS, IDEA_IDX, workers=a.workers)
    if not any([a.prep, a.score, a.status, a.smoke]):
        ap.print_help()


if __name__ == "__main__":
    main()
