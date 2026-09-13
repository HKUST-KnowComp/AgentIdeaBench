"""v3 — Subdomain-centric ideation (100 subdomains = 20 per broad domain).

New design (user 2026-06-29), replacing the paper-centric / broad-domain setup:
  - Unit of evaluation = 100 curated subdomains (experiments/subdomains_100.json).
  - Static refs  = SS search top-10 for the subdomain (table subdomain_refs).
  - Static idea  = IdeaLLM given (subdomain as topic + its top-10 refs)  [track B]
  - Active idea  = run_active_agent(subdomain, budget=10 tool calls)       [track C]
    NB: Active is now given the SUBDOMAIN (not the broad domain) — fixes the E11
    granularity mismatch so Static and Active explore the same topic space.
  - Open-source models only (default key). 1 idea per (model, subdomain).

Old Active (v1_paper_refs / broad-domain) is NOT used here, but is PRESERVED in
the results table (raw-data rule). This writes to NEW tables only.

SAFETY: new tables subdomain_refs + subdomain_ideas. Idempotent. modern open
models → default OPENROUTER_API_KEY. SS throttled (active_agent global limiter).

Usage:
  /usr/bin/python3 experiments/v3_subdomain_ideation.py --fetch-refs
  /usr/bin/python3 experiments/v3_subdomain_ideation.py --static --workers 8
  /usr/bin/python3 experiments/v3_subdomain_ideation.py --active --workers 6
  /usr/bin/python3 experiments/v3_subdomain_ideation.py --audit
"""
import argparse
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from generation.generate_ideas import _build_generation_payload, _clean_idea_text
from generation.active_agent import run_active_agent, _search_papers

SUBDOMAINS_FILE = ROOT / "experiments" / "subdomains_100.json"
STATIC_REFS_N = 10
ACTIVE_BUDGET = 10
MIN_WORDS = 30
BROKEN = {"mistralai/devstral-medium", "mistralai/mistral-7b-instruct-v0.1",
          "minimax/minimax-m1",          # delisted (404) / never protocol-compliant
          # thinking models that mandate reasoning -> IdeaLLM (reasoning disabled)
          # gets HTTP 400 on the Static path; excluded for a consistent B+C roster.
          "qwen/qwen3-235b-a22b-thinking-2507", "qwen/qwen3-vl-8b-thinking"}


def load_subdomains():
    d = json.load(open(SUBDOMAINS_FILE))
    out = []
    for dom in ("Biology", "CS", "Chemistry", "Medicine", "Physics"):
        for sub in d[dom]:
            out.append((dom, sub))
    return out


def roster():
    """Open-source models: benchmark-existing open ∪ official IDEA_MODELS open,
    minus broken ones."""
    c = sqlite3.connect(str(cfg.RESULTS_DB))
    existing = {r[0] for r in c.execute(
        "SELECT DISTINCT idea_model FROM results WHERE track='B' AND idea_text!=''")}
    c.close()
    cand = {m for m in existing if not m.startswith("baseline/") and not cfg.is_us_key_model(m)}
    cand |= {m for m in cfg.IDEA_MODELS if not cfg.is_us_key_model(m)}
    return sorted(cand - BROKEN)


def ensure_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS subdomain_refs (
        subdomain TEXT PRIMARY KEY, domain TEXT, refs_json TEXT, n_refs INTEGER, fetched_at TEXT
    );
    CREATE TABLE IF NOT EXISTS subdomain_ideas (
        idea_model TEXT NOT NULL, domain TEXT, subdomain TEXT NOT NULL,
        track TEXT NOT NULL, idea_index INTEGER NOT NULL,
        idea_text TEXT, telemetry TEXT, created_at TEXT NOT NULL,
        PRIMARY KEY (idea_model, subdomain, track, idea_index)
    );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
def do_fetch_refs():
    subs = load_subdomains()
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_tables(conn)
    have = {r[0] for r in conn.execute("SELECT subdomain FROM subdomain_refs")}
    todo = [(d, s) for d, s in subs if s not in have]
    print(f"subdomains={len(subs)} refs_to_fetch={len(todo)}")
    ok = empty = 0
    for i, (dom, sub) in enumerate(todo, 1):
        try:
            res = _search_papers(query=sub, limit=STATIC_REFS_N)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {sub[:50]}: SS error {e}"); continue
        with_abs = [r for r in res if (r.get("abstract") or "").strip()][:STATIC_REFS_N]
        if not with_abs:
            empty += 1; print(f"  [{i}/{len(todo)}] {sub[:50]}: 0 refs w/ abstract"); continue
        conn.execute("INSERT OR REPLACE INTO subdomain_refs VALUES (?,?,?,?,?)",
                     (sub, dom, json.dumps(with_abs, ensure_ascii=False), len(with_abs),
                      datetime.now(timezone.utc).isoformat()))
        conn.commit(); ok += 1
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] ok={ok} empty={empty}", flush=True)
    conn.close()
    print(f"done fetch-refs: ok={ok} empty={empty}")


def _synthetic_paper(dom, sub, refs):
    return {"title": sub, "domain": dom,
            "ranked_refs_json": json.dumps(refs, ensure_ascii=False),
            "paper_id": "sub:" + sub}


def do_static(workers):
    subs = load_subdomains()
    models = roster()
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tables(conn)
    refs = {r[0]: (r[1], json.loads(r[2])) for r in
            conn.execute("SELECT subdomain, domain, refs_json FROM subdomain_refs")}
    done = {(r[0], r[1]) for r in conn.execute(
        "SELECT idea_model, subdomain FROM subdomain_ideas WHERE track='B' "
        "AND idea_text IS NOT NULL AND TRIM(idea_text)!=''")}
    tasks = [(m, dom, sub) for m in models for dom, sub in subs
             if sub in refs and (m, sub) not in done]
    print(f"models={len(models)} subdomains={len(subs)} static_tasks={len(tasks)} workers={workers}")
    if not tasks:
        print("static: nothing to do."); conn.close(); return

    def _w(t):
        from utils.LLM import IdeaLLM
        m, dom, sub = t
        paper = _synthetic_paper(dom, sub, refs[sub][1])
        prompt, fallback, system = _build_generation_payload(paper, "B")
        try:
            out = IdeaLLM(model_name=m).generate_idea(prompt, fallback_prompt=fallback, system_prompt=system)
            txt = _clean_idea_text(out["idea"])
            if not txt or len(txt.split()) < MIN_WORDS:
                return (t, None)
            return (t, txt)
        except Exception as e:
            return (t, ("ERR:" + str(e)[:120]))

    ts = datetime.now(timezone.utc).isoformat()
    gen = blank = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            (m, dom, sub), txt = f.result()
            if txt and not txt.startswith("ERR:"):
                conn.execute("INSERT OR IGNORE INTO subdomain_ideas VALUES (?,?,?,?,?,?,?,?)",
                             (m, dom, sub, "B", 1, txt, None, ts)); conn.commit(); gen += 1
            elif txt and txt.startswith("ERR:"):
                err += 1
            else:
                blank += 1
            if i % 25 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] gen={gen} blank={blank} err={err}", flush=True)
    conn.close()
    print(f"done static: gen={gen} blank={blank} err={err}")


def do_active(workers):
    subs = load_subdomains()
    models = roster()
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tables(conn)
    done = {(r[0], r[1]) for r in conn.execute(
        "SELECT idea_model, subdomain FROM subdomain_ideas WHERE track='C' "
        "AND idea_text IS NOT NULL AND TRIM(idea_text)!=''")}
    conn.close()
    tasks = [(m, dom, sub) for m in models for dom, sub in subs if (m, sub) not in done]
    print(f"models={len(models)} subdomains={len(subs)} active_tasks={len(tasks)} workers={workers}")
    if not tasks:
        print("active: nothing to do."); return

    def _w(t):
        m, dom, sub = t
        try:
            res = run_active_agent(sub, m, max_iters=ACTIVE_BUDGET)
            err = res.get("error") or ""
            if "limit exceeded" in err.lower():
                return (t, "KEYLIMIT", None)
            txt = _clean_idea_text(res.get("hypothesis") or "")
            if not txt or len(txt.split()) < MIN_WORDS:
                return (t, None, None)
            tele = {"trace": res.get("trace", []), "n_tool_calls": res.get("n_tool_calls"),
                    "iters_used": res.get("iters_used"), "error": res.get("error")}
            return (t, txt, json.dumps(tele, ensure_ascii=False))
        except Exception as e:
            return (t, None, "ERR:" + str(e)[:120])

    import threading
    abort = threading.Event()
    ts = datetime.now(timezone.utc).isoformat()
    gen = blank = 0
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            (m, dom, sub), txt, tele = f.result()
            if txt == "KEYLIMIT":
                if not abort.is_set():
                    abort.set(); print("\n!!! ABORT: OpenRouter key limit (403). Top up & rerun.\n", flush=True)
                continue
            if txt:
                conn.execute("INSERT OR IGNORE INTO subdomain_ideas VALUES (?,?,?,?,?,?,?,?)",
                             (m, dom, sub, "C", 1, txt, tele, ts)); conn.commit(); gen += 1
            else:
                blank += 1
            if i % 25 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] gen={gen} blank={blank}", flush=True)
    conn.close()
    print(f"done active: gen={gen} blank={blank}")


def do_audit():
    subs = load_subdomains(); models = roster()
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    nref = conn.execute("SELECT COUNT(*) FROM subdomain_refs").fetchone()[0]
    print(f"subdomains={len(subs)} | open models={len(models)} | subdomain_refs fetched={nref}/100")
    for tr, name in (("B", "Static"), ("C", "Active")):
        n = conn.execute("SELECT COUNT(*) FROM subdomain_ideas WHERE track=? AND TRIM(idea_text)!=''",
                         (tr,)).fetchone()[0]
        full = conn.execute("SELECT COUNT(DISTINCT idea_model) FROM subdomain_ideas WHERE track=?", (tr,)).fetchone()[0]
        print(f"  {name}: {n} ideas / target {len(models)*len(subs)} ({full} models)")
    # per-model completeness
    print(f"\n{'model':<40}{'B/100':>7}{'C/100':>7}")
    for m in models:
        b = conn.execute("SELECT COUNT(*) FROM subdomain_ideas WHERE idea_model=? AND track='B' AND TRIM(idea_text)!=''", (m,)).fetchone()[0]
        cc = conn.execute("SELECT COUNT(*) FROM subdomain_ideas WHERE idea_model=? AND track='C' AND TRIM(idea_text)!=''", (m,)).fetchone()[0]
        print(f"{m:<40}{b:>7}{cc:>7}")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-refs", action="store_true")
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--active", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if not (a.fetch_refs or a.static or a.active or a.audit):
        a.audit = True
    if a.fetch_refs:
        do_fetch_refs()
    if a.static:
        do_static(a.workers)
    if a.active:
        do_active(a.workers)
    if a.audit:
        do_audit()


if __name__ == "__main__":
    main()
