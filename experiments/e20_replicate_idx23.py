"""E20 — replicate every seed to 3 ideas: add idea_index 2 and 3 for every
existing (model, subdomain, track) that currently has idx=1.

Reduces per-seed sampling noise (each seed currently has n=1). Reuses the v3
generation path (Static = IdeaLLM on subdomain+refs; Active = run_active_agent).
Idempotent: only generates missing (model, subdomain, track, idx) cells.

New rows go into the existing subdomain_ideas table with idea_index in {2,3};
idx=1 rows are untouched (raw-data rule).

Usage (staged):
  /usr/bin/python3 experiments/e20_replicate_idx23.py --static --workers 8
  /usr/bin/python3 experiments/e20_replicate_idx23.py --active --workers 6
  /usr/bin/python3 experiments/e20_replicate_idx23.py --audit
"""
import argparse, json, sqlite3, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from experiments.v3_subdomain_ideation import _synthetic_paper, MIN_WORDS, ACTIVE_BUDGET
from generation.generate_ideas import _build_generation_payload, _clean_idea_text
from generation.active_agent import run_active_agent

REPS = (2, 3)


def _base_cells(conn, track):
    """(model, domain, subdomain) that have an idx=1 idea in this track."""
    return [(r[0], r[1], r[2]) for r in conn.execute(
        "SELECT DISTINCT idea_model, domain, subdomain FROM subdomain_ideas "
        "WHERE track=? AND idea_index=1 AND TRIM(idea_text)!=''", (track,))]


def _missing(conn, track):
    have = {(r[0], r[1], r[2]) for r in conn.execute(
        "SELECT idea_model, subdomain, idea_index FROM subdomain_ideas "
        "WHERE track=? AND TRIM(idea_text)!=''", (track,))}
    tasks = []
    for m, dom, sub in _base_cells(conn, track):
        for idx in REPS:
            if (m, sub, idx) not in have:
                tasks.append((m, dom, sub, idx))
    return tasks


def do_static(workers):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    refs = {r[0]: json.loads(r[1]) for r in conn.execute("SELECT subdomain, refs_json FROM subdomain_refs")}
    tasks = [t for t in _missing(conn, "B") if t[2] in refs]
    print(f"static replicate tasks (idx 2,3): {len(tasks)}", flush=True)
    if not tasks:
        conn.close(); return

    def _w(t):
        from utils.LLM import IdeaLLM
        m, dom, sub, idx = t
        paper = _synthetic_paper(dom, sub, refs[sub])
        prompt, fallback, system = _build_generation_payload(paper, "B")
        try:
            out = IdeaLLM(model_name=m).generate_idea(prompt, fallback_prompt=fallback, system_prompt=system)
            txt = _clean_idea_text(out["idea"])
            if not txt or len(txt.split()) < MIN_WORDS:
                return (t, None)
            return (t, txt)
        except Exception as e:
            return (t, "ERR:" + str(e)[:100])

    ts = datetime.now(timezone.utc).isoformat(); gen = blank = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_w, t) for t in tasks]), 1):
            (m, dom, sub, idx), txt = f.result()
            if txt and not txt.startswith("ERR:"):
                conn.execute("INSERT OR IGNORE INTO subdomain_ideas VALUES (?,?,?,?,?,?,?,?)",
                             (m, dom, sub, "B", idx, txt, None, ts)); conn.commit(); gen += 1
            elif txt: err += 1
            else: blank += 1
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] gen={gen} blank={blank} err={err}", flush=True)
    conn.close(); print(f"done static replicate: gen={gen} blank={blank} err={err}")


def do_active(workers):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    tasks = _missing(conn, "C")
    conn.close()
    print(f"active replicate tasks (idx 2,3): {len(tasks)}", flush=True)
    if not tasks:
        return

    def _w(t):
        m, dom, sub, idx = t
        try:
            res = run_active_agent(sub, m, max_iters=ACTIVE_BUDGET)
            err = (res.get("error") or "")
            if "limit exceeded" in err.lower():
                return (t, "KEYLIMIT", None)
            txt = _clean_idea_text(res.get("hypothesis") or "")
            if not txt or len(txt.split()) < MIN_WORDS:
                return (t, None, None)
            tele = {"trace": res.get("trace", []), "n_tool_calls": res.get("n_tool_calls"),
                    "iters_used": res.get("iters_used"), "error": res.get("error")}
            return (t, txt, json.dumps(tele, ensure_ascii=False))
        except Exception as e:
            return (t, None, "ERR:" + str(e)[:100])

    import threading
    abort = threading.Event()
    ts = datetime.now(timezone.utc).isoformat(); gen = blank = 0
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60); conn.execute("PRAGMA busy_timeout=60000")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_w, t) for t in tasks]), 1):
            (m, dom, sub, idx), txt, tele = f.result()
            if txt == "KEYLIMIT":
                if not abort.is_set():
                    abort.set(); print("\n!!! ABORT: OpenRouter key limit (403). Top up & rerun.\n", flush=True)
                continue
            if txt:
                conn.execute("INSERT OR IGNORE INTO subdomain_ideas VALUES (?,?,?,?,?,?,?,?)",
                             (m, dom, sub, "C", idx, txt, tele, ts)); conn.commit(); gen += 1
            else: blank += 1
            if i % 25 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] gen={gen} blank={blank}", flush=True)
    conn.close(); print(f"done active replicate: gen={gen} blank={blank}")


def do_audit():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    for tr, nm in (("B", "Static"), ("C", "Active")):
        rows = conn.execute("SELECT idea_index, COUNT(*) FROM subdomain_ideas WHERE track=? "
                            "AND TRIM(idea_text)!='' GROUP BY idea_index ORDER BY idea_index", (tr,)).fetchall()
        print(f"{nm}: " + " ".join(f"idx{r[0]}={r[1]}" for r in rows))
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    for f in ("static", "active", "audit"): ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.static: do_static(a.workers)
    if a.active: do_active(a.workers)
    if a.audit: do_audit()


if __name__ == "__main__":
    main()
