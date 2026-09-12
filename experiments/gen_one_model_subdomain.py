"""Generate v3 subdomain Static + Active ideas for ONE specified model.

Reuses the v3 pipeline helpers so the output is identical in format to the
27-model roster run. New rows only (idempotent: skips existing non-blank cells).
Static uses IdeaLLM on the synthetic subdomain paper; Active uses run_active_agent
with the same subdomain + 10 tool-call budget.

Usage:
  /usr/bin/python3 experiments/gen_one_model_subdomain.py --model deepseek/deepseek-v4-flash --static --workers 4
  /usr/bin/python3 experiments/gen_one_model_subdomain.py --model deepseek/deepseek-v4-flash --active --workers 4
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
from experiments.v3_subdomain_ideation import (
    load_subdomains, _synthetic_paper, _build_generation_payload,
    _clean_idea_text, ensure_tables, MIN_WORDS, ACTIVE_BUDGET,
)
from generation.active_agent import run_active_agent


def do_static(model, workers):
    subs = load_subdomains()
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tables(conn)
    refs = {r[0]: (r[1], json.loads(r[2])) for r in
            conn.execute("SELECT subdomain, domain, refs_json FROM subdomain_refs")}
    done = {r[0] for r in conn.execute(
        "SELECT subdomain FROM subdomain_ideas WHERE idea_model=? AND track='B' "
        "AND TRIM(idea_text)!=''", (model,))}
    tasks = [(dom, sub) for dom, sub in subs if sub in refs and sub not in done]
    print(f"[static] model={model} tasks={len(tasks)} workers={workers}", flush=True)
    if not tasks:
        print("static: nothing to do."); conn.close(); return

    def _w(t):
        from utils.LLM import IdeaLLM
        dom, sub = t
        paper = _synthetic_paper(dom, sub, refs[sub][1])
        prompt, fb, system = _build_generation_payload(paper, "B")
        try:
            out = IdeaLLM(model_name=model).generate_idea(prompt, fallback_prompt=fb, system_prompt=system)
            txt = _clean_idea_text(out["idea"])
            if not txt or len(txt.split()) < MIN_WORDS:
                return (t, None)
            return (t, txt)
        except Exception as e:
            return (t, "ERR:" + str(e)[:120])

    ts = datetime.now(timezone.utc).isoformat()
    gen = blank = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            (dom, sub), txt = f.result()
            if txt and not txt.startswith("ERR:"):
                conn.execute("INSERT OR IGNORE INTO subdomain_ideas VALUES (?,?,?,?,?,?,?,?)",
                             (model, dom, sub, "B", 1, txt, None, ts)); conn.commit(); gen += 1
            elif txt and txt.startswith("ERR:"):
                err += 1
            else:
                blank += 1
            if i % 20 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] gen={gen} blank={blank} err={err}", flush=True)
    conn.close()
    print(f"done static: gen={gen} blank={blank} err={err}")


def do_active(model, workers):
    subs = load_subdomains()
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tables(conn)
    dom_by_sub = {r[1]: r[0] for r in [(d, s) for d, s in subs]}
    done = {r[0] for r in conn.execute(
        "SELECT subdomain FROM subdomain_ideas WHERE idea_model=? AND track='C' "
        "AND TRIM(idea_text)!=''", (model,))}
    conn.close()
    tasks = [(dom, sub) for dom, sub in subs if sub not in done]
    print(f"[active] model={model} tasks={len(tasks)} workers={workers}", flush=True)
    if not tasks:
        print("active: nothing to do."); return

    def _w(t):
        dom, sub = t
        try:
            res = run_active_agent(sub, model, max_iters=ACTIVE_BUDGET)
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

    ts = datetime.now(timezone.utc).isoformat()
    gen = blank = 0
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            (dom, sub), txt, tele = f.result()
            if txt == "KEYLIMIT":
                print("!!! key limit; stop.", flush=True); break
            if txt:
                conn.execute("INSERT OR IGNORE INTO subdomain_ideas VALUES (?,?,?,?,?,?,?,?)",
                             (model, dom, sub, "C", 1, txt, tele, ts)); conn.commit(); gen += 1
            else:
                blank += 1
            if i % 20 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] gen={gen} blank={blank}", flush=True)
    conn.close()
    print(f"done active: gen={gen} blank={blank}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--active", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.static:
        do_static(a.model, a.workers)
    if a.active:
        do_active(a.model, a.workers)


if __name__ == "__main__":
    main()
