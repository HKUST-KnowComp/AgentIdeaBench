"""E33 — Cross-family Gemini extension.

Generate Static(B) + Active(C) ideas for native-Gemini models on the SAME 40
scored subdomains (e19 idx {0,5,10,15} + e24 idx {2,7,12,17}), idea_index 1..3,
reusing the v3 subdomain pipeline verbatim. Writes `subdomain_ideas`
(INSERT OR IGNORE — idempotent, raw-data safe). Scoring is done AFTERWARDS by
`e21_pilot20_3seed.py` + `e24_extend_scoring.py`, which auto-include any model
with >50 ideas (open-weight critics, unaffected by Gemini).

Why Gemini adds value (honest scope): (1) fills the currently-missing closed
frontier family — the live 30-model set is all open-weight; (2) F2 family
robustness + within-family capability spectrum (2.5→3.6). NOT an F3 cutoff
ladder: all Gemini generations share a ~2025-01 knowledge cutoff.

Routing: google/gemini* → native Google endpoint (own quota) IFF env
GEMINI_API_KEY is set (see config.use_gemini_native). No seed (native rejects it).

Usage:
  GEMINI_API_KEY=... e33_gemini_extend.py --gen --limit 60 --workers 4   # batch
  GEMINI_API_KEY=... e33_gemini_extend.py --gen --smoke                  # 1 model x 2 subs
  e33_gemini_extend.py --status
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
    _synthetic_paper, _build_generation_payload,
    _clean_idea_text, ensure_tables, MIN_WORDS, ACTIVE_BUDGET,
)
from experiments.e19_pilot20_lit8d import pick_subdomains   # idx (0,5,10,15)
from experiments.e24_extend_scoring import pick_ext          # idx (2,7,12,17)
from generation.active_agent import run_active_agent

# Capability spectrum (lite->pro), ALL knowledge-cutoff ~2025-01 (SAFE: benchmark
# papers are 2025-04..2026-01, so the models predate them → no leakage).
# gemini-3.6-flash is EXCLUDED: its cutoff is 2026-03 (DeepMind model card),
# which postdates every benchmark paper → leakage risk (may have memorized the
# ground-truth papers). Any 3.6-flash rows already in subdomain_ideas are kept as
# raw data but MUST be excluded from scoring/analysis (see EXCLUDE_LEAKAGE below).
GEMINI_MODELS = [
    "google/gemini-2.5-flash-lite",   # weakest tier
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview",
    "google/gemini-3.5-flash",
    "google/gemini-3.1-pro-preview",  # flagship
]
# Models generated but excluded from analysis due to post-benchmark cutoff.
EXCLUDE_LEAKAGE = ["google/gemini-3.6-flash"]
N_IDX = 3


def scored_subs(conn):
    """The 40 scored (domain, subdomain) cells — e19 picks + e24 picks, deduped,
    order-preserving. Identical selection to the main lit8d_scores_3seed set."""
    picks = list(pick_subdomains(conn)) + list(pick_ext(conn))
    seen, out = set(), []
    for d, s in picks:
        if (d, s) not in seen:
            seen.add((d, s))
            out.append((d, s))
    return out


def _refs_map(conn):
    return {r[0]: (r[1], json.loads(r[2])) for r in
            conn.execute("SELECT subdomain, domain, refs_json FROM subdomain_refs")}


def _done_set(conn, track):
    return {(r[0], r[1], r[2]) for r in conn.execute(
        "SELECT idea_model, subdomain, idea_index FROM subdomain_ideas "
        "WHERE track=? AND TRIM(idea_text)!=''", (track,))}


def gen(models, workers, limit, smoke, track_filter="both"):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_tables(conn)
    subs = scored_subs(conn)
    refs = _refs_map(conn)
    if smoke:
        models = models[:1]
        subs = subs[:2]
    doneB = _done_set(conn, "B")
    doneC = _done_set(conn, "C")

    want_b = track_filter in ("both", "B")
    want_c = track_filter in ("both", "C")
    tasks = []   # (model, dom, sub, track, idx)
    for m in models:
        for dom, sub in subs:
            for idx in range(1, N_IDX + 1):
                if want_b and sub in refs and (m, sub, idx) not in doneB:
                    tasks.append((m, dom, sub, "B", idx))
                if want_c and (m, sub, idx) not in doneC:
                    tasks.append((m, dom, sub, "C", idx))

    total = len(tasks)
    if limit:
        tasks = tasks[:limit]
    print(f"gen tasks remaining: {total}; running {len(tasks)} this batch "
          f"(limit={limit}, models={len(models)}, subs={len(subs)})", flush=True)
    if not tasks:
        print("gen: nothing to do.")
        conn.close()
        return

    def _w(task):
        m, dom, sub, track, idx = task
        try:
            if track == "B":
                from utils.LLM import IdeaLLM
                paper = _synthetic_paper(dom, sub, refs[sub][1])
                prompt, fb, system = _build_generation_payload(paper, "B")
                out = IdeaLLM(model_name=m).generate_idea(
                    prompt, fallback_prompt=fb, system_prompt=system)
                txt = _clean_idea_text(out["idea"])
                tele = None
            else:
                res = run_active_agent(sub, m, max_iters=ACTIVE_BUDGET)
                err = res.get("error") or ""
                if "limit exceeded" in err.lower():
                    return (task, "KEYLIMIT", None)
                # Retrieval-quality gate: require >=1 non-empty SS result. When
                # SS is down (504/429) the agent falls back to parametric-only
                # generation, which is NOT valid Active-mode data — skip it so a
                # later batch regenerates the cell once SS recovers.
                _tr = res.get("trace", [])
                if sum(1 for s in _tr if len((s.get("result_preview") or "")) > 5) == 0:
                    return (task, None, "NORETR")
                txt = _clean_idea_text(res.get("hypothesis") or "")
                tele = json.dumps({"trace": res.get("trace", []),
                                   "n_tool_calls": res.get("n_tool_calls"),
                                   "iters_used": res.get("iters_used"),
                                   "error": res.get("error")}, ensure_ascii=False)
            if not txt or len(txt.split()) < MIN_WORDS:
                return (task, None, None)
            return (task, txt, tele)
        except Exception as e:
            return (task, None, "ERR:" + str(e)[:120])

    ts = datetime.now(timezone.utc).isoformat()
    g = b = e = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            task, txt, tele = f.result()
            m, dom, sub, track, idx = task
            if txt == "KEYLIMIT":
                print("!!! key limit exceeded; stopping batch.", flush=True)
                break
            if txt and not (isinstance(tele, str) and tele.startswith("ERR:")):
                conn.execute(
                    "INSERT OR IGNORE INTO subdomain_ideas VALUES (?,?,?,?,?,?,?,?)",
                    (m, dom, sub, track, idx, txt, tele, ts))
                conn.commit()
                g += 1
            elif isinstance(tele, str) and tele.startswith("ERR:"):
                e += 1
            else:
                b += 1
            if i % 20 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] gen={g} blank={b} err={e}", flush=True)
    conn.close()
    print(f"done gen: gen={g} blank={b} err={e}")


def status(models):
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    subs = scored_subs(conn)
    tgt = len(subs) * N_IDX
    print(f"scored subdomains: {len(subs)}  (target per model per track = {tgt})")
    for m in models:
        b = conn.execute("SELECT COUNT(*) FROM subdomain_ideas WHERE idea_model=? "
                         "AND track='B' AND TRIM(idea_text)!=''", (m,)).fetchone()[0]
        c = conn.execute("SELECT COUNT(*) FROM subdomain_ideas WHERE idea_model=? "
                         "AND track='C' AND TRIM(idea_text)!=''", (m,)).fetchone()[0]
        print(f"  {m:34s} B={b}/{tgt}  C={c}/{tgt}")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of GEMINI_MODELS (default: all 5)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap tasks this batch (0=all)")
    ap.add_argument("--smoke", action="store_true", help="1 model x 2 subs")
    ap.add_argument("--track", choices=["both", "B", "C"], default="both",
                    help="B=static only (no SS needed), C=active only, both=default")
    a = ap.parse_args()
    models = a.models if a.models else GEMINI_MODELS
    if a.status:
        status(models)
    if a.gen:
        gen(models, a.workers, a.limit, a.smoke, a.track)


if __name__ == "__main__":
    main()
