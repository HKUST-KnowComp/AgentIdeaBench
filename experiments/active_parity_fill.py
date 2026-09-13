"""Active parity fill — bring Track C (Active-Blind) coverage up to match Static.

Goal (user, overnight 2026-06-27): every OPEN-SOURCE model that has Static
(Track B, v1_paper_refs) ideas must also have Active ideas on the SAME 25-paper
set with the SAME depth (3 ideas/paper → idea_index 1,2,3), so Static and Active
have identical n (same models, same papers per domain, same #ideas).

Closed models (御三家: openai/, anthropic/, google/gemini) are LEFT for later
per user instruction; only default-key open models run here (incl. google/gemma).

Why a custom orchestrator (not run.py --phase 2 --active):
  - run_active_phase is a serial for-loop → far too slow for ~1900 ideas.
  - This parallelizes across models (different OpenRouter rate buckets) while the
    shared SS limit is tolerated by the short [3,6,12]s active backoff.
  - idx=1 is generated for ALL models first (model+paper parity), then idx 2,3
    (depth), so even a partial run leaves a maximally-useful state.

SAFETY / raw-data rules:
  - INSERT-only, idempotent: skips any (model,paper,idx) that already has a
    NON-blank Track C idea. Never DELETEs / overwrites existing rows.
  - NEVER inserts a blank/short hypothesis (<30 words) — those are counted as
    errors and retried on the next run, so no blank placeholders enter the DB.
  - prompt_version defaults to 'v1_paper_refs' (matches existing Active).

Usage:
  /usr/bin/python3 experiments/active_parity_fill.py --audit          # report gaps + blanks
  /usr/bin/python3 experiments/active_parity_fill.py --run --max-workers 6
  /usr/bin/python3 experiments/active_parity_fill.py --run --only-idx 1   # idx=1 pass only
"""
import argparse
import json
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from generation.active_agent import run_active_agent
from generation.generate_ideas import _clean_idea_text

TRACK = "C"
PROMPT_VERSION = "v1_paper_refs"
MAX_ITERS = 10
TARGET_IDX = [1, 2, 3]
MIN_WORDS = 30

_print_lock = threading.Lock()


def smoke25():
    """Reproduce run_active_phase smoke set: first 5 filtered papers per domain
    (ordered by domain, paper_id). Returns list of (paper_id, domain)."""
    cp = sqlite3.connect(str(cfg.PAPERS_DB)); cp.row_factory = sqlite3.Row
    q = ("SELECT paper_id, domain FROM papers WHERE status='filtered' "
         "AND gt_hypothesis IS NOT NULL AND gt_hypothesis!='' "
         "AND ranked_refs_json IS NOT NULL ORDER BY domain, paper_id")
    by = defaultdict(list)
    for r in cp.execute(q):
        by[r["domain"]].append((r["paper_id"], r["domain"]))
    cp.close()
    return [pd for ps in by.values() for pd in ps[:5]]


def open_static_models():
    """Open-source (default-key) models that have Static B/v1 ideas, no baselines."""
    c = sqlite3.connect(str(cfg.RESULTS_DB))
    ms = [r[0] for r in c.execute(
        "SELECT DISTINCT idea_model FROM results WHERE track='B' "
        "AND prompt_version='v1_paper_refs' AND idea_text!=''")]
    c.close()
    return sorted(m for m in ms
                  if not m.startswith("baseline/") and not cfg.is_us_key_model(m))


def existing_active():
    """Set of (model, paper_id, idx) that already have a non-blank Track C idea."""
    c = sqlite3.connect(str(cfg.RESULTS_DB))
    have = set()
    for r in c.execute(
            "SELECT idea_model, paper_id, idea_index FROM results "
            "WHERE track='C' AND prompt_version=? AND critic_model='' "
            "AND idea_text IS NOT NULL AND TRIM(idea_text)!=''", (PROMPT_VERSION,)):
        have.add((r[0], r[1], r[2]))
    c.close()
    return have


def build_work(only_idx=None):
    papers = smoke25()
    models = open_static_models()
    have = existing_active()
    pdom = {pid: dom for pid, dom in papers}
    idxs = [only_idx] if only_idx else TARGET_IDX
    work = []
    # idx-major ordering (idx1 for everyone first), models interleaved within
    # each (idx, paper) so concurrent workers spread across OpenRouter buckets.
    for idx in idxs:
        for pid, dom in papers:
            for m in models:
                if (m, pid, idx) not in have:
                    work.append((m, pid, dom, idx))
    return work, len(models), len(papers)


def insert_idea(model, pid, idx, hypothesis, telemetry, ts):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    try:
        cur = conn.cursor()
        cur.execute("""INSERT OR IGNORE INTO results
            (paper_id, idea_model, track, idea_index, idea_text,
             critic_model, raw_response, created_at, prompt_version)
            VALUES (?,?,?,?,?,'',?,?,?)""",
            (pid, model, TRACK, idx, hypothesis,
             json.dumps(telemetry), ts, PROMPT_VERSION))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def do_run(max_workers, only_idx):
    work, nmodels, npapers = build_work(only_idx)
    print(f"open models={nmodels} papers={npapers} target_idx={only_idx or TARGET_IDX}")
    print(f"work items to fill: {len(work)} (workers={max_workers})")
    if not work:
        print("nothing to do — parity already met for this scope.")
        return
    counts = {"gen": 0, "blank": 0, "err": 0, "keylimit": 0}
    t_start = time.time()
    abort = threading.Event()  # set on OpenRouter key-limit 403 → short-circuit rest

    def _w(item):
        if abort.is_set():
            return ("skip", item, None)
        model, pid, dom, idx = item
        ts = datetime.now(timezone.utc).isoformat()
        try:
            res = run_active_agent(dom, model, max_iters=MAX_ITERS)
            err = res.get("error") or ""
            if "limit exceeded" in err.lower() or "insufficient credits" in err.lower():
                abort.set()
                return ("keylimit", item, err)
            hyp = _clean_idea_text(res.get("hypothesis") or "")
            if not hyp or len(hyp.split()) < MIN_WORDS:
                return ("blank", item, err)
            tele = {
                "trace": res.get("trace", []), "turns": res.get("turns", []),
                "iters_used": res.get("iters_used"), "n_tool_calls": res.get("n_tool_calls"),
                "n_turns": res.get("n_turns"), "final_prompt_tokens": res.get("final_prompt_tokens"),
                "final_total_tokens": res.get("final_total_tokens"),
                "budget_nudge_used": res.get("budget_nudge_used"), "error": res.get("error"),
            }
            ok = insert_idea(model, pid, idx, hyp, tele, ts)
            return ("gen" if ok else "dup", item, len(hyp.split()))
        except Exception as e:
            return ("err", item, str(e)[:200])

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_w, it) for it in work]
        for i, f in enumerate(as_completed(futs), 1):
            status, item, info = f.result()
            if status in counts:
                counts[status] += 1
            if status == "keylimit" and counts["keylimit"] == 1:
                with _print_lock:
                    print(f"\n!!! ABORT: OpenRouter key limit exceeded (403). "
                          f"Add credits / raise the default key limit, then re-run. "
                          f"Detail: {info}\n", flush=True)
            if i % 10 == 0 or i == len(work):
                rate = i / max(1e-9, (time.time() - t_start))
                eta_min = (len(work) - i) / max(1e-9, rate) / 60
                with _print_lock:
                    print(f"  [{i}/{len(work)}] gen={counts['gen']} "
                          f"blank={counts['blank']} err={counts['err']} "
                          f"| {rate*60:.1f}/min ETA {eta_min:.0f}min", flush=True)
    print(f"DONE run: {counts} in {(time.time()-t_start)/60:.1f} min")


def do_audit():
    papers = smoke25()
    models = open_static_models()
    have = existing_active()
    # blank/short audit on ALL existing Track C idea rows
    c = sqlite3.connect(str(cfg.RESULTS_DB)); c.row_factory = sqlite3.Row
    blanks = []
    for r in c.execute("SELECT idea_model, paper_id, idea_index, idea_text FROM results "
                       "WHERE track='C' AND critic_model='' AND prompt_version=?", (PROMPT_VERSION,)):
        t = (r["idea_text"] or "").strip()
        if not t or len(t.split()) < MIN_WORDS:
            blanks.append((r["idea_model"], r["paper_id"], r["idea_index"], len(t.split())))
    c.close()
    print(f"=== AUDIT (target: {len(models)} open models × {len(papers)} papers × {len(TARGET_IDX)} idx) ===")
    print(f"blank/short existing Track C idea rows: {len(blanks)}")
    for b in blanks[:20]:
        print("  blank:", b)
    print()
    print(f"{'model':<42}{'have/75':>10}{'missing':>9}")
    total_missing = 0
    for m in models:
        h = sum(1 for pid, _ in papers for idx in TARGET_IDX if (m, pid, idx) in have)
        miss = len(papers) * len(TARGET_IDX) - h
        total_missing += miss
        flag = "" if miss == 0 else "  <-"
        print(f"{m:<42}{h:>7}/75{miss:>9}{flag}")
    print(f"\nTOTAL missing to reach parity (25×3): {total_missing}")
    # idx=1 specific (model+paper parity)
    miss1 = sum(1 for m in models for pid, _ in papers if (m, pid, 1) not in have)
    print(f"missing idx=1 (model+paper parity): {miss1}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--only-idx", type=int, default=None, choices=[1, 2, 3])
    args = ap.parse_args()
    if not (args.run or args.audit):
        args.audit = True
    if args.audit:
        do_audit()
    if args.run:
        do_run(args.max_workers, args.only_idx)


if __name__ == "__main__":
    main()
