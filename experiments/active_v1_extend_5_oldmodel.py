"""Extend Active mode (v1, text protocol, no title) for 5 older-cutoff open-weight
models from 5 papers → 25 papers (1 → 5 per domain).

Used to strengthen Finding 1 (size/cutoff vs Active boost) with older-cutoff coverage.
Inserts into `results` with prompt_version='v1_paper_refs' (matches existing smoke v3 data).

Models (all open-weight, default OR key, no 御三家):
  - mistralai/mistral-7b-instruct-v0.1     cutoff 2023-09
  - meta-llama/llama-3.1-8b-instruct       cutoff 2023-12
  - qwen/qwen-2.5-72b-instruct             cutoff 2024-08
  - google/gemma-3-27b-it                  cutoff 2024-12
  - deepseek/deepseek-r1-0528              cutoff 2024-07

Phase: gen-active, critic, report.
"""
import argparse
import collections
import json
import logging
import sqlite3
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TARGET_MODELS = [
    "mistralai/mistral-7b-instruct-v0.1",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "google/gemma-3-27b-it",
    "deepseek/deepseek-r1-0528",
]

CRITIC_POOL = [
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5.1",
    "moonshotai/kimi-k2.6",
]

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in DIMS) / WS


def fetch_papers(n_per_domain=5):
    conn = sqlite3.connect(str(cfg.PAPERS_DB))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("""
        SELECT paper_id, title, domain, query
          FROM papers
         WHERE status='filtered'
         ORDER BY domain, paper_id
    """).fetchall()]
    conn.close()
    by_dom = collections.defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    return [p for ps in by_dom.values() for p in ps[:n_per_domain]]


def phase_gen_active(papers, workers=3):
    from generation.active_agent import run_active_agent

    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")

    tasks = []
    for p in papers:
        for m in TARGET_MODELS:
            exists = conn.execute(
                "SELECT 1 FROM results WHERE paper_id=? AND idea_model=? "
                "AND track='C' AND idea_index=1 AND critic_model='' "
                "AND prompt_version='v1_paper_refs'",
                (p["paper_id"], m)).fetchone()
            if not exists:
                tasks.append((p, m))

    logger.info(f"phase_gen_active: {len(tasks)} tasks (5 models × {len(papers)} papers)")
    if not tasks:
        return

    def _worker(task):
        paper, model = task
        try:
            result = run_active_agent(
                domain=paper["domain"], model_name=model, max_iters=10,
                # NO TITLE - matches v1 design (existing smoke v3 protocol)
            )
            hyp = (result.get("hypothesis") or "").strip()
            if not hyp or len(hyp.split()) < 30:
                return ("err_short", paper, model, None)
            return ("ok", paper, model, hyp, result)
        except Exception as e:
            return ("err", paper, model, str(e)[:300])

    ts = datetime.now(timezone.utc).isoformat()
    stats = collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            status = res[0]
            if status == "ok":
                _, paper, model, hyp, payload = res
                tel = {
                    "trace": payload.get("trace", []),
                    "iters_used": payload.get("iters_used"),
                    "n_tool_calls": payload.get("n_tool_calls"),
                    "final_prompt_tokens": payload.get("final_prompt_tokens"),
                }
                conn.execute("""
                    INSERT OR IGNORE INTO results
                      (paper_id, idea_model, track, idea_index, idea_text,
                       critic_model, raw_response, created_at, prompt_version)
                    VALUES (?, ?, 'C', 1, ?, '', ?, ?, 'v1_paper_refs')
                """, (paper["paper_id"], model, hyp,
                      json.dumps(tel, ensure_ascii=False), ts))
                conn.commit()
            stats[status] += 1
            logger.info(f"  [{i}/{len(tasks)}] {res[2]:<42} {res[1]['domain']:<10} → {status}")
    conn.close()
    logger.info(f"gen-active done: {dict(stats)}")


def phase_critic(papers, workers=6):
    from evaluation.absolute_scorer import score_idea
    from evaluation.critic_manager import _format_refs_for_judge

    conn_r = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn_r.execute("PRAGMA busy_timeout=30000")
    conn_p = sqlite3.connect(str(cfg.PAPERS_DB), timeout=30)
    conn_p.row_factory = sqlite3.Row
    papers_by_id = {p["paper_id"]: p for p in papers}

    paper_ids = list(papers_by_id)
    pidp = ",".join("?" * len(paper_ids))
    mp = ",".join("?" * len(TARGET_MODELS))
    ideas = list(conn_r.execute(f"""
        SELECT paper_id, idea_model, idea_text FROM results
        WHERE track='C' AND idea_index=1 AND critic_model=''
          AND idea_model IN ({mp}) AND paper_id IN ({pidp})
          AND prompt_version='v1_paper_refs'
          AND idea_text IS NOT NULL AND idea_text != ''
    """, list(TARGET_MODELS) + paper_ids))

    tasks = []
    for pid, idea_model, idea_text in ideas:
        for critic in CRITIC_POOL:
            if critic == idea_model:
                continue
            exists = conn_r.execute(
                "SELECT 1 FROM results WHERE paper_id=? AND idea_model=? "
                "AND track='C' AND idea_index=1 AND critic_model=? "
                "AND prompt_version='v1_paper_refs'",
                (pid, idea_model, critic)).fetchone()
            if not exists:
                tasks.append((pid, idea_model, idea_text, critic))

    logger.info(f"phase_critic: {len(tasks)} score tasks")
    if not tasks:
        return

    def _worker(task):
        pid, idea_model, idea_text, critic = task
        paper = papers_by_id.get(pid)
        if not paper:
            row = conn_p.execute("SELECT * FROM papers WHERE paper_id=?", (pid,)).fetchone()
            paper = dict(row) if row else {}
        refs = _format_refs_for_judge(paper)
        try:
            scores, raw, telem = score_idea(
                idea_text, critic,
                domain=paper.get("domain", ""),
                references=refs,
                judge_mode="static",
                db_papers=str(cfg.PAPERS_DB),
            )
            return ("ok", pid, idea_model, critic, scores, raw)
        except Exception as e:
            return ("err", pid, idea_model, critic, None, str(e)[:300])

    ts = datetime.now(timezone.utc).isoformat()
    stats = collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            status = res[0]
            if status == "ok":
                _, pid, idea_model, critic, scores, raw = res
                if scores:
                    conn_r.execute("""
                        INSERT OR IGNORE INTO results
                          (paper_id, idea_model, track, idea_index, idea_text,
                           critic_model, scores_json, reasoning_json,
                           raw_response, created_at, prompt_version)
                        VALUES (?, ?, 'C', 1, '', ?, ?, ?, ?, ?, 'v1_paper_refs')
                    """, (pid, idea_model, critic,
                          json.dumps({d: v["score"] for d, v in scores.items()}),
                          json.dumps({d: v["reasoning"] for d, v in scores.items()}),
                          (raw or "")[:4000], ts))
                    conn_r.commit()
                    stats["ok"] += 1
            else:
                stats["err"] += 1
            if i % 20 == 0 or i == len(tasks):
                logger.info(f"  [{i}/{len(tasks)}] {dict(stats)}")
    conn_r.close()
    conn_p.close()
    logger.info(f"critic done: {dict(stats)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["gen-active", "critic", "all"])
    ap.add_argument("--n-per-domain", type=int, default=5)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    papers = fetch_papers(n_per_domain=args.n_per_domain)
    logger.info(f"Loaded {len(papers)} papers")

    t0 = time.time()
    if args.phase in ("gen-active", "all"):
        phase_gen_active(papers, workers=args.workers)
    if args.phase in ("critic", "all"):
        phase_critic(papers, workers=args.workers)
    logger.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
