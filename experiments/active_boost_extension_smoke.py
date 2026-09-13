"""Smoke: extend Active-mode coverage to 5 older-cutoff models.

Goal: confirm Active boost (C-B) decreases with cutoff. Current 6-model data
shows GPT-5 (cutoff 2024-09) has anomalously SMALL boost (+0.225). User
wants to verify by adding 5 older-cutoff models:

  openai/gpt-4                          (cutoff 2021-09)  ← KILLER old anchor
  mistralai/mistral-7b-instruct-v0.1    (cutoff 2023-09)
  meta-llama/llama-3.1-8b-instruct      (cutoff 2023-12)
  qwen/qwen-2.5-72b-instruct            (cutoff 2024-08)
  google/gemma-3-27b-it                 (cutoff 2024-12)

Smoke = 1 paper per domain (5 paper total), 1 active idea each, 3 critic
each = 5 model × 5 paper × 1 idea × 3 critic = 75 critic rows + 25 active runs.

Stores in main results.db `results` table with track='C' (matches main pipeline
convention). Then computes weighted boost per model and prints.

Cost est: ~$3-4 + ~15 min.
"""
import argparse
import json
import logging
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# User direction 2026-05-18: no 御三家 (OpenAI/Anthropic/Gemini) for smokes.
# All targets must be OR default-key and verified to support tool use under
# the plain-text agent protocol (active_agent.run_active_agent).
#
# gpt-4 / claude-3.x / gemini-* removed from this list per that rule.
# Old func-call smoke data archived in `results_archive_funccall_smoke_5x5`.
TARGET_MODELS = [
    "mistralai/mistral-7b-instruct-v0.1",      # cutoff 2023-09 (old anchor)
    "meta-llama/llama-3.1-8b-instruct",        # cutoff 2023-12
    "qwen/qwen-2.5-72b-instruct",              # cutoff 2024-08
    "google/gemma-3-27b-it",                   # cutoff 2024-12
    "deepseek/deepseek-r1-0528",               # cutoff 2024-07 (added — tool-use ✓)
]

CRITIC_POOL = [
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5.1",
    "moonshotai/kimi-k2.5",
]

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in DIMS) / WS


def trim_mean(vals):
    if len(vals) < 2:
        return statistics.mean(vals) if vals else 0
    return statistics.mean(sorted(vals, reverse=True)[1:])


def fetch_smoke_papers(n_per_domain=1):
    """1 paper per domain = 5 papers (matches active_qwen_vs_claude smoke)."""
    import collections
    conn = sqlite3.connect(str(cfg.PAPERS_DB))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("""
        SELECT * FROM papers
        WHERE status='filtered' AND gt_hypothesis IS NOT NULL AND gt_hypothesis != ''
          AND ranked_refs_json IS NOT NULL
        ORDER BY domain, paper_id
    """).fetchall()]
    conn.close()
    by_dom = collections.defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    return [p for ps in by_dom.values() for p in ps[:n_per_domain]]


def phase_gen(papers, workers=3):
    """Generate active ideas for the 5 target models on the 5 papers."""
    from generation.active_agent import run_active_agent

    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")

    tasks = []
    for p in papers:
        for m in TARGET_MODELS:
            exists = conn.execute(
                "SELECT 1 FROM results WHERE paper_id=? AND idea_model=? "
                "AND track='C' AND idea_index=1 AND critic_model=''",
                (p["paper_id"], m)).fetchone()
            if not exists:
                tasks.append((p, m))

    logger.info(f"phase_gen: {len(tasks)} tasks ({len(papers)} papers × {len(TARGET_MODELS)} models)")
    if not tasks:
        return 0

    def _worker(task):
        paper, model = task
        t0 = time.time()
        try:
            result = run_active_agent(
                domain=paper["domain"],
                model_name=model,
                max_iters=10,
            )
            hypothesis = (result.get("hypothesis") or "").strip()
            if not hypothesis or len(hypothesis.split()) < 30:
                return ("err_short", paper, model, None, result)
            return ("ok", paper, model, hypothesis, result)
        except Exception as e:
            return ("err", paper, model, None, str(e)[:300])

    ts = datetime.now(timezone.utc).isoformat()
    stats = {"ok": 0, "err": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            status, paper, model, hyp, payload = f.result()
            if status == "ok":
                # Persist trace + telemetry (no truncation) into raw_response
                tel = {
                    "trace": payload.get("trace", []),
                    "iters_used": payload.get("iters_used"),
                    "n_tool_calls": payload.get("n_tool_calls"),
                    "final_prompt_tokens": payload.get("final_prompt_tokens"),
                }
                conn.execute("""
                    INSERT OR IGNORE INTO results
                      (paper_id, idea_model, track, idea_index, idea_text,
                       critic_model, raw_response, created_at)
                    VALUES (?, ?, 'C', 1, ?, '', ?, ?)
                """, (paper["paper_id"], model, hyp,
                      json.dumps(tel, ensure_ascii=False), ts))
                conn.commit()
                stats["ok"] += 1
            else:
                stats["err"] += 1
            logger.info(f"  [{i}/{len(tasks)}] {model:<45} {paper['domain']:<10} → {status}")
    conn.close()
    return stats


def phase_critic(papers, workers=4):
    """Critic-score the new Active ideas."""
    from evaluation.absolute_scorer import score_idea
    from evaluation.critic_manager import _format_refs_for_judge

    conn_r = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn_r.execute("PRAGMA busy_timeout=30000")
    conn_p = sqlite3.connect(str(cfg.PAPERS_DB), timeout=30)
    conn_p.row_factory = sqlite3.Row
    papers_by_id = {p["paper_id"]: p for p in papers}

    # Find ideas needing scoring
    paper_ids = list(papers_by_id)
    placeholders = ",".join("?" * len(paper_ids))
    m_ph = ",".join("?" * len(TARGET_MODELS))
    ideas = list(conn_r.execute(f"""
        SELECT paper_id, idea_model, idea_text FROM results
        WHERE track='C' AND idea_index=1 AND critic_model=''
          AND idea_model IN ({m_ph})
          AND paper_id IN ({placeholders})
          AND idea_text IS NOT NULL
    """, list(TARGET_MODELS) + paper_ids))

    tasks = []
    for pid, idea_model, idea_text in ideas:
        for critic in CRITIC_POOL:
            exists = conn_r.execute(
                "SELECT 1 FROM results WHERE paper_id=? AND idea_model=? "
                "AND track='C' AND idea_index=1 AND critic_model=?",
                (pid, idea_model, critic)).fetchone()
            if not exists:
                tasks.append((pid, idea_model, idea_text, critic))

    logger.info(f"phase_critic: {len(tasks)} score tasks")
    if not tasks:
        return 0

    def _worker(task):
        pid, idea_model, idea_text, critic = task
        paper = papers_by_id.get(pid)
        if not paper:
            paper_row = conn_p.execute("SELECT * FROM papers WHERE paper_id=?", (pid,)).fetchone()
            paper = dict(paper_row) if paper_row else {}
        refs = _format_refs_for_judge(paper)
        try:
            scores, raw, telem = score_idea(
                idea_text, critic,
                domain=paper.get("domain", ""),
                references=refs,
                judge_mode="static",
                db_papers=str(cfg.PAPERS_DB),
            )
            return ("ok", pid, idea_model, critic, scores, raw, telem)
        except Exception as e:
            return ("err", pid, idea_model, critic, None, None, str(e)[:300])

    ts = datetime.now(timezone.utc).isoformat()
    stats = {"ok": 0, "err": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            status, pid, idea_model, critic, scores, raw, telem = f.result()
            if status == "ok" and scores:
                conn_r.execute("""
                    INSERT OR IGNORE INTO results
                      (paper_id, idea_model, track, idea_index, idea_text,
                       critic_model, scores_json, reasoning_json,
                       raw_response, created_at)
                    VALUES (?,?,'C',1,'',?,?,?,?,?)
                """, (pid, idea_model, critic,
                      json.dumps({d: v["score"] for d, v in scores.items()}),
                      json.dumps({d: v["reasoning"] for d, v in scores.items()}),
                      (raw or "")[:4000], ts))
                conn_r.commit()
                stats["ok"] += 1
            else:
                stats["err"] += 1
            if i % 10 == 0 or i == len(tasks):
                logger.info(f"  [{i}/{len(tasks)}] {stats['ok']} ok / {stats['err']} err")
    conn_r.close()
    conn_p.close()
    return stats


def report(papers):
    """Compute weighted Active score per model, compare to Static, print boost."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    paper_ids = list(p["paper_id"] for p in papers)
    placeholders = ",".join("?" * len(paper_ids))

    # Active scores per (model, paper) trimmed-mean
    by_combo = defaultdict(list)
    for r in conn.execute(f"""
        SELECT paper_id, idea_model, critic_model, scores_json
        FROM results WHERE track='C' AND idea_index=1 AND critic_model != ''
          AND idea_model IN ({",".join("?"*len(TARGET_MODELS))})
          AND paper_id IN ({placeholders})
          AND scores_json IS NOT NULL
    """, list(TARGET_MODELS) + paper_ids):
        s = json.loads(r[3])
        by_combo[(r[1], r[0])].append(weighted(s))

    active_per_model = defaultdict(list)
    for (m, p), vals in by_combo.items():
        active_per_model[m].append(trim_mean(vals))

    # Static scores from model_scores aggregate (full dataset, not just these 5 papers)
    static_scores = {}
    for r in conn.execute(f"""
        SELECT idea_model, AVG(mean_absolute_score)
        FROM model_scores WHERE track='B' AND idea_model IN ({",".join("?"*len(TARGET_MODELS))})
        GROUP BY idea_model
    """, TARGET_MODELS):
        static_scores[r[0]] = r[1]

    # Also compute Static restricted to same 5 paper smoke set (apples-to-apples)
    static_5paper = defaultdict(list)
    by_combo_b = defaultdict(list)
    for r in conn.execute(f"""
        SELECT paper_id, idea_model, critic_model, scores_json
        FROM results WHERE track='B' AND critic_model != ''
          AND idea_model IN ({",".join("?"*len(TARGET_MODELS))})
          AND paper_id IN ({placeholders})
          AND scores_json IS NOT NULL
    """, list(TARGET_MODELS) + paper_ids):
        s = json.loads(r[3])
        by_combo_b[(r[1], r[0], r[2])].append(weighted(s))  # last key = critic_model

    # For static, also need to handle multiple ideas per paper (Track B has 3 ideas)
    # Use best-of-3 per paper (matches leaderboard) — read from results with idea_index 1/2/3
    by_combo_b2 = defaultdict(list)  # (model, paper, idea_idx) -> critic weighted scores
    for r in conn.execute(f"""
        SELECT paper_id, idea_model, idea_index, critic_model, scores_json
        FROM results WHERE track='B' AND critic_model != ''
          AND idea_model IN ({",".join("?"*len(TARGET_MODELS))})
          AND paper_id IN ({placeholders})
          AND scores_json IS NOT NULL
    """, list(TARGET_MODELS) + paper_ids):
        s = json.loads(r[4])
        by_combo_b2[(r[1], r[0], r[2])].append(weighted(s))

    # Per (model, paper): best-of-3 idea
    static_per_model_paper = defaultdict(dict)
    for (m, p, idx), vals in by_combo_b2.items():
        tm = trim_mean(vals)
        if tm > static_per_model_paper[m].get(p, 0):
            static_per_model_paper[m][p] = tm

    conn.close()

    print()
    print("=== Smoke result: Active vs Static (5 paper, 1 idea/paper) ===")
    print(f"{'Model':<45} {'Cutoff':<9} {'Static_5p':>10} {'Active_5p':>10} {'Boost':>8}")
    for m in TARGET_MODELS:
        a_vals = active_per_model.get(m, [])
        s_vals = list(static_per_model_paper.get(m, {}).values())
        a_mean = statistics.mean(a_vals) if a_vals else None
        s_mean = statistics.mean(s_vals) if s_vals else None
        if a_mean is None or s_mean is None:
            print(f"  {m:<43} ?         {s_mean or '?':>10} {a_mean or '?':>10} {'?':>8}")
            continue
        boost = a_mean - s_mean
        print(f"  {m:<43} {'?':<9} {s_mean:>10.3f} {a_mean:>10.3f} {boost:>+8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-domain", type=int, default=1)
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-critic", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    papers = fetch_smoke_papers(args.n_per_domain)
    logger.info(f"Smoke: {len(papers)} papers (1 per domain)")
    for p in papers:
        logger.info(f"  [{p['domain']}] {p['paper_id'][:18]} | {p['title'][:60]}")

    if not args.skip_gen:
        stats = phase_gen(papers, workers=args.workers)
        logger.info(f"gen stats: {stats}")

    if not args.skip_critic:
        stats = phase_critic(papers, workers=args.workers)
        logger.info(f"critic stats: {stats}")

    report(papers)


if __name__ == "__main__":
    main()
