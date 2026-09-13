"""
Critic Manager  (Phase 3-E)

For each (paper, idea_model, track):
  1. Randomly select NUM_CRITICS_PER_EVAL critics from CRITIC_MODELS
     (excluding the idea_model itself)
  2. Schedule one bounded-concurrency task per (paper, track, critic)
  3. Each task scores all 3 ideas via one judge call (5 dims)
  4. Write one results.db row per (idea_index, critic)

Skips combos already scored (idempotent).

Usage:
    python evaluation/critic_manager.py --model openai/gpt-4o
    python evaluation/critic_manager.py --model openai/gpt-4o --smoke
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Critic selection
# ---------------------------------------------------------------------------

def select_critics(idea_model: str, critic_pool: List[str],
                   n: int = 3, seed: Optional[int] = None) -> List[str]:
    """Randomly pick n critics, excluding the idea model."""
    available = [m for m in critic_pool if m != idea_model]
    if len(available) < n:
        logger.warning(f"Only {len(available)} critics available (need {n}), using all")
        return available
    rng = random.Random(seed)
    return rng.sample(available, n)


# ---------------------------------------------------------------------------
# Reference formatting (same refs that Track B generation sees)
# ---------------------------------------------------------------------------

def _format_refs_for_judge(paper: dict) -> str:
    """Format ranked references for judge context.

    Uses ranked_refs_json. Shuffles order deterministically (same seed as
    generation) so judge sees the same order the model saw — removes position
    bias from citation-count sorting.
    """
    refs_json = paper.get("ranked_refs_json")
    if not refs_json:
        return "(no background literature available)"
    try:
        refs = json.loads(refs_json)
    except (json.JSONDecodeError, TypeError):
        return "(no background literature available)"
    if not refs:
        return "(no background literature available)"

    pid = paper.get("paper_id") or ""
    seed = int(hashlib.sha1(pid.encode("utf-8")).hexdigest()[:8], 16) if pid else 42
    shuffled = list(refs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    parts = []
    for i, ref in enumerate(shuffled, 1):
        title = ref.get("title") or "Untitled"
        abstract = (ref.get("abstract") or "").strip()
        entry = f"[{i}] {title}"
        if abstract:
            entry += f"\n{abstract}"
        parts.append(entry)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Score one (paper, idea_model, track) combo
# ---------------------------------------------------------------------------

def score_combo(paper: dict, idea_model: str, track: str,
                ideas: List[dict],   # Phase 2 rows from results table for this combo
                critic_model: str,
                judge_mode: str = "static",
                db_papers: str = "") -> dict:
    """Run one combined judge pass for one critic.

    judge_mode forwarded to absolute_scorer.score_idea. db_papers needed
    for dynamic_cited mode (to resolve cite strings against local titles).

    Returns rows to be persisted by the caller:
      {
        "scored": int,
        "errors": int,
        "rows": [{"kind": "success"|"error", ...}, ...]
      }
    """
    from evaluation.absolute_scorer import score_idea

    refs_text = _format_refs_for_judge(paper)
    stats = {"scored": 0, "errors": 0, "rows": []}

    for idea_row in ideas:
        idea_text  = idea_row["idea_text"]
        idea_index = idea_row["idea_index"]

        try:
            # ── 5-dim absolute score ──
            scores_dict, raw_resp, telem = score_idea(
                idea_text,
                critic_model,
                domain=paper.get("domain", ""),
                references=refs_text,
                judge_mode=judge_mode,
                db_papers=db_papers,
            )

            stats["rows"].append({
                "kind": "success",
                "paper_id": paper["paper_id"],
                "idea_model": idea_model,
                "track": track,
                "idea_index": idea_index,
                "idea_text": idea_text,
                "critic_model": critic_model,
                "scores_json": (
                    json.dumps({d: v["score"] for d, v in scores_dict.items()})
                    if scores_dict else None
                ),
                "reasoning_json": (
                    json.dumps({d: v["reasoning"] for d, v in scores_dict.items()})
                    if scores_dict else None
                ),
                "raw_response": raw_resp[:4000] if raw_resp else None,
                "telemetry": json.dumps(telem) if telem else None,
                "semantic_overlap": None,
                "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            stats["scored"] += 1

        except Exception as e:
            logger.error(f"    error scoring idea {idea_index}: {e}")
            stats["rows"].append({
                "kind": "error",
                "paper_id": paper["paper_id"],
                "idea_model": idea_model,
                "track": track,
                "idea_index": idea_index,
                "idea_text": idea_text,
                "critic_model": critic_model,
                "scores_json": None,
                "reasoning_json": None,
                "raw_response": None,
                "telemetry": None,
                "semantic_overlap": None,
                "error": str(e)[:500],
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            stats["errors"] += 1
    return stats


def persist_rows(conn: sqlite3.Connection, rows: List[dict]) -> None:
    """Persist scored/error rows from completed worker tasks."""
    cur = conn.cursor()
    for row in rows:
        if row["kind"] == "success":
            cur.execute("""
                INSERT OR REPLACE INTO results
                  (paper_id, idea_model, track, idea_index, idea_text,
                   critic_model, scores_json, reasoning_json, raw_response,
                   telemetry, semantic_overlap, error, created_at)
                VALUES (?,?,?,?,?, ?,?,?,?,?,?,?,?)
            """, (
                row["paper_id"],
                row["idea_model"],
                row["track"],
                row["idea_index"],
                row["idea_text"],
                row["critic_model"],
                row["scores_json"],
                row["reasoning_json"],
                row["raw_response"],
                row.get("telemetry"),
                row["semantic_overlap"],
                row["error"],
                row["created_at"],
            ))
        else:
            cur.execute("""
                INSERT OR IGNORE INTO results
                  (paper_id, idea_model, track, idea_index, idea_text,
                   critic_model, error, created_at)
                VALUES (?,?,?,?,?, ?,?,?)
            """, (
                row["paper_id"],
                row["idea_model"],
                row["track"],
                row["idea_index"],
                row["idea_text"],
                row["critic_model"],
                row["error"],
                row["created_at"],
            ))
    conn.commit()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(idea_model: str, db_papers: str, db_results: str,
                   critic_pool: List[str],
                   n_critics: int = 3,
                   max_workers: Optional[int] = None,
                   smoke: bool = False,
                   domain_filter: Optional[str] = None,
                   judge_mode: str = "static") -> dict:
    """Evaluate all generated ideas for one idea_model.

    For each (paper, track), picks n_critics randomly and scores all 3 ideas.

    judge_mode: "static" (default, uses survey refs from papers.db),
                "dynamic_search" (critic searches SS at score time), or
                "dynamic_cited" (critic uses cited refs parsed from idea_text).
    """
    # Step 1: get paper_ids that have generated ideas (from results.db)
    conn_r0 = sqlite3.connect(db_results, timeout=30)
    pid_rows = conn_r0.execute(
        "SELECT DISTINCT paper_id FROM results WHERE idea_model=?", (idea_model,)
    ).fetchall()
    conn_r0.close()
    generated_pids = {r[0] for r in pid_rows}

    # Step 2: fetch paper metadata from papers.db
    conn_p = sqlite3.connect(db_papers, timeout=30)
    conn_p.row_factory = sqlite3.Row

    q = "SELECT * FROM papers WHERE status='filtered'"
    params: list = []
    if domain_filter:
        q += " AND domain=?"
        params.append(domain_filter)

    all_papers = [dict(row) for row in conn_p.execute(q, params).fetchall()]
    conn_p.close()

    papers = [p for p in all_papers if p["paper_id"] in generated_pids]
    if smoke:
        # 5 papers per domain for balanced coverage
        from collections import defaultdict
        by_domain = defaultdict(list)
        for p in papers:
            by_domain[p["domain"]].append(p)
        papers = [p for ps in by_domain.values() for p in ps[:5]]
        papers.sort(key=lambda p: (p["domain"], p["paper_id"]))
        logger.info(f"[smoke] evaluating {len(papers)} papers ({len(by_domain)} domains) × {idea_model!r}")
    else:
        logger.info(f"run_evaluation: {len(papers)} papers × {idea_model!r}")

    total_stats = {"papers": len(papers), "tasks": 0, "scored": 0, "errors": 0}

    conn_r = sqlite3.connect(db_results, timeout=30)
    conn_r.row_factory = sqlite3.Row
    conn_r.execute("PRAGMA busy_timeout=30000")

    tasks = []

    for i, paper in enumerate(papers, 1):
        pid = paper["paper_id"]
        logger.info(f"  [{i}/{len(papers)}] {paper.get('domain','?')} | "
                    f"{paper['title'][:55]!r}")

        # Pick critics once per paper (same critics for both tracks).
        # Use MD5 for a stable, process-independent seed (avoids PYTHONHASHSEED randomisation).
        seed_int = int(hashlib.md5((pid + idea_model).encode()).hexdigest(), 16) % (2**32)
        critics = select_critics(idea_model, critic_pool, n=n_critics, seed=seed_int)
        logger.info(f"    critics: {[c.split('/')[-1] for c in critics]}")

        for track in ("B", "C"):
            ideas = [dict(r) for r in conn_r.execute(
                "SELECT * FROM results WHERE paper_id=? AND idea_model=? "
                "AND track=? AND critic_model=''",
                (pid, idea_model, track)
            ).fetchall()]

            if not ideas:
                logger.debug(f"    no unscored ideas for track {track}, skipping")
                continue

            for critic in critics:
                pending_ideas = []
                for idea_row in ideas:
                    exists = conn_r.execute(
                        "SELECT 1 FROM results WHERE paper_id=? AND idea_model=? "
                        "AND track=? AND idea_index=? AND critic_model=? "
                        "AND scores_json IS NOT NULL",
                        (pid, idea_model, track, idea_row["idea_index"], critic)
                    ).fetchone()
                    if not exists:
                        pending_ideas.append(idea_row)
                if not pending_ideas:
                    continue
                tasks.append({
                    "paper": paper,
                    "track": track,
                    "ideas": pending_ideas,
                    "critic": critic,
                })

    conn_r.close()

    if not tasks:
        logger.info("run_evaluation: nothing to score")
        return total_stats

    if max_workers is None:
        import config as cfg
        max_workers = int(cfg.PARALLEL.get("phase3_max_workers", 12))
    max_workers = max(1, min(max_workers, len(tasks)))
    total_stats["tasks"] = len(tasks)

    logger.info(
        f"run_evaluation: scheduling {len(tasks)} task(s) "
        f"with max_workers={max_workers}"
    )

    completed = 0
    conn_w = sqlite3.connect(db_results, timeout=30)
    conn_w.execute("PRAGMA journal_mode=WAL")
    conn_w.execute("PRAGMA synchronous=NORMAL")
    conn_w.execute("PRAGMA busy_timeout=30000")
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="phase3") as executor:
        future_to_task = {
            executor.submit(
                score_combo,
                task["paper"],
                idea_model,
                task["track"],
                task["ideas"],
                task["critic"],
                judge_mode,
                db_papers,
            ): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            try:
                st = future.result()
                persist_rows(conn_w, st.get("rows", []))
                total_stats["scored"] += st["scored"]
                total_stats["errors"] += st["errors"]
            except Exception as e:
                logger.error(
                    f"task failed: {task['paper']['paper_id']} "
                    f"{task['track']} {task['critic']}: {e}"
                )
                total_stats["errors"] += len(task["ideas"])

            if completed == len(tasks) or completed % 10 == 0:
                logger.info(
                    f"  progress: {completed}/{len(tasks)} tasks "
                    f"({total_stats['scored']} scored, {total_stats['errors']} errors)"
                )
    conn_w.close()

    logger.info(
        f"run_evaluation done: {total_stats['scored']} scored, "
        f"{total_stats['errors']} errors"
    )
    return total_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    import config as cfg

    parser = argparse.ArgumentParser(description="Score generated ideas with critic models")
    parser.add_argument("--model",  required=True,            help="Idea model to evaluate")
    parser.add_argument("--smoke",  action="store_true",      help="2 papers only")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--papers-db",  default=str(cfg.PAPERS_DB))
    parser.add_argument("--results-db", default=str(cfg.RESULTS_DB))
    args = parser.parse_args()

    stats = run_evaluation(
        idea_model=args.model,
        db_papers=args.papers_db,
        db_results=args.results_db,
        critic_pool=cfg.CRITIC_MODELS,
        n_critics=cfg.NUM_CRITICS_PER_EVAL,
        max_workers=(args.max_workers if args.max_workers is not None
                     else cfg.PARALLEL.get("phase3_max_workers", 12)),
        smoke=args.smoke,
        domain_filter=args.domain,
    )
    print(f"\nDone: {stats['scored']} idea-critic pairs scored, "
          f"{stats['errors']} errors")
