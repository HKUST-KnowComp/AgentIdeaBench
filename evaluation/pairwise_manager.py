"""
Pairwise Manager

For a given track (e.g. Active Mode outputs), runs pairwise comparisons
between all model pairs, for each paper, with swap-order de-biasing.

Writes to pairwise_results table in results.db.

Usage:
    from evaluation.pairwise_manager import run_pairwise
    run_pairwise(db_results, db_papers, critic_pool, track="C", smoke=True)
"""

import hashlib
import json
import logging
import random
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# DB schema ensure
# ---------------------------------------------------------------------------

PAIRWISE_SCHEMA = """
CREATE TABLE IF NOT EXISTS pairwise_results (
    paper_id      TEXT NOT NULL,
    track         TEXT NOT NULL,
    model_a       TEXT NOT NULL,
    model_b       TEXT NOT NULL,
    critic_model  TEXT NOT NULL,
    overall_winner      TEXT,
    originality_winner  TEXT,
    feasibility_winner  TEXT,
    specificity_winner  TEXT,
    raw_forward   TEXT,
    raw_swap      TEXT,
    created_at    TEXT,
    PRIMARY KEY (paper_id, track, model_a, model_b, critic_model)
);
CREATE INDEX IF NOT EXISTS idx_pw_model_a ON pairwise_results(model_a);
CREATE INDEX IF NOT EXISTS idx_pw_track   ON pairwise_results(track);
"""


def _ensure_schema(db_results: str) -> None:
    conn = sqlite3.connect(db_results, timeout=30)
    conn.executescript(PAIRWISE_SCHEMA)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Refs formatting (identical to critic_manager for consistency)
# ---------------------------------------------------------------------------

def _format_refs_for_judge(paper: dict) -> str:
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
# Main runner
# ---------------------------------------------------------------------------

def _select_critic(model_a: str, model_b: str, critic_pool: List[str]) -> str:
    """Deterministic critic pick per pair, excluding both models if they're in the pool."""
    available = [c for c in critic_pool if c not in (model_a, model_b)]
    if not available:
        available = critic_pool
    seed_str = f"{model_a}:{model_b}"
    seed = int(hashlib.sha1(seed_str.encode()).hexdigest(), 16)
    return available[seed % len(available)]


def run_pairwise(
    db_results: str,
    db_papers: str,
    critic_pool: List[str],
    track: str = "C",
    model_filter: Optional[List[str]] = None,
    smoke: bool = False,
    max_workers: int = 6,
    n_critics: int = 1,
) -> dict:
    """Run pairwise comparisons between models for given track.

    For each (paper, model_a, model_b) where model_a < model_b lexicographically,
    runs n_critics pairwise comparisons with swap-order de-biasing.

    Returns:
        {"papers": int, "pairs": int, "completed": int, "errors": int}
    """
    from evaluation.pairwise_scorer import resolve_pair

    _ensure_schema(db_results)

    # Load papers
    conn_p = sqlite3.connect(db_papers, timeout=30)
    conn_p.row_factory = sqlite3.Row
    papers = [dict(r) for r in conn_p.execute(
        "SELECT * FROM papers WHERE status='filtered' "
        "AND ranked_refs_json IS NOT NULL ORDER BY domain, paper_id"
    ).fetchall()]
    conn_p.close()

    if smoke:
        from collections import defaultdict
        by_domain = defaultdict(list)
        for p in papers:
            by_domain[p["domain"]].append(p)
        papers = [p for ps in by_domain.values() for p in ps[:5]]

    # Load ideas per (paper, model, track)
    conn_r = sqlite3.connect(db_results, timeout=30)
    conn_r.execute("PRAGMA busy_timeout = 30000")

    idea_rows = conn_r.execute("""
        SELECT paper_id, idea_model, idea_text
        FROM results
        WHERE track = ? AND idea_text IS NOT NULL AND length(idea_text) > 30
          AND critic_model = ''
          AND idea_model NOT LIKE 'baseline/%'
        ORDER BY paper_id, idea_model
    """, (track,)).fetchall()

    # Build (paper, model) -> idea_text map, dedup per (paper, model)
    idea_map = {}
    models_set = set()
    for pid, model, text in idea_rows:
        if model_filter and model not in model_filter:
            continue
        key = (pid, model)
        if key not in idea_map:
            idea_map[key] = text
            models_set.add(model)

    models = sorted(models_set)
    paper_ids = sorted({pid for pid, _ in idea_map.keys()})
    # Only keep papers that have ideas for all models
    eligible_papers = [p for p in papers if p["paper_id"] in paper_ids
                       and all((p["paper_id"], m) in idea_map for m in models)]

    pairs = list(combinations(models, 2))
    logger.info(f"Pairwise: {len(eligible_papers)} papers, {len(pairs)} model pairs, "
                f"track={track}, {len(models)} models")

    tasks = []
    for p in eligible_papers:
        for ma, mb in pairs:
            for c_idx in range(n_critics):
                critic = _select_critic(ma, mb, critic_pool) if n_critics == 1 \
                         else critic_pool[c_idx % len(critic_pool)]
                # Check already exists
                exists = conn_r.execute(
                    "SELECT 1 FROM pairwise_results WHERE paper_id=? AND track=? "
                    "AND model_a=? AND model_b=? AND critic_model=?",
                    (p["paper_id"], track, ma, mb, critic)
                ).fetchone()
                if not exists:
                    tasks.append((p, ma, mb, critic))

    logger.info(f"  {len(tasks)} comparisons to run")

    stats = {"papers": len(eligible_papers), "pairs": len(pairs),
             "completed": 0, "errors": 0}
    ts = datetime.now(timezone.utc).isoformat()

    def _worker(task):
        paper, ma, mb, critic = task
        refs = _format_refs_for_judge(paper)
        idea_a = idea_map[(paper["paper_id"], ma)]
        idea_b = idea_map[(paper["paper_id"], mb)]
        try:
            res = resolve_pair(idea_a, idea_b, critic,
                               domain=paper.get("domain", ""),
                               references=refs)
            return ("ok", paper, ma, mb, critic, res)
        except Exception as e:
            logger.error(f"  task error: {e}")
            return ("err", paper, ma, mb, critic, str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            status, paper, ma, mb, critic, payload = fut.result()
            if status == "err":
                stats["errors"] += 1
                continue
            tel_json = (json.dumps(payload["telemetry"])
                        if payload.get("telemetry") else None)
            conn_r.execute("""
                INSERT OR REPLACE INTO pairwise_results
                (paper_id, track, model_a, model_b, critic_model,
                 overall_winner, originality_winner, feasibility_winner, specificity_winner,
                 raw_forward, raw_swap, telemetry, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                paper["paper_id"], track, ma, mb, critic,
                payload.get("overall"), payload.get("originality"),
                payload.get("feasibility"), payload.get("specificity"),
                payload.get("raw_forward", "")[:2000],
                payload.get("raw_swap", "")[:2000],
                tel_json,
                ts,
            ))
            conn_r.commit()
            stats["completed"] += 1
            if i % 10 == 0 or i == len(tasks):
                logger.info(f"  progress: {i}/{len(tasks)} "
                            f"({stats['completed']} ok, {stats['errors']} err)")

    conn_r.close()
    return stats
