"""
Score Aggregation  (Phase 4-A)

Reads results.db, computes for each (paper, idea_model, track):
  - best_idea:            idea with highest mean absolute score across critics
  - mean_absolute_score:  mean of 5 dims × all assigned critics
  - semantic_overlap:     E1 score of the best idea
  - context_gain:         score_B − score_A  (written at model level)

Results written to results.db → model_scores table.

Usage:
    python analysis/compute_scores.py
    python analysis/compute_scores.py --domain CS
"""

import argparse
import json
import logging
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as WEIGHTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> Optional[float]:
    return round(sum(vals) / len(vals), 4) if vals else None


def _trimmed_mean(vals: List[float]) -> Optional[float]:
    """Drop the highest score before averaging to suppress boilerplate
    rewarded by lenient critics; falls back to mean for n<=2."""
    if not vals:
        return None
    if len(vals) <= 2:
        return round(sum(vals) / len(vals), 4)
    trimmed = sorted(vals)[:-1]
    return round(sum(trimmed) / len(trimmed), 4)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_scores(db_results: str, db_papers: str,
                   domain_filter: Optional[str] = None) -> dict:
    """Aggregate all scores and write to model_scores table."""

    conn_r = sqlite3.connect(db_results, timeout=30)
    conn_r.row_factory = sqlite3.Row
    conn_p = sqlite3.connect(db_papers, timeout=30)
    conn_p.row_factory = sqlite3.Row

    # Load all scored results
    q = "SELECT * FROM results WHERE critic_model != '' AND scores_json IS NOT NULL"
    all_rows = [dict(r) for r in conn_r.execute(q).fetchall()]

    if domain_filter:
        domain_pids = {r["paper_id"] for r in conn_p.execute(
            "SELECT paper_id FROM papers WHERE domain=?", (domain_filter,)
        ).fetchall()}
        all_rows = [r for r in all_rows if r["paper_id"] in domain_pids]

    if not all_rows:
        logger.warning("No scored rows found — run critic_manager first")
        return {}

    # Group rows: (paper_id, idea_model, track, idea_index) → [scored rows]
    groups: Dict = defaultdict(list)
    for r in all_rows:
        key = (r["paper_id"], r["idea_model"], r["track"], r["idea_index"])
        groups[key].append(r)

    # Aggregate per (paper, idea_model, track, idea_index)
    combo_scores: Dict = defaultdict(dict)

    for (pid, model, track, idx), rows in groups.items():
        raw_means = []

        for r in rows:
            try:
                s = json.loads(r["scores_json"]) if isinstance(r["scores_json"], str) \
                    else r["scores_json"]
                # Weighted mean: O×2 + F×1 + C×0.5 + I×1.5 + S×0.5
                w_sum = sum(s[d] * WEIGHTS[d] for d in DIMS if d in s)
                w_total = sum(WEIGHTS[d] for d in DIMS if d in s)
                if w_total > 0:
                    raw_means.append(w_sum / w_total)
            except Exception:
                pass

        combo_scores[(pid, model, track)][idx] = {
            "raw":       _trimmed_mean(raw_means),
            "idea_text": rows[0]["idea_text"],
        }

    # For each (paper, model, track): pick best idea by raw score
    cur_r = conn_r.cursor()
    ts = datetime.now(timezone.utc).isoformat()
    written = 0

    for (pid, model, track), idx_scores in combo_scores.items():
        if not idx_scores:
            continue

        best_idx = max(idx_scores.keys(), key=lambda i: idx_scores[i]["raw"] or 0)
        best = idx_scores[best_idx]

        cur_r.execute("""
            INSERT INTO model_scores
              (paper_id, idea_model, track, best_idea_text, best_idea_index,
               mean_absolute_score, semantic_overlap, context_gain, computed_at)
            VALUES (?,?,?,?,?,?,NULL,NULL,?)
            ON CONFLICT(paper_id, idea_model, track) DO UPDATE SET
              best_idea_text      = excluded.best_idea_text,
              best_idea_index     = excluded.best_idea_index,
              mean_absolute_score = excluded.mean_absolute_score,
              computed_at         = excluded.computed_at
        """, (
            pid, model, track,
            best["idea_text"],
            best_idx,
            best["raw"],
            ts,
        ))
        written += 1

    # Context Gain removed — Track A no longer exists
    conn_r.commit()
    conn_r.close()
    conn_p.close()

    logger.info(f"compute_scores: wrote {written} model_scores rows")
    return {"rows_written": written}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    import config as cfg

    parser = argparse.ArgumentParser(description="Aggregate scores into model_scores table")
    parser.add_argument("--results-db", default=str(cfg.RESULTS_DB))
    parser.add_argument("--papers-db",  default=str(cfg.PAPERS_DB))
    parser.add_argument("--domain",     default=None)
    args = parser.parse_args()

    stats = compute_scores(args.results_db, args.papers_db, args.domain)
    print(f"\nDone: {stats.get('rows_written', 0)} rows written to model_scores")
