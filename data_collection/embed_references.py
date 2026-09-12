"""
Reference Selection (Phase 1-D)

For each filtered paper, selects up to 20 references ranked by how the ORIGINAL
paper uses them (not global fame). Signals from Semantic Scholar:
  - isInfluential: SS's "Highly Influential Citation" flag (strongest signal)
  - in_paper_citations: number of inline citations in the paper body
  - citationCount: global citation count (tiebreak only)

Only refs with non-empty abstracts are kept. Result stored as ranked_refs_json.

The name "embed_references" is historical — we no longer embed anything, since
embedding-similarity selection caused the top picks to overlap with the paper's
own contribution (ground-truth hypothesis), making GT look unoriginal.

Usage:
    python data_collection/embed_references.py
    python data_collection/embed_references.py --smoke
    python data_collection/embed_references.py --domain CS
"""

import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime import get_thread_sqlite_connection


# ---------------------------------------------------------------------------
# Reference selection strategy
# ---------------------------------------------------------------------------

def _dedupe_refs(refs: List[dict]) -> List[dict]:
    deduped = []
    seen = set()
    for ref in refs:
        paper_id = (ref.get("paperId") or "").strip()
        title = " ".join((ref.get("title") or "").strip().lower().split())
        key = f"id:{paper_id}" if paper_id else f"title:{title}"
        if not title and not paper_id:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _select_ranked_refs(
    refs: List[dict],
    max_refs: int = 20,
) -> List[dict]:
    """Select refs by how the ORIGINAL paper uses them, not their global fame.

    Strategy:
      1. Keep only refs with non-empty abstracts
      2. Sort by (isInfluential desc, in_paper_citations desc, global citationCount desc)
         - isInfluential: SS-marked "Highly Influential Citation" — strong signal
         - in_paper_citations: number of inline citations in the paper body
         - citationCount: tiebreak (global fame)
      3. Cap at max_refs

    This captures "refs that actually shaped this paper" rather than "refs that
    happen to be famous in general (ViT/CLIP/GPT-4)".
    """
    with_abs = [r for r in refs if (r.get("abstract") or "").strip()]
    with_abs.sort(
        key=lambda r: (
            1 if r.get("isInfluential") else 0,
            r.get("in_paper_citations") or 0,
            r.get("citationCount") or 0,
        ),
        reverse=True,
    )
    return with_abs[:max_refs]


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def _rank_one_paper(db_path: str, paper: dict, max_refs: int) -> dict:
    """Select top refs for one paper by in-paper citations + isInfluential."""
    pid      = paper["paper_id"]
    refs_raw = paper.get("references_json")

    if not refs_raw:
        return {"paper_id": pid, "status": "skipped", "reason": "no references_json"}

    try:
        refs = json.loads(refs_raw)
    except (json.JSONDecodeError, TypeError):
        return {"paper_id": pid, "status": "skipped", "reason": "invalid references_json"}

    candidate_refs = _dedupe_refs([r for r in refs if (r.get("title") or "").strip()])
    if not candidate_refs:
        return {"paper_id": pid, "status": "skipped", "reason": "no refs with titles"}

    # Select by in-paper citations + isInfluential, keeping only refs with abstracts
    selected = _select_ranked_refs(candidate_refs, max_refs=max_refs)

    if len(selected) < 3:
        return {"paper_id": pid, "status": "skipped",
                "reason": f"only {len(selected)} refs with abstract (need >=3)"}

    conn = get_thread_sqlite_connection(db_path, timeout=30, enable_wal=True)
    cur = conn.cursor()
    cur.execute(
        "UPDATE papers SET ranked_refs_json = ? WHERE paper_id = ?",
        (json.dumps(selected), pid)
    )
    conn.commit()

    return {
        "paper_id": pid,
        "status": "processed",
        "valid_refs": len(candidate_refs),
        "selected": len(selected),
        "refs_with_abstract": sum(1 for r in candidate_refs if (r.get("abstract") or "").strip()),
    }

def embed_references(db_path: str, cfg,
                     smoke: bool = False,
                     domain_filter: Optional[str] = None,
                     batch_size: int = 32,
                     max_workers: Optional[int] = None) -> dict:
    """Select top refs for each filtered paper.

    Ranks by (isInfluential, in_paper_citations, citationCount) and caps at
    cfg.DATASET["max_refs"]. No embedding — selection is purely from metadata.

    batch_size kept for signature compat (unused).

    Returns:
        dict with keys: processed, skipped, errors.
    """
    max_refs = int(cfg.DATASET.get("max_refs", 20))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    query = """
        SELECT paper_id, title, abstract, references_json
        FROM papers
        WHERE status = 'filtered'
          AND (ranked_refs_json IS NULL OR ranked_refs_json = '')
    """
    params: list = []
    if domain_filter:
        query += " AND domain = ?"
        params.append(domain_filter)
    query += " ORDER BY domain, paper_id"

    papers = [dict(r) for r in cur.execute(query, params).fetchall()]
    if smoke:
        papers = papers[:3]
        logger.info(f"[smoke] Processing {len(papers)} papers")
    else:
        logger.info(f"select_refs: {len(papers)} papers to process")

    stats = {"processed": 0, "skipped": 0, "errors": 0}
    max_workers = max(1, int(max_workers or cfg.PARALLEL.get("phase1_embed_max_workers", 4)))
    logger.info(f"select_refs workers: {max_workers}, max_refs={max_refs}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_rank_one_paper, db_path, paper, max_refs): (i, paper)
            for i, paper in enumerate(papers, 1)
        }

        for future in as_completed(futures):
            i, paper = futures[future]
            pid = paper["paper_id"]
            title = paper["title"] or ""
            logger.info(f"  [{i}/{len(papers)}] {pid[:16]} — {title[:60]!r}")
            try:
                result = future.result()
                if result["status"] == "processed":
                    stats["processed"] += 1
                    logger.info(
                        f"    ✓ {result['valid_refs']} refs "
                        f"({result['refs_with_abstract']} with abstract)"
                        f" → {result['selected']} selected (by citation count)"
                    )
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                    logger.warning(f"    SKIP: {result['reason']}")
                else:
                    stats["errors"] += 1
                    logger.error(f"    ERROR: {result['reason']}")
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"    ERROR embedding {pid[:16]}: {e}")

    conn.close()
    logger.info(
        f"embed_references done: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['errors']} errors"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(ROOT))
    import config as cfg

    parser = argparse.ArgumentParser(
        description="Rank references by embedding similarity to main paper"
    )
    parser.add_argument("--db",     default=str(cfg.PAPERS_DB), help="Path to papers.db")
    parser.add_argument("--smoke",  action="store_true",        help="Process first 3 papers only")
    parser.add_argument("--domain", default=None,               help="Restrict to one domain")
    parser.add_argument("--max-workers", type=int,
                        default=int(cfg.PARALLEL.get("phase1_embed_max_workers", 4)))
    args = parser.parse_args()

    result = embed_references(
        args.db,
        cfg,
        smoke=args.smoke,
        domain_filter=args.domain,
        max_workers=args.max_workers,
    )
    print(f"\nDone: {result['processed']} processed, "
          f"{result['skipped']} skipped, {result['errors']} errors")
