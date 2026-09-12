"""
Paper Quality Filter

Reads raw paper rows from papers.db (status='ss_enriched'), applies quality
filters, and marks surviving papers with status='filtered'.  Papers that fail
are marked status='rejected' with a reject_reason.

Filters applied (per config.json):
  1. Abstract word count in [min_abstract_words, max_abstract_words]
  2. Not a Review / Dataset / Editorial / Book (from SS publicationTypes)
  3. Valid reference count >= min_references
     (valid = any ref with a title; intent is NOT used for filtering because
      Semantic Scholar does not annotate intents on 2025 papers)
  4. Published date within config window: paper_date_start .. paper_date_end
"""

import json
import logging
import re
import sqlite3
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXCLUDED_TYPES = {"Review", "Dataset", "Editorial", "LettersAndComments", "News", "Book"}


def _word_count(text: str) -> int:
    return len(text.split())


def _is_excluded_type(pub_types_json: Optional[str]) -> bool:
    """Return True if any publication type is in the exclusion list."""
    if not pub_types_json:
        return False
    try:
        types = json.loads(pub_types_json) if isinstance(pub_types_json, str) else pub_types_json
        return bool(set(types) & EXCLUDED_TYPES)
    except (json.JSONDecodeError, TypeError):
        return False


def _count_valid_refs(references_json: Optional[str]) -> int:
    """Count references that have at least a title.

    Note: Semantic Scholar does not annotate citation intents for 2025 papers,
    so all refs have intent=''. We count any ref with a title as valid.
    """
    if not references_json:
        return 0
    try:
        refs = json.loads(references_json) if isinstance(references_json, str) else references_json
    except (json.JSONDecodeError, TypeError):
        return 0
    return sum(1 for r in refs if r.get("title"))


# ---------------------------------------------------------------------------
# Main filter function
# ---------------------------------------------------------------------------

def filter_papers(db_path: str, cfg, dry_run: bool = False) -> dict:
    """Apply all quality filters to papers in papers.db.

    Args:
        db_path:  Path to papers.db (must already have the papers table).
        cfg:      The config module (import config as cfg).
        dry_run:  If True, print stats but don't write to the DB.

    Returns:
        dict with keys: total, passed, rejected, reasons (counter dict).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Fetch all enriched papers (pipeline: ss_enriched → filtered)
    cur.execute("SELECT * FROM papers WHERE status = 'ss_enriched'")
    rows = [dict(r) for r in cur.fetchall()]   # convert to dicts once, reused below

    min_words = cfg.DATASET.get("min_abstract_words", 80)
    max_words = cfg.DATASET.get("max_abstract_words", 800)
    min_refs  = cfg.DATASET.get("min_references", 3)

    # Date window: fixed range from config (not dynamic anti-leakage computation)
    date_start_str = cfg.DATASET.get("paper_date_start", "2025-07-01")
    date_end_str   = cfg.DATASET.get("paper_date_end",   "2026-01-31")
    date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
    date_end   = datetime.strptime(date_end_str,   "%Y-%m-%d")
    logger.info(f"Date filter: {date_start_str} ≤ published_date ≤ {date_end_str}")

    stats = {"total": len(rows), "passed": 0, "rejected": 0, "reasons": {}}

    updates = []  # list of (status, reject_reason, paper_id)

    for row in rows:
        paper_id     = row["paper_id"]
        abstract     = row["abstract"] or ""
        pub_types    = row.get("pub_types_json")
        refs_json    = row.get("references_json")
        pub_date_str = row.get("published_date")

        reject_reason = None

        # --- Filter 1: abstract length ---
        wc = _word_count(abstract)
        if wc < min_words:
            reject_reason = f"abstract_too_short ({wc} words)"
        elif wc > max_words:
            reject_reason = f"abstract_too_long ({wc} words)"

        # --- Filter 2: publication type ---
        if reject_reason is None and _is_excluded_type(pub_types):
            reject_reason = "excluded_pub_type"

        # --- Filter 3: reference count (any ref with a title counts) ---
        if reject_reason is None:
            n_valid = _count_valid_refs(refs_json)
            if n_valid < min_refs:
                reject_reason = f"too_few_valid_refs ({n_valid})"

        # --- Filter 4: date range check ---
        if reject_reason is None and pub_date_str:
            try:
                pub_date = datetime.strptime(pub_date_str[:10], "%Y-%m-%d")
                if pub_date < date_start:
                    reject_reason = f"before_date_window ({pub_date_str[:10]} < {date_start_str})"
                elif pub_date > date_end:
                    reject_reason = f"after_date_window ({pub_date_str[:10]} > {date_end_str})"
            except ValueError:
                pass   # unparseable date — let it through

        if reject_reason:
            stats["rejected"] += 1
            stats["reasons"][reject_reason.split("(")[0].strip()] = \
                stats["reasons"].get(reject_reason.split("(")[0].strip(), 0) + 1
            updates.append(("rejected", reject_reason, paper_id))
            logger.debug(f"REJECT {paper_id}: {reject_reason}")
        else:
            stats["passed"] += 1
            updates.append(("filtered", None, paper_id))

    # --- Rule 5: per-domain cap (keep top papers_per_domain by valid ref count) ---
    per_domain = cfg.DATASET.get("papers_per_domain", 20)

    # Group passing paper_ids by domain
    passing_by_domain: dict = {}
    for status, reason, pid in updates:
        if status == "filtered":
            row_data = next((r for r in rows if r["paper_id"] == pid), None)
            if row_data:
                domain = row_data["domain"]
                nv = _count_valid_refs(row_data.get("references_json"))
                passing_by_domain.setdefault(domain, []).append((pid, nv))

    # For each domain, reject the overflow (lowest valid-ref-count papers)
    overflow_ids: set = set()
    for domain, pid_nv_list in passing_by_domain.items():
        if len(pid_nv_list) > per_domain:
            sorted_list = sorted(pid_nv_list, key=lambda x: -x[1])  # descending by nv
            overflow = sorted_list[per_domain:]
            for pid, _ in overflow:
                overflow_ids.add(pid)
                stats["rejected"] += 1
                stats["passed"]   -= 1
                stats["reasons"]["domain_cap_overflow"] = \
                    stats["reasons"].get("domain_cap_overflow", 0) + 1

    # Rebuild updates list with overflow items marked rejected
    final_updates = []
    for status, reason, pid in updates:
        if pid in overflow_ids:
            final_updates.append(("rejected", "domain_cap_overflow", pid))
        else:
            final_updates.append((status, reason, pid))
    updates = final_updates

    if not dry_run:
        cur.executemany(
            "UPDATE papers SET status=?, reject_reason=? WHERE paper_id=?",
            updates
        )
        # Also update n_valid_refs for passing papers
        for status, reason, pid in updates:
            if status == "filtered":
                row_data = next((r for r in rows if r["paper_id"] == pid), None)
                if row_data:
                    nv = _count_valid_refs(row_data.get("references_json"))
                    cur.execute("UPDATE papers SET n_valid_refs=? WHERE paper_id=?", (nv, pid))
        conn.commit()

    conn.close()

    logger.info(
        f"filter_papers: {stats['total']} enriched → {stats['passed']} passed, "
        f"{stats['rejected']} rejected"
    )
    if stats["reasons"]:
        for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    import config as cfg

    parser = argparse.ArgumentParser(description="Filter papers in papers.db")
    parser.add_argument("--db", default=str(cfg.PAPERS_DB), help="Path to papers.db")
    parser.add_argument("--dry-run", action="store_true", help="Report stats without writing")
    args = parser.parse_args()

    stats = filter_papers(args.db, cfg, dry_run=args.dry_run)
    print(f"\nResult: {stats['passed']} / {stats['total']} papers passed filters")
