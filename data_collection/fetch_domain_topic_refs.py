"""Pre-fetch SS top-10 search results per unique paper.query.

For the v2_topic_refs Track B prompt: each paper's static refs = SS search top-10
using the paper's seed query (e.g., "LLM agent planning tool use autonomy") as
the search keyword. Cached in results.db.domain_topic_refs to avoid re-running
SS calls when generating ideas across multiple models.

Usage:
    python data_collection/fetch_domain_topic_refs.py            # fetch all missing
    python data_collection/fetch_domain_topic_refs.py --force    # refetch all
"""
import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from generation.active_agent import _search_papers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Refetch all queries (default: only missing)")
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="Stop after N queries (smoke; 0=all)")
    args = ap.parse_args()

    conn_p = sqlite3.connect(str(cfg.PAPERS_DB))
    conn_p.row_factory = sqlite3.Row
    queries = [dict(r) for r in conn_p.execute("""
        SELECT DISTINCT query, domain FROM papers
         WHERE status='filtered' AND query IS NOT NULL AND query != ''
         ORDER BY domain, query
    """)]
    conn_p.close()

    conn_r = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    if not args.force:
        existing = {row[0] for row in conn_r.execute(
            "SELECT query FROM domain_topic_refs")}
        queries = [q for q in queries if q["query"] not in existing]

    if args.limit_queries:
        queries = queries[:args.limit_queries]

    logger.info(f"To fetch: {len(queries)} unique queries")
    if not queries:
        logger.info("Nothing to do.")
        return

    n_ok, n_err, n_empty = 0, 0, 0
    for i, q in enumerate(queries, 1):
        query_str = q["query"]
        domain = q["domain"]
        try:
            results = _search_papers(query=query_str, limit=10)
        except Exception as e:
            logger.warning(f"  [{i}/{len(queries)}] {query_str[:60]}: SS error {e}")
            n_err += 1
            continue

        # Filter out entries with no abstract (useless for ideation context)
        with_abs = [r for r in results if (r.get("abstract") or "").strip()]

        if not with_abs:
            logger.warning(f"  [{i}/{len(queries)}] {query_str[:60]}: 0 results with abstract (raw {len(results)})")
            n_empty += 1
            continue

        ts = datetime.now(timezone.utc).isoformat()
        conn_r.execute("""
            INSERT OR REPLACE INTO domain_topic_refs (query, domain, refs_json, n_refs, fetched_at)
            VALUES (?, ?, ?, ?, ?)
        """, (query_str, domain, json.dumps(with_abs, ensure_ascii=False), len(with_abs), ts))
        conn_r.commit()
        n_ok += 1
        logger.info(f"  [{i}/{len(queries)}] {query_str[:60]:<62}  {len(with_abs)} refs")

        time.sleep(0.3)  # rate limit politely

    conn_r.close()
    logger.info(f"Done. ok={n_ok}  err={n_err}  empty={n_empty}")


if __name__ == "__main__":
    main()
