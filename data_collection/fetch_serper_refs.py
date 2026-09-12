"""
Serper Scholar Reference Fallback (Phase 1-B2)

For papers with status='ss_enriched' and empty references_json (common for 2025 papers
that Semantic Scholar hasn't indexed yet), uses Serper Google Scholar to find semantically
related papers as proxy references.

Serper Scholar query = paper title → returns related papers from Google Scholar.
Results stored as references_json entries tagged with "_source": "serper_scholar".

NOTE: These are NOT the paper's actual bibliography. They are topically related papers
found via Google Scholar keyword search, used solely to enable the Phase 1-E embedding
ranking step for papers that would otherwise be rejected by the n_valid_refs filter.

The embed_references.py step still ranks these proxy refs by cosine similarity to the
main paper abstract, so only the most semantically relevant ones end up in Track B context.

Requires: SERPER_API_KEY in .env

Usage:
    python data_collection/fetch_serper_refs.py
    python data_collection/fetch_serper_refs.py --smoke
    python data_collection/fetch_serper_refs.py --domain CS
"""

import json
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SERPER_SCHOLAR_URL = "https://google.serper.dev/scholar"


# ---------------------------------------------------------------------------
# Serper Scholar helper
# ---------------------------------------------------------------------------

def _query_scholar(title: str, api_key: str, num: int = 20) -> list:
    """Query Serper Google Scholar for papers related to the given title.

    Response uses Serper's Scholar engine:
        POST https://google.serper.dev/scholar
        {"q": "...", "num": N}

    Each organic result contains:
        title, link, snippet (abstract snippet),
        publicationInfo.summary (e.g. "A Smith - Nature, 2024 - nature.com"),
        citedBy.total

    Returns a list of ref dicts compatible with the references_json schema:
        {
            "paperId":       None,
            "title":         str,
            "abstract":      str,   # Serper snippet (proxy)
            "year":          int | None,
            "intent":        "",
            "citationCount": int,
            "_source":       "serper_scholar",
        }
    """
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": title[:200], "num": num}

    for attempt in range(3):
        try:
            resp = requests.post(
                SERPER_SCHOLAR_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"  Serper request error (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(5)
            continue

        if resp.status_code == 200:
            data = resp.json()
            organic = data.get("organic") or []
            refs = []
            for item in organic:
                t = (item.get("title") or "").strip()
                if not t:
                    continue
                # Extract 4-digit year from summary, e.g. "Authors - Nature, 2024 - ..."
                summary = (item.get("publicationInfo") or {}).get("summary") or ""
                year_m = re.search(r'\b(20\d{2}|19\d{2})\b', summary)
                year = int(year_m.group(1)) if year_m else None

                cited_total = (item.get("citedBy") or {}).get("total", 0)

                refs.append({
                    "paperId":       None,
                    "title":         t,
                    "abstract":      (item.get("snippet") or "").strip(),
                    "year":          year,
                    "intent":        "",
                    "citationCount": cited_total or 0,
                    "_source":       "serper_scholar",
                })
            return refs

        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            logger.warning(f"  Serper 429 — backing off {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
            continue

        logger.error(f"  Serper HTTP {resp.status_code}: {resp.text[:200]}")
        return []

    return []


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def fetch_serper_refs(db_path: str, api_key: str,
                      smoke: bool = False,
                      domain_filter: Optional[str] = None,
                      num_results: int = 20,
                      delay: float = 1.0) -> dict:
    """Fill empty references_json with Serper Scholar proxy references.

    Targets papers with:
      status IN ('ss_enriched', 'raw')  AND  n_valid_refs = 0
      (i.e. papers whose SS references came back empty — typically 2025 papers)

    After this step, the filter_papers.py step (Phase 1-C) can pass these papers
    on the n_valid_refs criterion.  The embed_references.py step (Phase 1-E) then
    ranks the proxy refs by semantic similarity to each paper's own abstract.

    Args:
        db_path:       Path to papers.db.
        api_key:       Serper API key (SERPER_API_KEY).
        smoke:         If True, process first 5 papers only.
        domain_filter: Restrict to one domain (e.g. "CS").
        num_results:   Number of Scholar results to fetch per paper (default 20).
        delay:         Seconds to sleep between Serper requests (default 1.0).

    Returns:
        dict: total, processed, skipped, errors.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
        SELECT paper_id, title, domain
        FROM papers
        WHERE status IN ('ss_enriched', 'raw')
          AND (
            n_valid_refs = 0
            OR references_json IS NULL
            OR references_json = ''
            OR references_json = '[]'
          )
    """
    params: list = []
    if domain_filter:
        query += " AND domain = ?"
        params.append(domain_filter)
    query += " ORDER BY domain, paper_id"

    papers = [dict(r) for r in cur.execute(query, params).fetchall()]

    if smoke:
        papers = papers[:5]
        logger.info(f"[smoke] fetch_serper_refs: {len(papers)} papers")
    else:
        logger.info(f"fetch_serper_refs: {len(papers)} papers with empty references")

    stats = {"total": len(papers), "processed": 0, "skipped": 0, "errors": 0}

    for i, paper in enumerate(papers, 1):
        pid    = paper["paper_id"]
        title  = (paper["title"] or "").strip()
        domain = paper["domain"]
        logger.info(f"  [{i}/{len(papers)}] [{domain}] {title[:65]!r}")

        if not title:
            logger.warning(f"    SKIP: no title")
            stats["skipped"] += 1
            continue

        try:
            refs = _query_scholar(title, api_key, num=num_results)
        except Exception as e:
            logger.error(f"    ERROR: {e}")
            stats["errors"] += 1
            continue

        if not refs:
            logger.warning(f"    no Serper results — skipping")
            stats["skipped"] += 1
            continue

        n_with_snippet = sum(1 for r in refs if r.get("abstract"))
        cur.execute(
            "UPDATE papers SET references_json = ?, n_valid_refs = ? WHERE paper_id = ?",
            (json.dumps(refs), len(refs), pid)
        )
        conn.commit()

        stats["processed"] += 1
        logger.info(f"    ✓ {len(refs)} proxy refs ({n_with_snippet} with snippets)")

        time.sleep(delay)

    conn.close()
    logger.info(
        f"fetch_serper_refs done: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['errors']} errors"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg_mod
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fill empty references_json with Serper Scholar proxy refs"
    )
    parser.add_argument("--db",     default=str(cfg_mod.PAPERS_DB))
    parser.add_argument("--smoke",  action="store_true")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--num",    type=int, default=20,
                        help="Scholar results per paper (default 20)")
    parser.add_argument("--delay",  type=float, default=1.0,
                        help="Seconds between Serper requests (default 1.0)")
    args = parser.parse_args()

    if not cfg_mod.SERPER_API_KEY:
        print("ERROR: SERPER_API_KEY not set in .env")
        sys.exit(1)

    result = fetch_serper_refs(
        args.db,
        api_key=cfg_mod.SERPER_API_KEY,
        smoke=args.smoke,
        domain_filter=args.domain,
        num_results=args.num,
        delay=args.delay,
    )
    print(f"\nDone: {result['processed']} processed, "
          f"{result['skipped']} skipped, {result['errors']} errors")
