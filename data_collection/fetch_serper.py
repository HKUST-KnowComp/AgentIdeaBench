"""
Serper Scholar Search  (Phase 1-A)

Replaces fetch_ss_search.py + fetch_serper_refs.py.
For every (domain, query) in config.json:
  1. POST /scholar  →  top-N results (title, snippet, year, citedBy)
  2. Stratified sampling: 1 from rank 1-5, 1 from rank 6-10  (~2 per query)
  3. For each selected paper:
     a. POST /scholar with paper title  →  20 related papers as proxy refs
     b. INSERT into papers.db as status='serper_enriched'

All data comes from Serper Google Scholar — no Semantic Scholar calls.

Rate: ~1 req/s (Serper allows up to 100 req/s on paid plans).
Delay default: 1.1s between requests.

Usage:
    python data_collection/fetch_serper.py
    python data_collection/fetch_serper.py --smoke
    python data_collection/fetch_serper.py --domain CS
"""

import hashlib
import json
import logging
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SERPER_SCHOLAR_URL = "https://google.serper.dev/scholar"


# ---------------------------------------------------------------------------
# Serper API helpers
# ---------------------------------------------------------------------------

def _scholar_search(query: str, api_key: str, num: int = 10,
                    date_range: str = None, delay: float = 1.1) -> list:
    """POST to Serper Scholar, return organic results list."""
    payload = {"q": query, "num": num}
    if date_range:
        payload["tbs"] = date_range  # e.g. "cdr:1,cd_min:2025-04-01,cd_max:2026-01-31"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    backoffs = [30, 60, 120]
    for attempt, backoff in enumerate(backoffs):
        try:
            resp = requests.post(SERPER_SCHOLAR_URL, json=payload,
                                 headers=headers, timeout=20)
            if resp.status_code == 429:
                logger.warning(f"429 rate limit, backing off {backoff}s")
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            time.sleep(delay)
            return resp.json().get("organic", [])
        except requests.RequestException as e:
            logger.warning(f"Request error (attempt {attempt+1}): {e}")
            time.sleep(backoff)
    return []


def _parse_year(pub_info: str) -> Optional[int]:
    """Extract year from Serper publicationInfo.summary string."""
    if not pub_info:
        return None
    m = re.search(r"\b(20\d{2})\b", pub_info)
    return int(m.group(1)) if m else None


def _parse_date(pub_info: str) -> Optional[str]:
    """Return YYYY-MM-DD or YYYY-01-01 from publicationInfo string."""
    year = _parse_year(pub_info)
    return f"{year}-01-01" if year else None


def _make_paper_id(title: str) -> str:
    """Stable hash-based ID for papers without a real SS paperId."""
    return "serper_" + hashlib.sha1(title.lower().strip().encode()).hexdigest()[:16]


def _get_pub_info(result: dict) -> str:
    """Safely extract publicationInfo.summary — handles both dict and string."""
    pi = result.get("publicationInfo")
    if isinstance(pi, dict):
        return pi.get("summary", "")
    if isinstance(pi, str):
        return pi
    return ""


def _result_to_paper(result: dict, domain: str, query: str) -> dict:
    """Convert one Serper Scholar organic result to a paper dict."""
    title   = result.get("title", "").strip()
    snippet = result.get("snippet", "").strip()
    pub_info = _get_pub_info(result)
    year    = _parse_year(pub_info)
    pub_date = _parse_date(pub_info)
    cited_by_raw = result.get("citedBy")
    cited_by = (cited_by_raw.get("total", 0) if isinstance(cited_by_raw, dict) else 0) or 0

    return {
        "paper_id":      _make_paper_id(title),
        "title":         title,
        "abstract":      snippet,
        "domain":        domain,
        "query":         query,
        "published_date": pub_date,
        "venue":         "",
        "citation_count": cited_by,
        "pub_types_json": json.dumps(["JournalArticle"]),
        "year":          year,
    }


def _fetch_proxy_refs(query: str, self_title: str, api_key: str,
                      num: int = 20, delay: float = 1.1) -> list:
    """Search Serper Scholar by query to get proxy references.

    Uses a shorter query (e.g. first 7 words of title) to get broader results.
    Filters out the paper itself by comparing against self_title.
    """
    results = _scholar_search(query, api_key, num=num, delay=delay)
    logger.info(f"    proxy refs search for '{query[:50]}': {len(results)} raw results")
    refs = []
    for r in results:
        t = r.get("title", "").strip()
        if not t:
            continue
        if t.lower() == self_title.lower():
            continue  # skip self
        pub_info = _get_pub_info(r)
        cited_by_raw = r.get("citedBy")
        cited_by = (cited_by_raw.get("total", 0) if isinstance(cited_by_raw, dict) else 0) or 0
        refs.append({
            "paperId":        _make_paper_id(t),
            "title":          t,
            "abstract":       r.get("snippet", "").strip(),
            "year":           _parse_year(pub_info),
            "citationCount":  cited_by,
            "intents":        [],
            "_source":        "serper_scholar",
        })
    logger.info(f"    proxy refs for '{query[:50]}': {len(refs)} collected from {len(results)} results")
    return refs


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _stratified_sample(results: list, tier1: list, tier2: list) -> list:
    """Pick 1 from tier1 rank range, 1 from tier2 rank range (1-indexed)."""
    lo1, hi1 = tier1
    lo2, hi2 = tier2
    pool1 = results[lo1-1:hi1]
    pool2 = results[lo2-1:hi2]
    selected = []
    if pool1:
        selected.append(random.choice(pool1))
    if pool2:
        selected.append(random.choice(pool2))
    return selected


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _already_exists(conn: sqlite3.Connection, paper_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM papers WHERE paper_id=?", (paper_id,)
    ).fetchone() is not None


def _insert_paper(conn: sqlite3.Connection, paper: dict,
                  refs: list, date_start: str, date_end: str) -> bool:
    """Insert paper row. Returns True if inserted, False if skipped."""
    if _already_exists(conn, paper["paper_id"]):
        return False

    refs_json = json.dumps(refs)
    n_valid   = sum(1 for r in refs if r.get("title"))
    ts = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT OR IGNORE INTO papers
          (paper_id, title, abstract, abstract_words, domain, query,
           published_date, venue, citation_count, pub_types_json,
           references_json, n_valid_refs, status, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'serper_enriched',?)
    """, (
        paper["paper_id"],
        paper["title"],
        paper["abstract"],
        len(paper["abstract"].split()),
        paper["domain"],
        paper["query"],
        paper.get("published_date"),
        paper.get("venue", ""),
        paper.get("citation_count", 0),
        paper.get("pub_types_json"),
        refs_json,
        n_valid,
        ts,
    ))
    conn.commit()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_all(cfg, smoke: bool = False,
              domain_filter: str = None, delay: float = 1.1) -> dict:
    """Search Serper Scholar for all (domain, query) pairs in config.

    For each query:
      1. Fetch top-10 results from Serper Scholar (date-filtered)
      2. Stratified sample: 1 from rank 1-5, 1 from rank 6-10
      3. For each selected paper, fetch 20 proxy refs via title search
      4. Insert into papers.db

    Returns stats dict: inserted, skipped, refs_filled, errors
    """
    from utils.db_init import init_all
    init_all(str(cfg.PAPERS_DB), str(cfg.RESULTS_DB))

    api_key = cfg.SERPER_API_KEY
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not set in .env")

    dataset   = cfg.DATASET
    topics    = dataset.get("serper_topics", {})
    date_start = dataset.get("paper_date_start", "2025-04-01")
    date_end   = dataset.get("paper_date_end",   "2026-01-31")
    tier1      = dataset.get("ss_sample_tier1", [1, 5])
    tier2      = dataset.get("ss_sample_tier2", [6, 10])

    # Extract year boundaries for post-fetch date filtering
    year_start = int(date_start[:4])
    year_end   = int(date_end[:4])

    conn = sqlite3.connect(str(cfg.PAPERS_DB), timeout=30)

    stats = {"inserted": 0, "skipped": 0, "refs_filled": 0, "errors": 0}

    domains = list(topics.keys())
    if domain_filter:
        domains = [d for d in domains if d == domain_filter]

    if smoke:
        domains = domains[:1]

    for domain in domains:
        queries = topics.get(domain, [])
        if smoke:
            queries = queries[:2]

        for query in queries:
            logger.info(f"  [{domain}] {query[:60]}")

            # Step 1: search — append year to bias results toward recent papers
            # Serper Scholar tbs date filter is unreliable; year in query is more effective
            year_query = f"{query} {year_start}"
            results = _scholar_search(
                year_query, api_key, num=20, delay=delay
            )
            if not results:
                logger.warning(f"    no results for query: {query}")
                stats["errors"] += 1
                continue

            # Post-fetch date filter: keep only papers within the year window
            results_in_window = []
            for r in results:
                pub_info = _get_pub_info(r)
                yr = _parse_year(pub_info)
                if yr and year_start <= yr <= year_end:
                    results_in_window.append(r)
                elif yr:
                    logger.debug(f"    date-filter: {r.get('title','')[:40]} (year={yr})")

            logger.info(f"    {len(results)} results → {len(results_in_window)} in date window [{year_start}-{year_end}]")

            if not results_in_window:
                logger.warning(f"    no results in date window for: {query}")
                stats["errors"] += 1
                continue

            # Step 2: stratified sample from date-filtered results
            selected = _stratified_sample(results_in_window, tier1, tier2)
            if not selected:
                # If fewer than tier1 range results, just take all
                selected = results_in_window[:2]
            logger.info(f"    → {len(selected)} selected for insertion")

            for result in selected:
                try:
                    paper = _result_to_paper(result, domain, query)
                    if not paper["title"]:
                        continue

                    # Step 3: fetch proxy refs — use truncated title to avoid
                    # exact self-match returning only the paper itself
                    title_words = paper["title"].split()
                    proxy_query = " ".join(title_words[:7])
                    refs = _fetch_proxy_refs(
                        proxy_query, paper["title"], api_key, num=20, delay=delay
                    )

                    # Step 4: insert
                    inserted = _insert_paper(
                        conn, paper, refs, date_start, date_end
                    )
                    if inserted:
                        stats["inserted"] += 1
                        if refs:
                            stats["refs_filled"] += 1
                        logger.info(f"    + {paper['title'][:55]} ({len(refs)} refs)")
                    else:
                        stats["skipped"] += 1
                        logger.info(f"    ~ skip (dup): {paper['title'][:55]}")

                except Exception as e:
                    logger.error(f"    ERROR processing result: {e}")
                    stats["errors"] += 1

    conn.close()
    logger.info(
        f"fetch_serper done: {stats['inserted']} inserted, "
        f"{stats['skipped']} skipped, {stats['refs_filled']} got refs, "
        f"{stats['errors']} errors"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(ROOT))
    import config as cfg_mod

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Search Serper Scholar and populate papers.db"
    )
    parser.add_argument("--smoke",  action="store_true")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--delay",  type=float, default=1.1)
    args = parser.parse_args()

    if not cfg_mod.SERPER_API_KEY:
        print("ERROR: SERPER_API_KEY not set in .env")
        sys.exit(1)

    result = fetch_all(cfg_mod, smoke=args.smoke,
                       domain_filter=args.domain, delay=args.delay)
    print(f"\nDone: {result['inserted']} inserted, "
          f"{result['refs_filled']} got proxy refs, "
          f"{result['errors']} errors")
