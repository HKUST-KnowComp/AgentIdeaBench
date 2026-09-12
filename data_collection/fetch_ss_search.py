"""
Semantic Scholar Direct Search (Phase 1-A)

Replaces Serper with SS API keyword search + date range filter.
SS's ?year=YYYY-YYYY filter is reliable; exact date filtering is done downstream.

Per (domain, query):
  1. GET /paper/search?query=...&year=2025-2026&fields=...  (1 API call, fetches top-20)
  2. Stratified random sampling: 1 paper from top-5, 1 from rank 6-10 = ~2 per query
  3. GET /paper/{id}/references for each selected paper      (1 API call each)
  4. INSERT directly into papers.db as status='ss_enriched'

Sampling rationale: pure top-k yields near-duplicate papers within the same query.
Stratified sampling increases diversity while retaining relevance.

Rate: 1 req/s with key (delay=1.1s).

Usage:
    python data_collection/fetch_ss_search.py
    python data_collection/fetch_ss_search.py --smoke
    python data_collection/fetch_ss_search.py --domain Biology
"""

import json
import logging
import random
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SS_BASE = "https://api.semanticscholar.org/graph/v1"

_SEARCH_FIELDS = (
    "paperId,title,abstract,year,publicationDate,"
    "venue,citationCount,publicationTypes"
)
_REFS_FIELDS = (
    "citedPaper.paperId,citedPaper.title,citedPaper.abstract,"
    "citedPaper.year,citedPaper.citationCount,"
    "citedPaper.publicationTypes,intents,contexts,isInfluential"
)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime import get_thread_http_session


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _headers(api_key: str) -> dict:
    h = {"User-Agent": "SciSynthBench/1.0 (research)"}
    if api_key:
        h["x-api-key"] = api_key
    return h


def _get(url: str, params: dict, api_key: str, delay: float,
         backoffs: Optional[list] = None) -> Optional[dict]:
    """GET with 429 backoff.

    backoffs: retry wait schedule on 429. Default [60,120,300] suits Phase-1
    bulk collection. The active agent passes a SHORT schedule (e.g. [3,6,12])
    because SS recovers within ~1-2s and a 60s wait per 429 destroys agent
    throughput (each idea makes many SEARCH/FETCH calls).
    """
    if backoffs is None:
        backoffs = [60, 120, 300]
    session = get_thread_http_session("semantic_scholar")
    for attempt in range(len(backoffs) + 1):
        try:
            resp = session.get(
                url,
                params=params,
                headers=_headers(api_key),
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"  request error: {e}")
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt])
            continue

        time.sleep(delay)

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            if attempt < len(backoffs):
                wait = backoffs[attempt]
                logger.warning(f"  429 — backing off {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            else:
                logger.error("  429 persists after all retries")
                return None
        logger.error(f"  HTTP {resp.status_code}: {url}")
        return None
    return None


# ---------------------------------------------------------------------------
# References fetch
# ---------------------------------------------------------------------------

def _fetch_refs(paper_id: str, api_key: str, delay: float,
                backoffs: Optional[list] = None) -> list:
    """Fetch /references edges for a paper."""
    data = _get(
        f"{SS_BASE}/paper/{paper_id}/references",
        {"fields": _REFS_FIELDS, "limit": 500},
        api_key, delay, backoffs=backoffs,
    )
    if not data:
        return []
    out = []
    for edge in (data.get("data") or []):
        if edge is None:
            continue
        cited    = edge.get("citedPaper")   or {}
        intents  = edge.get("intents")      or []
        contexts = edge.get("contexts")     or []
        out.append({
            "paperId":       cited.get("paperId"),
            "title":         cited.get("title"),
            "abstract":      cited.get("abstract"),
            "year":          cited.get("year"),
            "publicationTypes": cited.get("publicationTypes") or [],
            "intents":       intents,
            "intent":        intents[0] if intents else "",
            "citationCount": cited.get("citationCount", 0),
            # Per-paper citation signals (how THIS paper cites this ref)
            "in_paper_citations": len(contexts),   # number of inline citations
            "contexts":      contexts,              # citation sentences in this paper
            "isInfluential": bool(edge.get("isInfluential")),
        })
    return out


def _pub_date_ok(paper: dict, date_start: datetime, date_end: datetime) -> bool:
    """Return True if paper publication date is within [date_start, date_end]."""
    raw = paper.get("publicationDate") or ""
    if not raw:
        year = paper.get("year")
        raw = f"{year}-01-01" if year else ""
    if not raw:
        return False
    try:
        d = datetime.strptime(raw[:10], "%Y-%m-%d")
        return date_start <= d <= date_end
    except ValueError:
        return False


def _process_query_task(domain: str, query: str, api_key: str, delay: float,
                        fetch_n: int, tier1: list, tier2: list,
                        year_str: str, date_start: str, date_end: str) -> dict:
    """Fetch, sample and enrich one (domain, query) task."""
    logger.info(f"  [{domain}] {query!r}")
    data = _get(
        f"{SS_BASE}/paper/search",
        {
            "query":  query,
            "year":   year_str,
            "fields": _SEARCH_FIELDS,
            "limit":  fetch_n,
        },
        api_key, delay,
    )

    result = {
        "domain": domain,
        "query": query,
        "raw_hits": 0,
        "selected_count": 0,
        "papers": [],
    }
    if not data:
        logger.warning("    no response")
        return result

    papers = data.get("data") or []
    result["raw_hits"] = len(papers)
    logger.info(f"    → {len(papers)} hits (fetched top-{fetch_n})")

    date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
    date_end_dt   = datetime.strptime(date_end, "%Y-%m-%d")
    dated_papers = [p for p in papers if p is not None and _pub_date_ok(p, date_start_dt, date_end_dt)]
    logger.info(f"    → {len(dated_papers)} in date window [{date_start} ~ {date_end}]")

    t1_lo, t1_hi = tier1[0] - 1, tier1[1] - 1
    t2_lo, t2_hi = tier2[0] - 1, tier2[1] - 1

    t1_pool = dated_papers[t1_lo : t1_hi + 1]
    t2_pool = dated_papers[t2_lo : t2_hi + 1]

    selected = []
    if t1_pool:
        selected.append(random.choice(t1_pool))
    if t2_pool:
        chosen = random.choice(t2_pool)
        if chosen not in selected:
            selected.append(chosen)
    if len(selected) < 2 and len(dated_papers) > len(selected):
        remaining = [p for p in dated_papers if p not in selected]
        selected.extend(random.sample(remaining, min(2 - len(selected), len(remaining))))

    result["selected_count"] = len(selected)
    logger.info(f"    → {len(selected)} selected via stratified sampling")

    for p in selected:
        if p is None:
            continue
        ss_id = p.get("paperId") or ""
        title = (p.get("title") or "").strip()
        if not ss_id or not title:
            continue

        refs = _fetch_refs(ss_id, api_key, delay)
        abstract = (p.get("abstract") or "").strip()
        pub_date = p.get("publicationDate") or (
            f"{p['year']}-01-01" if p.get("year") else None
        )
        n_refs = len(refs)
        result["papers"].append({
            "paper_id": ss_id,
            "title": title,
            "abstract": abstract,
            "abstract_words": len(abstract.split()) if abstract else 0,
            "domain": domain,
            "query": query,
            "published_date": pub_date,
            "venue": p.get("venue") or "",
            "citation_count": p.get("citationCount") or 0,
            "pub_types_json": json.dumps(p.get("publicationTypes") or []),
            "references_json": json.dumps(refs),
            "n_valid_refs": n_refs,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "n_refs": n_refs,
        })
        logger.info(
            f"    ✓ {ss_id[:16]}  ({n_refs} refs)  {title[:55]!r}"
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_all(cfg, smoke: bool = False,
              domain_filter: Optional[str] = None) -> dict:
    """Search SS for recent papers across all configured queries.

    Uses stratified random sampling per query to maximise diversity:
      - Fetches top ss_fetch_per_query (default 20) results from SS
      - Picks 1 paper randomly from rank 1-5 (tier 1)
      - Picks 1 paper randomly from rank 6-10 (tier 2)
      - ~2 papers retained per query

    Date range comes from config: paper_date_start .. paper_date_end.
    Exact date filtering is enforced downstream by filter_papers.py.

    Args:
        cfg:           config module (exposes SEMANTIC_SCHOLAR_API_KEY, DATASET, …)
        smoke:         If True, first domain × first query only.
        domain_filter: Restrict to one domain.

    Returns:
        stats dict: queries, raw_hits, selected, inserted, skipped, errors.
    """
    api_key = cfg.SEMANTIC_SCHOLAR_API_KEY
    if not api_key:
        raise ValueError("SEMANTIC_SCHOLAR_API_KEY not set in .env")

    delay   = 1.1   # 1 req/s limit
    topics  = dict(cfg.DATASET["serper_topics"])
    fetch_n = cfg.DATASET.get("ss_fetch_per_query", 20)   # how many SS results to request
    tier1   = cfg.DATASET.get("ss_sample_tier1", [1, 5])  # inclusive [lo, hi]
    tier2   = cfg.DATASET.get("ss_sample_tier2", [6, 10]) # inclusive [lo, hi]

    # ── Date range from config ─────────────────────────────────────────────
    date_start = cfg.DATASET.get("paper_date_start", "2025-07-01")
    date_end   = cfg.DATASET.get("paper_date_end",   "2026-01-31")
    # SS year filter accepts "YYYY-YYYY" for a range
    year_start = date_start[:4]
    year_end   = date_end[:4]
    year_str   = f"{year_start}-{year_end}" if year_start != year_end else f"{year_start}-"
    logger.info(f"SS search year filter: {year_str!r}  "
                f"(exact date window: {date_start} ~ {date_end})")

    # ── Scope restrictions ────────────────────────────────────────────────
    if smoke:
        first_d = list(topics.keys())[0]
        topics  = {first_d: [topics[first_d][0]]}
        logger.info(f"[smoke] {first_d!r} / {topics[first_d][0]!r}")
    if domain_filter:
        if domain_filter not in topics:
            raise ValueError(f"Unknown domain {domain_filter!r}")
        topics = {domain_filter: topics[domain_filter]}

    db_path = str(cfg.PAPERS_DB)
    conn    = sqlite3.connect(db_path)
    cur     = conn.cursor()

    stats = {"queries": 0, "raw_hits": 0, "selected": 0,
             "inserted": 0, "skipped": 0, "errors": 0}
    seen_ids_by_domain = {
        domain: {
            r[0] for r in cur.execute(
                "SELECT paper_id FROM papers WHERE domain=?", (domain,)
            )
        }
        for domain in topics.keys()
    }
    domain_inserted = {domain: 0 for domain in topics.keys()}

    query_tasks = [
        (domain, query)
        for domain, queries in topics.items()
        for query in queries
    ]
    stats["queries"] = len(query_tasks)
    max_workers = max(1, int(cfg.PARALLEL.get("phase1_fetch_max_workers", 6)))
    logger.info(f"SS fetch workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_query_task,
                domain,
                query,
                api_key,
                delay,
                fetch_n,
                tier1,
                tier2,
                year_str,
                date_start,
                date_end,
            ): (domain, query)
            for domain, query in query_tasks
        }

        for future in as_completed(futures):
            domain, query = futures[future]
            try:
                result = future.result()
                stats["raw_hits"] += result["raw_hits"]
                stats["selected"] += result["selected_count"]

                for item in result["papers"]:
                    paper_id = item["paper_id"]
                    if paper_id in seen_ids_by_domain[domain]:
                        stats["skipped"] += 1
                        continue

                    seen_ids_by_domain[domain].add(paper_id)
                    try:
                        cur.execute("""
                            INSERT OR IGNORE INTO papers
                              (paper_id, title, abstract, abstract_words,
                               domain, query, published_date, venue,
                               citation_count, pub_types_json,
                               references_json, n_valid_refs,
                               fetched_at, status)
                            VALUES (?,?,?,?, ?,?,?,?, ?,?, ?,?, ?,
                                    'ss_enriched')
                        """, (
                            item["paper_id"],
                            item["title"],
                            item["abstract"],
                            item["abstract_words"],
                            item["domain"],
                            item["query"],
                            item["published_date"],
                            item["venue"],
                            item["citation_count"],
                            item["pub_types_json"],
                            item["references_json"],
                            item["n_valid_refs"],
                            item["fetched_at"],
                        ))
                        if cur.rowcount:
                            domain_inserted[domain] += 1
                            stats["inserted"] += 1
                        else:
                            stats["skipped"] += 1
                    except sqlite3.Error as e:
                        logger.error(f"  DB error for {paper_id}: {e}")
                        stats["errors"] += 1

                conn.commit()
            except Exception as e:
                logger.error(f"  query task failed for [{domain}] {query!r}: {e}")
                stats["errors"] += 1

    for domain, inserted in domain_inserted.items():
        logger.info(f"  [{domain}] {inserted} papers inserted")

    conn.close()
    logger.info(
        f"fetch_ss_search done: "
        f"{stats['queries']} queries, {stats['raw_hits']} hits, "
        f"{stats['selected']} selected, {stats['inserted']} inserted, "
        f"{stats['skipped']} skipped"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Search Semantic Scholar for recent papers"
    )
    parser.add_argument("--smoke",  action="store_true")
    parser.add_argument("--domain", default=None)
    args = parser.parse_args()

    result = fetch_all(cfg, smoke=args.smoke, domain_filter=args.domain)
    print(f"\nDone: {result['inserted']} papers inserted into papers.db")
