"""
Semantic Scholar Enrichment

For every paper in papers.db with status='raw', searches Semantic Scholar
by title and enriches the row with:
  - SS canonical paperId  (replaces the temporary "serper_*" hash)
  - full abstract         (replaces Serper snippet)
  - year / publicationDate / venue / citationCount / publicationTypes
  - references list with intents  (JSON, stored in references_json)

Two-step fetch per paper:
  1. GET /paper/search?query={title}  →  find + verify paperId (rapidfuzz)
     Also returns full metadata (abstract, year, venue, etc.) — no extra call needed.
  2. GET /paper/{paperId}/references  →  reference edges WITH intents
     (intents live on the edge, not on the cited paper — must use /references endpoint)

Rate-limit handling:
  - Without SS key: ~100 req / 5 min  → use delay ≥ 3.5 s per request
  - With SS key:    ~10 req / s        → delay ≥ 0.2 s
  On 429, backs off 120 s and retries once.

Usage:
    python data_collection/fetch_semantic_scholar.py
    python data_collection/fetch_semantic_scholar.py --smoke
    python data_collection/fetch_semantic_scholar.py --delay 0.3  # with API key
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SS_BASE = "https://api.semanticscholar.org/graph/v1"

# Step 1: search fields — full metadata, no references (search endpoint supports these)
_SEARCH_FIELDS = (
    "paperId,title,abstract,year,publicationDate,"
    "venue,citationCount,publicationTypes"
)

# Step 2: /references endpoint fields
# NOTE: intents live on the citation EDGE, not on the cited paper.
#       Must use /paper/{id}/references — NOT /paper/{id}?fields=references.intents
_REFS_FIELDS = (
    "citedPaper.paperId,citedPaper.title,citedPaper.abstract,"
    "citedPaper.year,citedPaper.citationCount,"
    "intents"
)

TITLE_SIM_THRESHOLD = 0.80   # rapidfuzz token_set_ratio ÷ 100


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _headers(api_key: str) -> dict:
    h = {"User-Agent": "SciSynthBench/1.0 (research)"}
    if api_key:
        h["x-api-key"] = api_key
    return h


def _get_with_backoff(url: str, params: dict, api_key: str, delay: float) -> Optional[dict]:
    """GET with exponential-backoff retries on 429.

    Retry schedule on 429: 120s → 300s → 600s → give up (total ~17 min worst case).
    This covers the SS 5-minute rate-limit window even for unauthenticated requests.
    """
    backoffs = [120, 300, 600]

    for attempt in range(len(backoffs) + 1):
        try:
            resp = requests.get(url, params=params, headers=_headers(api_key), timeout=30)
        except requests.exceptions.RequestException as e:
            logger.warning(f"  SS request error (attempt {attempt+1}): {e}")
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt])
            continue

        time.sleep(delay)

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429:
            if attempt < len(backoffs):
                wait = backoffs[attempt]
                logger.warning(f"  SS rate-limit (429) — backing off {wait} s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            else:
                logger.error(f"  SS rate-limit persists after all retries: {url}")
                return None

        logger.error(f"  SS HTTP {resp.status_code}: {url}")
        return None

    return None


def _find_paper(title: str, api_key: str, delay: float) -> Optional[dict]:
    """Search SS by title; return best match dict (with paperId) or None."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        raise ImportError("rapidfuzz not installed — run: pip install rapidfuzz")

    data = _get_with_backoff(
        f"{SS_BASE}/paper/search",
        {"query": title, "fields": _SEARCH_FIELDS, "limit": 5},
        api_key, delay
    )
    if not data:
        return None

    candidates = data.get("data") or []
    best, best_score = None, 0.0
    for c in candidates:
        c_title = c.get("title") or ""
        score = fuzz.token_set_ratio(title.lower(), c_title.lower()) / 100.0
        if score > best_score:
            best_score, best = score, c

    if best_score < TITLE_SIM_THRESHOLD:
        logger.debug(f"  No SS match ({best_score:.2f}) for {title[:60]!r}")
        return None

    logger.debug(f"  SS matched ({best_score:.2f}): {best.get('title','')[:60]!r}")
    return best


def _fetch_references(paper_id: str, api_key: str, delay: float) -> list:
    """Fetch reference edges (with intents) via the dedicated /references endpoint.

    Response format:
        {"data": [{"citedPaper": {...}, "intents": ["background"]}, ...]}

    intents live on the EDGE, not on the citedPaper — this is why we cannot
    use /paper/{id}?fields=references.intents (that causes a 400).
    """
    data = _get_with_backoff(
        f"{SS_BASE}/paper/{paper_id}/references",
        {"fields": _REFS_FIELDS, "limit": 500},
        api_key, delay
    )
    if not data:
        return []

    out = []
    for edge in (data.get("data") or []):
        if edge is None:
            continue
        cited   = edge.get("citedPaper") or {}
        intents = edge.get("intents")    or []
        out.append({
            "paperId":       cited.get("paperId"),
            "title":         cited.get("title"),
            "abstract":      cited.get("abstract"),
            "year":          cited.get("year"),
            "intent":        intents[0] if intents else "",
            "citationCount": cited.get("citationCount", 0),
        })
    return out


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich_papers(db_path: str, api_key: str,
                  delay: float = 1.1,
                  smoke: bool = False) -> dict:
    """Enrich all status='raw' papers in papers.db with SS data.

    On success  → status set to 'ss_enriched', paper_id replaced with SS id.
    On no-match → status set to 'ss_failed'.
    On error    → status set to 'ss_failed', error logged.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    cur.execute("SELECT * FROM papers WHERE status = 'raw'")
    rows = list(cur.fetchall())

    if smoke:
        rows = rows[:5]
        logger.info(f"[smoke] enriching {len(rows)} papers")
    else:
        logger.info(f"fetch_semantic_scholar: enriching {len(rows)} papers")

    stats = {"total": len(rows), "enriched": 0, "no_match": 0, "errors": 0}

    for i, row in enumerate(rows, 1):
        old_id  = row["paper_id"]
        title   = row["title"]
        domain  = row["domain"]
        logger.info(f"  [{i}/{len(rows)}] [{domain}] {title[:65]!r}")

        try:
            # ── Step 1: find SS paperId ──────────────────────────────────
            match = _find_paper(title, api_key, delay)
            if not match:
                cur.execute(
                    "UPDATE papers SET status='ss_failed' WHERE paper_id=?", (old_id,)
                )
                conn.commit()
                stats["no_match"] += 1
                logger.info("    → no SS match")
                continue

            ss_id = match["paperId"]

            # ── Deduplicate: if SS id already in DB, drop this row ───────
            dup = cur.execute(
                "SELECT paper_id FROM papers WHERE paper_id=? AND paper_id!=?",
                (ss_id, old_id)
            ).fetchone()
            if dup:
                cur.execute("DELETE FROM papers WHERE paper_id=?", (old_id,))
                conn.commit()
                stats["no_match"] += 1
                logger.info(f"    → duplicate of existing {ss_id}, dropped")
                continue

            # ── Step 2: fetch references via /references endpoint ────────
            # (search result already has full metadata; only refs need a 2nd call)
            refs = _fetch_references(ss_id, api_key, delay)
            n_bg = sum(1 for r in refs if r["intent"] in ("background", ""))

            # Prefer full abstract from SS search result; fall back to Serper snippet
            abstract = (match.get("abstract") or row["abstract"] or "").strip()
            pub_date = match.get("publicationDate") or (
                f"{match['year']}-01-01" if match.get("year") else None
            )

            # ── Update row (swap paper_id to SS id) ──────────────────────
            cur.execute("""
                UPDATE papers SET
                    paper_id        = ?,
                    abstract        = ?,
                    abstract_words  = ?,
                    published_date  = ?,
                    venue           = ?,
                    citation_count  = ?,
                    pub_types_json  = ?,
                    references_json = ?,
                    n_valid_refs    = ?,
                    status          = 'ss_enriched'
                WHERE paper_id = ?
            """, (
                ss_id,
                abstract,
                len(abstract.split()) if abstract else 0,
                pub_date,
                match.get("venue") or "",
                match.get("citationCount") or 0,
                json.dumps(match.get("publicationTypes") or []),
                json.dumps(refs),
                n_bg,
                old_id,
            ))
            conn.commit()

            stats["enriched"] += 1
            logger.info(f"    ��� {ss_id}  ({len(refs)} refs, {n_bg} background)")

        except Exception as e:
            logger.error(f"    ERROR: {e}")
            try:
                cur.execute(
                    "UPDATE papers SET status='ss_failed' WHERE paper_id=?", (old_id,)
                )
                conn.commit()
            except Exception:
                pass
            stats["errors"] += 1

    conn.close()
    logger.info(
        f"fetch_semantic_scholar done: "
        f"{stats['enriched']} enriched, "
        f"{stats['no_match']} no-match/dup, "
        f"{stats['errors']} errors"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg

    parser = argparse.ArgumentParser(
        description="Enrich raw papers in papers.db with Semantic Scholar data"
    )
    parser.add_argument("--db",    default=str(cfg.PAPERS_DB), help="Path to papers.db")
    parser.add_argument("--delay", type=float, default=None,
                        help="Seconds between SS API calls (default: 1.1 with key, 3.5 without)")
    parser.add_argument("--smoke", action="store_true", help="Process first 5 papers only")
    args = parser.parse_args()

    delay = args.delay
    if delay is None:
        delay = 1.1 if cfg.SEMANTIC_SCHOLAR_API_KEY else 3.5

    stats = enrich_papers(
        args.db,
        api_key=cfg.SEMANTIC_SCHOLAR_API_KEY,
        delay=delay,
        smoke=args.smoke,
    )
    print(
        f"\nDone: {stats['enriched']} enriched, "
        f"{stats['no_match']} no-match, "
        f"{stats['errors']} errors"
    )
