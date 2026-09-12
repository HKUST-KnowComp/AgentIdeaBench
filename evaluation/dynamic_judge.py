"""Dynamic judge: per-idea reference retrieval for critic scoring.

Three judging modes are supported by absolute_scorer.score_idea:

  static            -- caller passes a fixed `references` block (the
                       canonical survey refs from papers.db). This is
                       the existing behavior.

  dynamic_search    -- critic-time SS search: extract a short query from
                       the hypothesis (first ~80 chars after "Hypothesis:"
                       or "We hypothesize"), search Semantic Scholar,
                       format top-N hits as the references block.

  dynamic_cited     -- the ideation prompt asked the generator to append
                       a "Cited:" footer listing the references it
                       grounded the hypothesis in. We parse that footer
                       and resolve each citation against papers.db (or
                       SS lookup) to build the references block.

The three modes are independent: any critic can run on any idea with any
of them. The mode is chosen at the call site (run.py CLI flag /
critic_manager argument).
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query extraction
# ---------------------------------------------------------------------------

_HYPOTH_PREFIX = re.compile(
    r"^\s*(?:Hypothesis:|We hypothesi[zs]e(?: that)?)\s*",
    re.IGNORECASE,
)


def extract_query_from_idea(idea_text: str, max_chars: int = 200) -> str:
    """Pull a search-friendly query from the first sentence of the idea.

    Strips the standard prefix ("Hypothesis:" / "We hypothesize that ...")
    and returns up to max_chars of the first sentence, suitable for an
    SS keyword search.
    """
    if not idea_text:
        return ""
    s = _HYPOTH_PREFIX.sub("", idea_text.strip(), count=1)
    # First sentence boundary
    m = re.search(r"[.!?]\s+", s)
    sentence = s[:m.start()] if m else s
    sentence = sentence.replace("\n", " ").strip()
    return sentence[:max_chars]


# ---------------------------------------------------------------------------
# dynamic_search: SS keyword search
# ---------------------------------------------------------------------------

def fetch_refs_by_search(idea_text: str, top_n: int = 10) -> List[dict]:
    """Run an SS keyword search using the hypothesis as a query.

    Returns up to top_n papers each with {paperId, title, abstract,
    year, citationCount}. Abstracts truncated to 1500 chars to match
    Active mode's tool-result format.
    """
    from data_collection.fetch_ss_search import _get, SS_BASE
    import config as cfg

    query = extract_query_from_idea(idea_text)
    if not query:
        return []
    api_key = getattr(cfg, "SEMANTIC_SCHOLAR_API_KEY", None)
    data = _get(
        f"{SS_BASE}/paper/search",
        {
            "query": query,
            "fields": "paperId,title,abstract,year,citationCount",
            "limit": min(max(top_n, 1), 20),
        },
        api_key,
        0.2,
    )
    if not data:
        return []
    out: List[dict] = []
    for p in (data.get("data") or [])[:top_n]:
        if not p:
            continue
        abstract = p.get("abstract") or ""
        out.append({
            "paperId": p.get("paperId"),
            "title": p.get("title") or "",
            "abstract": abstract[:1500] if abstract else "",
            "year": p.get("year"),
            "citationCount": p.get("citationCount") or 0,
        })
    return out


# ---------------------------------------------------------------------------
# dynamic_cited: parse "Cited:" footer
# ---------------------------------------------------------------------------

_CITED_HEADER = re.compile(
    r"(?:^|\n)\s*Cited(?:\s+references|\s+refs)?\s*:\s*",
    re.IGNORECASE,
)


def extract_cited_block(idea_text: str) -> Tuple[str, List[str]]:
    """Strip the trailing 'Cited: ...' block from idea_text.

    Returns (hypothesis_without_cites, list_of_cite_strings). Each cite
    string is one item (a title, a "First Author et al. (2023)" form,
    or any free-form short reference). Caller resolves them.

    If no Cited block, returns (original_text, []).
    """
    if not idea_text:
        return ("", [])
    m = _CITED_HEADER.search(idea_text)
    if not m:
        return (idea_text.strip(), [])
    body = idea_text[:m.start()].strip()
    cited_raw = idea_text[m.end():].strip()
    # Split on separators (newline / "; " / leading "- ") but keep [N] markers
    # so pure-index cites like "[13]; [10]; [16]" stay distinguishable.
    parts = re.split(r"\s*(?:\n|;)\s*", cited_raw)
    items = [p.lstrip("-* \t").strip(" .,;()") for p in parts if p.strip()]
    # If any item is just a bracketed index ("[13]"), keep that as the cite —
    # downstream resolvers can map index → ref via the papers' refs block.
    return (body, items)


def resolve_cited_refs(cite_strings: List[str], db_papers: Optional[str] = None,
                      top_n: int = 10) -> List[dict]:
    """Resolve each cite string to a paper record.

    Strategy:
      1. If db_papers given, try matching cite against existing
         papers.db titles (case-insensitive substring).
      2. Otherwise (or if no match), call SS keyword search with the
         cite string as query and take the top-1 result.

    Returns up to top_n resolved papers.
    """
    if not cite_strings:
        return []
    out: List[dict] = []
    db_titles: List[Tuple[str, dict]] = []
    if db_papers:
        try:
            conn = sqlite3.connect(db_papers)
            conn.row_factory = sqlite3.Row
            for r in conn.execute("SELECT paper_id, title, abstract FROM papers"):
                db_titles.append((
                    (r["title"] or "").lower(),
                    {
                        "paperId": r["paper_id"],
                        "title": r["title"] or "",
                        "abstract": (r["abstract"] or "")[:1500],
                    },
                ))
            conn.close()
        except Exception as e:
            logger.warning(f"resolve_cited_refs: papers.db read failed: {e}")

    for cite in cite_strings[:top_n]:
        cite_low = cite.lower()
        matched = None
        if db_titles:
            for low, rec in db_titles:
                if low and (cite_low in low or low in cite_low):
                    matched = rec
                    break
        if matched is None:
            # Fall back to SS search
            try:
                hits = fetch_refs_by_search(cite, top_n=1)
                if hits:
                    matched = hits[0]
            except Exception as e:
                logger.warning(f"resolve_cited_refs: SS lookup failed for '{cite[:40]}': {e}")
        if matched:
            out.append(matched)
    return out


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def build_references_block(
    idea_text: str,
    judge_mode: str = "static",
    static_references: str = "",
    db_papers: Optional[str] = None,
    top_n: int = 10,
) -> Tuple[str, dict]:
    """Build the references block to feed the critic.

    judge_mode:
      static          -> return static_references verbatim
      dynamic_search  -> SS search using idea as query, format top_n hits
      dynamic_cited   -> parse 'Cited:' footer in idea_text, resolve each

    Returns (references_block, meta) where meta records the mode + how
    many refs were retrieved (for telemetry).
    """
    if judge_mode == "static":
        return (static_references or "", {"mode": "static", "n_refs": None})

    if judge_mode == "dynamic_search":
        refs = fetch_refs_by_search(idea_text, top_n=top_n)
        block = _format_refs(refs)
        return (block, {"mode": "dynamic_search", "n_refs": len(refs)})

    if judge_mode == "dynamic_cited":
        _, cites = extract_cited_block(idea_text)
        refs = resolve_cited_refs(cites, db_papers=db_papers, top_n=top_n)
        block = _format_refs(refs)
        return (block, {
            "mode": "dynamic_cited",
            "n_refs": len(refs),
            "n_cite_strings": len(cites),
        })

    raise ValueError(
        f"Unknown judge_mode: {judge_mode!r}. "
        f"Expected one of: static / dynamic_search / dynamic_cited."
    )


def _format_refs(refs: List[dict]) -> str:
    """Format a list of {paperId, title, abstract, year, citationCount}
    into a numbered references block matching the static rendering."""
    if not refs:
        return "(no references retrieved)"
    lines = []
    for i, r in enumerate(refs, 1):
        title = r.get("title") or "(no title)"
        year = r.get("year")
        abstract = r.get("abstract") or ""
        head = f"[{i}] {title}"
        if year:
            head += f" ({year})"
        lines.append(head)
        if abstract:
            lines.append(f"    {abstract}")
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# CLI smoke (manual)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Hypothesis: We hypothesize that incorporating contrastive learning "
        "objectives into graph neural networks will improve molecular "
        "property prediction by 15-20% on MoleculeNet.\n"
        "Cited: Vaswani et al. Attention is All You Need; "
        "Brown et al. Language Models are Few-Shot Learners"
    )

    print("=== Query extraction ===")
    print(repr(extract_query_from_idea(sample)))

    body, cites = extract_cited_block(sample)
    print("\n=== Cited extraction ===")
    print("body:", body)
    print("cites:", cites)

    print("\n=== References block (static) ===")
    blk, meta = build_references_block(sample, judge_mode="static",
                                       static_references="(static refs here)")
    print("meta:", meta)
    print(blk[:200])

    print("\n=== Skipping dynamic_search smoke (requires SS API call)")
    print("    To test: build_references_block(sample, judge_mode='dynamic_search')")
