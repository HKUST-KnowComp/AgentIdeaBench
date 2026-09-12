"""Prior probe: how much of each test paper's "prior context" did each
idea_model see in training?

Two corpora (`--corpus`):
  - testset (default, v0): probe is the 25 Track B test papers themselves.
    Used to rule out training-set leakage. All 2026 flagship models
    correctly answered UNKNOWN_PAPER on these 2025-04..2025-12 papers, so
    leakage is not the concern.
  - refs (v1): probe is the ~96 unique top-20 references of those 25 test
    papers — i.e. the prior literature each test paper itself cites. This
    is the right measure for "did newer models read more of the domain?",
    because refs span 1997..2024 and cross the train-cutoff of every model
    in the sweep.

For every (idea_model, probe_paper) we prompt the model to reproduce the
abstract from memory (title + venue + date only). The model may output
"UNKNOWN_PAPER" to avoid fabrication. We log ROUGE-L F1/P/R + unigram
Jaccard + length ratio + the self-reported unknown flag.

Stored in data/results.db, tables `prior_probe` (testset) and
`prior_probe_refs` (refs). Analysis is in experiments/prior_probe_correlate.py.

Usage:
    python experiments/prior_probe.py --smoke
    python experiments/prior_probe.py --corpus testset
    python experiments/prior_probe.py --corpus refs
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from utils.LLM import IdeaLLM  # noqa: E402

# Cross-year models (must match reports/_make_cross_year_plot.YEAR_GROUPS).
from reports._make_cross_year_plot import YEAR_GROUPS, RELEASE_DATES  # noqa: E402

logger = logging.getLogger("prior_probe")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


PROBE_SYSTEM_PROMPT = (
    "You are tested on factual recall, not creative writing. Output only what "
    "is requested. If you do not know the paper, you must say so explicitly "
    "rather than fabricate a plausible-sounding abstract."
)

PROBE_USER_TEMPLATE = """Reproduce the abstract of the following paper from memory, as accurately as you can. Do not paraphrase if you remember the original wording. If you have never seen this paper, output exactly the single line:

UNKNOWN_PAPER

and nothing else.

PAPER TITLE: {title}
DOMAIN: {domain}
VENUE: {venue}
PUBLISHED: {published_date}

Output only the abstract text (or the UNKNOWN_PAPER sentinel). No preamble, no title line, no explanation, no citation markers."""


# ---------------------------------------------------------------------------
# Text overlap metrics (no external deps)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall((s or "").lower())


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    # Hirschberg-style row-only DP
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def overlap_metrics(recitation: str, ground_truth: str) -> dict:
    """ROUGE-L F1/P/R, unigram Jaccard, length ratio. All in [0,1]."""
    a = _tokenize(recitation)
    b = _tokenize(ground_truth)
    if not a or not b:
        return {"rouge_l_f1": 0.0, "rouge_l_p": 0.0, "rouge_l_r": 0.0,
                "jaccard_unigram": 0.0, "length_ratio": 0.0,
                "n_tokens_recitation": len(a), "n_tokens_gt": len(b)}
    lcs = _lcs_length(a, b)
    p = lcs / len(a) if a else 0.0
    r = lcs / len(b) if b else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    set_a, set_b = set(a), set(b)
    jacc = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 0.0
    lr = len(a) / len(b) if b else 0.0
    return {
        "rouge_l_f1": round(f1, 4),
        "rouge_l_p": round(p, 4),
        "rouge_l_r": round(r, 4),
        "jaccard_unigram": round(jacc, 4),
        "length_ratio": round(lr, 4),
        "n_tokens_recitation": len(a),
        "n_tokens_gt": len(b),
    }


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def _ddl(table: str) -> str:
    extra = ""
    if table == "prior_probe_refs":
        extra = "    ref_year INTEGER, ref_domain TEXT,\n"
    return f"""
CREATE TABLE IF NOT EXISTS {table} (
    id INTEGER PRIMARY KEY,
    paper_id TEXT NOT NULL,
    idea_model TEXT NOT NULL,
    recitation_text TEXT,
    rouge_l_f1 REAL,
    rouge_l_p REAL,
    rouge_l_r REAL,
    jaccard_unigram REAL,
    length_ratio REAL,
    n_tokens_recitation INTEGER,
    n_tokens_gt INTEGER,
{extra}    self_reported_unknown INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(paper_id, idea_model)
);
"""


def _upsert_sql(table: str) -> str:
    if table == "prior_probe_refs":
        cols = ("paper_id, idea_model, recitation_text, rouge_l_f1, rouge_l_p, "
                "rouge_l_r, jaccard_unigram, length_ratio, n_tokens_recitation, "
                "n_tokens_gt, ref_year, ref_domain, self_reported_unknown, error, created_at")
        placeholders = "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?"
        extra_update = "ref_year = excluded.ref_year, ref_domain = excluded.ref_domain, "
    else:
        cols = ("paper_id, idea_model, recitation_text, rouge_l_f1, rouge_l_p, "
                "rouge_l_r, jaccard_unigram, length_ratio, n_tokens_recitation, "
                "n_tokens_gt, self_reported_unknown, error, created_at")
        placeholders = "?,?,?,?,?,?,?,?,?,?,?,?,?"
        extra_update = ""
    return f"""
INSERT INTO {table}
  ({cols})
VALUES ({placeholders})
ON CONFLICT(paper_id, idea_model) DO UPDATE SET
  recitation_text       = excluded.recitation_text,
  rouge_l_f1            = excluded.rouge_l_f1,
  rouge_l_p             = excluded.rouge_l_p,
  rouge_l_r             = excluded.rouge_l_r,
  jaccard_unigram       = excluded.jaccard_unigram,
  length_ratio          = excluded.length_ratio,
  n_tokens_recitation   = excluded.n_tokens_recitation,
  n_tokens_gt           = excluded.n_tokens_gt,
  {extra_update}self_reported_unknown = excluded.self_reported_unknown,
  error                 = excluded.error,
  created_at            = excluded.created_at;
"""


def _load_test_papers(db_papers: Path, db_results: Path) -> List[dict]:
    """testset corpus: the 25 Track B test papers themselves."""
    conn_r = sqlite3.connect(db_results)
    pids = [r[0] for r in conn_r.execute(
        "SELECT DISTINCT paper_id FROM model_scores WHERE track='B'"
    )]
    conn_r.close()
    if not pids:
        raise SystemExit("No Track B papers in model_scores — run Phase 3 first")
    ph = "(" + ",".join("?" * len(pids)) + ")"
    conn_p = sqlite3.connect(db_papers)
    conn_p.row_factory = sqlite3.Row
    rows = conn_p.execute(
        f"SELECT paper_id, title, abstract, domain, venue, published_date "
        f"FROM papers WHERE paper_id IN {ph}", pids
    ).fetchall()
    conn_p.close()
    return [dict(r) for r in rows]


_PER_DOMAIN_QUOTA = 20  # refs corpus aims for ≥20 unique refs per domain


def _load_refs(db_papers: Path, db_results: Path) -> List[dict]:
    """refs corpus: ranked_refs of the 25 test papers, deduped by paperId,
    backfilled per-domain from each paper's `references_json` until every
    domain reaches `_PER_DOMAIN_QUOTA` unique refs. Returns dicts keyed by
    ref paperId."""
    conn_r = sqlite3.connect(db_results)
    test_pids = [r[0] for r in conn_r.execute(
        "SELECT DISTINCT paper_id FROM model_scores WHERE track='B'"
    )]
    conn_r.close()
    if not test_pids:
        raise SystemExit("No Track B papers in model_scores — run Phase 3 first")
    ph = "(" + ",".join("?" * len(test_pids)) + ")"
    conn_p = sqlite3.connect(db_papers)
    conn_p.row_factory = sqlite3.Row
    rows = conn_p.execute(
        f"SELECT paper_id, domain, ranked_refs_json, references_json "
        f"FROM papers WHERE paper_id IN {ph}",
        test_pids,
    ).fetchall()
    conn_p.close()

    def _ref_entry(rf: dict, test_domain: str) -> Optional[dict]:
        rid = rf.get("paperId") or rf.get("title", "")
        abstract = (rf.get("abstract") or "").strip()
        if not rid or not abstract or len(abstract.split()) < 30:
            return None
        year = rf.get("year")
        return {
            "paper_id": rid,
            "title": rf.get("title") or "",
            "abstract": abstract,
            "domain": test_domain,
            "venue": rf.get("venue") or "",
            "published_date": str(year) if year else "",
            "ref_year": year,
            "ref_domain": test_domain,
        }

    unique: dict = {}
    # Pass 1: take valid refs from ranked_refs_json (top-20 per test paper)
    for row in rows:
        refs = json.loads(row["ranked_refs_json"]) if row["ranked_refs_json"] else []
        for rf in refs:
            entry = _ref_entry(rf, row["domain"])
            if entry is None or entry["paper_id"] in unique:
                continue
            unique[entry["paper_id"]] = entry

    # Pass 2: backfill from references_json so each domain reaches the quota
    rows_by_domain: dict = {}
    for row in rows:
        rows_by_domain.setdefault(row["domain"], []).append(row)
    for domain, drows in rows_by_domain.items():
        have = sum(1 for v in unique.values() if v["ref_domain"] == domain)
        if have >= _PER_DOMAIN_QUOTA:
            continue
        for row in drows:
            if have >= _PER_DOMAIN_QUOTA:
                break
            extras = json.loads(row["references_json"]) if row["references_json"] else []
            for rf in extras:
                entry = _ref_entry(rf, domain)
                if entry is None or entry["paper_id"] in unique:
                    continue
                unique[entry["paper_id"]] = entry
                have += 1
                if have >= _PER_DOMAIN_QUOTA:
                    break

    return list(unique.values())


def _existing_keys(db_results: Path, table: str) -> set:
    conn = sqlite3.connect(db_results)
    conn.execute(_ddl(table))
    rows = conn.execute(
        f"SELECT paper_id, idea_model FROM {table} "
        f"WHERE error IS NULL AND recitation_text IS NOT NULL"
    ).fetchall()
    conn.close()
    return set(rows)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def probe_one(model_id: str, paper: dict) -> dict:
    """One (model, paper) recitation + metrics. Returns row dict."""
    user_prompt = PROBE_USER_TEMPLATE.format(
        title=paper["title"],
        domain=paper["domain"] or "unknown",
        venue=paper["venue"] or "unknown",
        published_date=paper["published_date"] or "unknown",
    )
    out = {"paper_id": paper["paper_id"], "idea_model": model_id,
           "error": None, "self_reported_unknown": 0,
           "recitation_text": None,
           "ref_year": paper.get("ref_year"),
           "ref_domain": paper.get("ref_domain")}
    try:
        llm = IdeaLLM(model_id)
        # Recitation needs ~300 tokens max; cheap.
        text = llm.completion(user_prompt, system_prompt=PROBE_SYSTEM_PROMPT)
        if isinstance(text, tuple):
            text = text[0]
        recitation = (text or "").strip()
        out["recitation_text"] = recitation
        first_line = recitation.splitlines()[0].upper() if recitation else ""
        if "UNKNOWN_PAPER" in first_line:
            out["self_reported_unknown"] = 1
            metrics = overlap_metrics("", paper["abstract"] or "")
        else:
            metrics = overlap_metrics(recitation, paper["abstract"] or "")
        out.update(metrics)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out.update(overlap_metrics("", paper["abstract"] or ""))
    return out


def _write_row(conn: sqlite3.Connection, row: dict, table: str) -> None:
    if table == "prior_probe_refs":
        conn.execute(_upsert_sql(table), (
            row["paper_id"], row["idea_model"], row["recitation_text"],
            row.get("rouge_l_f1"), row.get("rouge_l_p"), row.get("rouge_l_r"),
            row.get("jaccard_unigram"), row.get("length_ratio"),
            row.get("n_tokens_recitation"), row.get("n_tokens_gt"),
            row.get("ref_year"), row.get("ref_domain"),
            row.get("self_reported_unknown", 0), row.get("error"),
            datetime.now(timezone.utc).isoformat(),
        ))
    else:
        conn.execute(_upsert_sql(table), (
            row["paper_id"], row["idea_model"], row["recitation_text"],
            row.get("rouge_l_f1"), row.get("rouge_l_p"), row.get("rouge_l_r"),
            row.get("jaccard_unigram"), row.get("length_ratio"),
            row.get("n_tokens_recitation"), row.get("n_tokens_gt"),
            row.get("self_reported_unknown", 0), row.get("error"),
            datetime.now(timezone.utc).isoformat(),
        ))
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 model x 3 papers sanity run")
    ap.add_argument("--corpus", choices=["testset", "refs"], default="testset",
                    help="testset = 25 Track B papers (leakage probe). "
                         "refs = 96 unique top-20 references (training-breadth probe).")
    ap.add_argument("--models", type=str, default="",
                    help="Comma-separated model_ids to probe; default = all "
                         "cross-year YEAR_GROUPS models with RELEASE_DATES")
    ap.add_argument("--n-papers", type=int, default=0,
                    help="Cap n probes (0 = all).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent model-paper calls")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (paper_id, model) pairs already done")
    args = ap.parse_args()

    db_results = ROOT / "data" / "results.db"
    db_papers = ROOT / "data" / "papers.db"
    table = "prior_probe_refs" if args.corpus == "refs" else "prior_probe"

    # Init table
    conn = sqlite3.connect(db_results)
    conn.execute(_ddl(table))
    conn.commit()
    conn.close()

    # Papers / refs
    if args.corpus == "refs":
        papers = _load_refs(db_papers, db_results)
        logger.info(f"Loaded {len(papers)} unique refs (corpus=refs)")
    else:
        papers = _load_test_papers(db_papers, db_results)
        logger.info(f"Loaded {len(papers)} test papers (corpus=testset)")
    if args.n_papers and args.n_papers < len(papers):
        papers = papers[: args.n_papers]
    if args.smoke:
        papers = papers[:3]

    # Models
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = [mid for ylist in YEAR_GROUPS.values() for mid, _ in ylist
                  if mid in RELEASE_DATES]
    if args.smoke:
        models = models[:1]

    skip = _existing_keys(db_results, table) if args.resume else set()
    tasks: List[Tuple[str, dict]] = []
    for m in models:
        for p in papers:
            if (p["paper_id"], m) in skip:
                continue
            tasks.append((m, p))

    logger.info(f"Probe: {len(models)} models × {len(papers)} papers "
                f"= {len(tasks)} new calls (skipping {len(models)*len(papers)-len(tasks)} resumed)")
    if not tasks:
        logger.info("Nothing to do.")
        return

    t0 = time.time()
    n_done = 0
    n_err = 0
    n_unknown = 0
    conn = sqlite3.connect(db_results, timeout=60)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(probe_one, m, p): (m, p["paper_id"]) for m, p in tasks}
        for fut in cf.as_completed(futs):
            m, pid = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                paper_row = next((p for _m, p in tasks if p["paper_id"] == pid), {})
                row = {"paper_id": pid, "idea_model": m, "error": f"future: {e}",
                       "recitation_text": None, "self_reported_unknown": 0,
                       "ref_year": paper_row.get("ref_year"),
                       "ref_domain": paper_row.get("ref_domain"),
                       **overlap_metrics("", "")}
            _write_row(conn, row, table)
            n_done += 1
            if row.get("error"):
                n_err += 1
            if row.get("self_reported_unknown"):
                n_unknown += 1
            if n_done % 25 == 0 or n_done == len(tasks):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                logger.info(f"  [{n_done}/{len(tasks)}] err={n_err} unknown={n_unknown} "
                            f"({rate:.1f}/s, eta {(len(tasks)-n_done)/max(rate,0.01):.0f}s)")
    conn.close()
    logger.info(f"Done: {n_done} rows ({n_err} errors, {n_unknown} self-unknown)")


if __name__ == "__main__":
    main()
