"""
Database Initialisation — SciSynthBench

Creates two SQLite databases with their full schema:

  papers.db  — paper library (Serper + Semantic Scholar data + GT hypotheses)
  results.db — evaluation results (ideas, scores, comparisons, aggregates)

Safe to run multiple times (CREATE TABLE IF NOT EXISTS).

Usage:
    python utils/db_init.py                    # use paths from config.py
    python utils/db_init.py --papers /tmp/p.db --results /tmp/r.db
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

PAPERS_SCHEMA = """
-- ── papers ──────────────────────────────────────────────────────────────
-- One row per target paper (after Serper → SS fetch, before filtering).
CREATE TABLE IF NOT EXISTS papers (
    paper_id        TEXT PRIMARY KEY,      -- Semantic Scholar paperId (or Serper URL hash)
    title           TEXT NOT NULL,
    abstract        TEXT NOT NULL,
    abstract_words  INTEGER,               -- cached word count
    domain          TEXT NOT NULL,         -- CS / Biology / Physics / Chemistry / Medicine
    query           TEXT NOT NULL,         -- search query that found this paper
    published_date  TEXT,                  -- YYYY-MM-DD (from SS or Serper)
    venue           TEXT,                  -- journal / conference name
    citation_count  INTEGER DEFAULT 0,
    pub_types_json  TEXT,                  -- JSON list of SS publicationTypes
    references_json TEXT,                  -- JSON list of {paperId,title,abstract,year,
                                           --               citationCount,publicationTypes,
                                           --               intents,intent}
    n_valid_refs    INTEGER DEFAULT 0,     -- count of refs with titles (intent filter removed)
    ranked_refs_json TEXT,                 -- JSON: refs ranked by embedding cosine sim to main paper
                                           --   (Phase 1-E output; used by Track B generation)
    gt_hypothesis   TEXT,                  -- GPT-4o rewrite: abstract → hypothesis form
    status          TEXT DEFAULT 'raw',    -- raw | filtered | rejected
    reject_reason   TEXT,                  -- why it was rejected (if status=rejected)
    leakage_flag    INTEGER DEFAULT 0,     -- 1 if flagged by L3 leakage probe
    fetched_at      TEXT NOT NULL          -- ISO timestamp of data collection
);

CREATE INDEX IF NOT EXISTS idx_papers_domain  ON papers (domain);
CREATE INDEX IF NOT EXISTS idx_papers_status  ON papers (status);
CREATE INDEX IF NOT EXISTS idx_papers_date    ON papers (published_date);
"""

RESULTS_SCHEMA = """
-- ── results ──────────────────────────────────────────────────────────────
-- One row per (paper × idea_model × track × idea_index × critic_model).
CREATE TABLE IF NOT EXISTS results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        TEXT    NOT NULL,
    idea_model      TEXT    NOT NULL,
    track           TEXT    NOT NULL,    -- 'A' (keyword-only) or 'B' (with references)
    idea_index      INTEGER NOT NULL,    -- 1, 2, or 3
    idea_text       TEXT    NOT NULL,
    critic_model    TEXT    NOT NULL,
    scores_json     TEXT,               -- {"originality":X,"feasibility":X,"clarity":X,"impact":X,"specificity":X}
    reasoning_json  TEXT,               -- {"originality":"...", ...}  per-dimension reasoning
    raw_response    TEXT,               -- full critic LLM output
    semantic_overlap REAL,              -- E1: LLM semantic overlap vs gt_hypothesis (0–1)
    error           TEXT,               -- error message if scoring failed
    created_at      TEXT    NOT NULL    -- ISO timestamp
);

CREATE INDEX IF NOT EXISTS idx_results_paper_model  ON results (paper_id, idea_model);
CREATE INDEX IF NOT EXISTS idx_results_track        ON results (track);
CREATE INDEX IF NOT EXISTS idx_results_critic       ON results (critic_model);

-- Unique constraint: one row per (paper × idea_model × track × idea_index × critic_model)
-- Phase-2 source rows have critic_model=''; Phase-3 critic rows have critic_model=<model>.
CREATE UNIQUE INDEX IF NOT EXISTS idx_results_unique
    ON results (paper_id, idea_model, track, idea_index, critic_model);

-- ── model_scores ─────────────────────────────────────────────────────────
-- Aggregated per (paper × idea_model × track): best idea + mean scores.
CREATE TABLE IF NOT EXISTS model_scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id            TEXT    NOT NULL,
    idea_model          TEXT    NOT NULL,
    track               TEXT    NOT NULL,   -- 'A' or 'B'
    best_idea_text      TEXT,               -- highest-scoring idea among 3
    best_idea_index     INTEGER,
    mean_absolute_score REAL,               -- mean of 5 dims × 3 critics
    semantic_overlap    REAL,               -- E1 for best idea
    context_gain        REAL,               -- score_B − score_A (NULL for Track A rows)
    computed_at         TEXT    NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_scores_key
    ON model_scores (paper_id, idea_model, track);

-- ── pairwise_results ─────────────────────────────────────────────────────
-- One row per (paper × track × model_a × model_b × critic × swap_order).
-- Phase 3-P: pairwise comparison of best ideas between two idea models.
CREATE TABLE IF NOT EXISTS pairwise_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        TEXT    NOT NULL,
    track           TEXT    NOT NULL,      -- 'A' or 'B'
    model_a         TEXT    NOT NULL,      -- first idea model
    model_b         TEXT    NOT NULL,      -- second idea model
    idea_a_text     TEXT    NOT NULL,
    idea_b_text     TEXT    NOT NULL,
    critic_model    TEXT    NOT NULL,
    swap_order      INTEGER NOT NULL DEFAULT 0,  -- 0=original, 1=swapped
    winner          TEXT,                  -- 'A', 'B', or 'tie'
    reasoning       TEXT,
    raw_response    TEXT,
    error           TEXT,
    created_at      TEXT    NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_pairwise_unique
    ON pairwise_results (paper_id, track, model_a, model_b, critic_model, swap_order);
"""


# ---------------------------------------------------------------------------
# Init functions
# ---------------------------------------------------------------------------

def init_papers_db(db_path: str) -> None:
    """Create papers.db with full schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(PAPERS_SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit()
    conn.close()
    logger.info(f"papers.db initialised → {db_path}")


def init_results_db(db_path: str) -> None:
    """Create results.db with full schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(RESULTS_SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_telemetry_column(conn)
    _ensure_cited_refs_column(conn)
    _ensure_pairwise_telemetry_column(conn)
    conn.commit()
    conn.close()
    logger.info(f"results.db initialised → {db_path}")


def _ensure_telemetry_column(conn: sqlite3.Connection) -> None:
    """Add `telemetry` TEXT column to results if missing. JSON blob with
    {latency_ms, usage:{prompt_tokens, completion_tokens, reasoning_tokens}}
    per LLM call. Populated by phase-2 (idea gen) and phase-3 (critic) rows.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(results)")
    cols = {row[1] for row in cur.fetchall()}
    if "telemetry" not in cols:
        cur.execute("ALTER TABLE results ADD COLUMN telemetry TEXT")
        conn.commit()


def _ensure_cited_refs_column(conn: sqlite3.Connection) -> None:
    """Add `cited_refs` TEXT column to results if missing. JSON array of
    short reference identifiers (e.g. ['[3]', '[7]', 'Vaswani et al']) that
    the idea model claimed to ground the hypothesis in. Parsed from the
    Cited: footer in idea_text (REQUIRE_CITES path).
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(results)")
    cols = {row[1] for row in cur.fetchall()}
    if "cited_refs" not in cols:
        cur.execute("ALTER TABLE results ADD COLUMN cited_refs TEXT")
        conn.commit()


def _ensure_pairwise_telemetry_column(conn: sqlite3.Connection) -> None:
    """Add `telemetry` TEXT column to pairwise_results if missing. JSON blob
    with {latency_ms, attempts, forward:{...}, swap:{...}} for the two-call
    pairwise comparison (forward + swap order)."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(pairwise_results)")
    cols = {row[1] for row in cur.fetchall()}
    if "telemetry" not in cols:
        cur.execute("ALTER TABLE pairwise_results ADD COLUMN telemetry TEXT")
        conn.commit()


def init_all(papers_path: str, results_path: str) -> None:
    """Initialise both databases."""
    init_papers_db(papers_path)
    init_results_db(results_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg

    parser = argparse.ArgumentParser(description="Initialise SciSynthBench databases")
    parser.add_argument("--papers",  default=str(cfg.PAPERS_DB),  help="Path for papers.db")
    parser.add_argument("--results", default=str(cfg.RESULTS_DB), help="Path for results.db")
    args = parser.parse_args()

    init_all(args.papers, args.results)
    print(f"✓ papers.db  → {args.papers}")
    print(f"✓ results.db → {args.results}")
