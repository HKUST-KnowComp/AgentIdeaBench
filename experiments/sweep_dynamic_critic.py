"""Sweep A — dynamic critic mode ablation.

For a fixed set of already-generated ideas (phase-2 output), re-score each
under three judge_mode values:
  - static         (canonical baseline: survey refs from papers.db)
  - dynamic_search (critic searches SS at scoring time with the hypothesis)
  - dynamic_cited  (critic reads the "Cited:" footer from idea_text)

Writes to NEW table `dynamic_critic_sweep` in results.db so the main
results.results table is preserved (CLAUDE.md §9.5 — raw data must not be
overwritten). The unique key includes judge_mode.

Model selection (per user instruction: two vendors with multiple sizes):
  - qwen/qwen3.5-9b
  - qwen/qwen3.5-27b
  - qwen/qwen3.5-397b-a17b
  - xiaomi/mimo-v2.5
  - xiaomi/mimo-v2.5-pro
Two-family design: Qwen 3.5 (3 sizes) tests size-scaling within one family;
adding MiMo v2.5 + v2.5-pro tests family-agnosticism — if dynamic_search > static
spread holds across both families, the discriminator effect is not Qwen-specific.

Usage:
    # 1) Make sure phase-2 for the 3 idea models is done (REQUIRE_CITES=True)
    python experiments/sweep_dynamic_critic.py --check

    # 2) Run sweep (resumable; --skip-existing default true)
    python experiments/sweep_dynamic_critic.py --smoke
    python experiments/sweep_dynamic_critic.py
"""
import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from evaluation.absolute_scorer import score_idea
from evaluation.critic_manager import _format_refs_for_judge

IDEA_MODELS = [
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.5-397b-a17b",
    "xiaomi/mimo-v2.5",
    "xiaomi/mimo-v2.5-pro",
]

# Use the same fixed critic pool as the main bench — but only 3 fast critics
# to keep wall-clock manageable. Minimax-m2.7 (thinking-ON, p50 35s) excluded;
# kimi-k2.6 already fixed (thinking-OFF p50 17s) is OK to include.
CRITIC_MODELS = [
    "deepseek/deepseek-v4-flash",   # p50 ~8s
    "z-ai/glm-5.1",                 # p50 ~13s
    "qwen/qwen3.6-plus",            # p50 ~15s
]

JUDGE_MODES = ["static", "dynamic_search", "dynamic_cited"]

SCHEMA = """
-- Sweep A ideas: regenerated with REQUIRE_CITES=True so every idea has a
-- Cited: footer. Lives in its own table so the main results.results raw data
-- (v2 cross-year rerun) is preserved (CLAUDE.md §9.5).
CREATE TABLE IF NOT EXISTS dynamic_critic_sweep_ideas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        TEXT    NOT NULL,
    idea_model      TEXT    NOT NULL,
    track           TEXT    NOT NULL,
    idea_index      INTEGER NOT NULL,
    idea_text       TEXT    NOT NULL,
    cited_refs      TEXT,
    telemetry       TEXT,
    created_at      TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dcsi_unique
    ON dynamic_critic_sweep_ideas (paper_id, idea_model, track, idea_index);

-- Sweep A critic verdicts: 5-dim scores per (idea, critic, judge_mode).
-- Unique key includes judge_mode so the three modes don't overwrite each other.
CREATE TABLE IF NOT EXISTS dynamic_critic_sweep (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        TEXT    NOT NULL,
    idea_model      TEXT    NOT NULL,
    track           TEXT    NOT NULL,
    idea_index      INTEGER NOT NULL,
    critic_model    TEXT    NOT NULL,
    judge_mode      TEXT    NOT NULL,
    scores_json     TEXT,
    reasoning_json  TEXT,
    raw_response    TEXT,
    telemetry       TEXT,
    error           TEXT,
    created_at      TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dcs_unique
    ON dynamic_critic_sweep (paper_id, idea_model, track, idea_index,
                             critic_model, judge_mode);
CREATE INDEX IF NOT EXISTS idx_dcs_idea ON dynamic_critic_sweep (idea_model, judge_mode);
"""


def init_sweep_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def fetch_papers(db_papers: str, smoke: bool = False, n_papers: int = None) -> dict:
    """Return {paper_id: paper_row_dict} for filtered papers.

    - smoke=True overrides everything → LIMIT 5
    - else n_papers caps the count if provided (default = no cap = all filtered)
    """
    conn = sqlite3.connect(db_papers)
    conn.row_factory = sqlite3.Row
    if smoke:
        limit = "LIMIT 5"
    elif n_papers:
        limit = f"LIMIT {int(n_papers)}"
    else:
        limit = ""
    rows = conn.execute(f"""
        SELECT * FROM papers
        WHERE status='filtered' AND gt_hypothesis IS NOT NULL AND gt_hypothesis != ''
        ORDER BY paper_id
        {limit}
    """).fetchall()
    return {r["paper_id"]: dict(r) for r in rows}


def fetch_sweep_ideas(db_results: str, paper_ids: list) -> list:
    """Fetch already-generated sweep ideas (with cited_refs) from sweep table."""
    conn = sqlite3.connect(db_results)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(IDEA_MODELS))
    p_placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(f"""
        SELECT paper_id, idea_model, track, idea_index, idea_text, cited_refs
        FROM dynamic_critic_sweep_ideas
        WHERE idea_model IN ({placeholders}) AND paper_id IN ({p_placeholders})
    """, list(IDEA_MODELS) + list(paper_ids)).fetchall()
    return [dict(r) for r in rows]


def generate_one_idea(paper: dict, idea_model: str, idea_index: int) -> dict:
    """Generate one fresh idea with REQUIRE_CITES=True. Returns sweep idea row."""
    from utils.LLM import IdeaLLM
    from generation.generate_ideas import (
        _build_generation_payload, _extract_cited_refs,
    )
    import generation.generate_ideas as gen_mod
    gen_mod.REQUIRE_CITES = True   # force cited footer

    llm = IdeaLLM(model_name=idea_model)
    prompt, fallback, system = _build_generation_payload(paper, "B")
    try:
        out = llm.generate_idea(prompt, fallback_prompt=fallback, system_prompt=system)
        idea_text = out["idea"].strip()
        usage = out.get("usage")
        latency_ms = getattr(llm, "last_latency_ms", None)
    except Exception as e:
        idea_text = f"[generation error: {e}]"
        usage = None
        latency_ms = None
    return {
        "paper_id": paper["paper_id"],
        "idea_model": idea_model,
        "track": "B",
        "idea_index": idea_index,
        "idea_text": idea_text,
        "cited_refs": json.dumps(_extract_cited_refs(idea_text)),
        "telemetry": json.dumps({"latency_ms": latency_ms, "usage": usage}),
    }


def persist_idea(conn: sqlite3.Connection, row: dict) -> bool:
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute("""
        INSERT OR IGNORE INTO dynamic_critic_sweep_ideas
          (paper_id, idea_model, track, idea_index, idea_text, cited_refs,
           telemetry, created_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        row["paper_id"], row["idea_model"], row["track"], row["idea_index"],
        row["idea_text"], row["cited_refs"], row["telemetry"], ts,
    ))
    conn.commit()
    return cur.rowcount > 0


def regen_ideas(db_results: str, papers: dict, n_ideas: int = 3,
                workers: int = 4) -> int:
    """Generate fresh ideas for IDEA_MODELS × papers × n_ideas, REQUIRE_CITES=True."""
    conn = sqlite3.connect(db_results, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")

    tasks = []
    for paper in papers.values():
        for model in IDEA_MODELS:
            for idx in range(1, n_ideas + 1):
                existing = conn.execute("""
                    SELECT 1 FROM dynamic_critic_sweep_ideas
                    WHERE paper_id=? AND idea_model=? AND track='B' AND idea_index=?
                """, (paper["paper_id"], model, idx)).fetchone()
                if existing:
                    continue
                tasks.append((paper, model, idx))

    total = len(tasks)
    print(f"\n## Idea regen: {total} tasks (workers={workers})")
    if total == 0:
        return 0

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_one_idea, p, m, i): (p, m, i)
                   for (p, m, i) in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            persist_idea(conn, row)
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total - done)
                print(f"  [{done}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

    return done


def already_scored(conn: sqlite3.Connection, paper_id: str, idea_model: str,
                   track: str, idea_index: int, critic_model: str,
                   judge_mode: str) -> bool:
    row = conn.execute("""
        SELECT 1 FROM dynamic_critic_sweep
        WHERE paper_id=? AND idea_model=? AND track=? AND idea_index=?
          AND critic_model=? AND judge_mode=?
    """, (paper_id, idea_model, track, idea_index, critic_model, judge_mode)).fetchone()
    return row is not None


def score_one(papers: dict, idea: dict, critic_model: str, judge_mode: str,
              db_papers: str) -> dict:
    paper = papers[idea["paper_id"]]
    refs_text = _format_refs_for_judge(paper)
    try:
        scores, raw, telem = score_idea(
            idea["idea_text"], critic_model,
            domain=paper.get("domain", ""),
            references=refs_text,
            judge_mode=judge_mode,
            db_papers=db_papers,
        )
        return {
            "paper_id": paper["paper_id"],
            "idea_model": idea["idea_model"],
            "track": idea["track"],
            "idea_index": idea["idea_index"],
            "critic_model": critic_model,
            "judge_mode": judge_mode,
            "scores_json": (json.dumps({d: v["score"] for d, v in scores.items()})
                            if scores else None),
            "reasoning_json": (json.dumps({d: v["reasoning"] for d, v in scores.items()})
                               if scores else None),
            "raw_response": (raw or "")[:4000],
            "telemetry": json.dumps(telem) if telem else None,
            "error": None,
        }
    except Exception as e:
        return {
            "paper_id": paper["paper_id"], "idea_model": idea["idea_model"],
            "track": idea["track"], "idea_index": idea["idea_index"],
            "critic_model": critic_model, "judge_mode": judge_mode,
            "scores_json": None, "reasoning_json": None,
            "raw_response": None, "telemetry": None, "error": str(e)[:500],
        }


def persist(conn: sqlite3.Connection, row: dict) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO dynamic_critic_sweep
          (paper_id, idea_model, track, idea_index, critic_model, judge_mode,
           scores_json, reasoning_json, raw_response, telemetry, error, created_at)
        VALUES (?,?,?,?,?,?, ?,?,?,?,?,?)
    """, (
        row["paper_id"], row["idea_model"], row["track"], row["idea_index"],
        row["critic_model"], row["judge_mode"],
        row["scores_json"], row["reasoning_json"],
        row["raw_response"], row["telemetry"], row["error"], ts,
    ))
    conn.commit()


def check_idea_coverage(db_results: str) -> None:
    """Print how many sweep ideas exist per IDEA_MODEL × cited_refs status."""
    conn = sqlite3.connect(db_results)
    print("\n## Main results.results coverage (Qwen v2 rerun, no Cited footer)")
    print(f"{'model':<35} {'total':>6} {'with_cited':>11}")
    for m in IDEA_MODELS:
        total = conn.execute(
            "SELECT COUNT(*) FROM results WHERE critic_model='' AND idea_model=?",
            (m,)).fetchone()[0]
        with_cites = conn.execute("""
            SELECT COUNT(*) FROM results WHERE critic_model='' AND idea_model=?
              AND cited_refs IS NOT NULL AND cited_refs != '[]'
        """, (m,)).fetchone()[0]
        print(f"  {m:<35} {total:>6} {with_cites:>11}")
    print("\n## Sweep A ideas (regenerated with REQUIRE_CITES=True)")
    print(f"{'model':<35} {'total':>6} {'with_cited':>11}")
    for m in IDEA_MODELS:
        total = conn.execute(
            "SELECT COUNT(*) FROM dynamic_critic_sweep_ideas WHERE idea_model=?",
            (m,)).fetchone()[0]
        with_cites = conn.execute("""
            SELECT COUNT(*) FROM dynamic_critic_sweep_ideas WHERE idea_model=?
              AND cited_refs IS NOT NULL AND cited_refs != '[]'
        """, (m,)).fetchone()[0]
        print(f"  {m:<35} {total:>6} {with_cites:>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="5 papers × N idea_model × 3 critic × 3 judge_mode")
    ap.add_argument("--check", action="store_true",
                    help="Just print sweep idea coverage and exit")
    ap.add_argument("--regen-ideas", action="store_true",
                    help="Phase 1: regenerate ideas with REQUIRE_CITES=True")
    ap.add_argument("--score", action="store_true",
                    help="Phase 2: score sweep ideas under 3 judge_modes")
    ap.add_argument("--n-ideas", type=int, default=3)
    ap.add_argument("--n-papers", type=int, default=25,
                    help="Cap papers (default 25 = plan.md scope; ignored if --smoke)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    db_results = str(cfg.RESULTS_DB)
    db_papers = str(cfg.PAPERS_DB)

    init_sweep_table(db_results)

    if args.check:
        check_idea_coverage(db_results)
        return

    papers = fetch_papers(db_papers, smoke=args.smoke, n_papers=args.n_papers)
    print(f"papers: {len(papers)}")

    # Phase 1: regenerate ideas with Cited footer (if requested or implied)
    if args.regen_ideas or (not args.score):
        n_new = regen_ideas(db_results, papers, n_ideas=args.n_ideas,
                            workers=args.workers)
        print(f"\n✓ Idea regen done: {n_new} new rows")

    if args.regen_ideas and not args.score:
        return

    # Phase 2: scoring under 3 judge_modes
    ideas = fetch_sweep_ideas(db_results, list(papers.keys()))
    print(f"sweep ideas to score: {len(ideas)}")
    if not ideas:
        print("⚠ No sweep ideas yet. Re-run with --regen-ideas first.")
        return

    conn = sqlite3.connect(db_results, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")

    tasks = []
    for idea in ideas:
        for critic in CRITIC_MODELS:
            for mode in JUDGE_MODES:
                if mode == "dynamic_cited" and not idea.get("cited_refs"):
                    continue   # skip dynamic_cited when idea has no Cited footer
                if already_scored(conn, idea["paper_id"], idea["idea_model"],
                                  idea["track"], idea["idea_index"], critic, mode):
                    continue
                tasks.append((idea, critic, mode))

    total = len(tasks)
    print(f"to do: {total} (paper × idea × critic × mode), workers={args.workers}")
    if total == 0:
        print("Nothing to do — all combinations already scored.")
        return

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_one, papers, i, c, m, db_papers): (i, c, m)
                   for (i, c, m) in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            persist(conn, row)
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total - done) if done else 0
                print(f"  [{done}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

    print(f"\n✓ Done. {done} rows written. Total wall-clock: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
