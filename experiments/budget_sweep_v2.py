"""Sweep B — Active mode tool budget saturation.

For each model in IDEA_MODELS, run `run_active_agent` with budget ∈ BUDGETS
on the same paper set, record both the final score AND actual tool_calls used.

Writes to two new tables:
  budget_sweep_ideas  (idea generation + tool usage telemetry)
  budget_sweep_scores (critic verdicts per (idea, critic) — static judge)

Roster: a same-vendor Qwen3.5 size ladder plus a cross-family arm, so the
saturation claim can be stated across model families rather than within one.
See IDEA_MODELS below.

Usage:
    python experiments/budget_sweep_v2.py --check
    python experiments/budget_sweep_v2.py --smoke           # 2 paper × 3 model × 4 budget × 1 idea = 24
    python experiments/budget_sweep_v2.py --gen             # generate ideas only
    python experiments/budget_sweep_v2.py --score           # score generated ideas only
    python experiments/budget_sweep_v2.py                   # gen + score
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
    # Same-vendor size ladder (original 2026-05-17 sweep).
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.5-397b-a17b",
    # Cross-family arm added 2026-09-01: three non-Qwen open-weight backbones
    # spanning the leaderboard, chosen so no idea model is also a CRITIC_MODEL
    # (that would let a critic score its own output). Active totals and mean
    # turns from Table tab:leaderboard: kimi 6.17/9.8, gemma 5.49/5.0,
    # mimo 5.46/7.8, so the arm spans both quality and tool-use intensity.
    "moonshotai/kimi-k2.6",
    "google/gemma-4-31b-it",
    "xiaomi/mimo-v2.5",
]

# 20 added 2026-06-23 per user E9 sweep request; 2 added 2026-09-01 to resolve
# the flat 1->5 segment. Every entry is idempotent: a new budget or a new model
# only ever inserts new (paper, model, budget, idea_index) rows, and gen_phase
# skips any cell already present, so re-running never touches existing raw data.
BUDGETS = [1, 2, 5, 10, 15, 20]

CRITIC_MODELS = [
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5.1",
    "qwen/qwen3.6-plus",
]

# Table names are parameterised by --suffix so a re-run can land in a fresh
# pair of tables instead of mutating rows that are already published raw data.
# CLAUDE.md forbids DELETE on experiment rows; a new table is the sanctioned
# way to redo cells. Default suffix "" keeps the original names.
IDEAS_T = "budget_sweep_ideas"
SCORES_T = "budget_sweep_scores"


def _schema(ideas: str, scores: str) -> str:
    return SCHEMA_TMPL.format(ideas=ideas, scores=scores)


SCHEMA_TMPL = """
CREATE TABLE IF NOT EXISTS {ideas} (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        TEXT    NOT NULL,
    domain          TEXT    NOT NULL,
    idea_model      TEXT    NOT NULL,
    budget          INTEGER NOT NULL,
    idea_index      INTEGER NOT NULL,
    idea_text       TEXT,
    iters_used      INTEGER,
    n_tool_calls    INTEGER,
    n_turns         INTEGER,
    budget_nudge    INTEGER,
    final_prompt_tokens  INTEGER,
    final_total_tokens   INTEGER,
    error           TEXT,
    trace_json      TEXT,
    created_at      TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_{ideas}_unique
    ON {ideas} (paper_id, idea_model, budget, idea_index);

CREATE TABLE IF NOT EXISTS {scores} (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        TEXT    NOT NULL,
    idea_model      TEXT    NOT NULL,
    budget          INTEGER NOT NULL,
    idea_index      INTEGER NOT NULL,
    critic_model    TEXT    NOT NULL,
    scores_json     TEXT,
    reasoning_json  TEXT,
    raw_response    TEXT,
    telemetry       TEXT,
    error           TEXT,
    created_at      TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_{scores}_unique
    ON {scores} (paper_id, idea_model, budget, idea_index, critic_model);
"""


def init_tables(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(_schema(IDEAS_T, SCORES_T))
    conn.commit()
    conn.close()


def fetch_papers(db_papers: str, smoke: bool = False) -> list:
    conn = sqlite3.connect(db_papers)
    conn.row_factory = sqlite3.Row
    limit = "LIMIT 2" if smoke else "LIMIT 10"
    rows = conn.execute(f"""
        SELECT * FROM papers
        WHERE status='filtered' AND gt_hypothesis IS NOT NULL AND gt_hypothesis != ''
        ORDER BY paper_id {limit}
    """).fetchall()
    return [dict(r) for r in rows]


def gen_one(paper: dict, idea_model: str, budget: int, idea_index: int) -> dict:
    from generation.active_agent import run_active_agent
    try:
        out = run_active_agent(paper["domain"], idea_model, max_iters=budget)
        return {
            "paper_id": paper["paper_id"],
            "domain": paper["domain"],
            "idea_model": idea_model,
            "budget": budget,
            "idea_index": idea_index,
            "idea_text": out.get("hypothesis", ""),
            "iters_used": out.get("iters_used", 0),
            "n_tool_calls": out.get("n_tool_calls", 0),
            "n_turns": out.get("n_turns", 0),
            "budget_nudge": 1 if out.get("budget_nudge_used") else 0,
            "final_prompt_tokens": out.get("final_prompt_tokens"),
            "final_total_tokens": out.get("final_total_tokens"),
            "error": out.get("error"),
            "trace_json": json.dumps(out.get("trace", []))[:8000],
        }
    except Exception as e:
        return {
            "paper_id": paper["paper_id"], "domain": paper["domain"],
            "idea_model": idea_model, "budget": budget, "idea_index": idea_index,
            "idea_text": "", "iters_used": 0, "n_tool_calls": 0, "n_turns": 0,
            "budget_nudge": 0, "final_prompt_tokens": None,
            "final_total_tokens": None,
            "error": str(e)[:500], "trace_json": None,
        }


def persist_idea(conn: sqlite3.Connection, row: dict) -> bool:
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(f"""
        INSERT OR IGNORE INTO {IDEAS_T}
          (paper_id, domain, idea_model, budget, idea_index, idea_text,
           iters_used, n_tool_calls, n_turns, budget_nudge,
           final_prompt_tokens, final_total_tokens, error, trace_json, created_at)
        VALUES (?,?,?,?,?,?, ?,?,?,?, ?,?,?,?,?)
    """, (
        row["paper_id"], row["domain"], row["idea_model"], row["budget"],
        row["idea_index"], row["idea_text"],
        row["iters_used"], row["n_tool_calls"], row["n_turns"], row["budget_nudge"],
        row["final_prompt_tokens"], row["final_total_tokens"],
        row["error"], row["trace_json"], ts,
    ))
    conn.commit()
    return cur.rowcount > 0


def already_gen(conn: sqlite3.Connection, paper_id: str, idea_model: str,
                budget: int, idea_index: int) -> bool:
    row = conn.execute(f"""
        SELECT 1 FROM {IDEAS_T}
        WHERE paper_id=? AND idea_model=? AND budget=? AND idea_index=?
    """, (paper_id, idea_model, budget, idea_index)).fetchone()
    return row is not None


def gen_phase(db_results: str, papers: list, n_ideas: int, workers: int) -> int:
    conn = sqlite3.connect(db_results, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    tasks = []
    for p in papers:
        for m in IDEA_MODELS:
            for b in BUDGETS:
                for i in range(1, n_ideas + 1):
                    if already_gen(conn, p["paper_id"], m, b, i):
                        continue
                    tasks.append((p, m, b, i))
    total = len(tasks)
    print(f"\n## Gen: {total} active-agent calls (workers={workers})")
    if total == 0:
        return 0

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(gen_one, p, m, b, i): (p, m, b, i)
                   for (p, m, b, i) in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            persist_idea(conn, row)
            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total - done)
            print(f"  [{done}/{total}] {row['idea_model'].split('/')[-1]:>20} "
                  f"b={row['budget']:<2} tools={row['n_tool_calls']:<2} "
                  f"chars={len(row['idea_text']):<5} elapsed={elapsed:.0f}s eta={eta:.0f}s")
    return done


def score_one(papers_by_id: dict, idea: dict, critic_model: str,
              db_papers: str) -> dict:
    paper = papers_by_id[idea["paper_id"]]
    refs_text = _format_refs_for_judge(paper)
    try:
        scores, raw, telem = score_idea(
            idea["idea_text"], critic_model,
            domain=paper.get("domain", ""), references=refs_text,
            judge_mode="static", db_papers=db_papers,
        )
        return {
            "paper_id": paper["paper_id"], "idea_model": idea["idea_model"],
            "budget": idea["budget"], "idea_index": idea["idea_index"],
            "critic_model": critic_model,
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
            "budget": idea["budget"], "idea_index": idea["idea_index"],
            "critic_model": critic_model,
            "scores_json": None, "reasoning_json": None, "raw_response": None,
            "telemetry": None, "error": str(e)[:500],
        }


def persist_score(conn: sqlite3.Connection, row: dict) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(f"""
        INSERT OR IGNORE INTO {SCORES_T}
          (paper_id, idea_model, budget, idea_index, critic_model,
           scores_json, reasoning_json, raw_response, telemetry, error, created_at)
        VALUES (?,?,?,?,?, ?,?,?,?,?,?)
    """, (
        row["paper_id"], row["idea_model"], row["budget"], row["idea_index"],
        row["critic_model"], row["scores_json"], row["reasoning_json"],
        row["raw_response"], row["telemetry"], row["error"], ts,
    ))
    conn.commit()


def score_phase(db_results: str, db_papers: str, papers: list, workers: int) -> int:
    conn_r = sqlite3.connect(db_results, timeout=30)
    conn_r.execute("PRAGMA busy_timeout=30000")
    conn_r.row_factory = sqlite3.Row
    papers_by_id = {p["paper_id"]: p for p in papers}

    # Fetch sweep ideas that need scoring
    p_placeholders = ",".join("?" * len(papers))
    ideas = conn_r.execute(f"""
        SELECT paper_id, idea_model, budget, idea_index, idea_text
        FROM {IDEAS_T}
        WHERE paper_id IN ({p_placeholders}) AND idea_text != ''
    """, [p["paper_id"] for p in papers]).fetchall()
    ideas = [dict(r) for r in ideas]

    tasks = []
    for idea in ideas:
        for critic in CRITIC_MODELS:
            existing = conn_r.execute(f"""
                SELECT 1 FROM {SCORES_T}
                WHERE paper_id=? AND idea_model=? AND budget=? AND idea_index=?
                  AND critic_model=?
            """, (idea["paper_id"], idea["idea_model"], idea["budget"],
                  idea["idea_index"], critic)).fetchone()
            if existing:
                continue
            tasks.append((idea, critic))
    total = len(tasks)
    print(f"\n## Score: {total} critic calls (workers={workers})")
    if total == 0:
        return 0

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_one, papers_by_id, i, c, db_papers): (i, c)
                   for (i, c) in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            persist_score(conn_r, row)
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total - done)
                print(f"  [{done}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s")
    return done


def check(db_results: str) -> None:
    conn = sqlite3.connect(db_results)
    print(f"\n## {IDEAS_T} coverage")
    print(f"{'model':<30} {'budget':>7} {'n_ideas':>8} {'mean_tools':>11}")
    for m in IDEA_MODELS:
        for b in BUDGETS:
            r = conn.execute(f"""
                SELECT COUNT(*), AVG(n_tool_calls) FROM {IDEAS_T}
                WHERE idea_model=? AND budget=?
            """, (m, b)).fetchone()
            print(f"  {m.split('/')[-1]:<28} {b:>7} {r[0]:>8} "
                  f"{(r[1] or 0):>11.1f}")


def seed(db_results: str, src_suffix: str, op: str, ts: str) -> None:
    """Copy a time-slice of one table pair into the suffixed target pair.

    Pure INSERT: the source tables are never read-modify-written and never
    deleted from. Rows outside the slice are intentionally left behind, which
    is how a batch gets excluded from a redo without destroying it -- it stays
    in its source table as the historical record.

    op is "<" to keep rows before `ts` or ">=" to keep rows at or after it.
    """
    if op not in ("<", ">="):
        raise ValueError(f"op must be '<' or '>=', got {op!r}")
    src_i = f"budget_sweep_ideas{src_suffix}"
    src_s = f"budget_sweep_scores{src_suffix}"
    if src_i == IDEAS_T:
        raise ValueError("refusing to seed a table pair from itself")

    conn = sqlite3.connect(db_results, timeout=30)
    cols_i = "paper_id, domain, idea_model, budget, idea_index, idea_text, \
iters_used, n_tool_calls, n_turns, budget_nudge, final_prompt_tokens, \
final_total_tokens, error, trace_json, created_at"
    cols_s = "paper_id, idea_model, budget, idea_index, critic_model, \
scores_json, reasoning_json, raw_response, telemetry, error, created_at"
    n_i = conn.execute(
        f"INSERT OR IGNORE INTO {IDEAS_T} ({cols_i}) SELECT {cols_i} "
        f"FROM {src_i} WHERE created_at {op} ?", (ts,)).rowcount
    n_s = conn.execute(
        f"INSERT OR IGNORE INTO {SCORES_T} ({cols_s}) SELECT {cols_s} "
        f"FROM {src_s} WHERE created_at {op} ?", (ts,)).rowcount
    conn.commit()
    inv = ">=" if op == "<" else "<"
    left = conn.execute(
        f"SELECT COUNT(*) FROM {src_i} WHERE created_at {inv} ?",
        (ts,)).fetchone()[0]
    conn.close()
    print(f"seeded from {src_i} ({op} {ts}): {n_i} ideas, {n_s} scores")
    print(f"left behind in {src_i} (to be regenerated): {left} ideas")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gen", action="store_true", help="Run gen phase only")
    ap.add_argument("--score", action="store_true", help="Run score phase only")
    ap.add_argument("--n-ideas", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--suffix", default="",
                    help="write to budget_sweep_{ideas,scores}<suffix> instead "
                         "of the base tables; used to redo cells without ever "
                         "deleting a published row")
    ap.add_argument("--seed-src-suffix", default="",
                    help="suffix of the table pair to seed FROM (default: the "
                         "base tables)")
    ap.add_argument("--seed-before", metavar="TS",
                    help="copy source rows created before this ISO timestamp "
                         "into the suffixed tables, so an existing good cell "
                         "is not regenerated. Requires --suffix.")
    ap.add_argument("--seed-after", metavar="TS",
                    help="same, but keeps rows at or after the timestamp")
    args = ap.parse_args()

    global IDEAS_T, SCORES_T
    if args.suffix:
        IDEAS_T = f"budget_sweep_ideas{args.suffix}"
        SCORES_T = f"budget_sweep_scores{args.suffix}"
        print(f"tables: {IDEAS_T} / {SCORES_T}")

    db_results = str(cfg.RESULTS_DB)
    db_papers = str(cfg.PAPERS_DB)
    init_tables(db_results)

    if args.seed_before or args.seed_after:
        if not args.suffix:
            ap.error("--seed-before/--seed-after require --suffix")
        if args.seed_before and args.seed_after:
            ap.error("pass only one of --seed-before / --seed-after")
        op, ts = ((("<", args.seed_before)) if args.seed_before
                  else (">=", args.seed_after))
        seed(db_results, args.seed_src_suffix, op, ts)

    if args.check:
        check(db_results)
        return

    papers = fetch_papers(db_papers, smoke=args.smoke)
    n_ideas = 1 if args.smoke else args.n_ideas
    print(f"papers: {len(papers)} | budgets: {BUDGETS} | n_ideas/combo: {n_ideas}")

    do_gen = args.gen or not args.score
    do_score = args.score or not args.gen

    if do_gen:
        n = gen_phase(db_results, papers, n_ideas, args.workers)
        print(f"✓ gen done: {n} new rows")
    if do_score:
        n = score_phase(db_results, db_papers, papers, args.workers)
        print(f"✓ score done: {n} new rows")


if __name__ == "__main__":
    main()
