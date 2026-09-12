"""Rerun Static-Mode cross-year sweep with the corrected prompt.

Background:
  generation/generate_ideas.py Static-Mode (Track B) prompts used to phrase
  the task as "synthesize / extend the background papers", turning idea
  generation into summarization. After the 2026-05-14 prompt rewrite the
  Track B prompt now asks the model to "propose a novel hypothesis in the
  field; refs are context for what NOT to repeat, not targets to extend".

  This script reruns the cross-year nspindle plot with the new prompt:
    - 25 test papers (top-5 per domain, same set the older sweep used)
    - 22 cross-year models from reports/_make_cross_year_plot.YEAR_GROUPS
      (claude-3.7-sonnet excluded — OR 404)
    - n_ideas = 3 per (paper, model), best_idx chosen by Phase 4 aggregation
    - n_critics = 3 per (paper, model), drawn from the current 5-model pool

Steps:
  1. select 25 papers (top-5 / domain by domain, paper_id sort)
  2. DELETE results + model_scores rows for (those papers, those models, B)
  3. Phase 2: parallel _generate_single_idea_task → results.db
  4. Phase 3: parallel score_combo (3 critics / combo, judge_mode=static)
  5. Phase 4: compute_scores → model_scores
  6. Regenerate cross_year_scatter / cross_year_static / per_dim plots

Usage:
  python experiments/rerun_cross_year_static.py --smoke   # 1 paper × 2 model sanity
  python experiments/rerun_cross_year_static.py           # full sweep
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

from reports._make_cross_year_plot import YEAR_GROUPS  # noqa: E402
from generation.generate_ideas import _generate_single_idea_task  # noqa: E402
from evaluation.critic_manager import (  # noqa: E402
    select_critics, score_combo, persist_rows,
)
from analysis.compute_scores import compute_scores  # noqa: E402

DB_PAPERS = str(cfg.PAPERS_DB)
DB_RESULTS = str(cfg.RESULTS_DB)

CROSS_YEAR_MODELS: List[str] = [
    mid for ylist in YEAR_GROUPS.values() for mid, _ in ylist
    if mid != "anthropic/claude-3.7-sonnet"
]

N_PAPERS_PER_DOMAIN = 5  # matches the 25-paper bench (5 domains × 5 papers)
N_IDEAS_PER_COMBO = 3
N_CRITICS_PER_COMBO = 3
TRACK = "B"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rerun_cross_year")


def select_papers() -> List[dict]:
    conn = sqlite3.connect(DB_PAPERS)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM papers WHERE status='filtered' "
        "AND gt_hypothesis IS NOT NULL AND gt_hypothesis != '' "
        "ORDER BY domain, paper_id"
    ).fetchall()
    conn.close()
    by_domain: dict = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(dict(r))
    selected = []
    for d in sorted(by_domain.keys()):
        selected.extend(by_domain[d][:N_PAPERS_PER_DOMAIN])
    return selected


def clear_old_rows(paper_ids: List[str], models: List[str]) -> tuple:
    conn = sqlite3.connect(DB_RESULTS)
    ph_p = ",".join("?" * len(paper_ids))
    ph_m = ",".join("?" * len(models))
    args = paper_ids + models
    n1 = conn.execute(
        f"DELETE FROM results WHERE track='{TRACK}' "
        f"AND paper_id IN ({ph_p}) AND idea_model IN ({ph_m})",
        args,
    ).rowcount
    n2 = conn.execute(
        f"DELETE FROM model_scores WHERE track='{TRACK}' "
        f"AND paper_id IN ({ph_p}) AND idea_model IN ({ph_m})",
        args,
    ).rowcount
    conn.commit()
    conn.close()
    return n1, n2


def phase2_generate(papers: List[dict], models: List[str]) -> None:
    # Build the full Cartesian set, then filter to ONLY rows that are
    # missing or have idea_text starting with '[generation error'. This
    # avoids the previous bug where every (paper, model, idx) sent a fresh
    # LLM call even when a success row already existed, wasting tokens and
    # — through INSERT OR REPLACE — clobbering raw_data (CLAUDE.md §9.5).
    conn = sqlite3.connect(DB_RESULTS)
    paper_ids = [p["paper_id"] for p in papers]
    placeholders_p = ",".join("?" * len(paper_ids))
    placeholders_m = ",".join("?" * len(models))
    existing = {
        (pid, m, idx): txt
        for pid, m, idx, txt in conn.execute(
            f"SELECT paper_id, idea_model, idea_index, idea_text FROM results "
            f"WHERE critic_model='' AND track='{TRACK}' "
            f"AND paper_id IN ({placeholders_p}) AND idea_model IN ({placeholders_m})",
            paper_ids + models,
        )
    }
    conn.close()

    tasks = []
    skipped_ok = 0
    for p in papers:
        for m in models:
            for idx in range(1, N_IDEAS_PER_COMBO + 1):
                txt = existing.get((p["paper_id"], m, idx))
                if txt is None:
                    tasks.append((p, m, idx))    # missing → generate
                elif txt.startswith("[generation error"):
                    tasks.append((p, m, idx))    # error → retry
                else:
                    skipped_ok += 1              # success → skip
    logger.info(
        f"Phase 2: {len(papers)} papers × {len(models)} models × "
        f"{N_IDEAS_PER_COMBO} ideas = {len(papers)*len(models)*N_IDEAS_PER_COMBO} total "
        f"combos; {skipped_ok} already-success skipped; {len(tasks)} to run."
    )
    workers = cfg.PARALLEL.get("phase2_max_workers", 6)
    t0 = time.time()
    done = errs = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                _generate_single_idea_task, DB_RESULTS, p, m, TRACK, idx
            ): (p["paper_id"], m, idx)
            for p, m, idx in tasks
        }
        for fut in cf.as_completed(futs):
            done += 1
            try:
                res = fut.result()
                if res.get("had_error"):
                    errs += 1
            except Exception as e:
                errs += 1
                logger.error(f"  task error: {e}")
            if done % 25 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 0.01)
                eta = (len(tasks) - done) / max(rate, 0.01)
                logger.info(
                    f"  Phase 2 [{done}/{len(tasks)}] err={errs} "
                    f"({rate:.1f}/s, eta {eta:.0f}s)"
                )
    logger.info(f"Phase 2 done in {time.time() - t0:.0f}s ({errs} errors)")


def _critics_already_done(conn: sqlite3.Connection,
                          paper_id: str, idea_model: str,
                          n_ideas: int) -> set:
    """Return critic_model names that have scored all n_ideas non-error ideas
    for (paper_id, idea_model) Track B.

    Critics that scored only [generation error ...] placeholder ideas are
    NOT counted as done — they need to re-run against the newly regenerated
    real ideas. (Critic-idea timing mismatch — e.g. critic scored an older
    idea text that was later regenerated with new content — is left as
    accepted residue; the aggregate is computed across all critics for that
    (paper, idea_index) cell so the bias is bounded.)
    """
    rows = conn.execute(
        "SELECT r.critic_model FROM results r "
        "JOIN results s ON s.paper_id=r.paper_id AND s.idea_model=r.idea_model "
        "    AND s.track=r.track AND s.idea_index=r.idea_index AND s.critic_model='' "
        "WHERE r.paper_id=? AND r.idea_model=? AND r.track=? "
        "  AND r.scores_json IS NOT NULL "
        "  AND s.idea_text NOT LIKE '[generation error%' "
        "GROUP BY r.critic_model "
        "HAVING COUNT(DISTINCT r.idea_index) >= ?",
        (paper_id, idea_model, TRACK, n_ideas),
    ).fetchall()
    return {r[0] for r in rows}


def phase3_score(papers: List[dict], models: List[str]) -> None:
    # Drop kimi-k2.6 and minimax-m2.7 from this rerun's critic pool.
    # Both need thinking ON to respond at all (4000-token reasoning trace
    # required), and each takes ~30-180s/call — they dominate worker time
    # and cut total throughput by ~3-4x. Other 3 critics (qwen3.6-plus,
    # glm-5.1, deepseek-v4-flash) run thinking-OFF at ~10s/call.
    # config.json's official critic_models list is unchanged.
    slow_critics = {"moonshotai/kimi-k2.6", "minimax/minimax-m2.7"}
    critic_pool = [m for m in cfg.CRITIC_MODELS if m not in slow_critics]
    logger.info(
        f"Phase 3: critic pool (kimi excluded) = {critic_pool}, "
        f"n_critics_per_combo = {N_CRITICS_PER_COMBO}"
    )

    tasks = []
    skipped_critics = 0
    for p in papers:
        for m in models:
            conn = sqlite3.connect(DB_RESULTS)
            conn.row_factory = sqlite3.Row
            ideas = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM results WHERE paper_id=? AND idea_model=? "
                    "AND track=? AND critic_model=''",
                    (p["paper_id"], m, TRACK),
                ).fetchall()
            ]
            if not ideas:
                conn.close()
                logger.warning(f"  no ideas for {p['paper_id'][:12]} × {m}")
                continue
            seed = abs(hash((p["paper_id"], m))) % (10 ** 9)
            critics = select_critics(
                m, critic_pool, n=N_CRITICS_PER_COMBO, seed=seed,
            )
            done_critics = _critics_already_done(
                conn, p["paper_id"], m, n_ideas=len(ideas)
            )
            conn.close()
            for c in critics:
                if c in done_critics:
                    skipped_critics += 1
                    continue
                tasks.append((p, m, ideas, c))
    logger.info(f"Phase 3: skipped {skipped_critics} already-scored critic combos")

    logger.info(f"Phase 3 total = {len(tasks)} critic calls")
    # 6 workers: balance between throughput and avoiding the CLOSE_WAIT
    # pile-up that hung the original 12-worker run.
    workers = min(cfg.PARALLEL.get("phase3_max_workers", 12), 6)
    t0 = time.time()
    done = scored = errs = 0
    conn_r = sqlite3.connect(DB_RESULTS, timeout=60)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                score_combo, p, m, TRACK, ideas, c,
                judge_mode="static", db_papers=DB_PAPERS,
            ): (p["paper_id"], m, c)
            for p, m, ideas, c in tasks
        }
        for fut in cf.as_completed(futs):
            done += 1
            try:
                res = fut.result()
                scored += res.get("scored", 0)
                errs += res.get("errors", 0)
                persist_rows(conn_r, res.get("rows", []))
            except Exception as e:
                errs += 1
                logger.error(f"  scoring task error: {e}")
            if done % 50 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 0.01)
                eta = (len(tasks) - done) / max(rate, 0.01)
                logger.info(
                    f"  Phase 3 [{done}/{len(tasks)}] scored={scored} err={errs} "
                    f"({rate:.1f}/s, eta {eta:.0f}s)"
                )
    conn_r.close()
    logger.info(
        f"Phase 3 done in {time.time() - t0:.0f}s: "
        f"{scored} scored, {errs} errors"
    )


def regenerate_plots() -> None:
    logger.info("Regenerating plots …")
    for script in [
        "reports/_make_cross_year_plot.py",
        "reports/_make_per_dim_plot.py",
    ]:
        if (ROOT / script).exists():
            subprocess.run([sys.executable, script], check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 paper × 2 models sanity run")
    ap.add_argument("--skip-delete", action="store_true",
                    help="Don't pre-delete existing rows (resume mode)")
    ap.add_argument("--skip-phase2", action="store_true")
    ap.add_argument("--skip-phase3", action="store_true")
    args = ap.parse_args()

    papers = select_papers()
    models = CROSS_YEAR_MODELS[:]

    if args.smoke:
        papers = papers[:1]
        models = models[:2]

    logger.info(
        f"Cross-year rerun: {len(papers)} papers × {len(models)} models, "
        f"track={TRACK}"
    )
    logger.info(f"Papers: {[p['paper_id'][:12]+' '+p['domain'] for p in papers]}")
    logger.info(f"Models: {models}")

    if not args.skip_delete:
        n1, n2 = clear_old_rows([p["paper_id"] for p in papers], models)
        logger.info(f"Cleared old rows: {n1} results, {n2} model_scores")

    if not args.skip_phase2:
        phase2_generate(papers, models)
    if not args.skip_phase3:
        phase3_score(papers, models)

    logger.info("Phase 4: compute_scores")
    stats = compute_scores(DB_RESULTS, DB_PAPERS)
    logger.info(f"  wrote {stats.get('rows_written', 0)} model_scores rows")

    regenerate_plots()

    logger.info("Done. Updated artifacts:")
    logger.info("  reports/cross_year_scatter.{pdf,png}")
    logger.info("  reports/cross_year_static.{pdf,png,json}")
    logger.info("  reports/per_dim_*.{pdf,png}")


if __name__ == "__main__":
    main()
