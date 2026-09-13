"""
SciSynthBench — Main Pipeline Entry Point

Runs the full pipeline or individual phases:

  Phase 1 (data):        fetch(SS) → filter → prepare_context → select refs
  Phase 2 (generation):  generate ideas (Track B only, all models + baselines)
  Phase 3 (evaluation):  5-dim absolute scoring (critic_manager)
  Phase 4 (analysis):    compute scores + leaderboard

Usage:
    python run.py --phase 1                           # full data collection
    python run.py --phase 1 --smoke                   # smoke (1 domain, 2 queries)
    python run.py --phase 1 --skip-fetch              # skip SS fetch
    python run.py --phase 1 --skip-embed              # skip embedding step
    python run.py --phase 2                           # all idea models
    python run.py --phase 2 --model qwen/qwen3.5-9b  # one model only
    python run.py --phase 2 --smoke                   # 1 paper per domain
    python run.py --phase 3 --model qwen/qwen3.5-9b  # evaluate one model
    python run.py --phase 4                           # compute scores + leaderboard
    python run.py --phase 1 2 3 4                     # full pipeline
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config as cfg
from utils.db_init import init_all

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _run_models_parallel(phase_name: str, models: list, worker_fn,
                         max_workers: int) -> None:
    """Run per-model work serially or in parallel with bounded concurrency."""
    if len(models) <= 1 or max_workers <= 1:
        for model in models:
            logger.info(f"  model: {model!r}")
            stats = worker_fn(model)
            logger.info(f"  → {stats}")
        return

    logger.info(f"{phase_name} ─ model-level parallelism enabled (max_workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_fn, model): model
            for model in models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                stats = future.result()
                logger.info(f"  model {model!r} → {stats}")
            except Exception as e:
                logger.error(f"  model {model!r} failed: {e}")


# ---------------------------------------------------------------------------
# Phase 1 — Data collection
# ---------------------------------------------------------------------------

def run_phase1(smoke: bool = False, domain: str = None,
               skip_fetch: bool = False,
               skip_filter: bool = False,
               skip_context: bool = False,
               skip_embed: bool = False) -> None:
    """Full data-collection pipeline.

    Steps:
      1-A  Semantic Scholar search (date-filtered, stratified sampling + real references)
      1-B  Quality filter (abstract length, pub type, ref count, date range)
      1-C  prepare_context: generate gt_hypothesis for filtered papers
      1-D  embed_references: rank refs by embedding similarity (for Track B)
    """

    # ── 0. Ensure DBs exist ──────────────────────────────────────────────
    logger.info("Phase 1 ─ initialising databases")
    init_all(str(cfg.PAPERS_DB), str(cfg.RESULTS_DB))

    # ── 1-A. Semantic Scholar search + real references ────────────────────
    if not skip_fetch:
        logger.info("Phase 1-A ─ Semantic Scholar search (stratified sampling + real references)")
        from data_collection.fetch_ss_search import fetch_all as ss_fetch
        stats = ss_fetch(cfg, smoke=smoke, domain_filter=domain)
        logger.info(f"  → {stats['inserted']} papers inserted, {stats['skipped']} skipped")
    else:
        logger.info("Phase 1-A skipped (--skip-fetch)")

    # ── 1-B. Quality filter ───────────────────────────────────────────────
    if not skip_filter:
        logger.info("Phase 1-B ─ filter_papers (abstract len / pub type / ref count / date)")
        from data_collection.filter_papers import filter_papers
        stats = filter_papers(str(cfg.PAPERS_DB), cfg)
        logger.info(f"  → {stats['passed']} passed, {stats['rejected']} rejected")
        if stats.get('reasons'):
            for reason, count in sorted(stats['reasons'].items(), key=lambda x: -x[1]):
                logger.info(f"    {reason}: {count}")
    else:
        logger.info("Phase 1-B skipped (--skip-filter)")

    # ── 1-C. prepare_context (gt_hypothesis) ─────────────────────────────
    if not skip_context:
        logger.info("Phase 1-C ─ prepare_context (gt_hypothesis generation)")
        from data_collection.prepare_context import prepare_context
        stats = prepare_context(
            str(cfg.PAPERS_DB),
            smoke=smoke,
            max_workers=cfg.PARALLEL.get("phase1_context_max_workers", 4),
        )
        logger.info(f"  → {stats['processed']} gt_hypothesis generated")
    else:
        logger.info("Phase 1-C skipped (--skip-context)")

    # ── 1-D. Embedding-based reference ranking (for Track B) ─────────────
    if not skip_embed:
        logger.info("Phase 1-D ─ embed_references (embedding similarity ranking)")
        from data_collection.embed_references import embed_references
        stats = embed_references(str(cfg.PAPERS_DB), cfg,
                                  smoke=smoke, domain_filter=domain,
                                  max_workers=cfg.PARALLEL.get("phase1_embed_max_workers", 4))
        logger.info(f"  → {stats['processed']} papers got ranked_refs_json")
    else:
        logger.info("Phase 1-D skipped (--skip-embed)")

    # ── Summary ──────────────────────────────────────────────────────────
    import sqlite3
    conn = sqlite3.connect(str(cfg.PAPERS_DB))
    rows = conn.execute(
        "SELECT domain, COUNT(*) as n FROM papers WHERE status='filtered' GROUP BY domain"
    ).fetchall()
    total = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE status='filtered'"
    ).fetchone()[0]
    gt_done = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE gt_hypothesis IS NOT NULL AND gt_hypothesis != ''"
    ).fetchone()[0]
    embed_done = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE ranked_refs_json IS NOT NULL AND ranked_refs_json != ''"
    ).fetchone()[0]
    conn.close()

    logger.info("─" * 50)
    logger.info("Phase 1 complete — papers.db summary:")
    for domain_name, n in rows:
        logger.info(f"  {domain_name:12s}: {n} papers")
    logger.info(f"  {'TOTAL':12s}: {total} papers")
    logger.info(f"  gt_hypothesis   : {gt_done} / {total}")
    logger.info(f"  ranked_refs     : {embed_done} / {total}")
    logger.info("─" * 50)


# ---------------------------------------------------------------------------
# Phase 2 — Idea generation
# ---------------------------------------------------------------------------

def run_phase2(models: list = None, smoke: bool = False,
               domain: str = None, active: bool = False) -> None:
    """Generate ideas for all (or specified) idea models.

    Routes baseline/* models to baselines.generate_baselines.
    Routes real LLM models to generation.generate_ideas (Static Mode, Track B).
    If active=True, runs Active Mode via generation.active_agent (Track C) —
    model gets only domain name + SS search tools.
    """
    from generation.generate_ideas import run_generation
    from baselines.generate_baselines import generate_baselines, ALL_BASELINES

    models = models or cfg.IDEA_MODELS
    mode_str = "Active" if active else "Static"
    logger.info(f"Phase 2 ({mode_str}) ─ generating ideas for {len(models)} model(s)")

    def _worker(model: str) -> dict:
        if active:
            from generation.active_agent import run_active_phase
            return run_active_phase(
                db_papers=str(cfg.PAPERS_DB),
                db_results=str(cfg.RESULTS_DB),
                idea_model=model,
                smoke=smoke,
            )
        if model in ALL_BASELINES or model.startswith("baseline/"):
            return generate_baselines(
                db_papers=str(cfg.PAPERS_DB),
                db_results=str(cfg.RESULTS_DB),
                smoke=smoke,
                idea_model=model,
            )
        return run_generation(
            idea_model=model,
            db_papers=str(cfg.PAPERS_DB),
            db_results=str(cfg.RESULTS_DB),
            smoke=smoke,
            domain_filter=domain,
            max_workers=cfg.PARALLEL.get("phase2_max_workers", 4),
        )

    _run_models_parallel(
        "Phase 2",
        models,
        _worker,
        int(cfg.PARALLEL.get("phase2_model_max_workers", 2)),
    )


# ---------------------------------------------------------------------------
# Phase 3 — Evaluation
# ---------------------------------------------------------------------------

def run_phase3(models: list = None, smoke: bool = False,
               domain: str = None,
               judge_mode: str = "static") -> None:
    """Score generated ideas with random critics.

    judge_mode: "static" (default), "dynamic_search", "dynamic_cited".
    See evaluation/dynamic_judge.py for details.
    """
    from evaluation.critic_manager import run_evaluation
    models = models or cfg.IDEA_MODELS
    logger.info(f"Phase 3 ─ evaluating {len(models)} model(s) [judge_mode={judge_mode}]")

    def _worker(model: str) -> dict:
        return run_evaluation(
            idea_model=model,
            db_papers=str(cfg.PAPERS_DB),
            db_results=str(cfg.RESULTS_DB),
            critic_pool=cfg.CRITIC_MODELS,
            n_critics=cfg.NUM_CRITICS_PER_EVAL,
            smoke=smoke,
            domain_filter=domain,
            max_workers=cfg.PARALLEL.get("phase3_max_workers", 12),
            judge_mode=judge_mode,
        )

    _run_models_parallel(
        "Phase 3",
        models,
        _worker,
        int(cfg.PARALLEL.get("phase3_model_max_workers", 2)),
    )


# ---------------------------------------------------------------------------
# Phase 4 — Analysis
# ---------------------------------------------------------------------------

def run_phase4(output_dir: str = None) -> None:
    """Aggregate scores and generate leaderboard."""
    from analysis.compute_scores import compute_scores
    from analysis.leaderboard import generate_leaderboard

    logger.info("Phase 4-A ─ computing aggregated scores")
    stats = compute_scores(str(cfg.RESULTS_DB), str(cfg.PAPERS_DB))
    logger.info(f"  → {stats.get('rows_written', 0)} model_score rows written")

    logger.info("Phase 4-B ─ generating leaderboard")
    out = output_dir or str(ROOT / "reports")
    generate_leaderboard(str(cfg.RESULTS_DB), out)
    logger.info(f"  → leaderboard written to {out}/")


# ---------------------------------------------------------------------------
# Phase 5 — Pairwise Comparison + Bradley-Terry
# ---------------------------------------------------------------------------

def run_phase5(smoke: bool = False, output_dir: str = None, track: str = "C") -> None:
    """Run pairwise comparisons between idea models for the given track,
    then aggregate to Bradley-Terry ranking + win-rate matrix."""
    from evaluation.pairwise_manager import run_pairwise
    from analysis.pairwise_aggregate import aggregate_pairwise

    logger.info(f"Phase 5-A ─ pairwise comparisons (track={track})")
    stats = run_pairwise(
        db_results=str(cfg.RESULTS_DB),
        db_papers=str(cfg.PAPERS_DB),
        critic_pool=cfg.CRITIC_MODELS,
        track=track,
        smoke=smoke,
        n_critics=1,
    )
    logger.info(f"  → pairwise stats: {stats}")

    logger.info("Phase 5-B ─ aggregating to BT ranking")
    out = output_dir or str(ROOT / "reports")
    aggregate_pairwise(db_results=str(cfg.RESULTS_DB), out_dir=out, track=track)
    logger.info(f"  → pairwise report written to {out}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SciSynthBench pipeline runner"
    )
    parser.add_argument("--phase",   type=int, nargs="+", default=[1],
                        help="Phase(s) to run: 1 2 3 4 5  (space-separated)")
    parser.add_argument("--smoke",   action="store_true",
                        help="Minimal run for testing")
    parser.add_argument("--model",   default=None,
                        help="Restrict Phase 2/3 to one idea model")
    parser.add_argument("--domain",  default=None,
                        help="Restrict to one domain")
    # Phase 1 specific
    parser.add_argument("--skip-fetch",    action="store_true")
    parser.add_argument("--skip-filter",   action="store_true")
    parser.add_argument("--skip-context",  action="store_true")
    parser.add_argument("--skip-embed",    action="store_true",
                        help="Skip Phase 1-D embedding reference ranking")
    # Phase 2 specific
    parser.add_argument("--active", action="store_true",
                        help="Phase 2: run Active Mode (agent w/ SS search tools) instead of Static Mode")
    parser.add_argument("--judge-mode", default="static",
                        choices=["static", "dynamic_search", "dynamic_cited"],
                        help="Phase 3: how the critic gets references. "
                             "static (default) = canonical survey refs from papers.db. "
                             "dynamic_search = critic searches SS using the hypothesis. "
                             "dynamic_cited = critic uses Cited: footer parsed from idea text.")
    parser.add_argument("--no-require-cites", dest="no_require_cites",
                        action="store_true",
                        help="Phase 2: DISABLE the default 'Cited:' footer "
                             "instruction (ablation runs). REQUIRE_CITES is "
                             "True by default — Cited footer is always asked "
                             "for unless this flag is passed.")
    parser.add_argument("--require-cites", action="store_true",
                        help="(legacy, default True) Kept for backwards compat. "
                             "No effect — Cited footer is on by default.")
    # Phase 4 specific
    parser.add_argument("--output", default=None,
                        help="Output dir for leaderboard (default: reports/)")
    args = parser.parse_args()

    models = [args.model] if args.model else None

    # REQUIRE_CITES defaults to True in generate_ideas.py. --no-require-cites
    # disables it for ablation runs.
    if args.no_require_cites:
        import generation.generate_ideas as _gen
        _gen.REQUIRE_CITES = False

    for phase in sorted(set(args.phase)):
        if phase == 1:
            run_phase1(
                smoke=args.smoke, domain=args.domain,
                skip_fetch=args.skip_fetch,
                skip_filter=args.skip_filter, skip_context=args.skip_context,
                skip_embed=args.skip_embed,
            )
        elif phase == 2:
            run_phase2(models=models, smoke=args.smoke, domain=args.domain, active=args.active)
        elif phase == 3:
            run_phase3(models=models, smoke=args.smoke, domain=args.domain,
                       judge_mode=args.judge_mode)
        elif phase == 4:
            run_phase4(output_dir=args.output)
        elif phase == 5:
            run_phase5(smoke=args.smoke, output_dir=args.output)
        else:
            parser.error(f"Unknown phase: {phase}")


if __name__ == "__main__":
    main()
