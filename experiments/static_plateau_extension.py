"""Extend Static (Track B) to 9 new open-source models to strengthen plateau finding.

Strategy: add data points in cutoff range 2025-Q1 to 2025-Q4 (the plateau region),
expecting Static scores to cluster around the asymptote ~6.25 — strengthening the
quadratic concavity signal and the late-half slope=−0.13 plateau evidence.

Models (per user 2026-05-19 校对):
  cutoff-disclosed (go into both cutoff + release axes):
    qwen/qwen3-30b-a3b-instruct-2507    cutoff 2025-04, release 2025-07-28
    qwen/qwen3-30b-a3b-thinking-2507    cutoff 2025-04, release 2025-07-29
    z-ai/glm-4.5-air                    cutoff 2025-03, release 2025-07-28
    mistralai/mistral-medium-3.1        cutoff 2025-05, release 2025-08-12
    mistralai/devstral-medium           cutoff 2025-05, release 2025-07-10
  cutoff-unknown (release axis only):
    qwen/qwen3-coder-480b-a35b-instruct release 2025-07-22
    moonshotai/kimi-dev-72b             release 2025-06-16
    minimax/minimax-m1                  release 2025-06-17
    z-ai/glm-4.6                        release 2025-09-30

Reuses generation/generate_ideas.run_generation (Track B, paragraph format,
prompt_version=v1_paper_refs) and evaluation/critic_manager.run_evaluation.

Usage:
  python experiments/static_plateau_extension.py --phase gen   --smoke
  python experiments/static_plateau_extension.py --phase critic --smoke
  python experiments/static_plateau_extension.py --phase all   --smoke
  python experiments/static_plateau_extension.py --phase all          # full 25 paper
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TARGET_MODELS = [
    "qwen/qwen3-30b-a3b-instruct-2507",
    "z-ai/glm-4.5-air",
    "mistralai/mistral-medium-3.1",
    "mistralai/devstral-medium",
    "qwen/qwen3-coder",  # = qwen3-coder-480b-a35b per OR catalog
    "minimax/minimax-m1",
    "z-ai/glm-4.6",
    # DROPPED 2026-05-19:
    #   qwen/qwen3-30b-a3b-thinking-2507  → reasoning param mandatory, all 390 calls 400'd
    #   moonshotai/kimi-dev-72b           → not in OpenRouter catalog
]


def phase_gen(models, smoke):
    from generation.generate_ideas import run_generation
    for m in models:
        logger.info(f"=== gen Static for {m} (smoke={smoke}) ===")
        try:
            run_generation(
                idea_model=m,
                db_papers=str(cfg.PAPERS_DB),
                db_results=str(cfg.RESULTS_DB),
                smoke=smoke,
                max_workers=cfg.PARALLEL.get("phase2_max_workers", 4),
            )
        except Exception as e:
            logger.error(f"GEN FAIL {m}: {e}")


def phase_critic(models, smoke):
    from evaluation.critic_manager import run_evaluation
    for m in models:
        logger.info(f"=== critic Static for {m} (smoke={smoke}) ===")
        try:
            run_evaluation(
                idea_model=m,
                db_papers=str(cfg.PAPERS_DB),
                db_results=str(cfg.RESULTS_DB),
                critic_pool=cfg.CRITIC_MODELS,
                n_critics=cfg.NUM_CRITICS_PER_EVAL,
                smoke=smoke,
                max_workers=cfg.PARALLEL.get("phase3_max_workers", 12),
                judge_mode="static",
            )
        except Exception as e:
            logger.error(f"CRITIC FAIL {m}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "critic", "all"], default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="1 paper per domain (= 5 papers total). Default: full 25 paper.")
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    models = args.models or TARGET_MODELS
    logger.info(f"Models ({len(models)}): {models}")
    logger.info(f"Mode: {'smoke' if args.smoke else 'full'}")

    if args.phase in ("gen", "all"):
        phase_gen(models, smoke=args.smoke)
    if args.phase in ("critic", "all"):
        phase_critic(models, smoke=args.smoke)


if __name__ == "__main__":
    main()
