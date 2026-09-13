"""
Seed-variance experiment: measure within-model stochasticity vs between-model gap.

For each (model, paper, seed) triple we:
  1. Run Active Mode agent with that seed
  2. Score the output with ONE fixed critic (eliminates critic-pool randomness)
  3. Store hypothesis + 5-dim scores to JSON

Comparison: |397b_mean(paper) - 9b_mean(paper)|   vs   within-model sigma across seeds.
Ratio > 2 = signal clear. Ratio < 1 = pure noise, scaling papers won't help.

Output: experiments/seed_variance_results.json (idempotent — skips existing entries).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from generation.active_agent import run_active_agent
from generation.generate_ideas import _clean_idea_text
from evaluation.absolute_scorer import score_idea
from evaluation.critic_manager import _format_refs_for_judge

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("seed_variance")

MODELS = [
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3.5-9b",
]

# One paper per domain (diverse). Picked from filtered smoke-eligible set.
PAPER_IDS = [
    "050675de50181e735c187cb612b5f83e62d31cbe",  # CS - S4-Driver
    "05f31ad3176a5df162e4d041b8736bf123f4b628",  # Biology - kidney transplant
    "0142bf7cfaac534da831178cf27e49c31c16b70c",  # Physics - microwave photon detector
]

SEEDS = [42, 1337, 2024]

FIXED_CRITIC = "google/gemini-2.5-flash"

OUT_PATH = ROOT / "experiments" / "seed_variance_results.json"


def load_papers() -> dict:
    conn = sqlite3.connect(cfg.PAPERS_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join(["?"] * len(PAPER_IDS))
    rows = conn.execute(
        f"SELECT * FROM papers WHERE paper_id IN ({placeholders})",
        PAPER_IDS,
    ).fetchall()
    conn.close()
    return {r["paper_id"]: dict(r) for r in rows}


def load_existing() -> dict:
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            return json.load(f)
    return {"runs": []}


def save(state: dict) -> None:
    OUT_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def key(model: str, pid: str, seed: int) -> str:
    return f"{model}|{pid}|{seed}"


def main():
    papers = load_papers()
    missing = set(PAPER_IDS) - set(papers.keys())
    if missing:
        logger.error(f"missing papers: {missing}")
        sys.exit(1)

    state = load_existing()
    done = {key(r["model"], r["paper_id"], r["seed"]) for r in state["runs"]}
    logger.info(f"resuming with {len(done)} runs already complete")

    total = len(MODELS) * len(PAPER_IDS) * len(SEEDS)
    idx = 0

    for model in MODELS:
        for pid in PAPER_IDS:
            paper = papers[pid]
            domain = paper["domain"]
            refs_text = _format_refs_for_judge(paper)
            for seed in SEEDS:
                idx += 1
                k = key(model, pid, seed)
                if k in done:
                    logger.info(f"[{idx}/{total}] skip {k}")
                    continue
                logger.info(f"[{idx}/{total}] GEN {model} | {pid} | {domain} | seed={seed}")
                t0 = time.time()
                agent_out = run_active_agent(domain, model, max_iters=6, seed=seed)
                hyp = _clean_idea_text(agent_out.get("hypothesis") or "")
                gen_secs = round(time.time() - t0, 1)
                if not hyp or len(hyp.split()) < 30:
                    logger.warning(f"  empty/short hypothesis; err={agent_out.get('error')}")
                    state["runs"].append({
                        "model": model, "paper_id": pid, "domain": domain, "seed": seed,
                        "hypothesis": hyp, "iters_used": agent_out.get("iters_used", 0),
                        "error": agent_out.get("error") or "empty_hypothesis",
                        "scores": None, "gen_secs": gen_secs,
                    })
                    save(state)
                    continue

                logger.info(f"  SCORE ({FIXED_CRITIC})")
                t1 = time.time()
                scores, raw, _ = score_idea(
                    hyp, FIXED_CRITIC,
                    domain=domain, title="", references=refs_text,
                )
                score_secs = round(time.time() - t1, 1)

                state["runs"].append({
                    "model": model, "paper_id": pid, "domain": domain, "seed": seed,
                    "hypothesis": hyp, "iters_used": agent_out.get("iters_used", 0),
                    "error": None,
                    "scores": scores,
                    "gen_secs": gen_secs, "score_secs": score_secs,
                })
                save(state)
                ws = hyp.split()
                logger.info(f"  done: {len(ws)}w, iters={agent_out.get('iters_used')}, "
                            f"gen={gen_secs}s, score={score_secs}s")

    logger.info(f"all runs saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
