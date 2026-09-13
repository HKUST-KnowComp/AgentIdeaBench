"""
Context Preparation — gt_hypothesis generation  (adapted from IdeaBench generate_summaries.py)

For each filtered paper in papers.db:
  1. Reads the paper's abstract.
  2. Calls GPT-4o to rewrite it as a "ground-truth hypothesis" (blinded form —
     strips concrete results so it reads like a research proposal, not a paper).
  3. Writes the result back to papers.db as gt_hypothesis.

This is the "one-time build" step; skip already-filled rows automatically.

Usage:
    python data_collection/prepare_context.py [--db path/to/papers.db] [--smoke]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sqlite3
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime import get_thread_sqlite_connection

# ---------------------------------------------------------------------------
# Prompt template (mirrors IdeaBench's SummaryGenerator prompt, generalised
# beyond biomedical to cover CS / Physics / Chemistry / Medicine / Biology)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PARAGRAPH = (
    "You are a scientific writing assistant. "
    "Your task is to rewrite a paper abstract as a research hypothesis or proposal. "
    "Write in first-person future tense ('We propose...', 'We hypothesize...', "
    "'Given that...'). "
    "Do NOT include specific numerical results, p-values, or benchmark scores. "
    "Keep the core research idea, motivation, and high-level approach. "
    "Output exactly one paragraph, beginning with 'Hypothesis: ' or 'Given that '."
)

USER_TEMPLATE_PARAGRAPH = """\
Rewrite the following abstract as a research hypothesis paragraph.

Abstract:
{abstract}

Hypothesis:"""

SYSTEM_PROMPT_STRUCTURED = (
    "You are a scientific writing assistant. "
    "Your task is to rewrite a paper abstract as a structured research proposal. "
    "Write in first-person future tense. "
    "Do NOT include specific numerical results, p-values, or benchmark scores. "
    "Keep the core research idea, motivation, methodology, and high-level approach. "
    "Output exactly eight sections with the headers shown below."
)

USER_TEMPLATE_STRUCTURED = """\
Rewrite the following abstract as a structured research proposal with these eight sections:

## Hypothesis (80-120 words)
The core research hypothesis in future tense.

## Experimental Design (120-200 words)
The primary methodology: variables, techniques, data, and controls.

## Expected Outcomes (60-100 words)
Predicted results and falsification criteria.

## Contingency Plan (60-100 words)
Backup approach if the primary experiment fails.

## Broader Implications (40-80 words)
Downstream consequences if the hypothesis is confirmed.

## Assumption Audit (80-120 words)
Identify 2-3 implicit assumptions in the research approach and why they might be wrong.

## Potential Confounds & Controls (60-100 words)
Identify likely confounding variables and how to control for them.

## Quantitative Predictions (60-100 words)
Predict specific numerical outcomes the hypothesis implies.

Abstract:
{abstract}"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _prepare_one_context(db_path: str, model: str,
                         paper_id: str, abstract: str,
                         delay: float = 0.0) -> dict:
    """Generate and persist one gt_hypothesis row."""
    import config as cfg
    from utils.LLM import BaseLLM

    if not abstract.strip():
        return {"paper_id": paper_id, "status": "skipped", "chars": 0}

    structured = getattr(cfg, "GENERATION_FORMAT", "paragraph") == "structured"

    if structured:
        system = SYSTEM_PROMPT_STRUCTURED
        user = USER_TEMPLATE_STRUCTURED
    else:
        system = SYSTEM_PROMPT_PARAGRAPH
        user = USER_TEMPLATE_PARAGRAPH

    llm = BaseLLM(model_name=model)
    prompt = user.format(abstract=abstract.strip())
    response = llm.completion(prompt, system_prompt=system)

    if isinstance(response, tuple):
        hypothesis = response[0]
    else:
        hypothesis = response

    hypothesis = hypothesis.strip()
    if not structured:
        if not (hypothesis.startswith("Hypothesis:") or
                hypothesis.lower().startswith("given that")):
            hypothesis = "Hypothesis: " + hypothesis

    conn = get_thread_sqlite_connection(db_path, timeout=30, enable_wal=True)
    cur = conn.cursor()
    cur.execute(
        "UPDATE papers SET gt_hypothesis=? WHERE paper_id=?",
        (hypothesis, paper_id)
    )
    conn.commit()

    if delay > 0:
        time.sleep(delay)

    return {"paper_id": paper_id, "status": "processed", "chars": len(hypothesis)}


def prepare_context(db_path: str, model: str = "openai/gpt-4o",
                    smoke: bool = False, delay: float = 1.0,
                    max_workers: int = 1) -> dict:
    """Generate gt_hypothesis for all filtered papers without one.

    Args:
        db_path:  Path to papers.db.
        model:    LLM model name for rewriting (default: gpt-4o via OpenRouter).
        smoke:    If True, process only the first 3 papers (smoke test).
        delay:    Seconds to sleep between API calls (rate-limit courtesy).

    Returns:
        dict with keys: processed, skipped, errors.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        "SELECT paper_id, abstract FROM papers "
        "WHERE status = 'filtered' AND (gt_hypothesis IS NULL OR gt_hypothesis = '')"
    )
    rows = cur.fetchall()

    if smoke:
        rows = rows[:3]
        logger.info(f"Smoke mode — processing {len(rows)} papers")
    else:
        logger.info(f"prepare_context: {len(rows)} papers to process")

    stats = {"processed": 0, "skipped": 0, "errors": 0}
    conn.close()

    effective_delay = delay if max_workers <= 1 else 0.0
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {
            executor.submit(
                _prepare_one_context,
                db_path,
                model,
                row["paper_id"],
                row["abstract"] or "",
                effective_delay,
            ): row["paper_id"]
            for row in rows
        }
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                result = future.result()
                if result["status"] == "processed":
                    stats["processed"] += 1
                    logger.info(f"  ✓ {paper_id} ({result['chars']} chars)")
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                    logger.warning(f"  SKIP {paper_id}: empty abstract")
            except Exception as e:
                logger.error(f"  ✗ {paper_id}: {e}")
                stats["errors"] += 1

    logger.info(
        f"prepare_context done: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['errors']} errors"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg

    parser = argparse.ArgumentParser(description="Generate gt_hypothesis for filtered papers")
    parser.add_argument("--db",    default=str(cfg.PAPERS_DB), help="Path to papers.db")
    parser.add_argument("--model", default="openai/gpt-4o",    help="LLM model for rewriting")
    parser.add_argument("--smoke", action="store_true",        help="Process first 3 papers only")
    parser.add_argument("--delay", type=float, default=1.0,    help="Seconds between API calls")
    parser.add_argument("--max-workers", type=int,
                        default=int(cfg.PARALLEL.get("phase1_context_max_workers", 4)),
                        help="Concurrent paper workers")
    args = parser.parse_args()

    stats = prepare_context(
        args.db,
        model=args.model,
        smoke=args.smoke,
        delay=args.delay,
        max_workers=args.max_workers,
    )
    print(f"\nDone: {stats['processed']} processed, {stats['skipped']} skipped, {stats['errors']} errors")
