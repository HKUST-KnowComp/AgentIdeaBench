"""
Semantic Overlap Scorer  (E1 metric)

Compares a model's best_idea against the gt_hypothesis using an LLM judge.
Returns a 1–10 conceptual overlap score.

Anchors:
  1–2:  Completely different direction
  3–4:  Same broad area, different specific question
  5–6:  Related — some shared concepts or mechanisms
  7–8:  Substantially similar core idea
  9–10: Nearly identical hypothesis

Output parsed from "Rating: X/10 | Reason: ..." format.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert scientific evaluator. "
    "You will compare two scientific hypotheses and rate how conceptually "
    "similar they are on a scale of 1 to 10."
)

USER_TEMPLATE = """\
Compare the following two scientific hypotheses and rate their conceptual overlap.

Hypothesis A (model-generated):
{idea}

Hypothesis B (reference):
{gt}

Scoring guide:
  1–2:  Completely different direction — different topic, mechanism, and goal
  3–4:  Same broad research area but different specific question
  5–6:  Related — some shared concepts, mechanisms, or systems
  7–8:  Substantially similar core idea — same mechanism and goal, different details
  9–10: Nearly identical — same specific hypothesis, mechanism, and prediction

Respond in EXACTLY this format (one line):
Rating: X/10 | Reason: <one sentence explaining the score>"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_rating(text: str) -> Optional[int]:
    """Extract the integer rating from 'Rating: X/10 | ...' or fallback patterns."""
    # Primary: "Rating: X/10"
    m = re.search(r'[Rr]ating\s*:\s*(\d+)\s*/\s*10', text)
    if m:
        return int(m.group(1))
    # Fallback: first standalone integer 1-10
    m = re.search(r'\b([1-9]|10)\b', text)
    if m:
        return int(m.group(1))
    return None


def _parse_reason(text: str) -> str:
    """Extract the reason clause."""
    m = re.search(r'[Rr]eason\s*:\s*(.+)', text)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_overlap(idea: str, gt_hypothesis: str,
                  judge_model: str,
                  max_retries: int = 2) -> Tuple[Optional[float], str, str]:
    """Compute semantic overlap between idea and gt_hypothesis.

    Returns:
        (score_0_to_1, raw_response, reason_str)
        score is None on failure.
    """
    from utils.LLM import BaseLLM

    if not idea.strip() or not gt_hypothesis.strip():
        return None, "", "empty input"

    llm = BaseLLM(model_name=judge_model)
    prompt = USER_TEMPLATE.format(idea=idea.strip(), gt=gt_hypothesis.strip())

    for attempt in range(max_retries):
        try:
            resp = llm.completion(prompt, system_prompt=SYSTEM_PROMPT)
            if isinstance(resp, tuple):
                resp = resp[0]

            rating = _parse_rating(resp)
            reason = _parse_reason(resp)

            if rating is not None:
                score = round(rating / 10.0, 4)
                return score, resp, reason

            logger.warning(f"  overlap parse failed (attempt {attempt+1})")

        except Exception as e:
            logger.error(f"  overlap error (attempt {attempt+1}): {e}")

    return None, "", "parse failed"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    import config as cfg

    idea = (
        "We hypothesize that applying step-level RL rewards with formal "
        "verification feedback will increase LLM mathematical reasoning accuracy."
    )
    gt = (
        "Hypothesis: We propose that integrating dense, step-level reinforcement "
        "signals derived from automated theorem provers into LLM training will "
        "substantially improve multi-step mathematical reasoning performance."
    )

    judge = cfg.CRITIC_MODELS[0]
    print(f"Judge: {judge!r}")
    score, raw, reason = score_overlap(idea, gt, judge)
    print(f"Score: {score}  ({int(score*10)}/10)")
    print(f"Reason: {reason}")
