"""
Pairwise Scorer

Compares two candidate hypotheses head-to-head. Used as a more discriminating
alternative to absolute scoring when models cluster near a ceiling (common in
Active Mode where all outputs look "specific").

Given (hypothesis_A, hypothesis_B, refs), the critic must pick a winner for:
  1. Originality — which is more novel beyond the refs?
  2. Feasibility — which has a clearer experimental path?
  3. Specificity — which names concrete mechanisms (not just technical terms)?
  4. Overall — all things considered

To mitigate position bias, each pair is evaluated twice (A-first then B-first).
"""

import json
import logging
import re
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_PAIRWISE = """\
You are a senior scientific reviewer. You will compare two research hypotheses \
(A and B) proposed in the same field, given the same background literature. \
For each criterion, you MUST pick a winner — no ties allowed.

Use chain-of-thought reasoning: first analyze each criterion separately, then \
pick the overall winner.

COHERENCE CHECK: If one proposal names specific techniques (e.g. "AlphaFold", \
"Loihi 2", "CsV3Sb5") without logically justifying WHY those components are \
needed, treat it as keyword stuffing and prefer the more coherently motivated \
proposal — even if it uses fewer specific terms.

Criteria:
  1. ORIGINALITY — which proposal goes further beyond the background literature?
  2. FEASIBILITY — which has a clearer and more realistic experimental path?
  3. SPECIFICITY — which names coherent, justified mechanisms (not just \
technical vocabulary)?
  4. OVERALL — taking all criteria together, which is the stronger hypothesis?

Output JSON only:
{
  "originality_winner": "A" or "B",
  "feasibility_winner": "A" or "B",
  "specificity_winner": "A" or "B",
  "overall_winner":     "A" or "B",
  "reasoning": "<2-3 sentences explaining the overall choice>"
}"""


USER_TEMPLATE_PAIRWISE = """\
Field: {domain}

Background literature:
{references}

Proposal A:
---
{idea_a}
---

Proposal B:
---
{idea_b}
---

Compare them and return ONLY the JSON object."""


JSON_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. "
    "Return ONLY valid JSON — no markdown, no prose, just the raw JSON object."
)


def _extract_json(text: str) -> Optional[Dict]:
    """Try several strategies to parse JSON from critic output."""
    if not text:
        return None
    # Strategy 1: raw parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        pass
    # Strategy 2: ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Strategy 3: first {...} block (greedy to handle nested)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _normalize_winner(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw = str(raw).strip().upper()
    if raw.startswith("A") or raw == "PROPOSAL A":
        return "A"
    if raw.startswith("B") or raw == "PROPOSAL B":
        return "B"
    return None


def compare_pair(idea_a: str, idea_b: str, critic_model: str,
                 domain: str = "", references: str = "",
                 max_retries: int = 3, retry_delay: float = 2.0
                 ) -> Tuple[Optional[Dict], str, Dict]:
    """Run one pairwise comparison (single order: A then B).

    Returns:
        (winners_dict, raw_response, telemetry)
        winners_dict: {"originality": "A"|"B", "feasibility": ..., "specificity": ...,
                       "overall": ..., "reasoning": str}
        telemetry:    {latency_ms, usage, attempts}
    """
    from utils.LLM import CriticLLM
    llm = CriticLLM(model_name=critic_model)

    prompt = USER_TEMPLATE_PAIRWISE.format(
        domain=domain or "General Science",
        references=references or "(no background literature available)",
        idea_a=(idea_a or "").strip(),
        idea_b=(idea_b or "").strip(),
    )
    raw = ""
    telemetry = {"latency_ms": None, "usage": None, "attempts": 0}

    for attempt in range(max_retries):
        telemetry["attempts"] = attempt + 1
        try:
            resp = llm.score_idea(
                prompt + (JSON_RETRY_SUFFIX if attempt > 0 else ""),
                system_prompt=SYSTEM_PROMPT_PAIRWISE,
            )
            telemetry["latency_ms"] = getattr(llm, "last_latency_ms", None)
            telemetry["usage"] = getattr(llm, "last_usage", None)

            if isinstance(resp, tuple):
                raw = f"<reasoning>{resp[1]}</reasoning>\n{resp[0]}"
                resp_text = resp[0]
            else:
                raw = resp
                resp_text = resp

            parsed = _extract_json(resp_text)
            if parsed:
                out = {}
                for key in ("originality", "feasibility", "specificity", "overall"):
                    winner = _normalize_winner(parsed.get(f"{key}_winner"))
                    if winner:
                        out[key] = winner
                if "overall" in out:
                    out["reasoning"] = parsed.get("reasoning", "")[:500]
                    return out, raw, telemetry

            logger.warning(f"  pairwise parse failed (attempt {attempt+1})")
            time.sleep(retry_delay)

        except Exception as e:
            logger.error(f"  pairwise error (attempt {attempt+1}): {e}")
            time.sleep(retry_delay)

    logger.error(f"  pairwise all retries failed for {critic_model!r}")
    return None, raw, telemetry


def resolve_pair(idea_a: str, idea_b: str, critic_model: str,
                 domain: str = "", references: str = "",
                 max_retries: int = 3
                 ) -> Dict:
    """Run pairwise BOTH directions to de-bias position.

    Returns:
        {
          "overall":    "A" | "B" | "split",     # split if A/B swap disagree
          "originality": same,
          "feasibility": same,
          "specificity": same,
          "raw_forward":  raw text of A-first call,
          "raw_swap":     raw text of B-first call,
        }
    """
    # Forward order
    fwd, fwd_raw, fwd_tel = compare_pair(idea_a, idea_b, critic_model,
                                          domain=domain, references=references,
                                          max_retries=max_retries)
    # Swap order
    swp, swp_raw, swp_tel = compare_pair(idea_b, idea_a, critic_model,
                                          domain=domain, references=references,
                                          max_retries=max_retries)

    # Combine telemetry: latency = sum (two calls), usage = element-wise sum
    combined_tel = {
        "latency_ms": (fwd_tel.get("latency_ms") or 0) + (swp_tel.get("latency_ms") or 0),
        "attempts": (fwd_tel.get("attempts") or 0) + (swp_tel.get("attempts") or 0),
        "forward": fwd_tel,
        "swap": swp_tel,
    }

    result = {"raw_forward": fwd_raw[:2000], "raw_swap": swp_raw[:2000],
              "telemetry": combined_tel}

    for key in ("originality", "feasibility", "specificity", "overall"):
        f = fwd.get(key) if fwd else None
        s = swp.get(key) if swp else None
        if f is None and s is None:
            result[key] = None
        elif f is None:
            # swp: swap's 'A' corresponds to original B; flip
            result[key] = "A" if s == "B" else "B"
        elif s is None:
            result[key] = f
        else:
            # Map swap's letter back to original positions
            s_original = "A" if s == "B" else "B"
            if f == s_original:
                result[key] = f
            else:
                result[key] = "split"

    return result
