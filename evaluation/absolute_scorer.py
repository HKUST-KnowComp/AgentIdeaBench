"""
5-Dimension Absolute Scorer  (Phase 3-B)

Scores a scientific idea on five dimensions using a strict anchor rubric:
  Originality / Feasibility / Clarity / Impact / Specificity  (each 1–10)

The critic persona is a senior Nature/Science reviewer. Scores ≥8 require
genuinely exciting ideas; the anchors are calibrated to make 5–6 the "decent
but not great" range.

Output format (JSON):
  {
    "originality":  {"score": 7, "reasoning": "..."},
    "feasibility":  {"score": 8, "reasoning": "..."},
    "clarity":      {"score": 6, "reasoning": "..."},
    "impact":       {"score": 7, "reasoning": "..."},
    "specificity":  {"score": 8, "reasoning": "..."}
  }

Four-layer JSON parse fallback:
  1. ```json ... ``` block
  2. First { ... } block
  3. Regex key:score patterns
  4. Re-prompt with explicit JSON instruction (up to max_retries)
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.constants import SCORING_DIMS as DIMS

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior reviewer for Nature and Science with 20+ years of experience \
evaluating scientific ideas.

CRITICAL: Use the FULL 1-10 range and differentiate clearly. A weak proposal \
and a strong proposal must NOT receive similar scores. Score each dimension \
independently based on its own rubric — do not let one dimension influence \
another. A well-written but unoriginal proposal should score high on clarity \
but low on originality.

COHERENCE CHECK (applies to Originality and Specificity): Before scoring, \
identify every named technique / method / dataset / hardware in the proposal \
(e.g. "AlphaFold", "Loihi 2", "CsV3Sb5", "Chain-of-Thought", "RLHF"). For each, \
ask: "Is this component LOGICALLY NECESSARY for the hypothesis, or is it \
name-dropped to sound impressive?" If the proposal stacks specific terms \
without justifying why the combination is needed, treat it as KEYWORD STUFFING \
— cap Originality at 5 and Specificity at 6, regardless of how technical it \
sounds. Real novelty requires a coherent mechanism, not vocabulary density.

FACTUAL CONSISTENCY CHECK (applies to ALL dimensions): Before scoring, extract \
at least TWO specific factual claims from the proposal that would need to be \
scientifically TRUE for the hypothesis to make sense (e.g. "Weyl semimetals \
have 4D surface intersections", "Moore-Read Pfaffian supports fractional \
charge e/2", "method X scales linearly with system size"). For each claim, \
label it as one of: ESTABLISHED (widely accepted textbook fact), SPECULATIVE \
(plausible but unverified), or INCORRECT (dimensionally inconsistent, violates \
a conservation law, confuses distinct concepts, wrong numerical value, or \
category error). If ANY extracted claim is INCORRECT, the proposal is \
FACTUALLY INCOHERENT — regardless of how technical it sounds — and you MUST \
cap Originality at 4, Feasibility at 3, and Specificity at 4. If you are not \
confident enough in a domain to judge, label as SPECULATIVE and cap Originality \
at 7 and Specificity at 7 (since you cannot verify the mechanism). You MUST \
include the list of extracted claims and their labels in your output under \
the `factual_check` field.

BOILERPLATE CHECK (applies to Originality and Impact): Before scoring, ask \
yourself the following three questions:
  (a) Could a competent ML graduate student produce this hypothesis in 30 \
minutes by combining standard tools (Transformer + GAN + contrastive learning \
+ LoRA + RLHF + GNN, etc.)? If yes — this is "boilerplate ML proposal" — cap \
Originality at 5.
  (b) Does the proposed mechanism introduce a NEW causal link, NEW measurement, \
or NEW phenomenon, or does it only re-arrange existing methods on a known \
problem? If only re-arrangement — cap Originality at 6.
  (c) If the hypothesis turns out to be true, would it FALSIFY any current \
belief in the field, or merely confirm what experts already suspect? If it \
only confirms suspicions — cap Impact at 6.
A hypothesis that is well-written, specific, and feasible can still be \
boilerplate; do not let writing quality leak into Originality / Impact.

When scoring ORIGINALITY, judge whether the proposal REFRAMES a problem or \
introduces a genuinely new mechanism/insight relative to common practice in the \
field. A strong idea that builds on prior work is NOT unoriginal: foundational \
ideas legitimately rest on existing literature. Do NOT penalize an idea merely \
because its components appear in the background; penalize only if the proposal \
adds no new conceptual angle. A clean problem reframing that an expert would call \
'I haven't seen it framed this way' is a 8-9 even if every component is known.

Score the research proposal on FIVE dimensions, each from 1 to 10:

ORIGINALITY (judge against the provided background literature)
  Score by checking: does the proposal go beyond what the background papers \
already cover? In your reasoning, cite which refs overlap and what is new.
  1 : Directly restates or paraphrases one of the background papers
  2 : Trivially combines two background papers without new insight
  3 : Addresses a known problem with a known method — no novelty
  4 : Minor twist on existing work; core idea already in the literature
  5 : Reasonable direction but an obvious next step any researcher would propose
  6 : Identifies a real gap in the literature; angle is plausible but incremental
  7 : Proposes a specific mechanism or reframing that an expert would find \
non-obvious and explains why existing approaches are insufficient
  8 : Introduces a testable hypothesis connecting ideas from different sub-fields \
in a way not suggested by the references; expert would say "that's interesting"
  9 : Highly creative; reframes the problem in a way no background paper considers; \
expert would say "I haven't seen this angle before"
  10: Paradigm-shifting; challenges a core assumption in the field
  NOTE: Simply combining two known techniques (e.g., "apply method A to problem B") \
is a 5, not a 7. Score 7+ requires the proposal to articulate WHY this combination \
reveals something the background papers missed.

FEASIBILITY
  1 : Requires technology or data that does not exist
  2 : Theoretically possible but no clear path to implementation
  3 : Major unresolved obstacles (cost, scale, ethics, access)
  4 : Could work in principle but requires significant new infrastructure
  5 : Feasible with substantial effort; realistic 3-5 year project
  6 : Achievable with current methods but non-trivial engineering required
  7 : Clear experimental path using established techniques
  8 : Could be executed by a well-equipped lab within one year; confounds are identified and controlled
  9 : Straightforward to test with off-the-shelf tools and public data
  10: Could be validated in a week with minimal resources

CLARITY
  1 : Incoherent or self-contradictory
  2 : Vague general direction with no specifics
  3 : Core idea is discernible but experimental plan is missing
  4 : Hypothesis is clear but methods are underspecified
  5 : Adequate description; a domain expert could infer what is intended
  6 : Clear hypothesis and methods, but some details need elaboration
  7 : Well-structured; could serve as a starting point for a grant proposal
  8 : Detailed enough to write an experimental protocol directly
  9 : Fully operationalized with all variables, controls, and metrics defined
  10: Publication-ready methodology section

IMPACT
  1 : No discernible contribution to any field
  2 : Marginal refinement of an already-solved problem
  3 : Incremental improvement within a narrow sub-community
  4 : Useful result but limited audience
  5 : Solid contribution to the field; would be cited by peers
  6 : Addresses an important open problem in the field
  7 : Could influence research directions across sub-fields
  8 : High-impact finding that would attract attention beyond the discipline
  9 : Would reshape how multiple fields approach the problem
  10: Paradigm-shifting; Nobel/Turing-level significance

SPECIFICITY (conceptual precision of the IDEA — this is an idea-level benchmark, \
do NOT require a full experimental protocol or numerical thresholds)
  1 : No concrete content whatsoever ("study X using AI")
  2 : Names a broad area but no specific mechanism, target, or claim
  3 : States a direction but the central mechanism is left vague
  4 : Mechanism gestured at but not clearly articulated
  5 : Core mechanism is identifiable; the specific change vs current practice is \
implied but not sharp
  6 : Mechanism is clearly stated and one can see what would be done differently
  7 : Mechanism + a concrete testable prediction or clear evaluation axis are \
both precisely stated (numbers NOT required for an idea)
  8 : Mechanism, the precise problem reframing, and how it would be tested are \
all unambiguous; an expert could design the study from this idea alone
  9 : All of the above plus the key variables / conditions that would decide the \
hypothesis are named
  10: Crisp, complete idea: mechanism, prediction, and decisive test all explicit

Output a JSON object ONLY — no prose before or after the JSON:
{
  "factual_check": {
    "claims": [
      {"claim": "<short quote or paraphrase>", "label": "ESTABLISHED"|"SPECULATIVE"|"INCORRECT", "note": "<why>"},
      {"claim": "...", "label": "...", "note": "..."}
    ],
    "verdict": "coherent"|"speculative"|"factually_incoherent"
  },
  "originality":  {"score": <int>, "reasoning": "<cite which background refs overlap and what is genuinely new>"},
  "feasibility":  {"score": <int>, "reasoning": "<one sentence>"},
  "clarity":      {"score": <int>, "reasoning": "<one sentence>"},
  "impact":       {"score": <int>, "reasoning": "<one sentence>"},
  "specificity":  {"score": <int>, "reasoning": "<one sentence>"}
}"""

USER_TEMPLATE = """\
Please score the following scientific research proposal.

Field: {domain}

Background literature:
{references}

Candidate proposal:
---
{idea}
---

Use the field and background literature to calibrate your assessment. \
For originality, judge whether the proposal goes beyond what the background \
papers already cover. A proposal that merely restates existing findings should \
score low on originality.

Return ONLY the JSON object described in the system prompt."""

SYSTEM_PROMPT_WITH_OVERLAP = """\
You are a senior reviewer for Nature and Science with 20+ years of experience \
evaluating scientific ideas.

CRITICAL: Use the FULL 1-10 range and differentiate clearly. A weak proposal \
and a strong proposal must NOT receive similar scores. Score each dimension \
independently based on its own rubric — do not let one dimension influence \
another. A well-written but unoriginal proposal should score high on clarity \
but low on originality.

When scoring ORIGINALITY, judge whether the proposal REFRAMES a problem or \
introduces a genuinely new mechanism/insight relative to common practice in the \
field. A strong idea that builds on prior work is NOT unoriginal: foundational \
ideas legitimately rest on existing literature. Do NOT penalize an idea merely \
because its components appear in the background; penalize only if the proposal \
adds no new conceptual angle. A clean problem reframing that an expert would call \
'I haven't seen it framed this way' is a 8-9 even if every component is known.

Score the candidate research proposal on SIX outputs (five quality dims + semantic overlap):

ORIGINALITY (judge against the provided background literature)
  Score by checking: does the proposal go beyond what the background papers \
already cover? In your reasoning, cite which refs overlap and what is new.
  1 : Directly restates or paraphrases one of the background papers
  2 : Trivially combines two background papers without new insight
  3 : Addresses a known problem with a known method — no novelty
  4 : Minor twist on existing work; core idea already in the literature
  5 : Reasonable direction but an obvious next step any researcher would propose
  6 : Identifies a real gap in the literature; angle is plausible but incremental
  7 : Proposes a specific mechanism or reframing that an expert would find \
non-obvious and explains why existing approaches are insufficient
  8 : Introduces a testable hypothesis connecting ideas from different sub-fields \
in a way not suggested by the references; expert would say "that's interesting"
  9 : Highly creative; reframes the problem in a way no background paper considers; \
expert would say "I haven't seen this angle before"
  10: Paradigm-shifting; challenges a core assumption in the field
  NOTE: Simply combining two known techniques (e.g., "apply method A to problem B") \
is a 5, not a 7. Score 7+ requires the proposal to articulate WHY this combination \
reveals something the background papers missed.

FEASIBILITY
  1 : Requires technology or data that does not exist
  2 : Theoretically possible but no clear path to implementation
  3 : Major unresolved obstacles (cost, scale, ethics, access)
  4 : Could work in principle but requires significant new infrastructure
  5 : Feasible with substantial effort; realistic 3-5 year project
  6 : Achievable with current methods but non-trivial engineering required
  7 : Clear experimental path using established techniques
  8 : Could be executed by a well-equipped lab within one year; confounds are identified and controlled
  9 : Straightforward to test with off-the-shelf tools and public data
  10: Could be validated in a week with minimal resources

CLARITY
  1 : Incoherent or self-contradictory
  2 : Vague general direction with no specifics
  3 : Core idea is discernible but experimental plan is missing
  4 : Hypothesis is clear but methods are underspecified
  5 : Adequate description; a domain expert could infer what is intended
  6 : Clear hypothesis and methods, but some details need elaboration
  7 : Well-structured; could serve as a starting point for a grant proposal
  8 : Detailed enough to write an experimental protocol directly
  9 : Fully operationalized with all variables, controls, and metrics defined
  10: Publication-ready methodology section

IMPACT
  1 : No discernible contribution to any field
  2 : Marginal refinement of an already-solved problem
  3 : Incremental improvement within a narrow sub-community
  4 : Useful result but limited audience
  5 : Solid contribution to the field; would be cited by peers
  6 : Addresses an important open problem in the field
  7 : Could influence research directions across sub-fields
  8 : High-impact finding that would attract attention beyond the discipline
  9 : Would reshape how multiple fields approach the problem
  10: Paradigm-shifting; Nobel/Turing-level significance

SPECIFICITY (conceptual precision of the IDEA — this is an idea-level benchmark, \
do NOT require a full experimental protocol or numerical thresholds)
  1 : No concrete content whatsoever ("study X using AI")
  2 : Names a broad area but no specific mechanism, target, or claim
  3 : States a direction but the central mechanism is left vague
  4 : Mechanism gestured at but not clearly articulated
  5 : Core mechanism is identifiable; the specific change vs current practice is \
implied but not sharp
  6 : Mechanism is clearly stated and one can see what would be done differently
  7 : Mechanism + a concrete testable prediction or clear evaluation axis are \
both precisely stated (numbers NOT required for an idea)
  8 : Mechanism, the precise problem reframing, and how it would be tested are \
all unambiguous; an expert could design the study from this idea alone
  9 : All of the above plus the key variables / conditions that would decide the \
hypothesis are named
  10: Crisp, complete idea: mechanism, prediction, and decisive test all explicit

SEMANTIC_OVERLAP (vs reference hypothesis only — independent of quality scores)
  1–2 : Completely different direction
  3–4 : Same broad area, different specific question
  5–6 : Related — some shared concepts or mechanisms
  7–8 : Substantially similar core idea
  9–10: Nearly identical hypothesis

Output a JSON object ONLY — no prose before or after the JSON:
{
  "originality":      {"score": <int>, "reasoning": "<cite which refs overlap and what is new>"},
  "feasibility":      {"score": <int>, "reasoning": "<one sentence>"},
  "clarity":          {"score": <int>, "reasoning": "<one sentence>"},
  "impact":           {"score": <int>, "reasoning": "<one sentence>"},
  "specificity":      {"score": <int>, "reasoning": "<one sentence>"},
  "semantic_overlap": {"score": <int>, "reasoning": "<one sentence>"}
}"""

USER_TEMPLATE_WITH_OVERLAP = """\
Please score the following scientific candidate proposal and compare it against the reference.

Field: {domain}

Background literature:
{references}

Candidate proposal:
---
{idea}
---

Reference hypothesis (for semantic overlap only):
---
{gt}
---

STEP 1: Score the five quality dimensions using the candidate proposal, \
field/topic, and background literature. For originality, judge whether the \
proposal goes beyond what the background papers already cover.
STEP 2: Score semantic_overlap based on conceptual similarity between \
candidate and reference hypothesis.

Return ONLY the JSON object described in the system prompt."""

JSON_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. "
    "Return ONLY valid JSON — no markdown, no explanation, just the raw JSON object."
)

# ---------------------------------------------------------------------------
# JSON parsing — 4-layer fallback
# ---------------------------------------------------------------------------

def _try_parse(text: str) -> Optional[Dict]:
    """Try to parse JSON from text; return dict or None."""
    try:
        d = json.loads(text.strip())
        if isinstance(d, dict) and any(k in d for k in DIMS):
            return d
    except json.JSONDecodeError:
        pass
    return None


def _extract_json(text: str) -> Optional[Dict]:
    """Layer 1-2: extract JSON from ```json block or first {} block."""
    # Layer 1: ```json ... ```
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        result = _try_parse(m.group(1))
        if result:
            return result

    # Layer 2: first standalone {...} block
    m = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\})', text, re.DOTALL)
    if m:
        result = _try_parse(m.group(1))
        if result:
            return result

    # Layer 3: regex key:value patterns
    scores: Dict = {}
    for dim in DIMS:
        m = re.search(
            rf'["\']?{dim}["\']?\s*[:\s]+.*?["\']?score["\']?\s*:\s*(\d+)',
            text, re.IGNORECASE | re.DOTALL
        )
        if not m:
            m = re.search(
                rf'["\']?{dim}["\']?\s*:\s*(\d+)',
                text, re.IGNORECASE
            )
        if m:
            scores[dim] = {"score": int(m.group(1)), "reasoning": ""}

    if len(scores) == len(DIMS):
        return scores

    return None


def _normalise(raw: Dict) -> Dict:
    """Ensure each dimension has {score: int, reasoning: str}."""
    out = {}
    for dim in DIMS:
        val = raw.get(dim, {})
        if isinstance(val, dict):
            score = val.get("score") or val.get("rating") or 5
            reasoning = val.get("reasoning") or val.get("reason") or ""
        elif isinstance(val, (int, float)):
            score, reasoning = int(val), ""
        else:
            score, reasoning = 5, ""
        out[dim] = {"score": int(score), "reasoning": str(reasoning)}
    return out


def _extract_overlap(raw: Dict) -> Tuple[Optional[float], str]:
    """Extract semantic overlap from a parsed JSON dict."""
    val = raw.get("semantic_overlap") or raw.get("overlap")
    if isinstance(val, dict):
        score = val.get("score") or val.get("rating")
        reasoning = val.get("reasoning") or val.get("reason") or ""
    elif isinstance(val, (int, float)):
        score, reasoning = val, ""
    else:
        return None, ""

    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return None, str(reasoning)

    if 0.0 <= score_f <= 1.0:
        return round(score_f, 4), str(reasoning)
    if 1.0 <= score_f <= 10.0:
        return round(score_f / 10.0, 4), str(reasoning)
    return None, str(reasoning)


def _extract_overlap_from_text(text: str) -> Tuple[Optional[float], str]:
    """Fallback parser for semantic overlap when JSON is incomplete."""
    m = re.search(
        r'["\']?(?:semantic_overlap|overlap)["\']?\s*[:\s]+.*?["\']?score["\']?\s*:\s*(\d+)',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        m = re.search(
            r'["\']?(?:semantic_overlap|overlap)["\']?\s*[:\s]+([1-9]|10)',
            text,
            re.IGNORECASE,
        )
    if not m:
        return None, ""

    try:
        return round(int(m.group(1)) / 10.0, 4), ""
    except (TypeError, ValueError):
        return None, ""


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_idea(idea_text: str, critic_model: str,
               domain: str = "", title: str = "",
               references: str = "",
               judge_mode: str = "static",
               db_papers: Optional[str] = None,
               max_retries: int = 3,
               retry_delay: float = 2.0) -> Tuple[Optional[Dict], str, Dict]:
    """Score one idea with the given critic model.

    judge_mode:
      "static"          (default) — use the `references` argument verbatim
                                    (the canonical survey refs from papers.db).
      "dynamic_search"  — at scoring time, search Semantic Scholar with the
                          hypothesis as the query and feed top-N hits to the
                          critic. Ignores `references`.
      "dynamic_cited"   — parse the trailing "Cited: ..." footer in the
                          idea_text, resolve each cite to a paper, and feed
                          to the critic. Ignores `references`.

    Returns:
        (scores_dict, raw_response, telemetry)
        scores_dict: {dim: {score, reasoning}} or None on total failure
        telemetry:   {latency_ms, usage, judge_mode, attempts}
    """
    from utils.LLM import CriticLLM

    # Resolve references block based on judge_mode
    if judge_mode != "static":
        from evaluation.dynamic_judge import build_references_block
        # For dynamic_cited we strip the cited block from the idea body so
        # the critic only scores the hypothesis itself (cites are moved
        # into the references block instead).
        from evaluation.dynamic_judge import extract_cited_block
        if judge_mode == "dynamic_cited":
            idea_body, _ = extract_cited_block(idea_text)
        else:
            idea_body = idea_text
        ref_block, ref_meta = build_references_block(
            idea_text=idea_text,
            judge_mode=judge_mode,
            static_references=references,
            db_papers=db_papers,
        )
        scoring_idea = idea_body
        scoring_refs = ref_block
        logger.info(f"  dynamic judge ({judge_mode}): {ref_meta}")
    else:
        scoring_idea = idea_text
        scoring_refs = references or "(no background literature available)"

    llm = CriticLLM(model_name=critic_model)
    prompt = USER_TEMPLATE.format(
        idea=scoring_idea.strip(),
        domain=domain or "General Science",
        references=scoring_refs,
    )
    raw_response = ""
    telemetry = {"latency_ms": None, "usage": None,
                 "judge_mode": judge_mode, "attempts": 0}

    for attempt in range(max_retries):
        telemetry["attempts"] = attempt + 1
        try:
            resp = llm.score_idea(
                prompt + (JSON_RETRY_SUFFIX if attempt > 0 else ""),
                system_prompt=SYSTEM_PROMPT
            )
            # Capture latency + usage of the *last* call regardless of parse outcome
            telemetry["latency_ms"] = getattr(llm, "last_latency_ms", None)
            telemetry["usage"] = getattr(llm, "last_usage", None)

            # Handle (content, reasoning) tuple from thinking models
            if isinstance(resp, tuple):
                raw_response = f"<reasoning>{resp[1]}</reasoning>\n{resp[0] or ''}"
                resp = resp[0]
            else:
                raw_response = resp or ""

            if not resp:
                logger.warning(f"  empty response (attempt {attempt+1}), retrying")
                time.sleep(retry_delay)
                continue

            parsed = _extract_json(resp)
            if parsed:
                return _normalise(parsed), raw_response, telemetry

            logger.warning(f"  score parse failed (attempt {attempt+1}), retrying")
            time.sleep(retry_delay)

        except Exception as e:
            logger.error(f"  critic error (attempt {attempt+1}): {e}")
            time.sleep(retry_delay)

    logger.error(f"  all {max_retries} attempts failed for {critic_model!r}")
    return None, raw_response, telemetry


def mean_score(scores: Optional[Dict]) -> Optional[float]:
    """Compute mean of 5 dimension scores."""
    if not scores:
        return None
    vals = [scores[d]["score"] for d in DIMS if d in scores]
    return round(sum(vals) / len(vals), 4) if vals else None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg

    test_idea = (
        "We hypothesize that integrating chain-of-thought supervision with "
        "step-level reinforcement learning signals will improve mathematical "
        "reasoning in large language models by 15–25% on GSM8K benchmarks. "
        "Specifically, we propose a reward model that assigns dense rewards "
        "at each intermediate reasoning step, penalizing logical inconsistencies "
        "identified via formal verification tools, rather than rewarding only "
        "final-answer correctness."
    )

    critic = cfg.CRITIC_MODELS[0]
    print(f"Scoring with {critic!r} ...")
    scores, raw, _ = score_idea(test_idea, critic)
    if scores:
        for dim, v in scores.items():
            print(f"  {dim:14s}: {v['score']:2d}  — {v['reasoning'][:60]}")
        print(f"  mean: {mean_score(scores):.2f}")
    else:
        print("Scoring failed. Raw response:")
        print(raw[:500])
