"""Ablation: re-critic Track C ideas with Coherence Check + Boilerplate Check REMOVED.

Tests Finding 5/F2 mechanism: if removing the caps changes Originality scores,
the caps were binding in the original.

Approach:
  1. Build modified SYSTEM_PROMPT removing Coherence Check + Boilerplate Check rules.
  2. Re-critic existing Track C ideas for 4 representative models.
  3. Store in new table `cap_ablation_scores` (preserve original `results` rows).
  4. Compare O score distributions: with-cap vs no-cap.

Models (4, ~25 papers each, 3 critics = 300 calls, ~$3, ~20 min):
  - meta-llama/llama-3.1-8b-instruct    (high cap-trigger, low O)
  - mistralai/mistral-7b-instruct-v0.1  (high cap-trigger, low O)
  - qwen/qwen3.5-9b                     (medium, with pile-up at 5)
  - google/gemma-4-31b-it               (low cap-trigger, high O)
"""
import argparse
import collections
import json
import logging
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from evaluation.absolute_scorer import USER_TEMPLATE, JSON_RETRY_SUFFIX, _extract_json, _normalise

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT_NO_CAP = """\
You are a senior reviewer for Nature and Science with 20+ years of experience evaluating scientific ideas.

CRITICAL: Use the FULL 1-10 range and differentiate clearly. A weak proposal and a strong proposal must NOT receive similar scores. Score each dimension independently based on its own rubric — do not let one dimension influence another. A well-written but unoriginal proposal should score high on clarity but low on originality.

FACTUAL CONSISTENCY CHECK (applies to ALL dimensions): Before scoring, extract at least TWO specific factual claims from the proposal that would need to be scientifically TRUE for the hypothesis to make sense. For each claim, label it as one of: ESTABLISHED (widely accepted textbook fact), SPECULATIVE (plausible but unverified), or INCORRECT (dimensionally inconsistent, violates a conservation law, confuses distinct concepts, wrong numerical value, or category error). If ANY extracted claim is INCORRECT, the proposal is FACTUALLY INCOHERENT — regardless of how technical it sounds — and you MUST cap Originality at 4, Feasibility at 3, and Specificity at 4. If you are not confident enough in a domain to judge, label as SPECULATIVE and cap Originality at 7 and Specificity at 7 (since you cannot verify the mechanism). You MUST include the list of extracted claims and their labels in your output under the `factual_check` field.

When scoring ORIGINALITY, compare the proposal against the provided background literature. If the proposal's core idea is already covered by or easily derivable from the references, score 4 or below.

Score the research proposal on FIVE dimensions, each from 1 to 10:

ORIGINALITY (judge against the provided background literature)
  Score by checking: does the proposal go beyond what the background papers already cover?
  1: Directly restates or paraphrases one of the background papers
  2: Trivially combines two background papers without new insight
  3: Addresses a known problem with a known method — no novelty
  4: Minor twist on existing work; core idea already in the literature
  5: Reasonable direction but an obvious next step any researcher would propose
  6: Identifies a real gap in the literature; angle is plausible but incremental
  7: Proposes a specific mechanism or approach NOT mentioned or implied by ANY of the background papers, AND explains why existing approaches are insufficient
  8: Introduces a testable hypothesis connecting ideas from different sub-fields in a way not suggested by the references; expert would say "that's interesting"
  9: Highly creative; reframes the problem in a way no background paper considers; expert would say "I haven't seen this angle before"
  10: Paradigm-shifting; challenges a core assumption in the field

FEASIBILITY
  1: Requires technology or data that does not exist
  2: Theoretically possible but no clear path to implementation
  3: Major unresolved obstacles (cost, scale, ethics, access)
  4: Could work in principle but requires significant new infrastructure
  5: Feasible with substantial effort; realistic 3-5 year project
  6: Achievable with current methods but non-trivial engineering required
  7: Clear experimental path using established techniques
  8: Could be executed by a well-equipped lab within one year; confounds are identified and controlled
  9: Straightforward to test with off-the-shelf tools and public data
  10: Could be validated in a week with minimal resources

CLARITY (1-10): Same scale as in the standard rubric (1=incoherent, 10=publication-ready methods section).

IMPACT (1-10): Same scale as in the standard rubric (1=no contribution, 10=Nobel-level).

SPECIFICITY (1-10): Same scale as in the standard rubric (1=no concrete details, 10=preregistration-quality).

NOTE: This ablation removes the Coherence Check (keyword stuffing) and Boilerplate Check rules that normally cap Originality at 5/6. Score Originality based purely on the rubric tier above.

Output a JSON object ONLY — no prose before or after the JSON:
{
  "factual_check": {
    "claims": [
      {"claim": "<short quote or paraphrase>", "label": "ESTABLISHED"|"SPECULATIVE"|"INCORRECT", "note": "<why>"}
    ],
    "verdict": "coherent"|"speculative"|"factually_incoherent"
  },
  "originality":  {"score": <int>, "reasoning": "<cite which background refs overlap and what is genuinely new>"},
  "feasibility":  {"score": <int>, "reasoning": "<one sentence>"},
  "clarity":      {"score": <int>, "reasoning": "<one sentence>"},
  "impact":       {"score": <int>, "reasoning": "<one sentence>"},
  "specificity":  {"score": <int>, "reasoning": "<one sentence>"}
}"""


TARGET_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.1",
    "qwen/qwen3.5-9b",
    "google/gemma-4-31b-it",
]
CRITIC_POOL = [
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5.1",
    "moonshotai/kimi-k2.6",
]


def ensure_ablation_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS cap_ablation_scores (
        paper_id      TEXT NOT NULL,
        idea_model    TEXT NOT NULL,
        track         TEXT NOT NULL,
        critic_model  TEXT NOT NULL,
        scores_json   TEXT,
        reasoning_json TEXT,
        raw_response  TEXT,
        created_at    TEXT NOT NULL,
        PRIMARY KEY (paper_id, idea_model, track, critic_model)
    );
    """)
    conn.commit()


def score_no_cap(idea_text: str, critic_model: str, domain: str, references: str):
    """Score with cap-removed prompt. Returns (scores_dict, raw_response)."""
    from utils.LLM import CriticLLM
    llm = CriticLLM(model_name=critic_model)
    prompt = USER_TEMPLATE.format(
        idea=idea_text.strip(),
        domain=domain or "General Science",
        references=references or "(no background literature available)",
    )
    raw = ""
    for attempt in range(3):
        try:
            resp = llm.score_idea(
                prompt + (JSON_RETRY_SUFFIX if attempt > 0 else ""),
                system_prompt=SYSTEM_PROMPT_NO_CAP,
            )
            if isinstance(resp, tuple):
                raw = resp[0] or ""
            else:
                raw = resp or ""
            if not raw:
                continue
            parsed = _extract_json(raw)
            if parsed:
                scores = _normalise(parsed)
                return scores, raw
        except Exception as e:
            logger.warning(f"  attempt {attempt+1} err: {e}")
        time.sleep(1)
    return None, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-per-domain", type=int, default=5)
    args = ap.parse_args()

    conn_r = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn_r.execute("PRAGMA busy_timeout=30000")
    ensure_ablation_table(conn_r)

    conn_p = sqlite3.connect(str(cfg.PAPERS_DB), timeout=30)
    conn_p.row_factory = sqlite3.Row

    # Get papers
    papers = [dict(r) for r in conn_p.execute("""
        SELECT * FROM papers WHERE status='filtered'
        ORDER BY domain, paper_id
    """)]
    by_dom = collections.defaultdict(list)
    for p in papers:
        by_dom[p["domain"]].append(p)
    papers = [p for ps in by_dom.values() for p in ps[:args.n_per_domain]]
    paper_ids = [p["paper_id"] for p in papers]
    papers_by_id = {p["paper_id"]: p for p in papers}
    logger.info(f"papers: {len(papers)}")

    # Find Track C ideas needing ablation re-critic
    pidp = ",".join("?" * len(paper_ids))
    mp = ",".join("?" * len(TARGET_MODELS))
    ideas = list(conn_r.execute(f"""
        SELECT paper_id, idea_model, idea_text FROM results
        WHERE prompt_version='v1_paper_refs' AND track='C' AND idea_index=1
          AND critic_model=''
          AND idea_model IN ({mp}) AND paper_id IN ({pidp})
          AND idea_text IS NOT NULL AND idea_text != ''
    """, list(TARGET_MODELS) + paper_ids))
    logger.info(f"Track C ideas to re-critic: {len(ideas)}")

    tasks = []
    for pid, im, it in ideas:
        for cr in CRITIC_POOL:
            if cr == im:
                continue
            exists = conn_r.execute(
                "SELECT 1 FROM cap_ablation_scores WHERE paper_id=? AND idea_model=? "
                "AND track='C' AND critic_model=?", (pid, im, cr)).fetchone()
            if not exists:
                tasks.append((pid, im, it, cr))

    logger.info(f"ablation critic tasks: {len(tasks)}")
    if not tasks:
        return

    from evaluation.critic_manager import _format_refs_for_judge

    def _worker(task):
        pid, im, idea_text, cr = task
        paper = papers_by_id.get(pid, {})
        refs = _format_refs_for_judge(paper)
        scores, raw = score_no_cap(
            idea_text, cr, domain=paper.get("domain", ""), references=refs,
        )
        return (pid, im, cr, scores, raw)

    ts = datetime.now(timezone.utc).isoformat()
    stats = collections.Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            pid, im, cr, scores, raw = f.result()
            if scores:
                conn_r.execute("""
                    INSERT OR IGNORE INTO cap_ablation_scores
                      (paper_id, idea_model, track, critic_model,
                       scores_json, reasoning_json, raw_response, created_at)
                    VALUES (?, ?, 'C', ?, ?, ?, ?, ?)
                """, (pid, im, cr,
                      json.dumps({d: v["score"] for d, v in scores.items()}),
                      json.dumps({d: v["reasoning"] for d, v in scores.items()}),
                      (raw or "")[:4000], ts))
                conn_r.commit()
                stats["ok"] += 1
            else:
                stats["err"] += 1
            if i % 20 == 0 or i == len(tasks):
                logger.info(f"  [{i}/{len(tasks)}] {dict(stats)}")
    conn_r.close()
    conn_p.close()
    logger.info(f"done: {dict(stats)}")


if __name__ == "__main__":
    main()
