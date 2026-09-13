"""
Dual-Track Idea Generation  (Phase 2-A)

For each (paper, idea_model) pair:
  Track A — keyword only:   3 hypotheses generated without refs
  Track B — with refs:      3 hypotheses generated using refs

Results written to results.db (table: results).
Skips combos already present (idempotent / checkpoint-friendly).

Usage:
    python generation/generate_ideas.py --model openai/gpt-4o
    python generation/generate_ideas.py --model openai/gpt-4o --smoke
    python generation/generate_ideas.py --model openai/gpt-4o --domain CS
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import hashlib
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime import get_thread_sqlite_connection

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_A = (
    "You are a creative and rigorous scientist with deep expertise in your field. "
    "Your task is to generate a novel, testable scientific hypothesis. "
    "Be specific, mechanistically grounded, and feasible."
)

_USER_A = """\
Generate a single novel scientific hypothesis based on the following:

Topic: {title}
Field: {domain}

Requirements:
1. Output exactly one paragraph
2. Total length: 80-150 words
3. Be specific — name the exact mechanism, molecule, algorithm, or system
4. Be feasible — it should be testable with current technology
5. Be novel — go beyond obvious or already-established ideas
6. Output ONLY the hypothesis, no preamble or explanation

Hypothesis:"""

_SYSTEM_B = (
    "You are a creative and rigorous scientist. "
    "Your task is to propose a single novel, testable scientific hypothesis "
    "in the given research area. You will also be given a set of background "
    "papers describing what has already been done in the field; use them only "
    "to understand the landscape and to AVOID duplicating existing work — your "
    "hypothesis should be a new research direction, not an extension or "
    "synthesis of these specific papers."
)

_USER_B = """\
Propose a single novel scientific hypothesis in the field below.

Field: {domain}

Background papers (these describe what has already been published — use them
to know what NOT to repeat, not as targets to extend):
{references}

Requirements:
1. Output exactly one paragraph
2. Total length: 80-150 words
3. Your hypothesis should be a new direction in the field, not a derivative
   of the background papers
4. Do NOT summarize, synthesize, or merely extend the background — propose
   a research idea that could stand on its own
5. Be specific — name the exact mechanism, molecule, algorithm, or system
6. Be feasible — it should be testable with current technology
7. Output ONLY the hypothesis, no preamble or explanation

Hypothesis:"""

_FALLBACK_A = """\
Imagine you are designing a groundbreaking experiment. Based on the topic below,
propose one specific scientific hypothesis in a single paragraph.
Total length 80-150 words.

Topic: {title}
Field: {domain}"""

_FALLBACK_B = """\
Based on the following research area and related work, propose one specific
scientific hypothesis in a single paragraph.
Total length 80-150 words.

Field: {domain}
Related work summary: {ref_summary}"""


# ---------------------------------------------------------------------------
# Structured Research Proposal prompts
# ---------------------------------------------------------------------------

_SYSTEM_A_STRUCTURED = (
    "You are a creative and rigorous scientist with deep expertise in your field. "
    "Your task is to generate a novel, testable scientific research proposal. "
    "Be specific, mechanistically grounded, and feasible. "
    "Critically examine your own assumptions and provide specific quantitative predictions. "
    "Write in a structured format with clearly labeled sections."
)

_USER_A_STRUCTURED = """\
Generate a structured scientific research proposal based on the following:

Topic: {title}
Field: {domain}

Write EXACTLY these eight sections with the specified section headers:

## Hypothesis (80-120 words)
State a single novel, testable scientific hypothesis. Name the exact mechanism, \
molecule, algorithm, or system under investigation.

## Experimental Design (120-200 words)
Describe the primary experiment to test this hypothesis. Specify: \
independent and dependent variables, methodology (technique, assay, model, or dataset), \
sample or data requirements, and key controls.

## Expected Outcomes (60-100 words)
What result would support the hypothesis? What result would falsify it? \
Predict the direction or magnitude of the expected effect.

## Contingency Plan (60-100 words)
If the primary experiment fails or produces null results, what is the backup \
approach? Describe an alternative methodology or a modified hypothesis to pursue.

## Broader Implications (40-80 words)
If the hypothesis is confirmed, what are the downstream consequences for the field?

## Assumption Audit (80-120 words)
Identify 2-3 implicit assumptions underlying your hypothesis. For each, \
explain why the assumption might be wrong and how that would change \
the interpretation of your results.

## Potential Confounds & Controls (60-100 words)
List 2-3 specific confounding variables or alternative explanations for \
your expected results. For each confound, describe a concrete control \
experiment or analysis to rule it out.

## Quantitative Predictions (60-100 words)
Provide 2-3 specific numerical predictions (effect sizes, accuracy ranges, \
fold-changes, or thresholds) that your hypothesis implies. Justify each \
number by referencing known baselines or analogous prior results.

Requirements:
- Total length: 560-920 words across all sections
- Be specific throughout — name concrete methods, targets, and measurable outcomes
- Output ONLY the eight sections with their headers, no other preamble or explanation"""

_SYSTEM_B_STRUCTURED = (
    "You are a creative and rigorous scientist. "
    "Your task is to propose a single novel, testable scientific research "
    "proposal in the given research area. You will also be given a set of "
    "background papers; treat them as a snapshot of what has already been "
    "published in the field — use them only to avoid duplicating prior work, "
    "not as targets to extend or synthesize. Your proposal should be a new "
    "research direction with specific quantitative predictions."
)

_USER_B_STRUCTURED = """\
Generate a structured scientific research proposal based on the topic and background below.

Topic: {title}
Field: {domain}

Background papers (these describe what has already been published — use them
to know what NOT to repeat, not as targets to extend):
{references}

Write EXACTLY these eight sections with the specified section headers:

## Hypothesis (80-120 words)
State a single novel, testable scientific hypothesis in this field. It should \
be a new research direction, not a derivative of the background papers.

## Experimental Design (120-200 words)
Describe the primary experiment to test this hypothesis. Specify: \
independent and dependent variables, methodology (technique, assay, model, or dataset), \
sample or data requirements, and key controls.

## Expected Outcomes (60-100 words)
What result would support the hypothesis? What result would falsify it? \
Predict the direction or magnitude of the expected effect.

## Contingency Plan (60-100 words)
If the primary experiment fails or produces null results, what is the backup \
approach? Describe an alternative methodology or a modified hypothesis to pursue.

## Broader Implications (40-80 words)
If the hypothesis is confirmed, what are the downstream consequences for the field?

## Differentiation from Prior Work (80-120 words)
Briefly explain how your hypothesis differs from the background papers — \
identify 2-3 specific claims or methods in those papers that your hypothesis \
does NOT merely extend. Keep this short; the hypothesis itself is the main \
contribution.

## Potential Confounds & Controls (60-100 words)
List 2-3 specific confounding variables or alternative explanations for \
your expected results. For each confound, describe a concrete control \
experiment or analysis to rule it out.

## Quantitative Predictions (60-100 words)
Provide 2-3 specific numerical predictions (effect sizes, accuracy ranges, \
fold-changes, or thresholds) that your hypothesis implies. Justify each \
number by referencing known baselines or analogous prior results.

Requirements:
- Total length: 560-920 words across all sections
- Be specific throughout — name concrete methods, targets, and measurable outcomes
- Output ONLY the eight sections with their headers, no other preamble or explanation"""

_FALLBACK_A_STRUCTURED = """\
Imagine you are writing a research proposal. Based on the topic below, \
produce a structured proposal with these eight sections: \
Hypothesis (80-120 words), Experimental Design (120-200 words), \
Expected Outcomes (60-100 words), Contingency Plan (60-100 words), \
Broader Implications (40-80 words), Assumption Audit (80-120 words), \
Potential Confounds & Controls (60-100 words), Quantitative Predictions (60-100 words).

Topic: {title}
Field: {domain}"""

_FALLBACK_B_STRUCTURED = """\
Based on the following research area and related work, produce a structured \
proposal with these eight sections: Hypothesis (80-120 words), \
Experimental Design (120-200 words), Expected Outcomes (60-100 words), \
Contingency Plan (60-100 words), Broader Implications (40-80 words), \
Literature Gap Analysis (80-120 words), Potential Confounds & Controls (60-100 words), \
Quantitative Predictions (60-100 words).

Topic: {title}
Field: {domain}
Related work summary: {ref_summary}"""


# ---------------------------------------------------------------------------
# Reference formatting
# ---------------------------------------------------------------------------

def _format_refs(paper: dict) -> str:
    """Format references for Track B prompt.

    Reads from ranked_refs_json. Shuffles order deterministically (by paper_id)
    so models don't get position bias from citation-count sorting — the
    original order put famous generic ML papers first, inducing all models
    to converge on "combine the top-cited techniques".
    Full abstracts are included — no word truncation.
    """
    refs_json = paper.get("ranked_refs_json") or paper.get("references_json")

    if not refs_json:
        return "(no references available)"
    try:
        refs = json.loads(refs_json)
    except (json.JSONDecodeError, TypeError):
        return "(no references available)"

    if not refs:
        return "(no references available)"

    # Deterministic shuffle per paper to decouple order from citation rank
    pid = paper.get("paper_id") or ""
    seed = int(hashlib.sha1(pid.encode("utf-8")).hexdigest()[:8], 16) if pid else 42
    shuffled = list(refs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    parts = []
    for i, ref in enumerate(shuffled, 1):
        title    = ref.get("title") or "Untitled"
        abstract = (ref.get("abstract") or "").strip()
        entry    = f"[{i}] {title}"
        if abstract:
            entry += f"\n{abstract}"
        parts.append(entry)

    return "\n\n".join(parts)


def _ref_summary(references_json: Optional[str], max_words: int = 150) -> str:
    """Short summary of ref titles for fallback prompt."""
    if not references_json:
        return "no references"
    try:
        refs = json.loads(references_json)
    except Exception:
        return "no references"
    titles = [r.get("title", "") for r in refs if r.get("title")][:5]
    summary = "; ".join(titles)
    words = summary.split()
    return " ".join(words[:max_words]) if len(words) > max_words else summary


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _clean_idea_text(raw: str) -> str:
    """Strip reasoning/thinking artifacts from LLM output.

    Some reasoning LLMs leak their thinking process despite the 'Output ONLY
    the hypothesis' instruction. This extracts the actual hypothesis by
    finding the LAST sentence-chunk that starts with a hypothesis marker
    and is 60-250 words. Draft markers (e.g. '*Draft 3:*', 'Trimmed:') are
    stripped from the start.
    """
    import re
    text = raw.strip()

    leak_markers = ("Thinking Process:", "Draft 1:", "Draft 2:", "Wait,", "Critique 1:",
                    "Final Polish", "Word Count", "Trimmed:")
    has_leak = any(m in text for m in leak_markers)

    if has_leak:
        # Find all hypothesis chunks: start at "We hypothesize/We propose/
        # Hypothesis:/I hypothesize/Given that" and end at a paragraph break
        # or start of a new draft/meta-marker.
        pattern = re.compile(
            r'(?:We hypothesize|We propose|Hypothesis:|I hypothesize|Given that)'
            r'[^\n]*(?:\n(?!\s*\*|\s*\d[\.\)]|\n)[^\n]*)*',
            re.MULTILINE,
        )
        candidates = []
        for m in pattern.finditer(text):
            chunk = m.group(0).strip()
            # Strip leading draft markers like "*Draft 3 (Final Polish):*"
            chunk = re.sub(r'^\*?(Draft \d+|Trimmed|Final Polish)[^:]*:\*?\s*', '', chunk)
            word_count = len(chunk.split())
            if 60 <= word_count <= 250:
                candidates.append(chunk)

        if candidates:
            # Prefer the last candidate (most polished draft)
            text = candidates[-1]
        else:
            # No clean hypothesis found — discard and signal regen upstream
            text = "[generation leaked reasoning trace; could not extract hypothesis]"

    # Safety cap
    if len(text.split()) > 250:
        words = text.split()
        text = ' '.join(words[:200]) + '...'

    return text.strip()


def generate_for_paper(paper: dict, model_name: str,
                       n_ideas: int = 3) -> dict:
    """Generate n_ideas for Track B for one paper.

    Returns generated text. Final best-idea selection is handled downstream
    in Phase 4 after scoring.
    """
    from utils.LLM import IdeaLLM

    llm = IdeaLLM(model_name=model_name)

    result = {}

    for track in ("B",):
        ideas = []
        for _ in range(n_ideas):
            prompt, fallback, system = _build_generation_payload(paper, track)

            try:
                out = llm.generate_idea(prompt, fallback_prompt=fallback,
                                        system_prompt=system)
                idea_text = _clean_idea_text(out["idea"])
            except Exception as e:
                logger.warning(f"    idea generation error ({track}): {e}")
                idea_text = f"[generation error: {e}]"

            ideas.append(idea_text)

        result[track] = {
            "ideas": ideas,
            "best_index": None,
            "best_idea": None,
        }

    return result


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

# Module-level flag — default True (every idea must emit a "Cited:" footer).
# Run.py can still flip this via --no-require-cites for ablation runs.
# The footer feeds two downstream consumers:
#   1) dynamic_judge.extract_cited_block (dynamic_cited judging)
#   2) the dynamic-critic sweep: distinguishes ideas that ground in refs
#      vs ideas that don't, without needing the active-mode agent trace
REQUIRE_CITES: bool = True

_CITES_FOOTER_INSTRUCTION = (
    "\n\nAfter your hypothesis paragraph, append a single line:\n"
    "    Cited: <ref1>; <ref2>; <ref3>\n"
    "listing 2-5 references from the background literature you actually "
    "grounded the hypothesis in. Each ref should be a short identifying "
    "string (first author + title fragment, or [N] index from the refs "
    "block). Do not include refs you did not use."
)


def _apply_cites_footer(prompt: str) -> str:
    """Append the Cited-footer instruction to a generation prompt if enabled."""
    if REQUIRE_CITES:
        return prompt + _CITES_FOOTER_INSTRUCTION
    return prompt


def _extract_cited_refs(idea_text: str) -> list:
    """Parse the trailing 'Cited:' footer and return list of cite strings.

    Returns [] if no footer present (REQUIRE_CITES disabled or model omitted it).
    Reuses evaluation.dynamic_judge.extract_cited_block for consistency.
    """
    try:
        from evaluation.dynamic_judge import extract_cited_block
        _, cites = extract_cited_block(idea_text or "")
        return cites
    except Exception:
        return []


def _build_generation_payload(paper: dict, track: str) -> tuple:
    """Build (prompt, fallback, system) for one (paper, track) generation task."""
    import config as cfg

    title  = paper["title"]
    domain = paper["domain"]
    structured = cfg.GENERATION_FORMAT == "structured"

    if track == "A":
        if structured:
            return (
                _apply_cites_footer(_USER_A_STRUCTURED.format(title=title, domain=domain)),
                _apply_cites_footer(_FALLBACK_A_STRUCTURED.format(title=title, domain=domain)),
                _SYSTEM_A_STRUCTURED,
            )
        return (
            _apply_cites_footer(_USER_A.format(title=title, domain=domain)),
            _apply_cites_footer(_FALLBACK_A.format(title=title, domain=domain)),
            _SYSTEM_A,
        )

    refs_text   = _format_refs(paper)
    ref_summary = _ref_summary(
        paper.get("ranked_refs_json") or paper.get("references_json")
    )
    if structured:
        return (
            _apply_cites_footer(_USER_B_STRUCTURED.format(title=title, domain=domain, references=refs_text)),
            _apply_cites_footer(_FALLBACK_B_STRUCTURED.format(title=title, domain=domain, ref_summary=ref_summary)),
            _SYSTEM_B_STRUCTURED,
        )
    return (
        _apply_cites_footer(_USER_B.format(title=title, domain=domain, references=refs_text)),
        _apply_cites_footer(_FALLBACK_B.format(title=title, domain=domain, ref_summary=ref_summary)),
        _SYSTEM_B,
    )


def _existing_idea_indices(cur: sqlite3.Cursor, idea_model: str) -> dict:
    """Return {(paper_id, track): {idea_index, ...}} for SUCCESSFUL Phase-2 rows.

    Rows whose idea_text starts with '[generation error' are treated as
    missing — they were transient API failures (e.g. 402 credit), and
    re-running phase 2 should retry them. The INSERT in
    _generate_single_idea_task uses INSERT OR IGNORE, so we must remove
    these from the "done" set for the row to be UPDATEable. Wait — actually
    INSERT OR IGNORE will still skip if the unique-key row exists. So we
    also need to UPDATE in-place when retrying. Done via REPLACE semantics
    below by switching from IGNORE to REPLACE when the prior row was an
    error.
    """
    rows = cur.execute(
        "SELECT paper_id, track, idea_index, idea_text FROM results "
        "WHERE idea_model=? AND critic_model=''",
        (idea_model,)
    ).fetchall()
    done = {}
    for paper_id, track, idea_index, idea_text in rows:
        if idea_text and idea_text.startswith("[generation error"):
            continue   # treat as missing → phase 2 will regenerate
        done.setdefault((paper_id, track), set()).add(int(idea_index))
    return done


def _generate_single_idea_task(db_results: str, paper: dict,
                               idea_model: str, track: str,
                               idea_index: int) -> dict:
    """Generate and insert one Phase-2 row.

    Static-mode telemetry: prompt/completion/reasoning token counts of the
    LLM call are captured in raw_response (JSON) for later analysis (matches
    Active mode telemetry format introduced 2026-05-05).
    """
    from utils.LLM import IdeaLLM

    # Pre-check: if a successful idea row already exists for this combo,
    # short-circuit. Only the [generation error...] placeholder rows are
    # eligible for regeneration. This guards against the previous bug where
    # INSERT OR REPLACE clobbered raw 22-model v2 data on retry runs.
    pre_conn = get_thread_sqlite_connection(db_results, timeout=30, enable_wal=True)
    pre_cur = pre_conn.cursor()
    pre_row = pre_cur.execute(
        "SELECT idea_text FROM results "
        "WHERE paper_id=? AND idea_model=? AND track=? AND idea_index=? AND critic_model=''",
        (paper["paper_id"], idea_model, track, idea_index)
    ).fetchone()
    if pre_row and pre_row[0] and not pre_row[0].startswith("[generation error"):
        return {
            "paper_id": paper["paper_id"],
            "track": track,
            "idea_index": idea_index,
            "inserted": False,
            "had_error": False,
            "idea_chars": len(pre_row[0]),
            "skipped_existing": True,
        }

    llm = IdeaLLM(model_name=idea_model)
    prompt, fallback, system = _build_generation_payload(paper, track)

    had_error = False
    usage = None
    latency_ms = None
    try:
        out = llm.generate_idea(prompt, fallback_prompt=fallback,
                                system_prompt=system)
        idea_text = out["idea"].strip()
        usage = out.get("usage")
        latency_ms = getattr(llm, "last_latency_ms", None)
    except Exception as e:
        had_error = True
        idea_text = f"[generation error: {e}]"
        logger.warning(
            f"    idea generation error ({track}{idea_index}, {paper['paper_id'][:12]}): {e}"
        )

    telemetry = {
        "usage": usage,                     # {prompt_tokens, completion_tokens, reasoning_tokens, total_tokens}
        "latency_ms": latency_ms,           # wall-clock latency of the final LLM call
        "first_was_rejected": (out.get("first_was_rejected") if not had_error else None),
        "used_fallback": (out.get("used_fallback") if not had_error else None),
        "had_error": had_error,
    }
    cited_refs_json = json.dumps(_extract_cited_refs(idea_text)) if not had_error else None

    conn_r = get_thread_sqlite_connection(db_results, timeout=30, enable_wal=True)
    cur_r = conn_r.cursor()
    ts = datetime.now(timezone.utc).isoformat()
    # Default semantics: INSERT OR IGNORE (skip if row exists, preserve raw
    # data per CLAUDE.md §9.5). Error rows are upgraded to success only via
    # the explicit error-retry path which targets idea_text LIKE '[generation
    # error%' before calling LLM — see _generate_single_idea_task's pre-check
    # above, plus the SQL trigger `protect_successful_idea` that converts
    # accidental REPLACEs on success rows into silent IGNOREs.
    cur_r.execute("""
        INSERT OR IGNORE INTO results
          (paper_id, idea_model, track, idea_index, idea_text,
           critic_model, raw_response, telemetry, cited_refs, created_at)
        VALUES (?, ?, ?, ?, ?, '', ?, ?, ?, ?)
    """, (paper["paper_id"], idea_model, track, idea_index, idea_text,
          json.dumps(telemetry), json.dumps(telemetry), cited_refs_json, ts))
    inserted_via_ignore = cur_r.rowcount > 0
    # If the row already existed (likely an error placeholder), UPDATE the
    # error idea_text in place with the freshly-generated content. The SQL
    # trigger protects non-error rows so this UPDATE is safe.
    if not inserted_via_ignore and not had_error:
        cur_r.execute("""
            UPDATE results SET idea_text=?, raw_response=?, telemetry=?,
                   cited_refs=?, created_at=?
            WHERE paper_id=? AND idea_model=? AND track=? AND idea_index=?
              AND critic_model='' AND idea_text LIKE '[generation error%'
        """, (idea_text, json.dumps(telemetry), json.dumps(telemetry),
              cited_refs_json, ts,
              paper["paper_id"], idea_model, track, idea_index))
    inserted = cur_r.rowcount > 0
    conn_r.commit()

    return {
        "paper_id": paper["paper_id"],
        "track": track,
        "idea_index": idea_index,
        "inserted": inserted,
        "had_error": had_error,
        "idea_chars": len(idea_text),
    }


def _already_done(cur: sqlite3.Cursor,
                  paper_id: str, idea_model: str, track: str,
                  n: int = 3) -> bool:
    row = cur.execute(
        "SELECT COUNT(*) FROM results WHERE paper_id=? AND idea_model=? AND track=?",
        (paper_id, idea_model, track)
    ).fetchone()
    return (row[0] if row else 0) >= n


def _insert_ideas(conn: sqlite3.Connection, cur: sqlite3.Cursor,
                  paper_id: str, idea_model: str,
                  track: str, ideas: list, best_index: int) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    for i, idea_text in enumerate(ideas, 1):
        cur.execute("""
            INSERT OR IGNORE INTO results
              (paper_id, idea_model, track, idea_index, idea_text,
               critic_model, created_at)
            VALUES (?, ?, ?, ?, ?, '', ?)
        """, (paper_id, idea_model, track, i, idea_text, ts))
    conn.commit()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_generation(idea_model: str, db_papers: str, db_results: str,
                   smoke: bool = False,
                   domain_filter: Optional[str] = None,
                   n_ideas: int = 3,
                   max_workers: Optional[int] = None) -> dict:
    """Generate ideas for all filtered papers.

    Returns stats: {papers, skipped, generated, errors}
    """
    import config as cfg

    conn_p = sqlite3.connect(db_papers, timeout=30)
    conn_p.row_factory = sqlite3.Row
    conn_r = sqlite3.connect(db_results, timeout=30)
    conn_r.execute("PRAGMA busy_timeout = 30000")
    cur_r  = conn_r.cursor()

    # Fetch filtered papers with gt_hypothesis
    query = """
        SELECT * FROM papers
        WHERE status = 'filtered'
          AND gt_hypothesis IS NOT NULL AND gt_hypothesis != ''
    """
    params: list = []
    if domain_filter:
        query += " AND domain = ?"
        params.append(domain_filter)
    query += " ORDER BY domain, paper_id"

    papers = [dict(r) for r in conn_p.execute(query, params).fetchall()]
    if smoke:
        # 5 papers per domain for balanced coverage
        from collections import defaultdict
        by_domain = defaultdict(list)
        for p in papers:
            by_domain[p["domain"]].append(p)
        papers = [p for ps in by_domain.values() for p in ps[:5]]
        papers.sort(key=lambda p: (p["domain"], p["paper_id"]))
        logger.info(f"[smoke] {len(papers)} papers ({len(by_domain)} domains) × model {idea_model!r}")
    else:
        logger.info(f"generate_ideas: {len(papers)} papers × model {idea_model!r}")

    stats = {"papers": len(papers), "skipped": 0, "generated": 0, "errors": 0, "tasks": 0}
    done_map = _existing_idea_indices(cur_r, idea_model)
    conn_r.close()

    tasks = []
    for i, paper in enumerate(papers, 1):
        pid = paper["paper_id"]
        logger.info(f"  [{i}/{len(papers)}] {paper['domain']} | {paper['title'][:60]!r}")

        pending = []
        for track in ("B",):
            existing = done_map.get((pid, track), set())
            if len(existing) >= n_ideas:
                continue
            for idea_index in range(1, n_ideas + 1):
                if idea_index not in existing:
                    pending.append((paper, track, idea_index))

        if not pending:
            stats["skipped"] += 1
            logger.info("    → already done, skipping")
            continue

        tasks.extend(pending)

    stats["tasks"] = len(tasks)
    if not tasks:
        conn_p.close()
        logger.info(
            f"generate_ideas done: {stats['generated']} generated, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )
        return stats

    max_workers = max(1, int(max_workers or cfg.PARALLEL.get("phase2_max_workers", 4)))
    logger.info(f"  submitting {len(tasks)} idea tasks with max_workers={max_workers}")

    generated_papers = set()
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_single_idea_task,
                db_results,
                paper,
                idea_model,
                track,
                idea_index,
            ): (paper["paper_id"], track, idea_index)
            for paper, track, idea_index in tasks
        }

        for future in as_completed(futures):
            pid, track, idea_index = futures[future]
            completed += 1
            try:
                result = future.result()
                if result["inserted"]:
                    generated_papers.add(result["paper_id"])
                if result["had_error"]:
                    stats["errors"] += 1
                logger.info(
                    f"    [{completed}/{len(tasks)}] {pid[:12]} {track}{idea_index} "
                    f"({'inserted' if result['inserted'] else 'duplicate'}) "
                    f"{result['idea_chars']} chars"
                )
            except Exception as e:
                logger.error(f"    ERROR {pid[:12]} {track}{idea_index}: {e}")
                stats["errors"] += 1

    conn_p.close()
    stats["generated"] = len(generated_papers)

    logger.info(
        f"generate_ideas done: {stats['generated']} generated, "
        f"{stats['skipped']} skipped, {stats['errors']} errors"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import config as cfg

    parser = argparse.ArgumentParser(description="Dual-track idea generation")
    parser.add_argument("--model",  required=True,             help="Idea model name")
    parser.add_argument("--smoke",  action="store_true",       help="3 papers only")
    parser.add_argument("--domain", default=None,              help="Restrict to domain")
    parser.add_argument("--papers-db",  default=str(cfg.PAPERS_DB))
    parser.add_argument("--results-db", default=str(cfg.RESULTS_DB))
    args = parser.parse_args()

    stats = run_generation(
        idea_model=args.model,
        db_papers=args.papers_db,
        db_results=args.results_db,
        smoke=args.smoke,
        domain_filter=args.domain,
    )
    print(f"\nDone: {stats['generated']} papers generated, "
          f"{stats['skipped']} skipped, {stats['errors']} errors")
