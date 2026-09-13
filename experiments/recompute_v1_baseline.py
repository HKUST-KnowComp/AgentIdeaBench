"""Recompute the v1 cross-year baseline that was overwritten on 2026-05-14.

Background:
  On 2026-05-09/10 a Static-Mode cross-year sweep ran over 23 idea_models
  with 5 critic_models, producing reports/cross_year_*.png that became the
  "v1" motivation figures. On 2026-05-14 a rerun (with corrected prompt)
  DELETEd those rows from results.db before writing new (v2) data, making
  v1 unrecoverable from DB.

This script regenerates a v1' baseline that mirrors v1's recipe as closely
as possible:

  - idea_models:  v1's 23 list MINUS `mistralai/mistral-small-2603`
                  (user dropped it; 22 total) MINUS `claude-3.7-sonnet`
                  (404 on OR) MINUS `qwen3-32b` (those two still have
                  un-deleted v1 rows from 5/9, reused as-is). 20 to regenerate.
  - critic_pool:  v1's 5 critics: deepseek-v4-flash, kimi-k2-0905,
                  qwen3-max, deepseek-r1, grok-4.3.
  - prompt:       v1 _SYSTEM_B / _USER_B verbatim ("synthesize/extend
                  background"), pasted in below as constants — does NOT
                  depend on generation/generate_ideas.py (which has the v2
                  "propose new" prompt).
  - thinking:     left at the model default (i.e. ON for OR
                  thinking-default models). We achieve this by NOT setting
                  `reasoning.enabled=False` — controlled via a per-call
                  env-var sentinel that LLM.py reads.
  - target DB:    data/results_v1.db (NOT data/results.db). The v2 db is
                  preserved untouched.

Run:
  python experiments/recompute_v1_baseline.py --smoke   # 1 paper × 2 model
  python experiments/recompute_v1_baseline.py           # full

Outputs:
  data/results_v1.db                                    raw rows
  reports/cross_year_static_v1recompute.{png,pdf,json}  v1' plots
  reports/cross_year_scatter_v1recompute.{png,pdf}      v1' scatter
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration: hardcoded to match v1
# ---------------------------------------------------------------------------

V1_IDEA_MODELS_TO_RECOMPUTE: List[str] = [
    "openai/gpt-4",
    "mistralai/mistral-7b-instruct-v0.1",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "google/gemma-2-27b-it",
    "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-4.1",
    "openai/gpt-5",
    "google/gemma-3-27b-it",
    "qwen/qwen3-8b",
    "mistralai/mistral-small-24b-instruct-2501",
    "meta-llama/llama-4-maverick",
    "anthropic/claude-sonnet-4",
    "openai/gpt-5.5",
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
    "google/gemma-4-31b-it",
    "anthropic/claude-sonnet-4.6",
]

V1_CRITIC_POOL: List[str] = [
    "deepseek/deepseek-v4-flash",
    "moonshotai/kimi-k2-0905",
    "qwen/qwen3-max",
    "deepseek/deepseek-r1",
    "x-ai/grok-4.3",
]

DB_PAPERS = str(cfg.PAPERS_DB)
DB_RESULTS_V1 = str(ROOT / "data" / "results_v1.db")

N_PAPERS_PER_DOMAIN = 5
N_IDEAS_PER_COMBO = 3
N_CRITICS_PER_COMBO = 3
TRACK = "B"

# ---------------------------------------------------------------------------
# v1 prompt verbatim (from git commit 96f4f26:generation/generate_ideas.py)
# ---------------------------------------------------------------------------

V1_SYSTEM_B = (
    "You are a creative and rigorous scientist. "
    "You will be given a research topic and a set of background papers. "
    "Your task is to synthesize these ideas into a genuinely novel scientific "
    "hypothesis that goes BEYOND what the background papers already say. "
    "Do not merely summarize the literature."
)

V1_USER_B = """\
Generate a single novel scientific hypothesis based on the background papers below.

Field: {domain}

Background papers (use these as inspiration, not as a summary):
{references}

Requirements:
1. Output exactly one paragraph
2. Total length: 80-150 words
3. Your hypothesis must EXTEND or COMBINE ideas from the background in a non-obvious way
4. Do NOT simply restate what the background papers already conclude
5. Be specific — name the exact mechanism, molecule, algorithm, or system
6. Be feasible — it should be testable with current technology
7. Output ONLY the hypothesis, no preamble or explanation

Hypothesis:"""

V1_FALLBACK_B = """\
Based on the following research area and related work, propose one specific
scientific hypothesis in a single paragraph.
Total length 80-150 words.

Field: {domain}
Related work summary: {ref_summary}"""


# ---------------------------------------------------------------------------
# Keep thinking ON for all OR models (v1 era behaviour) via env var that
# utils/LLM.py reads. This used to monkey-patch cfg.is_us_key_model, which
# unintentionally also routed all traffic to OPENROUTER_US_API_KEY (the
# same predicate gates API-key selection). Env var only touches the
# thinking-OFF check; key routing stays correct.
# ---------------------------------------------------------------------------

os.environ["SCISYNTH_KEEP_THINKING_ON"] = "1"


# ---------------------------------------------------------------------------
# v1 reference formatting (copied from git 96f4f26)
# ---------------------------------------------------------------------------

def _format_refs(paper: dict) -> str:
    """v1 ref formatting: deterministic shuffle of ranked_refs_json."""
    import hashlib
    import random
    refs_json = paper.get("ranked_refs_json")
    if not refs_json:
        return "(no background literature available)"
    try:
        refs = json.loads(refs_json)
    except (json.JSONDecodeError, TypeError):
        return "(no background literature available)"
    if not refs:
        return "(no background literature available)"
    pid = paper.get("paper_id") or ""
    seed = int(hashlib.sha1(pid.encode("utf-8")).hexdigest()[:8], 16) if pid else 42
    shuffled = list(refs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    parts = []
    for i, ref in enumerate(shuffled, 1):
        title = ref.get("title") or "Untitled"
        abstract = (ref.get("abstract") or "").strip()
        entry = f"[{i}] {title}"
        if abstract:
            entry += f"\n{abstract}"
        parts.append(entry)
    return "\n\n".join(parts)


def _ref_summary(refs_json, max_words: int = 150) -> str:
    if not refs_json:
        return ""
    try:
        refs = json.loads(refs_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    pieces = []
    for ref in refs:
        abstract = (ref.get("abstract") or "").strip()
        if abstract:
            pieces.append(abstract[:200])
    joined = " ".join(pieces)
    return " ".join(joined.split()[:max_words])


def _clean_idea_text(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.lower().startswith("hypothesis:"):
        raw = raw.split(":", 1)[1].strip()
    return raw


# ---------------------------------------------------------------------------
# Worker functions (mirror generation/generate_ideas._generate_single_idea_task
# and evaluation/critic_manager.score_combo, but use v1 prompt + DB_RESULTS_V1)
# ---------------------------------------------------------------------------

import hashlib
import random

from utils.LLM import IdeaLLM, CriticLLM
from evaluation.absolute_scorer import score_idea


def _generate_one_v1(paper: dict, idea_model: str, idea_index: int) -> dict:
    user_prompt = V1_USER_B.format(domain=paper["domain"], references=_format_refs(paper))
    fallback = V1_FALLBACK_B.format(domain=paper["domain"], ref_summary=_ref_summary(paper.get("ranked_refs_json")))
    out = {"paper_id": paper["paper_id"], "idea_model": idea_model,
           "track": TRACK, "idea_index": idea_index,
           "error": None, "idea_text": None}
    try:
        llm = IdeaLLM(idea_model)
        res = llm.generate_idea(user_prompt, fallback_prompt=fallback, system_prompt=V1_SYSTEM_B)
        out["idea_text"] = _clean_idea_text(res["idea"])
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["idea_text"] = f"[generation error: {e}]"
    return out


def _score_combo_v1(paper: dict, idea_model: str, ideas: List[dict], critic_model: str) -> dict:
    """Score 3 ideas for one (paper, model, critic) — uses v1 absolute_scorer
    which already supports static judge_mode + survey refs."""
    out_rows = []
    refs_text = _format_refs(paper)
    for idea in ideas:
        try:
            scores_dict, raw_resp, _ = score_idea(
                idea["idea_text"], critic_model,
                domain=paper.get("domain", ""), references=refs_text,
                judge_mode="static", db_papers=DB_PAPERS,
            )
            out_rows.append({
                "kind": "success",
                "paper_id": paper["paper_id"], "idea_model": idea_model,
                "track": TRACK, "idea_index": idea["idea_index"],
                "idea_text": idea["idea_text"], "critic_model": critic_model,
                "scores_json": json.dumps({d: v["score"] for d, v in scores_dict.items()}) if scores_dict else None,
                "reasoning_json": json.dumps({d: v["reasoning"] for d, v in scores_dict.items()}) if scores_dict else None,
                "raw_response": (raw_resp or "")[:4000],
                "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            out_rows.append({
                "kind": "error",
                "paper_id": paper["paper_id"], "idea_model": idea_model,
                "track": TRACK, "idea_index": idea["idea_index"],
                "idea_text": idea["idea_text"], "critic_model": critic_model,
                "scores_json": None, "reasoning_json": None, "raw_response": None,
                "error": str(e)[:500],
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
    return {"rows": out_rows}


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------

def _insert_idea(conn: sqlite3.Connection, row: dict) -> bool:
    """Returns True if inserted (False if duplicate idea_index existed)."""
    cur = conn.cursor()
    ts = datetime.now(timezone.utc).isoformat()
    res = cur.execute("""
        INSERT OR IGNORE INTO results
          (paper_id, idea_model, track, idea_index, idea_text,
           critic_model, raw_response, created_at)
        VALUES (?, ?, ?, ?, ?, '', '', ?)
    """, (row["paper_id"], row["idea_model"], row["track"], row["idea_index"],
          row["idea_text"], ts))
    return res.rowcount > 0


def _insert_critic_rows(conn: sqlite3.Connection, rows: List[dict]) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute("""
            INSERT INTO results
              (paper_id, idea_model, track, idea_index, idea_text,
               critic_model, scores_json, reasoning_json, raw_response,
               error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (r["paper_id"], r["idea_model"], r["track"], r["idea_index"],
              r["idea_text"], r["critic_model"], r.get("scores_json"),
              r.get("reasoning_json"), r.get("raw_response"), r.get("error"),
              r["created_at"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("v1_recompute")


def select_papers() -> List[dict]:
    conn = sqlite3.connect(DB_PAPERS)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM papers WHERE status='filtered' "
        "AND gt_hypothesis IS NOT NULL AND gt_hypothesis != '' "
        "ORDER BY domain, paper_id"
    ).fetchall()
    conn.close()
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(dict(r))
    selected = []
    for d in sorted(by_domain.keys()):
        selected.extend(by_domain[d][:N_PAPERS_PER_DOMAIN])
    return selected


def select_critics(idea_model: str, seed: int) -> List[str]:
    available = [m for m in V1_CRITIC_POOL if m != idea_model]
    if len(available) < N_CRITICS_PER_COMBO:
        return available
    return random.Random(seed).sample(available, N_CRITICS_PER_COMBO)


def phase2_generate(papers, models, workers):
    tasks = [(p, m, idx) for p in papers for m in models for idx in (1, 2, 3)]
    logger.info(f"Phase 2: {len(tasks)} idea calls ({len(papers)} papers × {len(models)} models × 3)")
    t0 = time.time(); done = errs = 0
    conn = sqlite3.connect(DB_RESULTS_V1, timeout=60)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_generate_one_v1, p, m, idx): (p["paper_id"], m, idx) for p, m, idx in tasks}
        for fut in cf.as_completed(futs):
            done += 1
            try:
                row = fut.result()
                if row["error"]:
                    errs += 1
                _insert_idea(conn, row)
                conn.commit()
            except Exception as e:
                errs += 1
                logger.error(f"  idea task error: {e}")
            if done % 25 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 0.01)
                logger.info(f"  Phase 2 [{done}/{len(tasks)}] err={errs} ({rate:.1f}/s)")
    conn.close()
    logger.info(f"Phase 2 done in {time.time() - t0:.0f}s ({errs} errors)")


def _critics_done(conn, paper_id, idea_model, n_ideas):
    rows = conn.execute("""
        SELECT critic_model FROM results
        WHERE paper_id=? AND idea_model=? AND track=? AND scores_json IS NOT NULL
        GROUP BY critic_model HAVING COUNT(DISTINCT idea_index) >= ?
    """, (paper_id, idea_model, TRACK, n_ideas)).fetchall()
    return {r[0] for r in rows}


def phase3_score(papers, models, workers):
    logger.info(f"Phase 3: critic pool = {V1_CRITIC_POOL}")
    tasks = []
    skipped = 0
    for p in papers:
        for m in models:
            conn = sqlite3.connect(DB_RESULTS_V1)
            conn.row_factory = sqlite3.Row
            ideas = [dict(r) for r in conn.execute(
                "SELECT idea_index, idea_text FROM results WHERE paper_id=? AND idea_model=? AND track=? AND critic_model=''",
                (p["paper_id"], m, TRACK)).fetchall()]
            done_critics = _critics_done(conn, p["paper_id"], m, len(ideas)) if ideas else set()
            conn.close()
            if not ideas:
                logger.warning(f"  no ideas for {p['paper_id'][:12]} × {m}")
                continue
            seed = abs(hash((p["paper_id"], m))) % (10**9)
            critics = select_critics(m, seed)
            for c in critics:
                if c in done_critics:
                    skipped += 1
                    continue
                tasks.append((p, m, ideas, c))
    logger.info(f"Phase 3: {len(tasks)} critic calls (skipped {skipped} already-done)")
    t0 = time.time(); done = errs = 0
    conn = sqlite3.connect(DB_RESULTS_V1, timeout=60)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_score_combo_v1, p, m, ideas, c): (p["paper_id"], m, c) for p, m, ideas, c in tasks}
        for fut in cf.as_completed(futs):
            done += 1
            try:
                res = fut.result()
                _insert_critic_rows(conn, res["rows"])
                errs += sum(1 for r in res["rows"] if r["kind"] == "error")
                conn.commit()
            except Exception as e:
                errs += 1
                logger.error(f"  critic task error: {e}")
            if done % 25 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 0.01)
                eta = (len(tasks) - done) / max(rate, 0.01)
                logger.info(f"  Phase 3 [{done}/{len(tasks)}] err={errs} ({rate:.2f}/s, eta {eta:.0f}s)")
    conn.close()
    logger.info(f"Phase 3 done in {time.time() - t0:.0f}s ({errs} errors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-phase2", action="store_true")
    ap.add_argument("--skip-phase3", action="store_true")
    args = ap.parse_args()

    papers = select_papers()
    models = V1_IDEA_MODELS_TO_RECOMPUTE[:]
    if args.smoke:
        papers = papers[:1]
        models = models[:2]

    logger.info(f"V1 recompute: {len(papers)} papers × {len(models)} new idea models")
    logger.info(f"  (claude-3.7-sonnet + qwen3-32b reuse existing v1 rows in {DB_RESULTS_V1})")

    if not args.skip_phase2:
        phase2_generate(papers, models, args.workers)
    if not args.skip_phase3:
        phase3_score(papers, models, args.workers)

    # Phase 4: compute model_scores for all v1 cross-year models (including
    # claude-3.7 + qwen3-32b which still have their original v1 rows).
    logger.info("Phase 4: compute_scores on results_v1.db")
    from analysis.compute_scores import compute_scores
    res = compute_scores(DB_RESULTS_V1, DB_PAPERS)
    logger.info(f"  model_scores: {res.get('rows_written', 0)} rows")

    logger.info("Done. Run reports/_make_cross_year_plot_v1.py separately for v1 plots.")


if __name__ == "__main__":
    main()
