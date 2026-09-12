"""Active Mode comparison: Qwen3.5-9B/27B vs Claude-3.7-Sonnet/Sonnet-4.

Motivation (user, 2026-05-17):
  Static-mode cross-year shows Qwen3.5-9B/27B ≈ Claude-3.7-Sonnet/Sonnet-4
  (within 0.2 weighted score). If Claude wins Active Mode by a margin, it
  supports "agentic capability stratifies even when static benchmarks compress."

Two critic modes (user, 2026-05-18):
  1. agent_refs       — refs = papers the agent ACTUALLY fetched during
                        ideation (parsed from trace's search_papers/get_paper)
  2. dynamic_title    — refs = SS keyword search of the agent's proposed TITLE,
                        top_n=10

Storage: dedicated tables — `active_comparison_ideas` (one row per
(paper, model) Active-mode run, with full trace stored without truncation)
and `active_comparison_scores` (one row per (idea, critic, mode)). This keeps
the existing results.db.results Track C data untouched per CLAUDE.md §9.5.

Smoke: 5 paper × 4 model × 2 mode × 4 critic = 160 score rows + 20 active runs.
"""
import argparse
import hashlib
import json
import logging
import random
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

TARGET_MODELS = [
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
    "anthropic/claude-sonnet-4",     # US-key (claude-3.7-sonnet was delisted from OR around 2026-05-18)
]

# Critic pool (user-confirmed 2026-05-18: keep qwen3.6-plus).
DEFAULT_CRITIC_POOL = [
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5.1",
    "moonshotai/kimi-k2.5",
    "qwen/qwen3.6-plus",
]

# Active-mode prompt variant — requires the agent to emit a Title line before
# the hypothesis paragraph. Critic Mode 2 uses this Title as the SS query.
ACTIVE_SYSTEM_PROMPT_WITH_TITLE = """\
You are a creative and rigorous scientist. You will be given a research domain \
name and you must propose a single novel, testable scientific hypothesis in \
that domain.

To do this, you have access to Semantic Scholar search tools. You should:
  1. Search the literature to identify recent work and open problems in the domain
  2. (Optional) Drill into a promising paper's references for deeper context
  3. Synthesize a genuinely novel hypothesis that goes BEYOND what exists

You have a budget of {max_iters} tool calls total. Explore broadly before \
committing to a direction: try multiple distinct angles (different sub-areas, \
different methods, different recent vs foundational works) and use \
`get_paper_references` to drill into the most promising lead. Aim to use most \
of your budget; only stop early if you have already found a genuine, \
non-obvious gap that is well-supported by what you read.

Your FINAL output (after all tool calls) MUST be EXACTLY two lines:
  Title: <a concise, paper-style title of your hypothesis, 8-15 words>
  Hypothesis: <one paragraph, 80-150 words, first-person future tense; be \
specific — name mechanisms, methods, datasets>

Do NOT include any other text, preamble, or explanation. Use the exact \
"Title:" and "Hypothesis:" labels at the start of each line."""

JUDGE_MODES = ["agent_refs", "dynamic_title"]
SS_TOP_N = 10   # Mode 2 fixed top_n (user-confirmed default)


# ────────────────────────────────────────────────────────────────────────────
# DB schema (separate tables; never touches results.db.results)
# ────────────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS active_comparison_ideas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        TEXT    NOT NULL,
    domain          TEXT    NOT NULL,
    idea_model      TEXT    NOT NULL,
    title           TEXT,                 -- parsed "Title: ..." line
    hypothesis      TEXT,                 -- parsed "Hypothesis: ..." paragraph
    raw_output      TEXT,                 -- full LLM final output (before parse)
    trace_json      TEXT,                 -- full agent trace (NO truncation)
    telemetry_json  TEXT,                 -- {latency_ms, iters_used, n_tool_calls, ...}
    error           TEXT,
    created_at      TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_aci_key
    ON active_comparison_ideas (paper_id, idea_model);
CREATE INDEX IF NOT EXISTS idx_aci_model ON active_comparison_ideas (idea_model);

CREATE TABLE IF NOT EXISTS active_comparison_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        TEXT    NOT NULL,
    idea_model      TEXT    NOT NULL,
    critic_model    TEXT    NOT NULL,
    judge_mode      TEXT    NOT NULL,    -- 'agent_refs' or 'dynamic_title'
    scores_json     TEXT,
    reasoning_json  TEXT,
    raw_response    TEXT,
    telemetry_json  TEXT,
    error           TEXT,
    created_at      TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_acs_key
    ON active_comparison_scores
    (paper_id, idea_model, critic_model, judge_mode);
"""


def init_tables(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


# ────────────────────────────────────────────────────────────────────────────
# Phase 2: Active-mode generation
# ────────────────────────────────────────────────────────────────────────────

def parse_title_and_hypothesis(raw: str) -> tuple:
    """Extract (title, hypothesis) from raw 'Title: ...\nHypothesis: ...'.

    Robust to extra whitespace / case variation. Returns (title, hypothesis);
    either may be None if not present.
    """
    if not raw:
        return (None, None)
    import re
    title_m = re.search(r"(?im)^\s*Title\s*:\s*(.+?)\s*(?:\n|$)", raw)
    hyp_m   = re.search(r"(?im)^\s*Hypothesis\s*:\s*(.+)$", raw, re.DOTALL)
    title = title_m.group(1).strip() if title_m else None
    hyp   = hyp_m.group(1).strip()   if hyp_m else None
    if hyp:
        # Strip any trailing extra content after a blank line (defensive)
        parts = re.split(r"\n\s*\n", hyp, maxsplit=1)
        hyp = parts[0].strip()
    return (title, hyp)


def fetch_smoke_papers(db_papers: str, n_per_domain: int = 1) -> list:
    """Smoke = 1 paper per domain → 5 papers total (5 domains)."""
    import collections
    conn = sqlite3.connect(db_papers)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("""
        SELECT * FROM papers
        WHERE status='filtered' AND gt_hypothesis IS NOT NULL AND gt_hypothesis != ''
          AND ranked_refs_json IS NOT NULL
        ORDER BY domain, paper_id
    """).fetchall()]
    by_domain = collections.defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
    out = []
    for d, ps in sorted(by_domain.items()):
        out.extend(ps[:n_per_domain])
    return out


def phase2_generate(db_path: str, db_papers: str, n_per_domain: int = 1,
                    max_iters: int = 10, workers: int = 4) -> dict:
    from generation.active_agent import run_active_agent

    init_tables(db_path)
    papers = fetch_smoke_papers(db_papers, n_per_domain=n_per_domain)
    logger.info(f"Phase 2: {len(papers)} papers × {len(TARGET_MODELS)} models")

    # Build task list: skip already-generated rows
    conn = sqlite3.connect(db_path, timeout=30)
    tasks = []
    for p in papers:
        for m in TARGET_MODELS:
            exists = conn.execute(
                "SELECT 1 FROM active_comparison_ideas WHERE paper_id=? AND idea_model=?",
                (p["paper_id"], m)).fetchone()
            if exists:
                continue
            tasks.append((p, m))
    logger.info(f"  tasks: {len(tasks)} to run, {len(papers)*len(TARGET_MODELS)-len(tasks)} already done")

    def _worker(task):
        paper, model = task
        t0 = time.time()
        try:
            result = run_active_agent(
                domain=paper["domain"],
                model_name=model,
                max_iters=max_iters,
                system_prompt_override=ACTIVE_SYSTEM_PROMPT_WITH_TITLE,
            )
            raw = result.get("hypothesis", "") or ""
            title, hyp = parse_title_and_hypothesis(raw)
            telemetry = {
                "latency_ms": int((time.time() - t0) * 1000),
                "iters_used": result.get("iters_used"),
                "n_tool_calls": result.get("n_tool_calls"),
                "n_turns": result.get("n_turns"),
                "final_prompt_tokens": result.get("final_prompt_tokens"),
                "budget_nudge_used": result.get("budget_nudge_used"),
                "error": result.get("error"),
            }
            return {
                "paper": paper, "model": model,
                "title": title, "hypothesis": hyp, "raw": raw,
                "trace": result.get("trace", []),
                "telemetry": telemetry,
                "error": None if (title and hyp) else "parse_failed",
            }
        except Exception as e:
            return {"paper": paper, "model": model, "error": str(e)[:500]}

    stats = {"ok": 0, "parse_err": 0, "agent_err": 0}
    ts = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            err = r.get("error")
            if err and err != "parse_failed":
                stats["agent_err"] += 1
            elif err == "parse_failed":
                stats["parse_err"] += 1
            else:
                stats["ok"] += 1
            conn.execute("""
                INSERT OR IGNORE INTO active_comparison_ideas
                  (paper_id, domain, idea_model, title, hypothesis,
                   raw_output, trace_json, telemetry_json, error, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                r["paper"]["paper_id"], r["paper"]["domain"], r["model"],
                r.get("title"), r.get("hypothesis"),
                r.get("raw"),
                json.dumps(r.get("trace", []), ensure_ascii=False)[:1_000_000],  # 1MB cap, far above need
                json.dumps(r.get("telemetry", {})),
                err, ts,
            ))
            conn.commit()
            logger.info(f"  [{i}/{len(tasks)}] {r['model']:<35} {r['paper']['domain']:<12} → {'OK' if not err else err}")
    conn.close()
    return stats


# ────────────────────────────────────────────────────────────────────────────
# Phase 3: critic scoring under 2 modes
# ────────────────────────────────────────────────────────────────────────────

def extract_refs_from_trace(trace_json: str) -> list:
    """Mode 1 (agent_refs): extract papers the agent actually fetched.

    Trace structure (from active_agent.py):
      [{"iter": int, "tool": "search_papers"|"get_paper_references",
        "args": {...}, "result_preview": JSON-stringified array}]

    Each result_preview is a JSON list of paper dicts:
      [{"paperId", "title", "abstract", ...}, ...]

    Returns deduplicated list (by paperId) preserving discovery order.
    """
    if not trace_json:
        return []
    try:
        trace = json.loads(trace_json)
    except Exception:
        return []
    seen = set()
    out = []
    for ev in trace:
        if not isinstance(ev, dict):
            continue
        if ev.get("tool") not in ("search_papers", "get_paper_references"):
            continue
        rp = ev.get("result_preview")
        if not rp:
            continue
        # result_preview is a JSON-stringified list (may be truncated for
        # very long results — handle parse errors gracefully)
        try:
            papers = json.loads(rp)
        except Exception:
            continue
        if not isinstance(papers, list):
            continue
        for p in papers:
            if not isinstance(p, dict):
                continue
            pid = p.get("paperId") or p.get("paper_id")
            title = p.get("title") or ""
            if not title:
                continue
            # Dedup key: paperId if available, else title (lowercased)
            key = pid if pid else title.lower()
            if key in seen:
                continue
            seen.add(key)
            abstract = (p.get("abstract") or "").strip()
            out.append({
                "paperId": pid,
                "title": title,
                "abstract": abstract[:1500],
            })
    return out


# Stop-word list for title → SS keyword query extraction (observed: full
# 11-word titles return 0 results on SS; 4-word keyword queries return 10).
_TITLE_STOPWORDS = frozenset({
    "a", "an", "the", "of", "for", "to", "in", "on", "with", "by", "and",
    "or", "is", "are", "as", "at", "from", "via", "using", "based", "into",
    "be", "can", "this", "that", "these", "those", "such", "their", "its",
})


def title_to_query(title: str, max_words: int = 7) -> str:
    """Extract a SS-friendly keyword query from a long title.

    Drops stopwords and common connectors, keeps content words in original
    order, caps at max_words. Hyphens split into tokens.
    """
    import re
    tokens = re.split(r"[\s\-/,:;()]+", title)
    kept = [t for t in tokens
            if t and t.lower() not in _TITLE_STOPWORDS and len(t) > 1]
    return " ".join(kept[:max_words])


def fetch_refs_by_title(title: str, top_n: int = SS_TOP_N) -> list:
    """Mode 2 (dynamic_title): SS keyword search using the agent-proposed title.

    Strategy:
      1. Try the FULL title first (may be too narrow, often 0 results).
      2. Fall back to keyword-extracted query (top-7 content words).
      3. Fall back to first 4 content words (last-resort).
    Returns up to top_n papers.
    """
    if not title:
        return []
    from data_collection.fetch_ss_search import _get, SS_BASE
    api_key = getattr(cfg, "SEMANTIC_SCHOLAR_API_KEY", None)

    queries = [
        title[:200],                        # full title first
        title_to_query(title, max_words=7), # stopword-stripped 7 words
        title_to_query(title, max_words=4), # last resort 4 words
    ]
    # Dedup while preserving order
    seen_q = set()
    queries = [q for q in queries if q and not (q in seen_q or seen_q.add(q))]

    for q in queries:
        data = _get(
            f"{SS_BASE}/paper/search",
            {
                "query": q,
                "fields": "paperId,title,abstract,year,citationCount",
                "limit": min(max(top_n, 1), 20),
            },
            api_key, 0.2,
        )
        if not data:
            continue
        raw = data.get("data") or []
        if not raw:
            continue
        out = []
        for p in raw[:top_n]:
            if not p:
                continue
            title_p = p.get("title") or ""
            abstract = (p.get("abstract") or "").strip()
            if not title_p:
                continue
            out.append({
                "paperId": p.get("paperId"),
                "title": title_p,
                "abstract": abstract[:1500],
            })
        if out:
            return out
    return []


def format_refs_block(refs: list) -> str:
    """Format refs list into the standard critic-prompt block."""
    if not refs:
        return "(no background literature retrieved)"
    parts = []
    for i, r in enumerate(refs, 1):
        entry = f"[{i}] {r['title']}"
        if r.get("abstract"):
            entry += f"\n{r['abstract']}"
        parts.append(entry)
    return "\n\n".join(parts)


def score_one(idea_row: dict, critic_model: str, mode: str,
              db_papers: str) -> dict:
    """Score a single (idea, critic, mode) combination."""
    from evaluation.absolute_scorer import score_idea

    paper_id = idea_row["paper_id"]
    domain = idea_row["domain"]
    hyp = idea_row["hypothesis"] or ""
    title = idea_row["title"] or ""

    if mode == "agent_refs":
        refs = extract_refs_from_trace(idea_row["trace_json"])
    elif mode == "dynamic_title":
        refs = fetch_refs_by_title(title, top_n=SS_TOP_N)
    else:
        raise ValueError(f"unknown mode: {mode}")

    refs_block = format_refs_block(refs)

    try:
        scores, raw, telem = score_idea(
            hyp, critic_model,
            domain=domain,
            references=refs_block,
            judge_mode="static",   # critic uses static prompt format (refs already prepared)
            db_papers=db_papers,
        )
        return {
            "paper_id": paper_id, "idea_model": idea_row["idea_model"],
            "critic_model": critic_model, "judge_mode": mode,
            "scores_json": json.dumps({d: v["score"] for d, v in scores.items()}) if scores else None,
            "reasoning_json": json.dumps({d: v["reasoning"] for d, v in scores.items()}) if scores else None,
            "raw_response": (raw or "")[:4000],
            "telemetry_json": json.dumps({**(telem or {}), "n_refs": len(refs)}),
            "error": None if scores else "no_scores",
        }
    except Exception as e:
        return {
            "paper_id": paper_id, "idea_model": idea_row["idea_model"],
            "critic_model": critic_model, "judge_mode": mode,
            "scores_json": None, "reasoning_json": None,
            "raw_response": None,
            "telemetry_json": json.dumps({"n_refs": len(refs)}),
            "error": str(e)[:500],
        }


def phase3_score(db_path: str, db_papers: str, critic_pool: list,
                 workers: int = 4) -> dict:
    init_tables(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row

    ideas = [dict(r) for r in conn.execute("""
        SELECT * FROM active_comparison_ideas
        WHERE hypothesis IS NOT NULL AND hypothesis != ''
        ORDER BY paper_id, idea_model
    """).fetchall()]
    logger.info(f"Phase 3: {len(ideas)} ideas × {len(critic_pool)} critic × {len(JUDGE_MODES)} mode")

    tasks = []
    for idea in ideas:
        for c in critic_pool:
            for mode in JUDGE_MODES:
                exists = conn.execute("""
                    SELECT 1 FROM active_comparison_scores
                    WHERE paper_id=? AND idea_model=? AND critic_model=? AND judge_mode=?
                """, (idea["paper_id"], idea["idea_model"], c, mode)).fetchone()
                if not exists:
                    tasks.append((idea, c, mode))

    logger.info(f"  tasks: {len(tasks)} to run")

    stats = {"ok": 0, "err": 0}
    ts = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(score_one, idea, c, mode, db_papers)
                for (idea, c, mode) in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r.get("error"):
                stats["err"] += 1
            else:
                stats["ok"] += 1
            conn.execute("""
                INSERT OR IGNORE INTO active_comparison_scores
                  (paper_id, idea_model, critic_model, judge_mode,
                   scores_json, reasoning_json, raw_response,
                   telemetry_json, error, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                r["paper_id"], r["idea_model"], r["critic_model"], r["judge_mode"],
                r["scores_json"], r["reasoning_json"], r["raw_response"],
                r["telemetry_json"], r["error"], ts,
            ))
            conn.commit()
            if i % 5 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                eta = (elapsed / i) * (len(tasks) - i)
                logger.info(f"  [{i}/{len(tasks)}] elapsed={elapsed:.0f}s eta={eta:.0f}s ({stats['ok']} ok, {stats['err']} err)")
    conn.close()
    return stats


# ────────────────────────────────────────────────────────────────────────────
# --check: coverage report
# ────────────────────────────────────────────────────────────────────────────

def check_coverage(db_path: str) -> None:
    init_tables(db_path)
    conn = sqlite3.connect(db_path)
    print("\n## active_comparison_ideas (per model)")
    print(f"{'model':<35} {'idea_n':>7} {'with_title':>11} {'with_hyp':>9} {'err':>5}")
    for m in TARGET_MODELS:
        n = conn.execute("SELECT COUNT(*) FROM active_comparison_ideas WHERE idea_model=?", (m,)).fetchone()[0]
        nt = conn.execute("SELECT COUNT(*) FROM active_comparison_ideas WHERE idea_model=? AND title IS NOT NULL", (m,)).fetchone()[0]
        nh = conn.execute("SELECT COUNT(*) FROM active_comparison_ideas WHERE idea_model=? AND hypothesis IS NOT NULL", (m,)).fetchone()[0]
        ne = conn.execute("SELECT COUNT(*) FROM active_comparison_ideas WHERE idea_model=? AND error IS NOT NULL", (m,)).fetchone()[0]
        print(f"  {m:<35} {n:>7} {nt:>11} {nh:>9} {ne:>5}")

    print("\n## active_comparison_scores (per model × mode)")
    print(f"{'model':<35} {'mode':<16} {'n':>5}")
    for m in TARGET_MODELS:
        for mode in JUDGE_MODES:
            n = conn.execute("""
                SELECT COUNT(*) FROM active_comparison_scores
                WHERE idea_model=? AND judge_mode=? AND scores_json IS NOT NULL
            """, (m, mode)).fetchone()[0]
            print(f"  {m:<35} {mode:<16} {n:>5}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--phase", type=int, choices=[2, 3])
    ap.add_argument("--smoke", action="store_true",
                    help="1 paper per domain (5 total) for smoke")
    ap.add_argument("--n-per-domain", type=int, default=None,
                    help="Override paper count per domain (--smoke = 1, full = 5)")
    ap.add_argument("--critic-pool", default=",".join(DEFAULT_CRITIC_POOL))
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--db", default=str(cfg.RESULTS_DB))
    ap.add_argument("--db-papers", default=str(cfg.PAPERS_DB))
    args = ap.parse_args()

    if args.check:
        check_coverage(args.db)
        return

    if args.phase == 2:
        if args.n_per_domain is not None:
            n_per_domain = args.n_per_domain
        else:
            n_per_domain = 1 if args.smoke else 5
        stats = phase2_generate(
            args.db, args.db_papers,
            n_per_domain=n_per_domain,
            max_iters=args.max_iters,
            workers=args.workers,
        )
        print(f"\n✓ Phase 2 done: {stats}")
    elif args.phase == 3:
        critic_pool = [c.strip() for c in args.critic_pool.split(",") if c.strip()]
        stats = phase3_score(args.db, args.db_papers, critic_pool, workers=args.workers)
        print(f"\n✓ Phase 3 done: {stats}")
    else:
        print("Use --check, or --phase {2,3} [--smoke]")


if __name__ == "__main__":
    main()
