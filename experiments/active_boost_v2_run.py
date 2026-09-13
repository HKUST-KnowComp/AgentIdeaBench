"""active_boost_v2_run.py — Active vs Static under the v2_topic_refs prompt design.

v2 design (vs v1_paper_refs):
  - Track B (Static): give paper TITLE + 10 SS-search refs (from domain_topic_refs cache,
    pre-fetched by data_collection/fetch_domain_topic_refs.py).
    Old v1 gave paper.ranked_refs_json (20 paper-specific refs) and NO title.
  - Track C (Active): give paper TITLE + agent freely SEARCH/FETCH via SS.
    Old v1 gave domain only, no title.
  - Critic for both tracks: use the same v2 refs (paper.query SS top-10) for fairness.

Open-source models only (no OpenAI/Anthropic/Gemini per user direction 2026-05-18).
Results tagged prompt_version='v2_topic_refs' in `results`. v1 data untouched.

Usage:
  python experiments/active_boost_v2_run.py --phase gen-static --smoke
  python experiments/active_boost_v2_run.py --phase gen-active --smoke
  python experiments/active_boost_v2_run.py --phase critic --smoke
  python experiments/active_boost_v2_run.py --phase report --smoke
  python experiments/active_boost_v2_run.py --phase all --smoke
  python experiments/active_boost_v2_run.py --phase all --full
"""
import argparse
import collections
import json
import logging
import sqlite3
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROMPT_VERSION = "v2_topic_refs"

# 13 open-source idea models. All routed via default OR key (no US-tier prefixes).
TARGET_MODELS = [
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-vl-8b-thinking",
    "mistralai/mistral-7b-instruct-v0.1",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-3-27b-it",
    "google/gemma-4-31b-it",
    "deepseek/deepseek-r1-0528",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
]

CRITIC_POOL = [
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5.1",
    "moonshotai/kimi-k2.6",
]

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())


# ───────────────────────────────────────────────────────────────────────────
# v2 prompts (Track B). Track C prompts live in active_agent.USER_PROMPT_TEXT_V2.
# ───────────────────────────────────────────────────────────────────────────
_SYSTEM_B_V2 = (
    "You are a creative and rigorous scientist. "
    "Your task is to propose a single novel, testable scientific hypothesis "
    "in the same area as the paper described below. You will be given the "
    "paper's title plus a set of background papers retrieved from a Semantic "
    "Scholar topic search. Use the background to understand the landscape and "
    "to AVOID duplicating existing work — your hypothesis should be a new "
    "research direction, not an extension or synthesis of these specific "
    "papers."
)

_USER_B_V2 = """\
Propose a single novel scientific hypothesis related to the paper below.

Field: {domain}
Paper title: {title}

Background papers (Semantic Scholar search results — title + abstract):
{references}

Requirements:
1. Output exactly one paragraph
2. Total length: 80-150 words
3. Your hypothesis should be a new direction in the same area, not a derivative
   of the listed background papers
4. Do NOT summarize, synthesize, or merely extend the background — propose a
   research idea that could stand on its own
5. Be specific — name the exact mechanism, molecule, algorithm, or system
6. Be feasible — it should be testable with current technology
7. Output ONLY the hypothesis, no preamble or explanation

Hypothesis:"""


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in DIMS) / WS


# ───────────────────────────────────────────────────────────────────────────
# Data loaders
# ───────────────────────────────────────────────────────────────────────────
def fetch_papers(smoke: bool, n_per_domain: int):
    conn = sqlite3.connect(str(cfg.PAPERS_DB))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("""
        SELECT paper_id, title, domain, query
          FROM papers
         WHERE status='filtered'
         ORDER BY domain, paper_id
    """).fetchall()]
    conn.close()
    by_dom = collections.defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    n = n_per_domain if not smoke else 1
    out = [p for ps in by_dom.values() for p in ps[:n]]
    logger.info(f"Loaded {len(out)} papers (smoke={smoke}, n_per_domain={n})")
    return out


def fetch_cache_refs(query: str, conn_r: sqlite3.Connection) -> list:
    row = conn_r.execute(
        "SELECT refs_json FROM domain_topic_refs WHERE query=?", (query,)).fetchone()
    if not row or not row[0]:
        return []
    try:
        return json.loads(row[0])
    except Exception:
        return []


def format_refs_block(refs: list) -> str:
    if not refs:
        return "(no background papers available)"
    parts = []
    for i, r in enumerate(refs, 1):
        title = r.get("title") or "(untitled)"
        abstract = (r.get("abstract") or "").strip()
        year = r.get("year") or "?"
        entry = f"[{i}] ({year}) {title}"
        if abstract:
            entry += f"\n{abstract}"
        parts.append(entry)
    return "\n\n".join(parts)


# ───────────────────────────────────────────────────────────────────────────
# Phase: gen-static (Track B v2)
# ───────────────────────────────────────────────────────────────────────────
def phase_gen_static(papers, workers=4):
    from utils.LLM import IdeaLLM

    conn_r = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn_r.execute("PRAGMA busy_timeout=30000")

    tasks = []
    for p in papers:
        for m in TARGET_MODELS:
            exists = conn_r.execute(
                "SELECT 1 FROM results WHERE paper_id=? AND idea_model=? "
                "AND track='B' AND idea_index=1 AND critic_model='' "
                "AND prompt_version=?",
                (p["paper_id"], m, PROMPT_VERSION)).fetchone()
            if not exists:
                tasks.append((p, m))

    logger.info(f"phase_gen_static (v2): {len(tasks)} idea tasks")
    if not tasks:
        return 0

    # Pre-warm cache refs to avoid repeated DB hits
    cache = {}
    for p in papers:
        cache[p["paper_id"]] = fetch_cache_refs(p["query"], conn_r)

    def _worker(task):
        paper, model = task
        refs = cache.get(paper["paper_id"], [])
        if not refs:
            return ("skip_no_refs", paper, model, None)
        refs_text = format_refs_block(refs)
        prompt = _USER_B_V2.format(domain=paper["domain"], title=paper["title"],
                                    references=refs_text)
        try:
            llm = IdeaLLM(model_name=model)
            result = llm.generate_idea(prompt=prompt, system_prompt=_SYSTEM_B_V2)
            text = (result.get("idea") or "").strip()
            if not text or len(text.split()) < 30:
                return ("err_short", paper, model, text)
            return ("ok", paper, model, text)
        except Exception as e:
            return ("err", paper, model, str(e)[:300])

    ts = datetime.now(timezone.utc).isoformat()
    stats = collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            status, paper, model, text = f.result()
            if status == "ok":
                conn_r.execute("""
                    INSERT OR IGNORE INTO results
                      (paper_id, idea_model, track, idea_index, idea_text,
                       critic_model, created_at, prompt_version)
                    VALUES (?, ?, 'B', 1, ?, '', ?, ?)
                """, (paper["paper_id"], model, text, ts, PROMPT_VERSION))
                conn_r.commit()
            stats[status] += 1
            if i % 10 == 0 or i == len(tasks):
                logger.info(f"  [{i}/{len(tasks)}] {dict(stats)}")
    conn_r.close()
    logger.info(f"phase_gen_static done: {dict(stats)}")
    return sum(stats.values())


# ───────────────────────────────────────────────────────────────────────────
# Phase: gen-active (Track C v2)
# ───────────────────────────────────────────────────────────────────────────
def phase_gen_active(papers, workers=3):
    from generation.active_agent import run_active_agent

    conn_r = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn_r.execute("PRAGMA busy_timeout=30000")

    tasks = []
    for p in papers:
        for m in TARGET_MODELS:
            exists = conn_r.execute(
                "SELECT 1 FROM results WHERE paper_id=? AND idea_model=? "
                "AND track='C' AND idea_index=1 AND critic_model='' "
                "AND prompt_version=?",
                (p["paper_id"], m, PROMPT_VERSION)).fetchone()
            if not exists:
                tasks.append((p, m))

    logger.info(f"phase_gen_active (v2): {len(tasks)} idea tasks")
    if not tasks:
        return 0

    def _worker(task):
        paper, model = task
        try:
            result = run_active_agent(
                domain=paper["domain"], model_name=model,
                title=paper["title"], max_iters=10,
            )
            hyp = (result.get("hypothesis") or "").strip()
            if not hyp or len(hyp.split()) < 30:
                return ("err_short", paper, model, None, result)
            return ("ok", paper, model, hyp, result)
        except Exception as e:
            return ("err", paper, model, None, str(e)[:300])

    ts = datetime.now(timezone.utc).isoformat()
    stats = collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            status, paper, model, hyp, payload = f.result()
            if status == "ok":
                tel = {
                    "trace": payload.get("trace", []),
                    "iters_used": payload.get("iters_used"),
                    "n_tool_calls": payload.get("n_tool_calls"),
                    "final_prompt_tokens": payload.get("final_prompt_tokens"),
                }
                conn_r.execute("""
                    INSERT OR IGNORE INTO results
                      (paper_id, idea_model, track, idea_index, idea_text,
                       critic_model, raw_response, created_at, prompt_version)
                    VALUES (?, ?, 'C', 1, ?, '', ?, ?, ?)
                """, (paper["paper_id"], model, hyp,
                      json.dumps(tel, ensure_ascii=False), ts, PROMPT_VERSION))
                conn_r.commit()
            stats[status] += 1
            logger.info(f"  [{i}/{len(tasks)}] {model:<42} {paper['domain']:<10} → {status}")
    conn_r.close()
    logger.info(f"phase_gen_active done: {dict(stats)}")
    return sum(stats.values())


# ───────────────────────────────────────────────────────────────────────────
# Phase: critic (both tracks)
# ───────────────────────────────────────────────────────────────────────────
def phase_critic(papers, workers=4):
    from evaluation.absolute_scorer import score_idea

    conn_r = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn_r.execute("PRAGMA busy_timeout=30000")
    conn_p = sqlite3.connect(str(cfg.PAPERS_DB), timeout=30)
    conn_p.row_factory = sqlite3.Row
    papers_by_id = {p["paper_id"]: p for p in papers}

    # Pre-warm cache refs (also used by critic to evaluate Track B and Track C
    # under the same v2 ref frame)
    cache = {p["paper_id"]: fetch_cache_refs(p["query"], conn_r) for p in papers}
    cache_text = {pid: format_refs_block(refs) for pid, refs in cache.items()}

    # Fetch all v2 ideas needing scoring
    paper_ids = list(papers_by_id)
    pidp = ",".join("?" * len(paper_ids))
    mp = ",".join("?" * len(TARGET_MODELS))
    ideas = list(conn_r.execute(f"""
        SELECT paper_id, idea_model, track, idea_text
        FROM results
        WHERE track IN ('B','C') AND idea_index=1 AND critic_model=''
          AND idea_model IN ({mp}) AND paper_id IN ({pidp})
          AND prompt_version=? AND idea_text IS NOT NULL AND idea_text != ''
    """, list(TARGET_MODELS) + paper_ids + [PROMPT_VERSION]))

    tasks = []
    for pid, idea_model, track, idea_text in ideas:
        for critic in CRITIC_POOL:
            if critic == idea_model:
                continue  # skip self-eval
            exists = conn_r.execute(
                "SELECT 1 FROM results WHERE paper_id=? AND idea_model=? "
                "AND track=? AND idea_index=1 AND critic_model=? AND prompt_version=?",
                (pid, idea_model, track, critic, PROMPT_VERSION)).fetchone()
            if not exists:
                tasks.append((pid, idea_model, track, idea_text, critic))

    logger.info(f"phase_critic (v2): {len(tasks)} score tasks")
    if not tasks:
        return 0

    def _worker(task):
        pid, idea_model, track, idea_text, critic = task
        paper = papers_by_id.get(pid)
        if not paper:
            row = conn_p.execute("SELECT * FROM papers WHERE paper_id=?", (pid,)).fetchone()
            paper = dict(row) if row else {}
        refs_text = cache_text.get(pid, "(no background)")
        try:
            scores, raw, telem = score_idea(
                idea_text, critic,
                domain=paper.get("domain", ""),
                references=refs_text,
                judge_mode="static",
                db_papers=str(cfg.PAPERS_DB),
            )
            return ("ok", pid, idea_model, track, critic, scores, raw)
        except Exception as e:
            return ("err", pid, idea_model, track, critic, None, str(e)[:300])

    ts = datetime.now(timezone.utc).isoformat()
    stats = collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            status = res[0]
            if status == "ok":
                _, pid, idea_model, track, critic, scores, raw = res
                if scores:
                    conn_r.execute("""
                        INSERT OR IGNORE INTO results
                          (paper_id, idea_model, track, idea_index, idea_text,
                           critic_model, scores_json, reasoning_json,
                           raw_response, created_at, prompt_version)
                        VALUES (?, ?, ?, 1, '', ?, ?, ?, ?, ?, ?)
                    """, (pid, idea_model, track, critic,
                          json.dumps({d: v["score"] for d, v in scores.items()}),
                          json.dumps({d: v["reasoning"] for d, v in scores.items()}),
                          (raw or "")[:4000], ts, PROMPT_VERSION))
                    conn_r.commit()
                    stats["ok"] += 1
                else:
                    stats["empty"] += 1
            else:
                stats["err"] += 1
            if i % 20 == 0 or i == len(tasks):
                logger.info(f"  [{i}/{len(tasks)}] {dict(stats)}")
    conn_r.close()
    conn_p.close()
    logger.info(f"phase_critic done: {dict(stats)}")
    return sum(stats.values())


# ───────────────────────────────────────────────────────────────────────────
# Phase: report
# ───────────────────────────────────────────────────────────────────────────
def phase_report(papers):
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    paper_ids = list(p["paper_id"] for p in papers)
    pidp = ",".join("?" * len(paper_ids))
    mp = ",".join("?" * len(TARGET_MODELS))

    by_combo = collections.defaultdict(list)  # (model, track) -> list of weighted-scores
    for r in conn.execute(f"""
        SELECT idea_model, track, scores_json
          FROM results
         WHERE track IN ('B','C') AND idea_index=1 AND critic_model != ''
           AND idea_model IN ({mp}) AND paper_id IN ({pidp})
           AND scores_json IS NOT NULL AND prompt_version=?
    """, list(TARGET_MODELS) + paper_ids + [PROMPT_VERSION]):
        model, track, sj = r
        try:
            s = json.loads(sj)
        except Exception:
            continue
        w = weighted(s)
        by_combo[(model, track)].append(w)
    conn.close()

    print("\n" + "=" * 76)
    print(f"v2 (topic refs + title) — RESULTS  [smoke={'yes' if len(papers)<10 else 'no'}]")
    print("=" * 76)
    print(f"{'Model':<42} {'B_v2':>7} {'C_v2':>7} {'Boost':>7} {'n_B':>4} {'n_C':>4}")
    rows = []
    for m in TARGET_MODELS:
        b_vals = by_combo.get((m, "B"), [])
        c_vals = by_combo.get((m, "C"), [])
        b_mean = statistics.mean(b_vals) if b_vals else None
        c_mean = statistics.mean(c_vals) if c_vals else None
        boost = (c_mean - b_mean) if (b_mean is not None and c_mean is not None) else None
        rows.append((m, b_mean, c_mean, boost, len(b_vals), len(c_vals)))
    rows.sort(key=lambda r: (r[3] is None, -(r[3] or 0)))
    for m, b, c, bo, nb, nc in rows:
        b_s = f"{b:>7.3f}" if b is not None else "    n/a"
        c_s = f"{c:>7.3f}" if c is not None else "    n/a"
        bo_s = f"{bo:>+7.3f}" if bo is not None else "    n/a"
        print(f"  {m:<40} {b_s} {c_s} {bo_s} {nb:>4} {nc:>4}")

    boosts = [r[3] for r in rows if r[3] is not None]
    if boosts:
        print(f"\n  Mean boost across {len(boosts)} models: "
              f"{statistics.mean(boosts):+.3f}")
        print(f"  Range: {min(boosts):+.3f} to {max(boosts):+.3f}")


# ───────────────────────────────────────────────────────────────────────────
# Entrypoint
# ───────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["gen-static", "gen-active", "critic", "report", "all"])
    ap.add_argument("--smoke", action="store_true",
                    help="1 paper/domain × 3 small models")
    ap.add_argument("--full", action="store_true",
                    help="5 paper/domain × 13 models")
    ap.add_argument("--n-per-domain", type=int, default=5)
    ap.add_argument("--models", nargs="+", help="Override TARGET_MODELS")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    if not (args.smoke or args.full or args.n_per_domain):
        ap.error("Need --smoke or --full")

    global TARGET_MODELS
    if args.models:
        TARGET_MODELS = args.models
    elif args.smoke and not args.full:
        # Use 3 small models for smoke (mostly to verify pipe + prompt rendering)
        TARGET_MODELS = [
            "qwen/qwen3.5-9b",
            "mistralai/mistral-7b-instruct-v0.1",
            "deepseek/deepseek-r1-0528",
        ]

    papers = fetch_papers(smoke=args.smoke, n_per_domain=args.n_per_domain)

    t0 = time.time()
    if args.phase in ("gen-static", "all"):
        phase_gen_static(papers, workers=args.workers)
    if args.phase in ("gen-active", "all"):
        phase_gen_active(papers, workers=args.workers)
    if args.phase in ("critic", "all"):
        phase_critic(papers, workers=args.workers)
    if args.phase in ("report", "all"):
        phase_report(papers)
    logger.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
