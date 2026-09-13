"""Uniform critic re-score — fix cross-model comparability.

7 models were scored with an OLD critic pool (grok-4 / qwen3-max / deepseek-r1 /
hunyuan / kimi-k2-0905), inconsistent with the modern pool the other 33 models
use. This re-scores their EXISTING ideas (no regeneration) with the modern
5-critic pool, using the PRODUCTION critic prompt (absolute_scorer.score_idea),
so scores become directly comparable.

SAFETY:
  - Reads idea_text from results (prompt_version='v1_paper_refs', critic_model='').
  - Writes ONLY to new table `uniform_critic_scores`. The `results` table is NEVER
    touched (raw-data death rule). Idempotent (PK skip).
  - Critics are all open-weight → default OPENROUTER_API_KEY. No idea regeneration,
    so no closed-weight generation cost even for claude-3.7-sonnet.

Scope (full): ~2925 critic calls, ~$3-8, ~30-60 min.

Usage:
  python experiments/uniform_critic_rescore.py --smoke --models qwen/qwen3-32b   # validate
  python experiments/uniform_critic_rescore.py                                    # full 7 models
"""
import argparse, collections, json, logging, sqlite3, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from evaluation.absolute_scorer import score_idea
from evaluation.critic_manager import _format_refs_for_judge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OLD_POOL_MODELS = [
    "anthropic/claude-3.7-sonnet",
    "mistralai/mistral-small-2603",
    "moonshotai/kimi-k2.6",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-32b",
    "qwen/qwen3-vl-8b-thinking",
    "z-ai/glm-5.1",
]
MODERN_CRITICS = list(cfg.CRITIC_MODELS)   # 5 modern critics from config
POOL_VERSION = "modern5_2026-06-23"


def ensure_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS uniform_critic_scores (
        paper_id      TEXT NOT NULL,
        idea_model    TEXT NOT NULL,
        track         TEXT NOT NULL,
        idea_index    INTEGER NOT NULL,
        idea_text     TEXT,
        critic_model  TEXT NOT NULL,
        scores_json   TEXT,
        reasoning_json TEXT,
        raw_response  TEXT,
        telemetry     TEXT,
        error         TEXT,
        critic_pool_version TEXT,
        created_at    TEXT NOT NULL,
        PRIMARY KEY (paper_id, idea_model, track, idea_index, critic_model)
    );
    """)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=OLD_POOL_MODELS)
    ap.add_argument("--all", action="store_true", help="re-score ALL idea models, not just old-pool 7")
    ap.add_argument("--tracks", nargs="*", default=["B", "C"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="2 papers per model")
    args = ap.parse_args()

    conn_r = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn_r.row_factory = sqlite3.Row
    conn_r.execute("PRAGMA busy_timeout=30000")
    ensure_table(conn_r)

    if args.all:
        args.models = [r[0] for r in conn_r.execute(
            "SELECT DISTINCT idea_model FROM results "
            "WHERE critic_model='' AND idea_text!='' AND prompt_version='v1_paper_refs' "
            "ORDER BY idea_model")]

    conn_p = sqlite3.connect(str(cfg.PAPERS_DB), timeout=30)
    conn_p.row_factory = sqlite3.Row
    papers_by_id = {r["paper_id"]: dict(r) for r in
                    conn_p.execute("SELECT * FROM papers WHERE status='filtered'")}
    conn_p.close()

    # Build tasks: (paper_id, idea_model, track, idea_index, idea_text, critic)
    tasks = []
    for im in args.models:
        for tr in args.tracks:
            ideas = conn_r.execute(
                "SELECT paper_id, idea_index, idea_text FROM results "
                "WHERE idea_model=? AND track=? AND prompt_version='v1_paper_refs' "
                "  AND critic_model='' AND idea_text IS NOT NULL AND idea_text!=''",
                (im, tr)).fetchall()
            if args.smoke:
                keep = sorted({r["paper_id"] for r in ideas})[:2]
                ideas = [r for r in ideas if r["paper_id"] in keep]
            for r in ideas:
                if r["paper_id"] not in papers_by_id:
                    continue
                for cr in MODERN_CRITICS:
                    if cr == im:
                        continue
                    exists = conn_r.execute(
                        "SELECT 1 FROM uniform_critic_scores WHERE paper_id=? AND idea_model=? "
                        "AND track=? AND idea_index=? AND critic_model=?",
                        (r["paper_id"], im, tr, r["idea_index"], cr)).fetchone()
                    if not exists:
                        tasks.append((r["paper_id"], im, tr, r["idea_index"], r["idea_text"], cr))

    logger.info(f"models={len(args.models)} tracks={args.tracks} tasks={len(tasks)}")
    if not tasks:
        logger.info("nothing to do"); conn_r.close(); return

    def _worker(t):
        pid, im, tr, idx, idea_text, cr = t
        paper = papers_by_id.get(pid, {})
        refs = _format_refs_for_judge(paper)
        try:
            scores, raw, telem = score_idea(idea_text, cr, domain=paper.get("domain", ""),
                                            references=refs, judge_mode="static",
                                            db_papers=str(cfg.PAPERS_DB))
            return (pid, im, tr, idx, idea_text, cr, scores, raw, telem, None)
        except Exception as e:
            return (pid, im, tr, idx, idea_text, cr, None, None, None, str(e)[:500])

    ts = datetime.now(timezone.utc).isoformat()
    stats = collections.Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            pid, im, tr, idx, idea_text, cr, scores, raw, telem, err = f.result()
            if scores:
                conn_r.execute("""INSERT OR IGNORE INTO uniform_critic_scores
                    (paper_id, idea_model, track, idea_index, idea_text, critic_model,
                     scores_json, reasoning_json, raw_response, telemetry, error,
                     critic_pool_version, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (pid, im, tr, idx, idea_text, cr,
                     json.dumps({d: v["score"] for d, v in scores.items()}),
                     json.dumps({d: v["reasoning"] for d, v in scores.items()}),
                     (raw or "")[:4000], json.dumps(telem) if telem else None, None,
                     POOL_VERSION, ts))
                conn_r.commit(); stats["ok"] += 1
            else:
                stats["err"] += 1
            if i % 25 == 0 or i == len(tasks):
                logger.info(f"  [{i}/{len(tasks)}] {dict(stats)}")
    conn_r.close()
    logger.info(f"done: {dict(stats)}")


if __name__ == "__main__":
    main()
