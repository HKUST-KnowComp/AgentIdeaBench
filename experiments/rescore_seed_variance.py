"""
Re-score the 18 seed_variance runs under new conditions:

  B = gemini-2.5-flash  + new rubric (with Factual Consistency Check)
  C = openai/gpt-5      + new rubric (strong critic + new rubric)

(The original baseline A = gemini-2.5-flash + old rubric is already in
experiments/seed_variance_results.json.)

Note: "new rubric" is already active in absolute_scorer.py; we just need to
re-call it. No extra flag needed.

Output: experiments/seed_variance_rescored.json
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from evaluation.absolute_scorer import score_idea
from evaluation.critic_manager import _format_refs_for_judge

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rescore")

SOURCE = ROOT / "experiments" / "seed_variance_results.json"
OUT = ROOT / "experiments" / "seed_variance_rescored.json"

CONDITIONS = [
    {"tag": "B_flash_newrubric",
     "critic": "google/gemini-2.5-flash"},
    {"tag": "C_gpt5_newrubric",
     "critic": "openai/gpt-5"},
]


def load_refs() -> dict:
    with open(SOURCE) as f:
        src = json.load(f)
    pids = {r["paper_id"] for r in src["runs"]}
    conn = sqlite3.connect(cfg.PAPERS_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join(["?"] * len(pids))
    rows = conn.execute(
        f"SELECT * FROM papers WHERE paper_id IN ({placeholders})",
        list(pids),
    ).fetchall()
    conn.close()
    return {r["paper_id"]: dict(r) for r in rows}, src


def load_state() -> dict:
    if OUT.exists():
        with open(OUT) as f:
            return json.load(f)
    return {"runs": []}


def save(state: dict) -> None:
    OUT.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def key(tag: str, model: str, pid: str, seed: int) -> str:
    return f"{tag}|{model}|{pid}|{seed}"


def main():
    papers, src = load_refs()
    state = load_state()
    done = {key(r["condition"], r["model"], r["paper_id"], r["seed"])
            for r in state["runs"]}
    logger.info(f"resuming with {len(done)} rescorings already done")

    scorable = [r for r in src["runs"] if r.get("hypothesis") and r.get("scores")]
    total = len(scorable) * len(CONDITIONS)
    idx = 0
    for cond in CONDITIONS:
        tag = cond["tag"]
        critic = cond["critic"]
        for r in scorable:
            idx += 1
            k = key(tag, r["model"], r["paper_id"], r["seed"])
            if k in done:
                continue
            paper = papers[r["paper_id"]]
            refs_text = _format_refs_for_judge(paper)
            logger.info(f"[{idx}/{total}] {tag} | {r['model'].split('/')[-1]} | "
                        f"{r['domain']} | seed={r['seed']}")
            t0 = time.time()
            try:
                scores, raw, _ = score_idea(
                    r["hypothesis"], critic,
                    domain=r["domain"], title="", references=refs_text,
                )
            except Exception as e:
                logger.error(f"  error: {e}")
                scores, raw = None, str(e)
            secs = round(time.time() - t0, 1)

            # Try to parse factual_check field from raw
            factual = None
            try:
                s = raw
                if "<reasoning>" in s:
                    s = s.split("</reasoning>", 1)[1]
                if "```json" in s:
                    s = s.split("```json", 1)[1].split("```", 1)[0]
                start = s.find("{")
                end = s.rfind("}")
                if start >= 0 and end > start:
                    parsed = json.loads(s[start:end+1])
                    factual = parsed.get("factual_check")
            except Exception:
                pass

            state["runs"].append({
                "condition": tag, "critic": critic,
                "model": r["model"], "paper_id": r["paper_id"],
                "domain": r["domain"], "seed": r["seed"],
                "scores": scores, "factual_check": factual,
                "score_secs": secs,
            })
            save(state)
            if scores:
                dims = {d: scores[d]["score"] for d in scores}
                logger.info(f"  {dims}  ({secs}s)")
            else:
                logger.warning(f"  SCORE FAILED ({secs}s)")

    logger.info(f"done. results -> {OUT}")


if __name__ == "__main__":
    main()
