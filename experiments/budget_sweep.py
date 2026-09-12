"""Tool-budget sweep experiment for Active Mode.

Runs the Active agent across multiple max_iters values for a small set of
idea models and papers, then collects:

  - tool-call counts (n_tool_calls, iters_used) per idea
  - per-turn token usage (prompt / completion / reasoning)
  - final_prompt_tokens distribution (context length at the last LLM call)
  - whether the budget-exhaustion nudge fired

Saves raw per-idea telemetry to reports/budget_sweep.json. Does NOT call
critic / phase 3 — speed-curve interpretation is left to a follow-up
analysis script.

Default sweep:
  models   = ["google/gemma-4-31b-it", "qwen/qwen3.5-397b-a17b"]
  budgets  = [1, 5, 10, 15]
  papers   = 5 (one per domain, smoke set)

To run (DO NOT auto-run; expensive):
  python experiments/budget_sweep.py
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from generation.active_agent import run_active_agent

logger = logging.getLogger(__name__)


MODELS = ["google/gemma-4-31b-it", "qwen/qwen3.5-397b-a17b"]
BUDGETS = [1, 5, 10, 15]


def load_smoke_papers(db_papers: str) -> List[dict]:
    import sqlite3
    conn = sqlite3.connect(db_papers, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT * FROM papers WHERE status='filtered' "
        "AND ranked_refs_json IS NOT NULL ORDER BY domain, paper_id"
    ))
    conn.close()
    # 1 paper per domain (smoke set)
    from collections import defaultdict
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(dict(r))
    return [ps[0] for ps in by_domain.values() if ps]


def run_sweep() -> Dict:
    """Run the cross-product (model × budget × paper) sweep.

    Returns a nested dict:
      {
        model: {
          budget: [
            { paper_id, domain, hypothesis, n_tool_calls, iters_used,
              n_turns, final_prompt_tokens, final_total_tokens,
              budget_nudge_used, turns, error },
            ... one per paper ...
          ]
        }
      }
    """
    papers = load_smoke_papers(str(cfg.PAPERS_DB))
    logger.info(f"Loaded {len(papers)} smoke papers ({[p['domain'] for p in papers]})")

    results: Dict = {}
    total = len(MODELS) * len(BUDGETS) * len(papers)
    counter = 0

    for model in MODELS:
        results[model] = {}
        for budget in BUDGETS:
            results[model][str(budget)] = []
            for paper in papers:
                counter += 1
                logger.info(
                    f"[{counter}/{total}] model={model} budget={budget} "
                    f"domain={paper['domain']} paper={paper['paper_id'][:10]}"
                )
                t0 = time.time()
                try:
                    out = run_active_agent(
                        domain=paper["domain"],
                        model_name=model,
                        max_iters=budget,
                        seed=42,
                    )
                except Exception as e:
                    logger.error(f"  agent error: {e}")
                    out = {"error": str(e)[:200]}
                dt = round(time.time() - t0, 1)

                rec = {
                    "paper_id": paper["paper_id"],
                    "domain": paper["domain"],
                    "wallclock": dt,
                    "hypothesis": (out.get("hypothesis") or "")[:1500],
                    "n_tool_calls": out.get("n_tool_calls"),
                    "iters_used": out.get("iters_used"),
                    "n_turns": out.get("n_turns"),
                    "final_prompt_tokens": out.get("final_prompt_tokens"),
                    "final_total_tokens": out.get("final_total_tokens"),
                    "budget_nudge_used": out.get("budget_nudge_used"),
                    "turns": out.get("turns") or [],
                    "error": out.get("error"),
                }
                results[model][str(budget)].append(rec)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = run_sweep()
    out = ROOT / "reports" / "budget_sweep.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    # Quick summary
    print("\n## Summary (per model, per budget)")
    print(f"{'model':40s} {'budget':>6s} {'mean_calls':>10s} {'mean_iters':>10s} "
          f"{'mean_final_ctx':>14s} {'nudge%':>7s}")
    print("-" * 95)
    for model, by_budget in results.items():
        for budget_str, recs in by_budget.items():
            calls = [r["n_tool_calls"] for r in recs if r.get("n_tool_calls") is not None]
            iters = [r["iters_used"] for r in recs if r.get("iters_used") is not None]
            ctxs = [r["final_prompt_tokens"] for r in recs if r.get("final_prompt_tokens") is not None]
            nudges = [r["budget_nudge_used"] for r in recs if r.get("budget_nudge_used") is not None]
            mc = sum(calls) / len(calls) if calls else 0
            mi = sum(iters) / len(iters) if iters else 0
            mctx = sum(ctxs) / len(ctxs) if ctxs else 0
            nudge_pct = (sum(1 for x in nudges if x) / len(nudges) * 100) if nudges else 0
            print(f"{model:40s} {budget_str:>6s} {mc:>10.2f} {mi:>10.2f} "
                  f"{mctx:>14.0f} {nudge_pct:>6.1f}%")


if __name__ == "__main__":
    main()
