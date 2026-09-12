"""Per-model mean Active tool-call count, aligned cell-for-cell with the leaderboard.

`n_tool_calls` is written by generation/active_agent.py into subdomain_ideas.telemetry
and counts SEARCH + FETCH turns before the final synthesis (the FINAL turn is not a
tool call, so a rollout issues n_tool_calls + 1 LLM turns).

Cells are restricted to exactly the (model, subdomain, idea_index) triples that carry
lit8d Track-C scores, so the turn means summarize the same rollouts the score columns
summarize. Aggregation nests the same way as e36 (idx -> subdomain mean -> model mean),
so a subdomain with a missing idx cannot outweigh a complete one.

Read-only on the DB. Writes reports/e43_active_turns.json.
"""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg                                    # noqa: E402
from experiments.e36_leaderboard_subscores import shorten  # noqa: E402

OUT = ROOT / "reports" / "e43_active_turns.json"


def main():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    conn.row_factory = sqlite3.Row

    scored = {(r["idea_model"], r["subdomain"], r["idea_index"])
              for r in conn.execute(
                  "SELECT DISTINCT idea_model, subdomain, idea_index "
                  "FROM lit8d_scores_3seed WHERE track='C' AND scores_json IS NOT NULL")}

    per_sub = defaultdict(list)      # (model, subdomain) -> [n_tool_calls per idx]
    missing = defaultdict(int)
    for r in conn.execute("SELECT idea_model, subdomain, idea_index, telemetry "
                          "FROM subdomain_ideas WHERE track='C'"):
        key = (r["idea_model"], r["subdomain"], r["idea_index"])
        if key not in scored:
            continue
        if not r["telemetry"]:
            missing[r["idea_model"]] += 1
            continue
        try:
            n = json.loads(r["telemetry"]).get("n_tool_calls")
        except (ValueError, TypeError):
            n = None
        if n is None:
            missing[r["idea_model"]] += 1
            continue
        per_sub[(r["idea_model"], r["subdomain"])].append(float(n))
    conn.close()

    by_model = defaultdict(list)
    cells = defaultdict(int)
    for (m, _sub), vals in per_sub.items():
        by_model[m].append(float(np.mean(vals)))
        cells[m] += len(vals)

    out = {}
    for m, sub_means in by_model.items():
        out[m] = {
            "short": shorten(m),
            "active_turns_mean": float(np.mean(sub_means)),
            "n_cells": cells[m],
            "n_subdomains": len(sub_means),
            "n_missing_telemetry": missing.get(m, 0),
        }

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {OUT.relative_to(ROOT)}  models={len(out)}")
    for m, v in sorted(out.items(), key=lambda kv: -kv[1]["active_turns_mean"]):
        print(f"  {v['short']:<24}{v['active_turns_mean']:5.2f}  "
              f"cells={v['n_cells']:>4}  subs={v['n_subdomains']:>3}  "
              f"missing={v['n_missing_telemetry']}")


if __name__ == "__main__":
    main()
