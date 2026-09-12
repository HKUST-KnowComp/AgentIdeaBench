r"""Every number Appendix app:budget states, recomputed from the DB.

The appendix used to be assembled from several ad-hoc queries, so dropping a
backbone from the panel meant re-deriving each figure by hand. This script is
the single source for that section: the roster it reports is imported from
_fig_turn_budget.py, so the prose and Figure 8 cannot disagree about who is in
the sweep.

Read-only on data/results.db. Prints; writes nothing.
Run with /usr/bin/python3.
"""
import importlib.util
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
N_BOOT = 10_000
STEPS = [(1, 2), (1, 5), (2, 5), (5, 10), (10, 15), (10, 20), (15, 20)]


def fig_module():
    spec = importlib.util.spec_from_file_location(
        "_fig_turn_budget", ROOT / "reports" / "_fig_turn_budget.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    F = fig_module()
    keys = [k for k, _, _, _ in F.MODELS]
    label = {k: lb for k, lb, _, _ in F.MODELS}
    conn = sqlite3.connect(f"file:{ROOT / 'data' / 'results.db'}?mode=ro",
                           uri=True)
    conn.row_factory = sqlite3.Row
    ph = ",".join("?" * len(keys))

    # ---- counts -------------------------------------------------------
    n_ideas = conn.execute(
        f"SELECT COUNT(*) FROM budget_sweep_ideas{F.SUFFIX} "
        f"WHERE idea_model IN ({ph})", keys).fetchone()[0]
    n_scores = conn.execute(
        f"SELECT COUNT(*) FROM budget_sweep_scores{F.SUFFIX} "
        f"WHERE idea_model IN ({ph})", keys).fetchone()[0]
    planned = len(keys) * len(F.BUDGETS) * 10 * 3
    print(f"backbones {len(keys)}   cells {len(keys) * len(F.BUDGETS)}")
    print(f"rollouts  {n_ideas} of a planned {planned}   critic scores {n_scores}")

    # ---- protocol failures --------------------------------------------
    fail = defaultdict(int)
    tot = defaultdict(int)
    for r in conn.execute(
            f"SELECT idea_model, budget, error FROM budget_sweep_ideas{F.SUFFIX} "
            f"WHERE idea_model IN ({ph})", keys):
        tot[(r["idea_model"], r["budget"])] += 1
        if r["error"] == "malformed_x3":
            fail[(r["idea_model"], r["budget"])] += 1
    print("\nmalformed_x3 rate by budget")
    for k in keys:
        row = "  ".join(
            f"b{b}={fail[(k, b)] / tot[(k, b)]:.0%}" for b in F.BUDGETS)
        print(f"  {label[k]:<13} {row}   total {sum(fail[(k, b)] for b in F.BUDGETS)}")

    # ---- per-paper values, nested critic -> idea -> paper ---------------
    per_idea = defaultdict(list)
    for r in conn.execute(
            f"SELECT idea_model, budget, paper_id, idea_index, scores_json "
            f"FROM budget_sweep_scores{F.SUFFIX} "
            f"WHERE scores_json IS NOT NULL AND idea_model IN ({ph})", keys):
        try:
            s = json.loads(r["scores_json"])
        except (TypeError, ValueError):
            continue
        if isinstance(s, dict):
            per_idea[(r["idea_model"], r["budget"], r["paper_id"],
                      r["idea_index"])].append(F.weighted(s))
    paper = defaultdict(dict)
    tmp = defaultdict(list)
    for (m, b, p, _), v in per_idea.items():
        tmp[(m, b, p)].append(float(np.mean(v)))
    for (m, b, p), v in tmp.items():
        paper[(m, b)][p] = float(np.mean(v))

    calls = defaultdict(list)
    for r in conn.execute(
            f"SELECT idea_model, budget, paper_id, n_tool_calls "
            f"FROM budget_sweep_ideas{F.SUFFIX} "
            f"WHERE idea_text IS NOT NULL AND TRIM(idea_text) <> '' "
            f"AND n_tool_calls IS NOT NULL AND idea_model IN ({ph})", keys):
        calls[(r["idea_model"], r["budget"], r["paper_id"])].append(
            float(r["n_tool_calls"]))
    conn.close()
    call_cell = defaultdict(list)
    for (m, b, _), v in calls.items():
        call_cell[(m, b)].append(float(np.mean(v)))

    print("\npapers retained per cell (10 = complete)")
    print("  " + "  ".join(f"b{b}" for b in F.BUDGETS))
    for k in keys:
        print(f"  {label[k]:<13} " +
              "  ".join(str(len(paper[(k, b)])) for b in F.BUDGETS))

    # ---- paired per-paper bootstrap ------------------------------------
    rng = np.random.default_rng(0)
    print(f"\npaired per-paper bootstrap, {N_BOOT} draws "
          f"(d = later budget minus earlier)")
    for lo, hi in STEPS:
        print(f"  {lo:>2} -> {hi:<2}")
        for k in keys:
            common = sorted(set(paper[(k, lo)]) & set(paper[(k, hi)]))
            d = np.array([paper[(k, hi)][p] - paper[(k, lo)][p]
                          for p in common])
            boot = np.array([rng.choice(d, d.size, replace=True).mean()
                             for _ in range(N_BOOT)])
            l, h = np.percentile(boot, [2.5, 97.5])
            # two-sided bootstrap p: how often the resampled mean crosses zero
            p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
            star = " *" if l > 0 or h < 0 else ""
            print(f"      {label[k]:<13} n={d.size:<3} d={d.mean():+.2f} "
                  f"[{l:+.2f},{h:+.2f}] p={min(p, 1.0):.3f}{star}")

    # ---- spread and best cell ------------------------------------------
    print("\nspread across budgets, and each backbone's best cell")
    for k in keys:
        ms = [float(np.mean(list(paper[(k, b)].values()))) for b in F.BUDGETS]
        best = F.BUDGETS[int(np.argmax(ms))]
        print(f"  {label[k]:<13} spread {max(ms) - min(ms):.2f}   "
              f"best b{best} ({max(ms):.2f})   " +
              " ".join(f"b{b}={m:.2f}" for b, m in zip(F.BUDGETS, ms)))

    # ---- utilization ----------------------------------------------------
    for b in (10, 20):
        print(f"\nutilization at budget {b}")
        for k in keys:
            m = float(np.mean(call_cell[(k, b)]))
            print(f"  {label[k]:<13} {m / b:5.1%}  ({m:.1f}/{b})")


if __name__ == "__main__":
    main()
