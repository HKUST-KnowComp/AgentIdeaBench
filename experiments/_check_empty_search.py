"""Empty-SEARCH rate in a budget-sweep ideas table, split by created_at slice.

A Semantic Scholar 429 that exhausts its retries returns no results, and the
agent then spends a tool call on a SEARCH it cannot learn from. The rate is
therefore a throttling health check, not a model property: it must be
comparable across slices of a run, otherwise concurrency level becomes a
confound in the same way collection date was.

A SEARCH is counted empty when its trace entry records an empty result list.

Usage:
    python experiments/_check_empty_search.py [--suffix _v3] [--since ISO_TS]

Read-only on the DB.
"""
import argparse
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
SEARCH_RE = re.compile(r'"result_preview":\s*"(\[\]|\[\{)')


def rate(rows):
    empty = total = 0
    for (trace,) in rows:
        if not trace:
            continue
        for m in SEARCH_RE.finditer(trace):
            total += 1
            empty += m.group(1) == "[]"
    return empty, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--since", default=None,
                    help="only rows with created_at >= this ISO timestamp")
    ap.add_argument("--before", default=None,
                    help="only rows with created_at < this ISO timestamp")
    a = ap.parse_args()

    conn = sqlite3.connect(f"file:{ROOT / 'data' / 'results.db'}?mode=ro",
                           uri=True)
    q = f"SELECT trace_json FROM budget_sweep_ideas{a.suffix} WHERE 1=1"
    params = []
    if a.since:
        q += " AND created_at >= ?"
        params.append(a.since)
    if a.before:
        q += " AND created_at < ?"
        params.append(a.before)
    rows = conn.execute(q, params).fetchall()
    conn.close()

    empty, total = rate(rows)
    span = f"since={a.since or '-'} before={a.before or '-'}"
    pct = f"{empty / total:.1%}" if total else "n/a"
    print(f"table=budget_sweep_ideas{a.suffix}  {span}")
    print(f"  rollouts={len(rows)}  searches={total}  empty={empty}  "
          f"rate={pct}")


if __name__ == "__main__":
    main()
