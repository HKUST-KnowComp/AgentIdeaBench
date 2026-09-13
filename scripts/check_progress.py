#!/usr/bin/env python3
"""
SciSynthBench — Progress Report

Shows current DB state: paper counts, idea counts, scoring progress.

Usage: python check_progress.py
"""
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config as cfg

DB_PAPERS  = str(cfg.PAPERS_DB)
DB_RESULTS = str(cfg.RESULTS_DB)


def main():
    # ── Papers DB ─────────────────────────────────────────────────
    print("=" * 60)
    print("  SciSynthBench — Progress Report")
    print("=" * 60)

    conn_p = sqlite3.connect(DB_PAPERS, timeout=10)

    total = conn_p.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    filtered = conn_p.execute(
        "SELECT COUNT(*) FROM papers WHERE status='filtered'"
    ).fetchone()[0]
    gt_done = conn_p.execute(
        "SELECT COUNT(*) FROM papers WHERE gt_hypothesis IS NOT NULL AND gt_hypothesis != ''"
    ).fetchone()[0]
    embed_done = conn_p.execute(
        "SELECT COUNT(*) FROM papers WHERE ranked_refs_json IS NOT NULL AND ranked_refs_json != ''"
    ).fetchone()[0]

    domain_rows = conn_p.execute(
        "SELECT domain, COUNT(*) FROM papers WHERE status='filtered' GROUP BY domain ORDER BY domain"
    ).fetchall()
    conn_p.close()

    print(f"\n  papers.db: {total} total, {filtered} filtered")
    for domain, n in domain_rows:
        print(f"    {domain:12s}: {n}")
    print(f"    gt_hypothesis : {gt_done}/{filtered}")
    print(f"    ranked_refs   : {embed_done}/{filtered}")

    # ── Results DB ────────────────────────────────────────────────
    conn_r = sqlite3.connect(DB_RESULTS, timeout=10)

    p2 = conn_r.execute(
        "SELECT COUNT(*) FROM results WHERE critic_model=''"
    ).fetchone()[0]
    p3_ok = conn_r.execute(
        "SELECT COUNT(*) FROM results WHERE critic_model!='' AND scores_json IS NOT NULL"
    ).fetchone()[0]
    p3_err = conn_r.execute(
        "SELECT COUNT(*) FROM results WHERE critic_model!='' AND scores_json IS NULL AND error IS NOT NULL"
    ).fetchone()[0]

    print(f"\n  results.db:")
    print(f"    Phase 2 (ideas)   : {p2}")
    print(f"    Phase 3 (scored)  : {p3_ok}")
    if p3_err:
        print(f"    Phase 3 (errors)  : {p3_err}")

    # Per-model breakdown
    model_rows = conn_r.execute(
        "SELECT idea_model, track, COUNT(*) FROM results "
        "WHERE critic_model!='' AND scores_json IS NOT NULL "
        "GROUP BY idea_model, track ORDER BY idea_model, track"
    ).fetchall()

    if model_rows:
        print(f"\n  Per-model scoring progress:")
        print(f"  {'Model':<40s} {'Track A':>8} {'Track B':>8}")
        print(f"  {'-'*56}")
        per_model = defaultdict(lambda: {"A": 0, "B": 0})
        for model, track, count in model_rows:
            per_model[model][track] = count
        for model in sorted(per_model):
            short = model.split("/")[-1]
            a = per_model[model]["A"]
            b = per_model[model]["B"]
            print(f"  {short:<40s} {a:>8} {b:>8}")

    # model_scores
    try:
        ms = conn_r.execute("SELECT COUNT(*) FROM model_scores").fetchone()[0]
        print(f"\n  model_scores: {ms} rows")
    except sqlite3.OperationalError:
        pass

    # pairwise
    try:
        pw = conn_r.execute(
            "SELECT COUNT(*) FROM pairwise_results WHERE winner IS NOT NULL"
        ).fetchone()[0]
        if pw:
            print(f"  pairwise_results: {pw} comparisons")
    except sqlite3.OperationalError:
        pass

    conn_r.close()
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
