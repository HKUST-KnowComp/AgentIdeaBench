#!/usr/bin/env python
"""Export the data behind the paper + the new experiments as a self-contained package.

Read-only. Dumps each selected table to JSONL (one row per line, all columns kept
verbatim) plus a schema file and a row-count manifest, then copies the pipeline
statistics JSONs. Retired tables (legacy v1/v2 pipeline, archives, backups) are
excluded by name -- see EXCLUDED below.

Usage:
    python reports/_export_data_package.py --out /path/to/stage
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Tables grouped by what they back. Every table not listed here is either in
# EXCLUDED or reported as unclassified so the roster never drifts silently.
GROUPS: dict[str, dict[str, list[str]]] = {
    "results.db": {
        "paper_core": [
            "subdomain_ideas",        # every generated hypothesis + Active telemetry
            "lit8d_scores_3seed",     # main analysis: 40 models x 40 subfields x 3 critics
            "lit8d_scores",           # E19 pilot pass (20 subfields)
            "e13_evidence",           # frozen prior-art evidence behind every critic call
            "subdomain_refs",         # curated Static-track references
        ],
        "paper_appendix": [
            "swm_ideas", "swm_scores",              # Scientific World Modeling section
            "wm_ideas", "wm_scores",                # E26 world-model scaffold
            "budget_sweep_ideas", "budget_sweep_scores",   # tool-budget saturation
            "cap_ablation_scores",                  # coherence-cap ablation
            "e7_bestpaper_scores",                  # critic rigor vs real papers
            "e10_idea_anchor_scores", "e10_model_idea_scores",
            "e10_deepseek_scores", "e10_fixv2_scores", "e10_fixv3_scores",
            "e12_gap_scores",                       # critic gap tuning
            "domain_anchor_pool", "e13_anchor_meta", "domain_topic_refs",
            "prior_probe", "prior_probe_refs",      # leakage / recency appendix
            "e29_nlp_scores", "e29_pairwise_llm",   # human-evaluation study
        ],
        "new_experiments": [
            "e31_bo3_ideas", "e31_bo3_scores",      # compute-matched best-of-3
            "e32_recall_ideas", "e32_recall_scores",  # recall-only control track R
            "e38_replay_ideas", "e38_replay_refs", "e38_replay_scores",  # replay control
        ],
    },
    "papers.db": {
        "source_data": ["papers", "survey_refs"],
    },
}

# Retired: legacy v1/v2 pipeline, archived error rows, pre-overwrite backups.
# Retained in the repo per the raw-data rule, but not part of this package.
EXCLUDED = {
    "results", "uniform_critic_scores", "model_scores", "pairwise_results",
    "results_archive_402errors", "results_archive_funccall_smoke_5x5",
    "results_archive_stale_critics", "swm_scores_bak_20260720",
    "active_comparison_ideas", "active_comparison_ideas_v1_archive",
    "active_comparison_scores", "dynamic_critic_sweep", "dynamic_critic_sweep_ideas",
}


def dump_table(conn: sqlite3.Connection, table: str, dest: Path) -> tuple[int, int]:
    """Write one table to JSONL. Returns (rows, bytes)."""
    conn.row_factory = sqlite3.Row
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with dest.open("w", encoding="utf-8") as fh:
        for row in conn.execute(f'SELECT * FROM "{table}"'):
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            rows += 1
    return rows, dest.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="staging directory to write into")
    args = ap.parse_args()

    out = Path(args.out)
    manifest: dict[str, object] = {"generated_from": str(REPO), "tables": []}
    schemas: dict[str, str] = {}

    for db_name, groups in GROUPS.items():
        db_path = REPO / "data" / db_name
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        present = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        listed = {t for tables in groups.values() for t in tables}
        unclassified = present - listed - EXCLUDED
        if unclassified:
            print(f"[warn] {db_name}: unclassified tables {sorted(unclassified)}")

        for group, tables in groups.items():
            for table in tables:
                if table not in present:
                    print(f"[warn] {db_name}: missing table {table}")
                    continue
                dest = out / "data" / group / f"{table}.jsonl"
                n, size = dump_table(conn, table, dest)
                cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
                sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone()
                schemas[table] = (sql[0] if sql else "") or ""
                manifest["tables"].append({
                    "table": table, "group": group, "source_db": db_name,
                    "rows": n, "bytes": size, "columns": cols,
                    "path": str(dest.relative_to(out)),
                })
                print(f"  {group:16s} {table:28s} rows={n:7d}  {size/1e6:8.2f} MB")
        conn.close()

    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / "SCHEMA.sql").write_text(
        "\n\n".join(f"-- {t}\n{s};" for t, s in sorted(schemas.items())), encoding="utf-8"
    )
    (out / "data" / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats_dir = out / "stats_json"
    stats_dir.mkdir(parents=True, exist_ok=True)
    n_json = 0
    for src in sorted((REPO / "reports").glob("*.json")):
        shutil.copy2(src, stats_dir / src.name)
        n_json += 1
    for sub in ("e11_ref_overlap",):
        for src in sorted((REPO / "reports" / sub).glob("*.json")):
            shutil.copy2(src, stats_dir / f"{sub}__{src.name}")
            n_json += 1

    total = sum(t["bytes"] for t in manifest["tables"])  # type: ignore[index]
    print(f"\ntables={len(manifest['tables'])}  jsonl={total/1e6:.1f} MB  stats_json={n_json}")


if __name__ == "__main__":
    main()
