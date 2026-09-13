#!/usr/bin/env python3
"""Archive-then-delete the gemini-3.6-flash leakage rows from results.db.

Why: google/gemini-3.6-flash has a knowledge cutoff (2026-03) that post-dates
the benchmark's target-literature date (2026-01), so its rows are a leakage
risk and are already EXCLUDE_LEAKAGE-filtered from all analysis. The user
authorized cleaning them out of the live tables (2026-07-24 "A1 直接清理").

Per CLAUDE.md raw-data rules: we ARCHIVE every deleted row to
archive/db_cleanup_20260724/leakage_gemini36flash.db first, verify the archive
row count matches the source, and only then DELETE from the live table. If the
archive verification fails, we abort WITHOUT deleting. Idempotent: if the live
rows are already gone, each table is reported as already-clean and skipped.

Run: /usr/bin/python3 experiments/cleanup_leakage_20260724.py
"""
import os
import sqlite3
import sys

LIVE_DB = "data/results.db"
ARCHIVE_DIR = "archive/db_cleanup_20260724"
ARCHIVE_DB = os.path.join(ARCHIVE_DIR, "leakage_gemini36flash.db")
PATTERN = "%gemini-3.6-flash%"

# (table, column used to match leakage rows, expected count for a sanity note)
TARGETS = [
    ("subdomain_ideas", "idea_model", 120),
    ("e13_evidence", "item_id", 120),
    ("lit8d_scores_3seed", "idea_model", 360),
]


def create_stmt(conn, table):
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not row:
        raise RuntimeError(f"table {table} not found in live DB")
    return row[0]


def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    live = sqlite3.connect(LIVE_DB)
    arch = sqlite3.connect(ARCHIVE_DB)
    manifest = []
    total_archived = 0
    total_deleted = 0

    for table, col, expected in TARGETS:
        src_n = live.execute(
            f'SELECT COUNT(*) FROM {table} WHERE "{col}" LIKE ?', (PATTERN,)
        ).fetchone()[0]
        if src_n == 0:
            print(f"[skip] {table}: 0 leakage rows (already clean)")
            manifest.append(f"{table}: already clean (0 rows)")
            continue

        # 1. recreate schema in archive DB (fresh copy each run is fine: same rows)
        arch.execute(f"DROP TABLE IF EXISTS {table}")
        arch.execute(create_stmt(live, table))
        arch.commit()

        # 2. copy matching rows into archive
        rows = live.execute(
            f'SELECT * FROM {table} WHERE "{col}" LIKE ?', (PATTERN,)
        ).fetchall()
        ncols = len(rows[0])
        arch.executemany(
            f"INSERT INTO {table} VALUES ({','.join('?' * ncols)})", rows
        )
        arch.commit()

        # 3. verify archive count == source count BEFORE any delete
        arch_n = arch.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if arch_n != src_n:
            print(
                f"[ABORT] {table}: archive={arch_n} != source={src_n}; "
                "NOT deleting anything."
            )
            live.close()
            arch.close()
            sys.exit(1)

        # 4. delete from live only after verified archive
        live.execute(f'DELETE FROM {table} WHERE "{col}" LIKE ?', (PATTERN,))
        live.commit()
        after_n = live.execute(
            f'SELECT COUNT(*) FROM {table} WHERE "{col}" LIKE ?', (PATTERN,)
        ).fetchone()[0]

        print(
            f"[done] {table}: archived {arch_n} rows -> {ARCHIVE_DB}, "
            f"deleted {src_n} from live (remaining leakage={after_n}, "
            f"expected~{expected})"
        )
        manifest.append(
            f"{table}: archived+deleted {src_n} rows (col={col})"
        )
        total_archived += arch_n
        total_deleted += src_n

    live.close()
    arch.close()

    with open(os.path.join(ARCHIVE_DIR, "MANIFEST.txt"), "w") as f:
        f.write("gemini-3.6-flash leakage cleanup (2026-07-24)\n")
        f.write(f"source DB: {LIVE_DB}\n")
        f.write(f"archive DB: {ARCHIVE_DB}\n")
        f.write(f"match pattern: {PATTERN}\n\n")
        f.write("\n".join(manifest) + "\n")
        f.write(f"\ntotal archived: {total_archived}\n")
        f.write(f"total deleted from live: {total_deleted}\n")

    print(f"\nMANIFEST written to {ARCHIVE_DIR}/MANIFEST.txt")
    print(f"total archived={total_archived} total deleted={total_deleted}")


if __name__ == "__main__":
    main()
