#!/usr/bin/env python
"""Build the public data release from the local SQLite databases.

Read-only against ``data/*.db``. Writes gzipped CSV into ``release_data/``.

The paper commits us to a redaction boundary (appendix I, "Ethical
considerations"): the released tables carry Semantic Scholar identifiers, DOIs,
retrieval timestamps and our own derived fields, and they carry no abstract
text, no reference-list text and no full text. Anyone reproducing the benchmark
refetches that content from the API under their own agreement.

Enforcing that boundary is the whole job of this script, so it works by
whitelist rather than by blacklist:

  * every column named in DROP_COLS is removed, because it is either upstream
    free text or a raw payload with upstream free text nested inside it;
  * the four nested payloads that carry information we do want (which papers a
    run retrieved, in what order, for which query) are exploded into side
    tables that keep only identifiers, counts and our own query strings;
  * ``scores_json`` is flattened to five numeric columns, which drops the
    per-dimension critic prose and is also what makes the release small enough
    to live in git -- ``lit8d_scores_3seed`` alone goes from 74 MB to ~3 MB.

``--verify`` then reads every output back and asserts the boundary held, by
shingling real abstracts out of the source databases and searching for them in
the products. Run it; do not trust the whitelist on its own.

Usage:
    /usr/bin/python3 scripts/export_release_data.py            # export + verify
    /usr/bin/python3 scripts/export_release_data.py --verify-only
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

csv.field_size_limit(1 << 30)

REPO = Path(__file__).resolve().parent.parent
PAPERS_DB = REPO / "data" / "papers.db"
RESULTS_DB = REPO / "data" / "results.db"

SCORE_DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]

# ---------------------------------------------------------------------------
# Table roster. Grouped by what each table backs, mirroring the internal
# roster in reports/_export_data_package.py so the two do not drift.
# ---------------------------------------------------------------------------

CORE = {
    RESULTS_DB: [
        "subdomain_ideas",       # every generated hypothesis, both tracks
        "lit8d_scores_3seed",    # main analysis: 3 critic seeds
        "lit8d_scores",          # E19 pilot pass
        "e13_evidence",          # frozen prior-art evidence behind each critic call
        "e13_anchor_meta",
        "subdomain_refs",        # curated Static-track reference sets
        "uniform_critic_scores",  # uniform critic pool rescore (cutoff analysis)
        "results",               # cross-year paper-level runs
        "model_scores",          # per-paper aggregates
        "pairwise_results",
    ],
    PAPERS_DB: ["papers", "survey_refs"],
}

APPENDIX = {
    RESULTS_DB: [
        "swm_ideas", "swm_scores",                     # scientific world modeling
        "wm_ideas", "wm_scores",                       # world-model scaffold
        "budget_sweep_ideas", "budget_sweep_scores",   # tool-budget saturation
        "budget_sweep_ideas_v2", "budget_sweep_scores_v2",
        "budget_sweep_ideas_v3", "budget_sweep_scores_v3",
        "cap_ablation_scores",                         # coherence-cap ablation
        "e7_bestpaper_scores",                         # critic rigor vs real papers
        "e10_idea_anchor_scores", "e10_model_idea_scores",
        "e10_deepseek_scores", "e10_fixv2_scores", "e10_fixv3_scores",
        "e12_gap_scores",
        "domain_anchor_pool", "domain_topic_refs",
        "prior_probe", "prior_probe_refs",             # leakage / recency probes
        "e29_nlp_scores", "e29_pairwise_llm",          # human-evaluation study
        "e31_bo3_ideas", "e31_bo3_scores",             # compute-matched best-of-3
        "e32_recall_ideas", "e32_recall_scores",       # recall-only control
        "e38_replay_ideas", "e38_replay_refs", "e38_replay_scores",
        "dynamic_critic_sweep", "dynamic_critic_sweep_ideas",
        "active_comparison_ideas", "active_comparison_scores",
    ],
}

# Pre-overwrite backups and archived error rows. Kept in the repo under the
# raw-data rule, but they are duplicates or failures and not part of a release.
EXCLUDED = {
    "sqlite_sequence",
    "results_archive_402errors", "results_archive_funccall_smoke_5x5",
    "results_archive_stale_critics", "swm_scores_bak_20260720",
    "active_comparison_ideas_v1_archive",
}

# ---------------------------------------------------------------------------
# Redaction rules
# ---------------------------------------------------------------------------

# Never released. Upstream free text, or a raw payload with upstream free text
# nested inside it. The information worth keeping from the payloads is lifted
# out into the side tables below.
DROP_COLS = {
    # upstream free text
    "abstract", "real_abstract", "idea_abstract",
    # gt_hypothesis is a GPT-4o rewrite of the source abstract, so republishing
    # it would republish the abstract's substance under another surface form.
    "gt_hypothesis",
    # recitation_text is a model's attempt to reproduce a held-out abstract;
    # the ROUGE/Jaccard columns are the actual finding and they stay.
    "recitation_text",
    # nested payloads (each contains upstream abstracts)
    "references_json", "ranked_refs_json", "paper_refs_backup",
    "refs_json", "evidence_json", "trace_json", "telemetry", "telemetry_json",
    # raw model transcripts: bulky, and critic prose quotes the evidence.
    # raw_output is the unparsed Active transcript; the parsed hypothesis stays.
    "raw_response", "raw_forward", "raw_swap", "reasoning_json", "raw_output",
}

# Columns allowed to hold long text. Everything here is text this project
# generated: model hypotheses and the titles of our own sampled papers.
LONG_TEXT_OK = {
    # model generations, which are the artifact this benchmark exists to release
    "idea_text", "hypothesis", "best_idea_text",
    "aux_world_model", "aux_analogies", "aux_swm_verdicts",
    # our own derived fields
    "title", "survey_title", "queries_json", "cited_refs",
    "returned_ss_ids", "reject_reason", "error",
}

# Columns holding model output. The redaction boundary governs the tables we
# assemble, not what a model happened to write; a weak model sometimes quotes a
# chunk of a reference abstract back into its own hypothesis. Rewriting that
# would corrupt the artifact, so --verify counts it and reports it instead of
# failing, and DATA_CARD.md carries the count.
MODEL_OUTPUT_COLS = {
    "idea_text", "hypothesis", "best_idea_text",
    "aux_world_model", "aux_analogies", "cited_refs",
}

MAX_SHORT_TEXT = 300  # any other text column longer than this fails --verify

# Text generated by Google's Gemini API models is withheld. The Gemini API
# additional terms assign output ownership to the developer while forbidding
# the use of that output to build or improve a competing model, so releasing
# the text under CC BY 4.0 would purport to grant rights we do not hold. The
# paper states this commitment for the five held-out models; we apply it to
# every gemini-* id in the databases, which is a superset and so strictly more
# conservative. Their scores, per-dimension breakdowns and derived statistics
# are released like everyone else's.
WITHHELD_TEXT_PREFIX = "google/gemini"
WITHHELD_PLACEHOLDER = "[withheld: Gemini API terms, see LICENSE-DATA]"

# columns holding model-generated text, keyed off whichever column names the model
GENERATED_TEXT_COLS = {"idea_text", "hypothesis", "best_idea_text", "query"}
MODEL_COLS = ("idea_model", "gen_model", "model", "judge_model")


def is_withheld(row):
    """True when this row's generating model is a Gemini API model."""
    for col in MODEL_COLS:
        v = row.get(col)
        if isinstance(v, str) and v.startswith(WITHHELD_TEXT_PREFIX):
            return True
    return False


def apply_withholding(row):
    """Blank the generated text of a withheld model, in place. Returns the row."""
    if not is_withheld(row):
        return row
    for col in GENERATED_TEXT_COLS:
        if row.get(col):
            row[col] = WITHHELD_PLACEHOLDER
    return row


def flatten_scores(raw):
    """scores_json -> five numeric columns.

    Two shapes exist in the databases: a flat ``{"originality": 8, ...}`` and a
    nested ``{"originality": {"score": 6, "reasoning": "..."}, ...}``. Only the
    number survives either way.
    """
    out = {f"score_{d}": None for d in SCORE_DIMS}
    if not raw:
        return out
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return out
    if not isinstance(d, dict):
        return out
    for dim in SCORE_DIMS:
        v = d.get(dim)
        if isinstance(v, dict):
            v = v.get("score")
        if isinstance(v, (int, float)):
            out[f"score_{dim}"] = v
    return out


def strip_cited_titles(raw):
    """A ``Cited:`` footer -> bracket indices only.

    The footer the generator emits is ``["[14] Highly accurate protein
    structure prediction with AlphaFold", ...]``. Which references a run cited
    is the analysable part; the titles are reference-list text, which the
    release does not carry. Weak models sometimes paste a sentence of the
    abstract in place of the title, which is the other reason to cut here.
    """
    if not raw:
        return raw
    try:
        items = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(items, list):
        return None
    idx = []
    for it in items:
        if not isinstance(it, str):
            continue
        m = re.match(r"\s*\[(\d+)\]", it)
        if m:
            idx.append(int(m.group(1)))
    return json.dumps(idx) if idx else None


def flatten_aux(raw):
    """aux_json -> scalar columns. Lists become compact JSON."""
    out = {}
    if not raw:
        return out
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return out
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        out[f"aux_{k}"] = (json.dumps(v, ensure_ascii=False)
                           if isinstance(v, (list, dict)) else v)
    return out


def transform_row(table, row):
    """Apply the column rules to one row. Returns the released dict."""
    out = {}
    for k, v in row.items():
        if k in DROP_COLS:
            continue
        if k == "cited_refs":
            out[k] = strip_cited_titles(v)
        elif k == "scores_json":
            out.update(flatten_scores(v))
        elif k == "aux_json":
            out.update(flatten_aux(v))
        else:
            out[k] = v
    return apply_withholding(out)


# ---------------------------------------------------------------------------
# Side tables lifted out of the dropped payloads
# ---------------------------------------------------------------------------

def _ids_from_refs(payload):
    """A reference list -> identifiers and counts only.

    Drops ``title``, ``abstract`` and ``contexts`` (citation contexts are
    excerpts of the citing paper's full text).
    """
    try:
        items = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    rows = []
    for rank, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        rows.append({
            "rank": rank,
            "ss_paper_id": it.get("paperId"),
            "year": it.get("year"),
            "citation_count": it.get("citationCount"),
            "in_paper_citations": it.get("in_paper_citations"),
            "is_influential": it.get("isInfluential"),
        })
    return rows


def side_paper_references(conn):
    """papers.references_json + papers.ranked_refs_json -> id-only rows."""
    conn.row_factory = sqlite3.Row
    for r in conn.execute("SELECT paper_id, references_json, ranked_refs_json FROM papers"):
        for kind, payload in (("reference", r["references_json"]),
                              ("ranked", r["ranked_refs_json"])):
            for row in _ids_from_refs(payload):
                yield {"paper_id": r["paper_id"], "list_kind": kind, **row}


def side_static_refs(conn):
    """The curated Static-track reference sets, as identifiers."""
    specs = [
        ("subdomain_refs", "subdomain", "subdomain, domain"),
        ("domain_topic_refs", "query", "query, domain"),
        ("e38_replay_refs", "subdomain", "idea_model, domain, subdomain"),
    ]
    conn.row_factory = sqlite3.Row
    for table, keycol, cols in specs:
        for r in conn.execute(f"SELECT {cols}, refs_json FROM {table}"):
            base = {"source_table": table, "key": r[keycol]}
            base.update({c.strip(): r[c.strip()] for c in cols.split(",")})
            for row in _ids_from_refs(r["refs_json"]):
                yield {**base, **row}


def _trace_calls(payload):
    """A tool-call trace -> query strings and returned identifiers.

    ``result_preview`` is the raw Semantic Scholar response, abstracts and all,
    so only the paperIds are lifted out of it and the text is discarded.
    """
    try:
        d = json.loads(payload)
    except (TypeError, ValueError):
        return []
    calls = d.get("trace") if isinstance(d, dict) else d
    if not isinstance(calls, list):
        return []
    out = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        args = c.get("args") or {}
        preview = c.get("result_preview") or ""
        ids = []
        try:
            hits = json.loads(preview)
            if isinstance(hits, list):
                ids = [h.get("paperId") for h in hits if isinstance(h, dict)]
        except (TypeError, ValueError):
            ids = []
        out.append({
            "iter": c.get("iter"),
            "tool": c.get("tool"),
            "query": args.get("query") if isinstance(args, dict) else None,
            "limit": args.get("limit") if isinstance(args, dict) else None,
            "n_results": len(ids),
            "returned_ss_ids": json.dumps(ids, ensure_ascii=False) if ids else None,
        })
    return out


def side_active_traces(conn):
    """Active-track retrieval behaviour, from telemetry and trace_json."""
    conn.row_factory = sqlite3.Row
    q = ("SELECT idea_model, domain, subdomain, track, idea_index, telemetry "
         "FROM subdomain_ideas WHERE telemetry IS NOT NULL")
    for r in conn.execute(q):
        base = {"source_table": "subdomain_ideas", "idea_model": r["idea_model"],
                "domain": r["domain"], "subdomain": r["subdomain"],
                "track": r["track"], "idea_index": r["idea_index"], "budget": None}
        for c in _trace_calls(r["telemetry"]):
            yield apply_withholding({**base, **c})
    for table in ("budget_sweep_ideas", "budget_sweep_ideas_v2", "budget_sweep_ideas_v3"):
        q = (f"SELECT paper_id, domain, idea_model, budget, idea_index, trace_json "
             f"FROM {table} WHERE trace_json IS NOT NULL")
        for r in conn.execute(q):
            base = {"source_table": table, "idea_model": r["idea_model"],
                    "domain": r["domain"], "subdomain": r["paper_id"],
                    "track": "active", "idea_index": r["idea_index"],
                    "budget": r["budget"]}
            for c in _trace_calls(r["trace_json"]):
                yield apply_withholding({**base, **c})


def side_evidence_hits(conn):
    """e13_evidence.evidence_json -> one row per retrieved paper, ids only."""
    conn.row_factory = sqlite3.Row
    q = "SELECT item_id, grp, cutoff_date, evidence_json FROM e13_evidence"
    for r in conn.execute(q):
        try:
            items = json.loads(r["evidence_json"] or "[]")
        except (TypeError, ValueError):
            continue
        if not isinstance(items, list):
            continue
        for rank, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            yield {
                "item_id": r["item_id"], "grp": r["grp"],
                "cutoff_date": r["cutoff_date"], "rank": rank,
                "query": it.get("query"),
                "ss_paper_id": it.get("paperId"),
                "year": it.get("year"),
                "publication_date": it.get("publicationDate"),
                "citation_count": it.get("citationCount"),
            }


SIDE_TABLES = [
    ("paper_references", PAPERS_DB, side_paper_references,
     "papers.references_json + papers.ranked_refs_json"),
    ("static_refs", RESULTS_DB, side_static_refs,
     "subdomain_refs / domain_topic_refs / e38_replay_refs .refs_json"),
    ("active_traces", RESULTS_DB, side_active_traces,
     "subdomain_ideas.telemetry + budget_sweep_ideas*.trace_json"),
    ("evidence_hits", RESULTS_DB, side_evidence_hits,
     "e13_evidence.evidence_json"),
]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def connect(db):
    # papers.db is left in WAL mode, which a plain mode=ro open cannot attach
    # to without writing a -shm file, so open it immutable instead.
    uri = f"file:{db}?immutable=1" if db == PAPERS_DB else f"file:{db}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def write_csv_gz(dest, rows):
    """Write dicts to a gzipped CSV.

    Materialises first so the header is the union of every row's keys, in first-
    seen order. Taking the header from the first row alone would silently drop
    columns wherever the key set varies between rows, which it does in two
    places: ``aux_json`` expands to different keys per condition, and the side
    tables merge several source tables with different key columns. Redacted,
    the largest product is tens of MB, so holding one table in memory is fine.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with gzip.open(dest, "wt", encoding="utf-8", newline="") as fh:
        if fields:
            writer = csv.DictWriter(fh, fieldnames=fields, restval="",
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return len(rows), fields, dest.stat().st_size


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(out):
    manifest = {
        "release": "AgentIdeaBench",
        "licence": "CC BY 4.0 (see LICENSE-DATA)",
        "redaction": ("No abstract text, no reference-list text, no full text. "
                      "Reference lists and retrieval traces are reduced to "
                      "Semantic Scholar identifiers; refetch content from the "
                      "API under your own agreement."),
        "dropped_columns": sorted(DROP_COLS),
        "withheld_text": ("Generated text from google/gemini* models is replaced by a placeholder; their scores and derived statistics are released."),
        "files": [],
    }

    for group, roster in (("core", CORE), ("appendix", APPENDIX)):
        for db, tables in roster.items():
            conn = connect(db)
            conn.row_factory = sqlite3.Row
            present = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            for table in tables:
                if table not in present:
                    print(f"  !! missing table {table} in {db.name}", file=sys.stderr)
                    continue
                dest = out / group / f"{table}.csv.gz"
                src_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
                rows = (transform_row(table, dict(r))
                        for r in conn.execute(f'SELECT * FROM "{table}"'))
                n, fields, size = write_csv_gz(dest, rows)
                manifest["files"].append({
                    "path": str(dest.relative_to(out)), "group": group,
                    "source": f"{db.name}:{table}", "rows": n,
                    "bytes": size, "sha256": sha256(dest),
                    "columns": fields,
                    "dropped": sorted(set(src_cols) & DROP_COLS),
                })
                print(f"  {group}/{table:34} {n:7d} rows  {size/1e6:7.2f} MB")
            conn.close()

    for name, db, fn, provenance in SIDE_TABLES:
        conn = connect(db)
        dest = out / "derived" / f"{name}.csv.gz"
        n, fields, size = write_csv_gz(dest, fn(conn))
        manifest["files"].append({
            "path": str(dest.relative_to(out)), "group": "derived",
            "source": provenance, "rows": n, "bytes": size,
            "sha256": sha256(dest), "columns": fields, "dropped": [],
        })
        print(f"  derived/{name:32} {n:7d} rows  {size/1e6:7.2f} MB")
        conn.close()

    total = sum(f["bytes"] for f in manifest["files"])
    manifest["total_bytes"] = total
    manifest["total_rows"] = sum(f["rows"] for f in manifest["files"])
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  {len(manifest['files'])} files, "
          f"{manifest['total_rows']:,} rows, {total/1e6:.1f} MB total")
    biggest = max(manifest["files"], key=lambda f: f["bytes"])
    print(f"  largest: {biggest['path']} at {biggest['bytes']/1e6:.1f} MB")
    return manifest


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def abstract_shingles(k=400, width=60, seed=0):
    """Distinctive substrings of real abstracts, to search the outputs for."""
    rng = random.Random(seed)
    shingles = []
    sources = [
        (PAPERS_DB, "SELECT abstract FROM papers WHERE abstract IS NOT NULL"),
        (PAPERS_DB, "SELECT references_json FROM papers WHERE references_json IS NOT NULL"),
        (PAPERS_DB, "SELECT ranked_refs_json FROM papers WHERE ranked_refs_json IS NOT NULL"),
        (RESULTS_DB, "SELECT refs_json FROM subdomain_refs"),
        (RESULTS_DB, "SELECT evidence_json FROM e13_evidence LIMIT 2000"),
        (RESULTS_DB, "SELECT telemetry FROM subdomain_ideas WHERE telemetry IS NOT NULL LIMIT 2000"),
        (RESULTS_DB, "SELECT real_abstract FROM domain_anchor_pool WHERE real_abstract IS NOT NULL"),
        (RESULTS_DB, "SELECT gt_hypothesis FROM results LIMIT 0"),  # column may not exist
    ]
    for db, q in sources:
        try:
            conn = connect(db)
            blobs = [r[0] for r in conn.execute(q) if r[0]]
            conn.close()
        except sqlite3.Error:
            continue
        rng.shuffle(blobs)
        for blob in blobs[:k // 4]:
            texts = []
            try:                                   # pull nested abstracts out
                d = json.loads(blob)
                stack = [d]
                while stack:
                    cur = stack.pop()
                    if isinstance(cur, dict):
                        for key, val in cur.items():
                            if key in ("abstract", "result_preview") and isinstance(val, str):
                                texts.append(val)
                            else:
                                stack.append(val)
                    elif isinstance(cur, list):
                        stack.extend(cur)
            except (TypeError, ValueError):
                texts = [blob]
            for t in texts:
                if isinstance(t, str) and len(t) > width * 3:
                    mid = len(t) // 2
                    shingles.append(t[mid:mid + width])
    rng.shuffle(shingles)
    return shingles[:k]


def verify(out):
    """Read every product back and assert the redaction boundary held."""
    failures = []
    regurgitation = {}

    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))

    # 1. no dropped column survived into any header
    for f in manifest["files"]:
        leaked = set(f["columns"]) & DROP_COLS
        if leaked:
            failures.append(f"{f['path']}: dropped column present: {sorted(leaked)}")

    # 2. no released text field is long, except the whitelisted generations
    # 3. no abstract shingle appears anywhere in the products
    shingles = abstract_shingles()
    if len(shingles) < 50:
        failures.append(f"only {len(shingles)} abstract shingles built; "
                        "verification would be vacuous")
    print(f"  searching {len(shingles)} abstract shingles across "
          f"{len(manifest['files'])} files ...")

    for f in manifest["files"]:
        path = out / f["path"]
        long_offenders = {}
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            assembled, model_out = [], []
            for row in csv.DictReader(fh):
                for k, v in row.items():
                    if not v:
                        continue
                    if len(v) > MAX_SHORT_TEXT and k not in LONG_TEXT_OK:
                        long_offenders[k] = max(long_offenders.get(k, 0), len(v))
                    (model_out if k in MODEL_OUTPUT_COLS else assembled).append(v)
            assembled_blob = "\n".join(assembled)
            model_blob = "\n".join(model_out)
        for k, n in long_offenders.items():
            failures.append(f"{f['path']}: column '{k}' holds {n} chars, "
                            f"over the {MAX_SHORT_TEXT}-char limit and not whitelisted")
        hits = [s for s in shingles if s in assembled_blob]
        if hits:
            failures.append(f"{f['path']}: {len(hits)} abstract shingle(s) in "
                            f"assembled fields, first: {hits[0][:40]!r}")
        quoted = [s for s in shingles if s in model_blob]
        if quoted:
            regurgitation[f["path"]] = len(quoted)

    # 4. no Gemini-generated text survived anywhere
    for f in manifest["files"]:
        cols = set(f["columns"])
        if not (cols & GENERATED_TEXT_COLS) or not (cols & set(MODEL_COLS)):
            continue
        leaks = 0
        with gzip.open(out / f["path"], "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if not is_withheld(row):
                    continue
                for col in GENERATED_TEXT_COLS & cols:
                    v = row.get(col)
                    if v and v != WITHHELD_PLACEHOLDER:
                        leaks += 1
        if leaks:
            failures.append(f"{f['path']}: {leaks} Gemini-generated text value(s) "
                            "released; Gemini API terms forbid this")

    # 5. checksums still match what the manifest recorded
    for f in manifest["files"]:
        if sha256(out / f["path"]) != f["sha256"]:
            failures.append(f"{f['path']}: sha256 mismatch against MANIFEST.json")

    if regurgitation:
        total = sum(regurgitation.values())
        print(f"\n  note: {total} abstract shingle(s) appear inside model-generated "
              f"text in {len(regurgitation)} file(s).")
        print("  These are quotations a model wrote into its own output, not "
              "tables we assembled.")
        print("  Left intact deliberately -- rewriting model output would corrupt "
              "the artifact.")
        for path, n in sorted(regurgitation.items(), key=lambda x: -x[1]):
            print(f"    {path}: {n}")

    if failures:
        print("\nVERIFY FAILED:")
        for x in failures:
            print("  -", x)
        return False
    print("  verify OK: no dropped columns, no long unwhitelisted text, "
          "no abstract shingles, checksums match")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "release_data"))
    ap.add_argument("--verify-only", action="store_true",
                    help="skip the export, just re-check an existing release_data/")
    args = ap.parse_args()
    out = Path(args.out)

    if not args.verify_only:
        out.mkdir(parents=True, exist_ok=True)
        print("exporting ...")
        export(out)
    print("\nverifying ...")
    if not verify(out):
        sys.exit(1)


if __name__ == "__main__":
    main()
