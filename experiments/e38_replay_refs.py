"""E38 — replay control: decompose the Active bundle (review point #5).

The Active advantage bundles three changes relative to Static: retrieval
control, multi-turn interaction, and tool-use competence. Track R (recall-only,
app:recall) already showed that mere reference *presence* is not the driver
(B~=R<C). This adds the complementary arm that separates the *content* the
agent retrieves from the *agentic process* of retrieving it:

  Arm R' (replay) = feed the model, in the passive single-pass Static format,
  the exact papers its own Active agent surfaced (extracted from the stored
  Track-C telemetry). Same prompt machinery as Static (_build_generation_payload
  "B"), same reference count (<=10), only the reference SOURCE differs
  (agent-surfaced vs designer-curated).

Reading:
  R'~=B<C  -> the gain is the agentic *process* (control + interaction), not the
             content of what gets retrieved.
  R'~=C    -> the gain is the retrieved *content*; passive access to it suffices.
Either way this decomposes the content-vs-process confound the paper flags.

Scope mirrors e32: 28 open-weight both-track models x 10 subdomains (per-domain
alphabetical index {0,5}) x 3 ideas = 840 cells. Gemini is held-out and
excluded (open-weight default OPENROUTER_API_KEY only). Active refs are read
from subdomain_ideas.telemetry (search_papers results carry full title+abstract,
so NO Semantic Scholar call is needed to rebuild them).

Raw-data rule: new tables e38_replay_refs / e38_replay_ideas / e38_replay_scores
(INSERT-only); evidence rows in e13_evidence grp='e38'. Idempotent, resume-safe.

Usage:
  /usr/bin/python3 experiments/e38_replay_refs.py --smoke
  /usr/bin/python3 experiments/e38_replay_refs.py --extract
  /usr/bin/python3 experiments/e38_replay_refs.py --gen   [--workers 8]
  /usr/bin/python3 experiments/e38_replay_refs.py --prep  [--workers 3]
  /usr/bin/python3 experiments/e38_replay_refs.py --score [--workers 6]
  /usr/bin/python3 experiments/e38_replay_refs.py --analyze
"""
import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from experiments.e10_idea_anchor_calibration import score_one, trimmed_mean
from experiments.e13_litverify_rubric8 import (
    ensure_tables, build_evidence_for_item, format_evidence_block,
    LIT8D_SYSTEM, V4_IDEA_CUTOFF)
from experiments.e32_recall_only import _seed_cells, _paired
from generation.generate_ideas import _build_generation_payload, _clean_idea_text
from experiments.v3_subdomain_ideation import MIN_WORDS

DOMAINS = ["Biology", "CS", "Chemistry", "Medicine", "Physics"]
SUB_IDX = (0, 5)          # per-domain alphabetical indices -> 10 subdomains
IDEA_IDX = (1, 2, 3)
MAX_REFS = 10             # match the Static top-10 reference count
OUT_JSON = ROOT / "reports" / "e38_replay_refs.json"


def _short(m):
    return m.split("/")[-1]


def _iid(short, idx, sub):
    return f"e38|{short}|{idx}|{sub[:40]}"


def ensure_tbls(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e38_replay_refs (
        idea_model TEXT, domain TEXT, subdomain TEXT,
        refs_json TEXT, n_refs INTEGER, created_at TEXT,
        PRIMARY KEY (idea_model, subdomain));
    CREATE TABLE IF NOT EXISTS e38_replay_ideas (
        idea_model TEXT, domain TEXT, subdomain TEXT, idea_index INTEGER,
        idea_text TEXT, n_refs INTEGER, created_at TEXT,
        PRIMARY KEY (idea_model, subdomain, idea_index));
    CREATE TABLE IF NOT EXISTS e38_replay_scores (
        item_id TEXT, idea_model TEXT, subdomain TEXT, domain TEXT, idea_index INTEGER,
        critic_model TEXT, scores_json TEXT, error TEXT, created_at TEXT,
        PRIMARY KEY (idea_model, subdomain, idea_index, critic_model));""")
    conn.commit()


def pick_subs(conn):
    out = []
    for dom in DOMAINS:
        rows = [r[0] for r in conn.execute(
            "SELECT subdomain FROM subdomain_refs WHERE domain=? ORDER BY subdomain", (dom,))]
        out += [(dom, rows[i]) for i in SUB_IDX]
    return out


def roster(conn):
    """28 open-weight both-track models (exclude google/gemini held-out)."""
    b = {r[0] for r in conn.execute(
        "SELECT DISTINCT idea_model FROM lit8d_scores_3seed WHERE track='B'")}
    c = {r[0] for r in conn.execute(
        "SELECT DISTINCT idea_model FROM lit8d_scores_3seed WHERE track='C'")}
    return sorted(m for m in (b & c) if not m.startswith("google/gemini-"))


# ------------------------------------------------------- extract active refs
def _active_refs(conn, model, sub):
    """Rebuild the papers the Active agent surfaced for (model, sub) from the
    stored Track-C telemetry: dedup search_papers hits by paperId in encounter
    order, keep title+abstract, cap at MAX_REFS. No Semantic Scholar call."""
    rows = conn.execute(
        "SELECT telemetry FROM subdomain_ideas WHERE idea_model=? AND subdomain=? "
        "AND track='C' AND telemetry IS NOT NULL AND TRIM(telemetry)!=''",
        (model, sub)).fetchall()
    seen, refs = set(), []
    for (tel_s,) in rows:
        try:
            tel = json.loads(tel_s)
        except (json.JSONDecodeError, TypeError):
            continue
        for step in tel.get("trace", []):
            if step.get("tool") != "search_papers":
                continue
            rp = step.get("result_preview") or step.get("result") or ""
            try:
                papers = json.loads(rp) if isinstance(rp, str) else rp
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(papers, list):
                continue
            for p in papers:
                pid = p.get("paperId") or p.get("paper_id")
                if not pid or pid in seen:
                    continue
                if not (p.get("abstract") or "").strip():
                    continue          # need an abstract to match Static refs
                seen.add(pid)
                refs.append({"paperId": pid, "title": p.get("title") or "Untitled",
                             "abstract": (p.get("abstract") or "").strip(),
                             "year": p.get("year"),
                             "citationCount": p.get("citationCount")})
    return refs[:MAX_REFS]


def extract(models=None, subs=None):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tbls(conn)
    models = models or roster(conn)
    subs = subs or pick_subs(conn)
    ts = datetime.now(timezone.utc).isoformat()
    have = {(r[0], r[1]) for r in conn.execute("SELECT idea_model, subdomain FROM e38_replay_refs")}
    n_ok = n_empty = 0
    empties = []
    for m in models:
        for dom, sub in subs:
            if (m, sub) in have:
                continue
            refs = _active_refs(conn, m, sub)
            conn.execute("INSERT OR IGNORE INTO e38_replay_refs VALUES (?,?,?,?,?,?)",
                         (m, dom, sub, json.dumps(refs, ensure_ascii=False), len(refs), ts))
            conn.commit()
            if refs:
                n_ok += 1
            else:
                n_empty += 1; empties.append((_short(m), sub[:24]))
    counts = [r[0] for r in conn.execute("SELECT n_refs FROM e38_replay_refs")]
    conn.close()
    print(f"extract done: cells_with_refs={n_ok} empty={n_empty} "
          f"| n_refs dist: min={min(counts)} median={int(np.median(counts))} max={max(counts)}")
    if empties:
        print(f"  EMPTY cells (no usable telemetry, will be skipped): {empties[:20]}"
              f"{' ...' if len(empties) > 20 else ''}")


# ---------------------------------------------------------------- generation
def _synthetic_paper(dom, sub, refs):
    return {"title": sub, "domain": dom,
            "ranked_refs_json": json.dumps(refs, ensure_ascii=False),
            "paper_id": "e38replay:" + sub}


def gen(workers=8, models=None, subs=None, idxs=IDEA_IDX):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tbls(conn)
    models = models or roster(conn)
    subs = subs or pick_subs(conn)
    subset = {s for _, s in subs}
    refmap = {(r[0], r[1]): (r[2], json.loads(r[3])) for r in conn.execute(
        "SELECT idea_model, subdomain, domain, refs_json FROM e38_replay_refs")
        if r[1] in subset and (models is None or r[0] in set(models))}
    tasks = [(m, refmap[(m, sub)][0], sub, i) for m in models for _, sub in subs for i in idxs
             if (m, sub) in refmap and refmap[(m, sub)][1]          # skip empty-ref cells
             and not conn.execute("SELECT 1 FROM e38_replay_ideas WHERE idea_model=? "
                                  "AND subdomain=? AND idea_index=? AND TRIM(idea_text)!=''",
                                  (m, sub, i)).fetchone()]
    print(f"models={len(models)} subs={len(subs)} gen tasks: {len(tasks)}", flush=True)
    ts = datetime.now(timezone.utc).isoformat()

    def _one(t):
        from utils.LLM import IdeaLLM
        m, dom, sub, i = t
        refs = refmap[(m, sub)][1]
        paper = _synthetic_paper(dom, sub, refs)
        prompt, fallback, system = _build_generation_payload(paper, "B")
        try:
            out = IdeaLLM(model_name=m).generate_idea(
                prompt, fallback_prompt=fallback, system_prompt=system)
            txt = _clean_idea_text(out["idea"])
            if not txt or len(txt.split()) < MIN_WORDS:
                return t, "", len(refs), "too_short_or_empty"
            return t, txt, len(refs), None
        except Exception as e:
            return t, "", len(refs), str(e)[:200]

    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(_one, t) for t in tasks]), 1):
            (m, dom, sub, i), txt, nref, e = f.result()
            if txt:
                conn.execute("INSERT OR IGNORE INTO e38_replay_ideas VALUES (?,?,?,?,?,?,?)",
                             (m, dom, sub, i, txt, nref, ts))
                conn.commit(); ok += 1
            else:
                err += 1
            if n % 25 == 0 or n == len(tasks):
                print(f"  [{n}/{len(tasks)}] ok={ok} err={err}", flush=True)
            if e and n <= 40:
                print(f"    fail {_short(m)}|{sub[:20]}|{i}: {e}", flush=True)
    conn.close(); print(f"gen done ok={ok} err={err}")


# ---------------------------------------------------------------- evidence
def _items(conn):
    return conn.execute("SELECT idea_model, subdomain, domain, idea_index, idea_text "
                        "FROM e38_replay_ideas WHERE TRIM(idea_text)!=''").fetchall()


def prep(workers=3, limit=0):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tables(conn); ensure_tbls(conn)
    items = _items(conn)
    todo = [it for it in items
            if not conn.execute("SELECT 1 FROM e13_evidence WHERE item_id=?",
                                (_iid(_short(it[0]), it[3], it[1]),)).fetchone()]
    total = len(todo)
    if limit:
        todo = todo[:limit]
    print(f"{len(items)} ideas; {total} need evidence; building {len(todo)} this batch", flush=True)
    ts = datetime.now(timezone.utc).isoformat()

    def _ev(it):
        m, sub, dom, i, txt = it
        iid = _iid(_short(m), i, sub)
        q, ev, ns = build_evidence_for_item(iid, "e38", txt, V4_IDEA_CUTOFF, None, None)
        return iid, q, ev, ns

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_ev, it) for it in todo]):
            iid, q, ev, ns = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                         (iid, "e38", V4_IDEA_CUTOFF, json.dumps(q), json.dumps(ev), len(ev), ns, ts))
            conn.commit(); done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}]", flush=True)
    conn.close(); print("prep done")


# ---------------------------------------------------------------- scoring
def score(workers=6, limit=0):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tbls(conn)
    critics = list(cfg.CRITIC_MODELS)[:3]
    items = _items(conn)
    have_ev = {r[0] for r in conn.execute("SELECT item_id FROM e13_evidence WHERE grp=?", ("e38",))}
    done = {(r[0], r[1], r[2], r[3]) for r in conn.execute(
        "SELECT idea_model, subdomain, idea_index, critic_model FROM e38_replay_scores "
        "WHERE error IS NULL AND scores_json IS NOT NULL")}
    tasks = [(m, sub, dom, i, txt, cr)
             for m, sub, dom, i, txt in items if _iid(_short(m), i, sub) in have_ev
             for cr in critics if (m, sub, i, cr) not in done]
    total = len(tasks)
    if limit:
        tasks = tasks[:limit]
    print(f"score tasks remaining: {total}; running {len(tasks)} this batch", flush=True)
    refs = {}
    for m, sub, dom, i, txt, cr in tasks:
        iid = _iid(_short(m), i, sub)
        if iid not in refs:
            r = conn.execute("SELECT evidence_json, cutoff_date FROM e13_evidence WHERE item_id=?",
                             (iid,)).fetchone()
            refs[iid] = format_evidence_block(json.loads(r[0] or "[]"), r[1]) if r else ""

    def _sc(t):
        m, sub, dom, i, txt, cr = t
        iid = _iid(_short(m), i, sub)
        s, _ = score_one(txt, cr, dom, LIT8D_SYSTEM, refs[iid])
        return iid, m, sub, dom, i, cr, s

    ts = datetime.now(timezone.utc).isoformat(); ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(_sc, t) for t in tasks]), 1):
            iid, m, sub, dom, i, cr, s = f.result()
            conn.execute("INSERT OR REPLACE INTO e38_replay_scores VALUES (?,?,?,?,?,?,?,?,?)",
                         (iid, m, sub, dom, i, cr, json.dumps(s) if s else None,
                          None if s else "parse_fail", ts))
            conn.commit(); ok += 1 if s else 0; err += 0 if s else 1
            if n % 25 == 0 or n == len(tasks):
                print(f"  [{n}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close(); print(f"score done ok={ok} err={err}")


# ---------------------------------------------------------------- analysis
def analyze():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    subs = pick_subs(conn)
    subset = {s for _, s in subs}
    models = set(roster(conn))
    rp_cells = _seed_cells([(m, sub, i, sj) for m, sub, i, sj in conn.execute(
        "SELECT idea_model, subdomain, idea_index, scores_json FROM e38_replay_scores "
        "WHERE scores_json IS NOT NULL")])
    bc_rows = [(m, sub, tr, i, sj) for m, sub, tr, i, sj in conn.execute(
        "SELECT idea_model, subdomain, track, idea_index, scores_json FROM lit8d_scores_3seed "
        "WHERE scores_json IS NOT NULL") if sub in subset and m in models]
    conn.close()
    b_cells = _seed_cells([(m, sub, i, sj) for m, sub, tr, i, sj in bc_rows if tr == "B"])
    c_cells = _seed_cells([(m, sub, i, sj) for m, sub, tr, i, sj in bc_rows if tr == "C"])

    rng = np.random.default_rng(20260802)
    shared = sorted(k for k in rp_cells if k in b_cells and k in c_cells)
    per_model = defaultdict(lambda: defaultdict(list))
    for (m, sub) in shared:
        per_model[m]["Rp"].append(rp_cells[(m, sub)])
        per_model[m]["B"].append(b_cells[(m, sub)])
        per_model[m]["C"].append(c_cells[(m, sub)])
    pm = {m.split("/")[-1]: {t: round(float(np.mean(v[t])), 4) for t in ("Rp", "B", "C")}
          for m, v in sorted(per_model.items())}
    md = lambda a, b: [np.mean(v[a]) - np.mean(v[b]) for v in per_model.values()]

    out = {
        "generated_by": "experiments/e38_replay_refs.py",
        "note": "Arm R' (replay) = Active-surfaced references fed in the passive "
                "Static (_build_generation_payload 'B') format, same top-<=10 "
                "reference count. B/C recomputed on the SAME 10 subdomains and "
                "shared models for exact pairing. Cell = (model, subdomain), mean "
                "of 3 ideas' weighted trimmed lit8d scores. "
                "R'~=B<C -> gain is agentic process not retrieved content; "
                "R'~=C -> gain is retrieved content.",
        "n_models": len(per_model), "n_subdomains": len({s for _, s in shared}),
        "n_shared_cells": len(shared),
        "means": {"Rp": round(float(np.mean([rp_cells[k] for k in shared])), 4),
                  "B": round(float(np.mean([b_cells[k] for k in shared])), 4),
                  "C": round(float(np.mean([c_cells[k] for k in shared])), 4)},
        "cell_level": {
            "Rp_minus_B": _paired([rp_cells[k] - b_cells[k] for k in shared], rng),
            "C_minus_Rp": _paired([c_cells[k] - rp_cells[k] for k in shared], rng),
            "C_minus_B": _paired([c_cells[k] - b_cells[k] for k in shared], rng)},
        "model_level": {
            "Rp_minus_B": _paired(md("Rp", "B"), rng),
            "C_minus_Rp": _paired(md("C", "Rp"), rng),
            "C_minus_B": _paired(md("C", "B"), rng)},
        "per_model": pm,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("n_models", "n_subdomains", "n_shared_cells", "means")},
                     indent=2))
    print(json.dumps(out["cell_level"], indent=2))
    print("wrote", OUT_JSON)


def main():
    ap = argparse.ArgumentParser()
    for flag in ("extract", "gen", "prep", "score", "analyze", "smoke"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        conn = sqlite3.connect(str(cfg.RESULTS_DB)); ensure_tbls(conn)
        subs = pick_subs(conn)[:1]; conn.close()
        models = ["qwen/qwen3.5-9b", "z-ai/glm-4.5-air"]
        print(f"SMOKE: {models} x {[s for _, s in subs]} x idx1")
        extract(models=models, subs=subs)
        gen(workers=2, models=models, subs=subs, idxs=(1,))
        prep(workers=2)
        score(workers=3)
        conn = sqlite3.connect(str(cfg.RESULTS_DB))
        for r in conn.execute("SELECT idea_model, subdomain, n_refs FROM e38_replay_refs"):
            print("  refs:", _short(r[0]), r[1][:24], "n_refs=", r[2])
        for r in conn.execute("SELECT idea_model, idea_index, critic_model, "
                              "scores_json IS NOT NULL FROM e38_replay_scores"):
            print("  scored:", _short(r[0]), r[1], _short(r[2]), r[3])
        conn.close()
        return
    if a.extract:
        extract()
    if a.gen:
        gen(workers=a.workers)
    if a.prep:
        prep(workers=min(a.workers, 3), limit=a.limit)
    if a.score:
        score(workers=a.workers, limit=a.limit)
    if a.analyze:
        analyze()
    if not any([a.extract, a.gen, a.prep, a.score, a.analyze, a.smoke]):
        ap.print_help()


if __name__ == "__main__":
    main()
