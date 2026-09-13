"""E32 — recall-only control track R (review point 1).

The paper contrasts Static (Track B: subdomain-scoped curated references) with
Active (Track C: agent-controlled retrieval). Neither is "recall": B also uses
external literature. This adds the missing third reference point:

  Track R = topic identity only (broad domain + subdomain name), NO references,
            NO tools — pure parametric-knowledge generation.

Prompting mirrors Track B (_SYSTEM_B/_USER_B) with the reference machinery
removed; no Cited footer (nothing to cite; Track C ideas also carry none).
Same sampling as B: 3 independent ideas per (model, subdomain), temp 0.7,
_clean_idea_text + MIN_WORDS=30 filter. Scoring = the exact lit8d pipeline.

Scope: the 28 both-track models x 10 subdomains (per-domain alphabetical
index {0,5}, a deterministic subset of the 40 scored) x 3 ideas = 840 cells.
B/C comparisons are recomputed on the SAME 10 subdomains for exact pairing.

Raw-data rule: new tables e32_recall_ideas / e32_recall_scores (INSERT-only);
evidence rows in e13_evidence grp='e32'. Idempotent, resume-safe. All 28 models
are open-weight (default OPENROUTER_API_KEY; google/gemma is not a US-key model).

Usage:
  /usr/bin/python3 experiments/e32_recall_only.py --smoke
  /usr/bin/python3 experiments/e32_recall_only.py --gen [--workers 8]
  /usr/bin/python3 experiments/e32_recall_only.py --prep [--workers 3]
  /usr/bin/python3 experiments/e32_recall_only.py --score [--workers 6]
  /usr/bin/python3 experiments/e32_recall_only.py --analyze
"""
import argparse, json, sqlite3, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as W
from experiments.e10_idea_anchor_calibration import score_one, trimmed_mean
from experiments.e13_litverify_rubric8 import (
    ensure_tables, build_evidence_for_item, format_evidence_block,
    LIT8D_SYSTEM, V4_IDEA_CUTOFF)
from generation.generate_ideas import _clean_idea_text
from experiments.v3_subdomain_ideation import MIN_WORDS

WSUM = sum(W.values())
DOMAINS = ["Biology", "CS", "Chemistry", "Medicine", "Physics"]
SUB_IDX = (0, 5)          # per-domain alphabetical indices -> 10 subdomains
IDEA_IDX = (1, 2, 3)
OUT_JSON = ROOT / "reports" / "e32_recall_only.json"

# Track B's _SYSTEM_B/_USER_B with the reference machinery removed; the
# subdomain is named explicitly (B conveys it via the refs, C receives it as
# the agent input), so R holds topic identity fixed while removing literature.
_SYSTEM_R = (
    "You are a creative and rigorous scientist. "
    "Your task is to propose a single novel, testable scientific hypothesis "
    "in the given research area, drawing only on your own knowledge of the "
    "field. Your hypothesis should be a new research direction, not a "
    "restatement or synthesis of well-known existing work."
)

_USER_R = """\
Propose a single novel scientific hypothesis in the field below.

Field: {domain}
Research area: {subdomain}

Requirements:
1. Output exactly one paragraph
2. Total length: 80-150 words
3. Your hypothesis should be a new direction in this research area, not a
   derivative of well-known published work
4. Be specific — name the exact mechanism, molecule, algorithm, or system
5. Be feasible — it should be testable with current technology
6. Output ONLY the hypothesis, no preamble or explanation

Hypothesis:"""

_FALLBACK_R = """\
Based on the following research area, propose one specific scientific
hypothesis in a single paragraph. Total length 80-150 words.

Field: {domain}
Research area: {subdomain}"""


def _iid(short, idx, sub):
    return f"e32|{short}|{idx}|{sub[:40]}"


def _short(m):
    return m.split("/")[-1]


def ensure_tbls(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e32_recall_ideas (
        idea_model TEXT, domain TEXT, subdomain TEXT, idea_index INTEGER,
        idea_text TEXT, created_at TEXT,
        PRIMARY KEY (idea_model, subdomain, idea_index));
    CREATE TABLE IF NOT EXISTS e32_recall_scores (
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
    """The 28 models with both Track B and C lit8d scores."""
    b = {r[0] for r in conn.execute(
        "SELECT DISTINCT idea_model FROM lit8d_scores_3seed WHERE track='B'")}
    c = {r[0] for r in conn.execute(
        "SELECT DISTINCT idea_model FROM lit8d_scores_3seed WHERE track='C'")}
    return sorted(b & c)


# ---------------------------------------------------------------- generation
def gen(workers=8, models=None, subs=None, idxs=IDEA_IDX):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tbls(conn)
    models = models or roster(conn)
    subs = subs or pick_subs(conn)
    tasks = [(m, dom, sub, i) for m in models for dom, sub in subs for i in idxs
             if not conn.execute("SELECT 1 FROM e32_recall_ideas WHERE idea_model=? "
                                 "AND subdomain=? AND idea_index=? AND TRIM(idea_text)!=''",
                                 (m, sub, i)).fetchone()]
    print(f"models={len(models)} subs={len(subs)} gen tasks: {len(tasks)}", flush=True)
    ts = datetime.now(timezone.utc).isoformat()

    def _one(t):
        from utils.LLM import IdeaLLM
        m, dom, sub, i = t
        try:
            out = IdeaLLM(model_name=m).generate_idea(
                _USER_R.format(domain=dom, subdomain=sub),
                fallback_prompt=_FALLBACK_R.format(domain=dom, subdomain=sub),
                system_prompt=_SYSTEM_R)
            txt = _clean_idea_text(out["idea"])
            if not txt or len(txt.split()) < MIN_WORDS:
                return t, "", "too_short_or_empty"
            return t, txt, None
        except Exception as e:
            return t, "", str(e)[:200]

    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(_one, t) for t in tasks]), 1):
            (m, dom, sub, i), txt, e = f.result()
            if txt:
                conn.execute("INSERT OR IGNORE INTO e32_recall_ideas VALUES (?,?,?,?,?,?)",
                             (m, dom, sub, i, txt, ts))
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
                        "FROM e32_recall_ideas WHERE TRIM(idea_text)!=''").fetchall()


def prep(workers=3, limit=0):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tables(conn); ensure_tbls(conn)
    items = _items(conn)
    todo = [it for it in items
            if not conn.execute("SELECT 1 FROM e13_evidence WHERE item_id=?",
                                (_iid(_short(it[0]), it[3], it[1]),)).fetchone()]
    total_remaining = len(todo)
    if limit:
        todo = todo[:limit]
    print(f"{len(items)} ideas; {total_remaining} need evidence; building {len(todo)} this batch"
          f"{' (limit=%d)' % limit if limit else ''}", flush=True)
    ts = datetime.now(timezone.utc).isoformat()

    def _ev(it):
        m, sub, dom, i, txt = it
        iid = _iid(_short(m), i, sub)
        q, ev, ns = build_evidence_for_item(iid, "e32", txt, V4_IDEA_CUTOFF, None, None)
        return iid, q, ev, ns

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_ev, it) for it in todo]):
            iid, q, ev, ns = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                         (iid, "e32", V4_IDEA_CUTOFF, json.dumps(q), json.dumps(ev), len(ev), ns, ts))
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
    # items that already have evidence (cheap id-only scan; refs are built later,
    # only for the items actually in this batch, to avoid re-formatting all 840 each run)
    have_ev = {r[0] for r in conn.execute("SELECT item_id FROM e13_evidence WHERE grp=?", ("e32",))}
    # already-scored (model, subdomain, idea_index, critic) tuples in one query (not 2520)
    done = {(r[0], r[1], r[2], r[3]) for r in conn.execute(
        "SELECT idea_model, subdomain, idea_index, critic_model FROM e32_recall_scores "
        "WHERE error IS NULL AND scores_json IS NOT NULL")}
    tasks = [(m, sub, dom, i, txt, cr)
             for m, sub, dom, i, txt in items if _iid(_short(m), i, sub) in have_ev
             for cr in critics
             if (m, sub, i, cr) not in done]
    total_remaining = len(tasks)
    if limit:
        tasks = tasks[:limit]
    print(f"score tasks remaining: {total_remaining}; running {len(tasks)} this batch"
          f"{' (limit=%d)' % limit if limit else ''}", flush=True)
    # build refs ONLY for the items in this batch
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
            conn.execute("INSERT OR REPLACE INTO e32_recall_scores VALUES (?,?,?,?,?,?,?,?,?)",
                         (iid, m, sub, dom, i, cr, json.dumps(s) if s else None,
                          None if s else "parse_fail", ts))
            conn.commit(); ok += 1 if s else 0; err += 0 if s else 1
            if n % 25 == 0 or n == len(tasks):
                print(f"  [{n}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close(); print(f"score done ok={ok} err={err}")


# ---------------------------------------------------------------- analysis
def _tm(vs):
    return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)


def _seed_cells(rows):
    """rows (m, sub, idx, scores_json) -> {(m, sub): mean weighted over ideas}."""
    per_idea = defaultdict(lambda: defaultdict(list))
    for m, sub, idx, sj in rows:
        s = json.loads(sj)
        for d in DIMS:
            per_idea[(m, sub, idx)][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    idea_w = {}
    for k, dv in per_idea.items():
        dims = {d: _tm(dv[d]) for d in DIMS}
        if all(dims[d] is not None for d in DIMS):
            idea_w[k] = sum(dims[d] * W[d] for d in DIMS) / WSUM
    cells = defaultdict(list)
    for (m, sub, _), w in idea_w.items():
        cells[(m, sub)].append(w)
    return {k: float(np.mean(v)) for k, v in cells.items()}


def _paired(delta, rng):
    from scipy import stats
    delta = np.asarray(delta, dtype=float)
    out = {"n": int(len(delta)), "delta_mean": round(float(delta.mean()), 4),
           "win_rate": round(float((delta > 0).mean()), 4)}
    if len(delta) >= 6:
        try:
            out["wilcoxon_p"] = round(float(stats.wilcoxon(delta).pvalue), 4)
        except ValueError:
            out["wilcoxon_p"] = None
        bs = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(5000)]
        out["delta_ci95"] = [round(float(np.percentile(bs, 2.5)), 4),
                             round(float(np.percentile(bs, 97.5)), 4)]
    return out


def analyze():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    subs = pick_subs(conn)
    subset = {s for _, s in subs}
    models = roster(conn)
    r_cells = _seed_cells([(m, sub, i, sj) for m, sub, i, sj in conn.execute(
        "SELECT idea_model, subdomain, idea_index, scores_json FROM e32_recall_scores "
        "WHERE scores_json IS NOT NULL")])
    bc_rows = [(m, sub, tr, i, sj) for m, sub, tr, i, sj in conn.execute(
        "SELECT idea_model, subdomain, track, idea_index, scores_json FROM lit8d_scores_3seed "
        "WHERE scores_json IS NOT NULL") if sub in subset and m in set(models)]
    conn.close()
    b_cells = _seed_cells([(m, sub, i, sj) for m, sub, tr, i, sj in bc_rows if tr == "B"])
    c_cells = _seed_cells([(m, sub, i, sj) for m, sub, tr, i, sj in bc_rows if tr == "C"])

    rng = np.random.default_rng(20260723)
    shared = sorted(k for k in r_cells if k in b_cells and k in c_cells)
    per_model = defaultdict(lambda: defaultdict(list))
    for (m, sub) in shared:
        per_model[m]["R"].append(r_cells[(m, sub)])
        per_model[m]["B"].append(b_cells[(m, sub)])
        per_model[m]["C"].append(c_cells[(m, sub)])
    pm = {m: {t: round(float(np.mean(v[t])), 4) for t in ("R", "B", "C")}
          for m, v in sorted(per_model.items())}
    model_dBR = [np.mean(v["B"]) - np.mean(v["R"]) for v in per_model.values()]
    model_dCR = [np.mean(v["C"]) - np.mean(v["R"]) for v in per_model.values()]
    model_dCB = [np.mean(v["C"]) - np.mean(v["B"]) for v in per_model.values()]

    out = {
        "note": "Track R = recall-only (broad domain + subdomain name, no refs, "
                "no tools). B/C recomputed on the SAME 10 subdomains and 28 "
                "models for exact pairing. Cell = (model, subdomain), mean of "
                "3 ideas' weighted trimmed lit8d scores.",
        "n_models": len(per_model), "n_subdomains": len({s for _, s in shared}),
        "n_shared_cells": len(shared),
        "means": {"R": round(float(np.mean([r_cells[k] for k in shared])), 4),
                  "B": round(float(np.mean([b_cells[k] for k in shared])), 4),
                  "C": round(float(np.mean([c_cells[k] for k in shared])), 4)},
        "cell_level": {
            "B_minus_R": _paired([b_cells[k] - r_cells[k] for k in shared], rng),
            "C_minus_R": _paired([c_cells[k] - r_cells[k] for k in shared], rng),
            "C_minus_B": _paired([c_cells[k] - b_cells[k] for k in shared], rng)},
        "model_level": {
            "B_minus_R": _paired(model_dBR, rng),
            "C_minus_R": _paired(model_dCR, rng),
            "C_minus_B": _paired(model_dCB, rng)},
        "per_model": pm,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("n_models", "n_subdomains", "n_shared_cells",
                                          "means")}, indent=2))
    print(json.dumps(out["cell_level"], indent=2))
    print("wrote", OUT_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 open models x 1 subdomain x idx 1: gen+prep+score")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N items this run (0=all); for small-batch resumable runs")
    a = ap.parse_args()
    if a.smoke:
        conn = sqlite3.connect(str(cfg.RESULTS_DB)); ensure_tbls(conn)
        subs = pick_subs(conn)[:1]; conn.close()
        models = ["qwen/qwen3.5-9b", "z-ai/glm-4.5-air"]
        print(f"SMOKE: {models} x {[s for _, s in subs]} x idx1")
        gen(workers=2, models=models, subs=subs, idxs=(1,))
        prep(workers=2)
        score(workers=3)
        conn = sqlite3.connect(str(cfg.RESULTS_DB))
        for r in conn.execute("SELECT idea_model, subdomain, idea_index, critic_model, "
                              "scores_json IS NOT NULL FROM e32_recall_scores"):
            print("  scored:", r)
        conn.close()
        return
    if a.gen:
        gen(workers=a.workers)
    if a.prep:
        prep(workers=min(a.workers, 3), limit=a.limit)
    if a.score:
        score(workers=a.workers, limit=a.limit)
    if a.analyze:
        analyze()
    if not any([a.gen, a.prep, a.score, a.analyze, a.smoke]):
        ap.print_help()


if __name__ == "__main__":
    main()
