"""E31 — compute-matched inference-scaling baseline for the SWM (review point 3).

The S4b condition spends ~21 LLM calls per cell (agent turns + internal panel)
vs ~7 for plain active_base — a ~3x compute gap. This experiment closes it with
best-of-3: two ADDITIONAL independent active_base runs per (backbone, subdomain)
cell (run 1 = the existing swm_ideas active_base row), each scored by the same
lit8d pipeline; best-of-3 = the max weighted cell score. 3 x 7 ~= 21 calls, so
best-of-3 is the equal-compute baseline for S4b.

Convention note: exactly as in E27, the agent input is the BROAD domain
(subdomains are paired replicate slots, not generation targets), max_iters=6.
Selection uses the same critic that defines the final metric — favorable to the
baseline, i.e. conservative for any SWM claim.

Raw-data rule: new tables e31_bo3_ideas / e31_bo3_scores only (INSERT-only);
evidence rows go to e13_evidence with grp='e31'. Idempotent, resume-safe.

Usage:
  /usr/bin/python3 experiments/e31_matched_compute.py --smoke
  /usr/bin/python3 experiments/e31_matched_compute.py --gen [--workers 6]
  /usr/bin/python3 experiments/e31_matched_compute.py --prep [--workers 3]
  /usr/bin/python3 experiments/e31_matched_compute.py --score [--workers 6]
  /usr/bin/python3 experiments/e31_matched_compute.py --analyze
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
from experiments.e27_swm import pick_subs, _short, AGENT_ITERS

WSUM = sum(W.values())
BACKBONES = ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b",
             "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"]
N_PER_DOMAIN = 9          # 45 subdomain slots, same grid as E27
RUNS = (2, 3)             # run 1 = existing swm_ideas active_base
RUN_SEED = {2: 43, 3: 44}
OUT_JSON = ROOT / "reports" / "e31_matched_compute.json"


def _iid(short, run, sub):
    return f"e31|{short}|r{run}|{sub[:40]}"


def ensure_tbls(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e31_bo3_ideas (
        gen_model TEXT, subdomain TEXT, domain TEXT, run_idx INTEGER,
        idea_text TEXT, aux_json TEXT, created_at TEXT,
        PRIMARY KEY (gen_model, subdomain, run_idx));
    CREATE TABLE IF NOT EXISTS e31_bo3_scores (
        item_id TEXT, gen_model TEXT, subdomain TEXT, domain TEXT, run_idx INTEGER,
        critic_model TEXT, scores_json TEXT, error TEXT, created_at TEXT,
        PRIMARY KEY (gen_model, subdomain, run_idx, critic_model));""")
    conn.commit()


# ---------------------------------------------------------------- generation
def gen(models, n_per_domain, runs, workers=6, subs=None, limit=0):
    from generation.active_agent import run_active_agent
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tbls(conn)
    subs = subs or pick_subs(conn, n_per_domain)
    tasks = []
    for m in models:
        for dom, sub in subs:
            for run in runs:
                if conn.execute("SELECT 1 FROM e31_bo3_ideas WHERE gen_model=? AND subdomain=? "
                                "AND run_idx=? AND TRIM(idea_text)!=''", (m, sub, run)).fetchone():
                    continue
                tasks.append((m, dom, sub, run))
    total_remaining = len(tasks)
    if limit:
        tasks = tasks[:limit]
    print(f"gen tasks remaining: {total_remaining}; running {len(tasks)} this batch "
          f"(models={len(models)} runs={runs})", flush=True)
    ts = datetime.now(timezone.utc).isoformat()

    def _one(t):
        m, dom, sub, run = t
        try:
            r = run_active_agent(dom, m, max_iters=AGENT_ITERS, seed=RUN_SEED[run])
            aux = {"n_tool_calls": r.get("n_tool_calls"), "iters_used": r.get("iters_used"),
                   "error": r.get("error")}
            return m, dom, sub, run, r.get("hypothesis", ""), aux, None
        except Exception as e:
            return m, dom, sub, run, "", None, str(e)[:200]

    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_one, t) for t in tasks]), 1):
            m, dom, sub, run, idea, aux, e = f.result()
            conn.execute("INSERT OR REPLACE INTO e31_bo3_ideas VALUES (?,?,?,?,?,?,?)",
                         (m, sub, dom, run, idea, json.dumps(aux) if aux else None, ts))
            conn.commit()
            ok += 1 if idea else 0; err += 0 if idea else 1
            print(f"  [{i}/{len(tasks)}] {_short(m)}|r{run}|{sub[:24]} -> "
                  f"{'OK '+str(len(idea.split()))+'w' if idea else 'EMPTY '+str(e)}", flush=True)
    conn.close(); print(f"gen done ok={ok} err={err}")


# ---------------------------------------------------------------- evidence
def _items(conn):
    return conn.execute("SELECT gen_model, subdomain, domain, run_idx, idea_text "
                        "FROM e31_bo3_ideas WHERE TRIM(idea_text)!=''").fetchall()


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
        m, sub, dom, run, txt = it
        iid = _iid(_short(m), run, sub)
        q, ev, ns = build_evidence_for_item(iid, "e31", txt, V4_IDEA_CUTOFF, None, None)
        return iid, q, ev, ns

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_ev, it) for it in todo]):
            iid, q, ev, ns = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                         (iid, "e31", V4_IDEA_CUTOFF, json.dumps(q), json.dumps(ev), len(ev), ns, ts))
            conn.commit(); done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}]", flush=True)
    conn.close(); print("prep done")


# ---------------------------------------------------------------- scoring
def score(workers=6, limit=0):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_tbls(conn)
    critics = list(cfg.CRITIC_MODELS)[:3]
    items = _items(conn)
    refs = {}
    for m, sub, dom, run, txt in items:
        iid = _iid(_short(m), run, sub)
        r = conn.execute("SELECT evidence_json, cutoff_date FROM e13_evidence WHERE item_id=?",
                         (iid,)).fetchone()
        if r:
            refs[iid] = format_evidence_block(json.loads(r[0] or "[]"), r[1])
    tasks = [(m, sub, dom, run, txt, cr)
             for m, sub, dom, run, txt in items if _iid(_short(m), run, sub) in refs
             for cr in critics
             if not conn.execute("SELECT 1 FROM e31_bo3_scores WHERE gen_model=? AND subdomain=? "
                                 "AND run_idx=? AND critic_model=? AND error IS NULL "
                                 "AND scores_json IS NOT NULL", (m, sub, run, cr)).fetchone()]
    total_remaining = len(tasks)
    if limit:
        tasks = tasks[:limit]
    print(f"score tasks remaining: {total_remaining}; running {len(tasks)} this batch"
          f"{' (limit=%d)' % limit if limit else ''}", flush=True)

    def _sc(t):
        m, sub, dom, run, txt, cr = t
        iid = _iid(_short(m), run, sub)
        s, _ = score_one(txt, cr, dom, LIT8D_SYSTEM, refs[iid])
        return iid, m, sub, dom, run, cr, s

    ts = datetime.now(timezone.utc).isoformat(); ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_sc, t) for t in tasks]), 1):
            iid, m, sub, dom, run, cr, s = f.result()
            conn.execute("INSERT OR REPLACE INTO e31_bo3_scores VALUES (?,?,?,?,?,?,?,?,?)",
                         (iid, m, sub, dom, run, cr, json.dumps(s) if s else None,
                          None if s else "parse_fail", ts))
            conn.commit(); ok += 1 if s else 0; err += 0 if s else 1
            if i % 20 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close(); print(f"score done ok={ok} err={err}")


# ---------------------------------------------------------------- analysis
def _tm(vs):
    return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)


def _wcells(rows):
    """rows of (model, sub, key, scores_json) -> {(model, sub, key): weighted}."""
    cell = defaultdict(lambda: defaultdict(list))
    for m, sub, key, sj in rows:
        s = json.loads(sj)
        for d in DIMS:
            cell[(m, sub, key)][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    out = {}
    for k, dv in cell.items():
        dims = {d: _tm(dv[d]) for d in DIMS}
        if all(dims[d] is not None for d in DIMS):
            out[k] = sum(dims[d] * W[d] for d in DIMS) / WSUM
    return out


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
    extra = _wcells([(m, sub, run, sj) for m, sub, run, sj in conn.execute(
        "SELECT gen_model, subdomain, run_idx, scores_json FROM e31_bo3_scores "
        "WHERE scores_json IS NOT NULL")])
    swm = _wcells([(m, sub, c, sj) for m, sub, c, sj in conn.execute(
        "SELECT gen_model, subdomain, condition, scores_json FROM swm_scores "
        "WHERE scores_json IS NOT NULL AND condition IN ('active_base','swm_S4b')")])
    conn.close()

    runs_of = defaultdict(dict)
    for (m, sub, run), w in extra.items():
        runs_of[(m, sub)][run] = w
    base1, s4b = {}, {}
    for (m, sub, c), w in swm.items():
        (base1 if c == "active_base" else s4b)[(m, sub)] = w

    bo3, complete = {}, 0
    for k, rd in runs_of.items():
        if k in base1 and 2 in rd and 3 in rd:
            bo3[k] = max(base1[k], rd[2], rd[3]); complete += 1
    rng = np.random.default_rng(20260723)

    def block(pairs_fn, keys):
        pooled = [pairs_fn(k) for k in keys]
        per_bb = defaultdict(list)
        for k, d in zip(keys, pooled):
            per_bb[k[0]].append(d)
        return {"pooled": _paired(pooled, rng),
                "per_backbone": {m: _paired(v, rng) for m, v in sorted(per_bb.items())}}

    keys_bo3 = sorted(bo3)
    keys_both = sorted(k for k in bo3 if k in s4b)
    out = {
        "note": "best-of-3 active_base (3x~7=21 LLM calls) = compute-matched "
                "baseline for S4b (~21 calls). Selection by the same lit8d "
                "weighted score (favorable to the baseline). Run 1 = existing "
                "swm_scores active_base cell.",
        "n_cells_complete_bo3": complete,
        "base_single_mean": round(float(np.mean([base1[k] for k in keys_bo3])), 4)
        if keys_bo3 else None,
        "bo3_mean": round(float(np.mean([bo3[k] for k in keys_bo3])), 4) if keys_bo3 else None,
        "s4b_mean_on_shared": round(float(np.mean([s4b[k] for k in keys_both])), 4)
        if keys_both else None,
        "bo3_minus_base_single": block(lambda k: bo3[k] - base1[k], keys_bo3),
        "s4b_minus_bo3": block(lambda k: s4b[k] - bo3[k], keys_both),
        "s4b_minus_base_single_on_shared": block(lambda k: s4b[k] - base1[k], keys_both),
        "calls_accounting": {"active_base_per_cell": "~7 (5.8 tool turns + final)",
                             "s4b_per_cell": "~21 (8.4 outer + ~13 internal panel)",
                             "bo3_per_cell": "~21 (3 independent base runs)"},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("n_cells_complete_bo3", "base_single_mean",
                                          "bo3_mean", "s4b_mean_on_shared")}, indent=2))
    print(json.dumps({"bo3_minus_base_single": out["bo3_minus_base_single"]["pooled"],
                      "s4b_minus_bo3": out["s4b_minus_bo3"]["pooled"]}, indent=2))
    print("wrote", OUT_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="1 backbone x 2 subdomains x run 2: gen+prep+score+report")
    ap.add_argument("--models", default=",".join(BACKBONES))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N items this run (0=all); for small-batch resumable runs")
    a = ap.parse_args()
    models = [x.strip() for x in a.models.split(",") if x.strip()]
    if a.smoke:
        conn = sqlite3.connect(str(cfg.RESULTS_DB)); ensure_tbls(conn)
        subs = pick_subs(conn, 1)[:2]; conn.close()
        print(f"SMOKE: qwen/qwen3.5-9b x {[s for _, s in subs]} x run2")
        gen(["qwen/qwen3.5-9b"], 1, (2,), workers=2, subs=subs)
        prep(workers=2)
        score(workers=3)
        conn = sqlite3.connect(str(cfg.RESULTS_DB))
        for r in conn.execute("SELECT gen_model, subdomain, run_idx, critic_model, "
                              "scores_json IS NOT NULL FROM e31_bo3_scores"):
            print("  scored:", r)
        conn.close()
        return
    if a.gen:
        gen(models, N_PER_DOMAIN, RUNS, workers=a.workers, limit=a.limit)
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
