"""E27 — closed-loop Scientific World Model (SWM) vs plain Active agent.

Conditions (all Active mode = agent searches Semantic Scholar itself):
  active_base : plain Active agent (SEARCH/FETCH/FINAL)                  [baseline]
  swm_S1      : Active + single-inference SWM (SIMULATE calls swm.py S1)
  swm_S2      : Active + multi-role SWM                                  [headline]
  swm_S3      : Active + tool-grounded SWM

Isolation: baseline and SWM conditions run the SAME agent/protocol with the SAME
SEARCH/FETCH budget on the SAME subdomains; the only difference is the SIMULATE command
(the SWM feedback loop). Any score delta is attributable to the SWM.

Scoring reuses the exact lit8d pipeline (live SS prior-art + LIT8D_SYSTEM + 3 critics +
weighted trimmed mean), so scores are comparable to the benchmark and to E26.

Backbones: qwen3.5-9b (small) + qwen3.5-397b-a17b (large) — the capability-gating pair.
Raw-data rule: own tables `swm_ideas` / `swm_scores`; evidence in `e13_evidence` grp='swm'.
Touches NO production tables. All stages idempotent. /usr/bin/python3.

Usage:
  /usr/bin/python3 experiments/e27_swm.py --gen [--models a,b] [--conditions active_base,swm_S2] [--n-sub 1]
  /usr/bin/python3 experiments/e27_swm.py --prep [--workers 3]
  /usr/bin/python3 experiments/e27_swm.py --score [--critic-idx 0|1|2]
  /usr/bin/python3 experiments/e27_swm.py --analyze
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

WSUM = sum(W.values())
DOMAINS = ["Biology", "CS", "Chemistry", "Medicine", "Physics"]
DEFAULT_MODELS = ["qwen/qwen3.5-9b", "qwen/qwen3.5-397b-a17b"]
DEFAULT_CONDS = ["active_base", "swm_S1", "swm_S2"]
AGENT_ITERS = 6
SWM_SIMS = 2


def _short(m):
    return m.split("/")[-1]


def _iid(short, cond, sub):
    return f"swm|{short}|{cond}|{sub[:40]}"


def pick_subs(conn, n_per_domain):
    out = []
    for dom in DOMAINS:
        rows = conn.execute("SELECT subdomain FROM subdomain_refs WHERE domain=? ORDER BY subdomain",
                            (dom,)).fetchall()
        for i, (sub,) in enumerate(rows):
            if i >= n_per_domain:
                break
            out.append((dom, sub))
    return out


def ensure_tbls(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS swm_ideas (
        gen_model TEXT, subdomain TEXT, domain TEXT, condition TEXT,
        idea_text TEXT, aux_json TEXT, created_at TEXT,
        PRIMARY KEY (gen_model, subdomain, condition));
    CREATE TABLE IF NOT EXISTS swm_scores (
        item_id TEXT, gen_model TEXT, subdomain TEXT, domain TEXT, condition TEXT,
        critic_model TEXT, scores_json TEXT, error TEXT, created_at TEXT,
        PRIMARY KEY (gen_model, subdomain, condition, critic_model));""")
    conn.commit()


# ---------------------------------------------------------------- generation
def _generate(model, domain, cond, seed):
    if cond == "active_base":
        from generation.active_agent import run_active_agent
        r = run_active_agent(domain, model, max_iters=AGENT_ITERS, seed=seed)
        aux = {"n_tool_calls": r.get("n_tool_calls"), "error": r.get("error")}
        return r.get("hypothesis", ""), aux
    design = cond.split("_", 1)[1]  # swm_S1 -> S1
    from generation.active_swm_agent import run_active_swm_agent
    r = run_active_swm_agent(domain, model, swm_design=design, max_iters=AGENT_ITERS,
                             max_sims=SWM_SIMS, seed=seed)
    verdicts = [s["feedback"]["mechanism_consistency"].get("verdict") for s in r.get("swm_trace", [])]
    aux = {"n_tool_calls": r.get("n_tool_calls"), "n_sims": r.get("n_sims"),
           "swm_verdicts": verdicts, "error": r.get("error")}
    return r.get("hypothesis", ""), aux


def gen(models, conds, n_per_domain, workers=3, seed=42):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    ensure_tbls(conn)
    subs = pick_subs(conn, n_per_domain)
    print(f"{len(subs)} subdomains x {len(models)} models x {len(conds)} conditions {conds}", flush=True)
    tasks = []
    for m in models:
        for dom, sub in subs:
            for c in conds:
                # skip only if a NON-EMPTY idea already exists; retry empty (failed) cells
                if conn.execute("SELECT 1 FROM swm_ideas WHERE gen_model=? AND subdomain=? "
                                "AND condition=? AND TRIM(idea_text)!=''", (m, sub, c)).fetchone():
                    continue
                tasks.append((m, dom, sub, c))
    print(f"gen tasks: {len(tasks)}", flush=True)
    ts = datetime.now(timezone.utc).isoformat()

    def _one(t):
        m, dom, sub, c = t
        try:
            idea, aux = _generate(m, dom, c, seed)
            return m, dom, sub, c, idea, aux, None
        except Exception as e:
            return m, dom, sub, c, "", None, str(e)[:200]

    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_one, t) for t in tasks]), 1):
            m, dom, sub, c, idea, aux, e = f.result()
            # OR REPLACE so a retry can overwrite an earlier EMPTY (failed) cell; never
            # overwrites a non-empty idea because those are filtered out of `tasks` above.
            conn.execute("INSERT OR REPLACE INTO swm_ideas VALUES (?,?,?,?,?,?,?)",
                         (m, sub, dom, c, idea, json.dumps(aux) if aux else None, ts))
            conn.commit()
            ok += 1 if idea else 0; err += 0 if idea else 1
            print(f"  [{i}/{len(tasks)}] {_short(m)}|{c}|{sub[:24]} -> "
                  f"{'OK '+str(len(idea.split()))+'w' if idea else 'EMPTY '+str(e)}", flush=True)
    conn.close(); print(f"gen done ok={ok} err={err}")


# ---------------------------------------------------------------- evidence
def _items(conn):
    return conn.execute("SELECT gen_model, subdomain, domain, condition, idea_text "
                        "FROM swm_ideas WHERE TRIM(idea_text)!=''").fetchall()


def prep(workers=3):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    ensure_tables(conn); ensure_tbls(conn)
    items = _items(conn)
    todo = [it for it in items
            if not conn.execute("SELECT 1 FROM e13_evidence WHERE item_id=?",
                                (_iid(_short(it[0]), it[3], it[1]),)).fetchone()]
    print(f"{len(items)} ideas; {len(todo)} need evidence", flush=True)
    ts = datetime.now(timezone.utc).isoformat()

    def _ev(it):
        m, sub, dom, c, txt = it
        iid = _iid(_short(m), c, sub)
        q, ev, ns = build_evidence_for_item(iid, "swm", txt, V4_IDEA_CUTOFF, None, None)
        return iid, q, ev, ns

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_ev, it) for it in todo]):
            iid, q, ev, ns = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                         (iid, "swm", V4_IDEA_CUTOFF, json.dumps(q), json.dumps(ev), len(ev), ns, ts))
            conn.commit(); done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}]", flush=True)
    conn.close(); print("prep done")


# ---------------------------------------------------------------- scoring
def score(n_critics=3, workers=6, only_critic=None):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    ensure_tbls(conn)
    critics = list(cfg.CRITIC_MODELS)[:n_critics]
    if only_critic is not None:
        critics = [critics[only_critic]]
    items = _items(conn)
    refs, miss = {}, 0
    for m, sub, dom, c, txt in items:
        iid = _iid(_short(m), c, sub)
        r = conn.execute("SELECT evidence_json, cutoff_date FROM e13_evidence WHERE item_id=?",
                         (iid,)).fetchone()
        if not r:
            miss += 1; continue
        refs[iid] = format_evidence_block(json.loads(r[0] or "[]"), r[1])
    if miss:
        print(f"WARN: {miss} ideas lack evidence (run --prep); scoring the rest")
    tasks = [(m, sub, dom, c, txt, cr)
             for m, sub, dom, c, txt in items if _iid(_short(m), c, sub) in refs
             for cr in critics
             if not conn.execute("SELECT 1 FROM swm_scores WHERE gen_model=? AND subdomain=? "
                                 "AND condition=? AND critic_model=? AND error IS NULL "
                                 "AND scores_json IS NOT NULL", (m, sub, c, cr)).fetchone()]
    print(f"score tasks: {len(tasks)}", flush=True)

    def _sc(t):
        m, sub, dom, c, txt, cr = t
        iid = _iid(_short(m), c, sub)
        s, _ = score_one(txt, cr, dom, LIT8D_SYSTEM, refs[iid])
        return iid, m, sub, dom, c, cr, s

    ts = datetime.now(timezone.utc).isoformat(); ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_sc, t) for t in tasks]), 1):
            iid, m, sub, dom, c, cr, s = f.result()
            conn.execute("INSERT OR REPLACE INTO swm_scores VALUES (?,?,?,?,?,?,?,?,?)",
                         (iid, m, sub, dom, c, cr, json.dumps(s) if s else None,
                          None if s else "parse_fail", ts))
            conn.commit(); ok += 1 if s else 0; err += 0 if s else 1
            if i % 20 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close(); print(f"score done ok={ok} err={err}")


# ---------------------------------------------------------------- analysis
def _weighted(d):
    return sum(d[k] * W[k] for k in DIMS) / WSUM


def analyze():
    from scipy import stats
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT gen_model,subdomain,domain,condition,scores_json "
                          "FROM swm_scores WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"]); k = (r["gen_model"], r["subdomain"], r["condition"])
        for d in DIMS:
            cell[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()

    def tm(vs): return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)
    dim_cell = {k: {d: tm(cell[k][d]) for d in DIMS} for k in cell}
    w_cell = {k: _weighted(v) for k, v in dim_cell.items()}
    by_ms = defaultdict(dict); dby_ms = defaultdict(dict)
    for (m, sub, c), w in w_cell.items():
        by_ms[(m, sub)][c] = w; dby_ms[(m, sub)][c] = dim_cell[(m, sub, c)]

    conds = sorted({c for (_, _, c) in w_cell})
    out = {"conditions": conds,
           "base_mean_overall": round(float(np.mean(
               [by_ms[k]["active_base"] for k in by_ms if "active_base" in by_ms[k]])), 4)
           if any("active_base" in by_ms[k] for k in by_ms) else None}

    def compare(cond):
        both = [(k, by_ms[k]["active_base"], by_ms[k][cond]) for k in by_ms
                if "active_base" in by_ms[k] and cond in by_ms[k]]
        if not both:
            return {"n_pairs": 0}
        base = np.array([b for _, b, _ in both]); alt = np.array([a for _, _, a in both])
        delta = alt - base
        r = {"n_pairs": len(both), "base_mean": round(float(base.mean()), 4),
             f"{cond}_mean": round(float(alt.mean()), 4),
             "delta_mean": round(float(delta.mean()), 4),
             "delta_win_rate": round(float((delta > 0).mean()), 4)}
        if len(both) >= 6:
            _, p = stats.wilcoxon(alt, base)
            r["wilcoxon_p"] = round(float(p), 4)
            lo, hi = np.percentile([delta[np.random.default_rng(s).integers(0, len(delta), len(delta))].mean()
                                    for s in range(2000)], [2.5, 97.5])
            r["delta_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
        pd = {}
        for d in DIMS:
            ds = [dby_ms[k][cond][d] - dby_ms[k]["active_base"][d] for k in dby_ms
                  if "active_base" in dby_ms[k] and cond in dby_ms[k]
                  and dby_ms[k]["active_base"][d] is not None and dby_ms[k][cond][d] is not None]
            pd[d] = round(float(np.mean(ds)), 4) if ds else None
        r["per_dim_delta"] = pd
        pm = defaultdict(list)
        for (k, b, a) in both:
            pm[k[0]].append(a - b)
        r["per_model_delta"] = {m: round(float(np.mean(v)), 4) for m, v in sorted(pm.items())}
        # per-model detail (n / delta / win / p) for the multi-backbone grid table
        pmd = {}
        for m, v in sorted(pm.items()):
            v = np.array(v)
            e = {"n": int(len(v)), "delta_mean": round(float(v.mean()), 4),
                 "win_rate": round(float((v > 0).mean()), 4)}
            if len(v) >= 6:
                try:
                    e["wilcoxon_p"] = round(float(stats.wilcoxon(v).pvalue), 4)
                except Exception:
                    pass
            pmd[m] = e
        r["per_model_detail"] = pmd
        return r

    for c in conds:
        if c != "active_base":
            out[c] = compare(c)
    (ROOT / "reports" / "e27_swm.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDS))
    ap.add_argument("--n-sub", type=int, default=1, help="subdomains per domain")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--critic-idx", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    models = [x.strip() for x in a.models.split(",") if x.strip()]
    conds = [x.strip() for x in a.conditions.split(",") if x.strip()]
    if a.gen:
        gen(models, conds, a.n_sub, workers=a.workers, seed=a.seed)
    if a.prep:
        prep(workers=a.workers)
    if a.score:
        score(workers=a.workers, only_critic=a.critic_idx)
    if a.analyze:
        analyze()
    if not any([a.gen, a.prep, a.score, a.analyze]):
        ap.print_help()


if __name__ == "__main__":
    main()
