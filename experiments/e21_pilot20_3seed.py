"""E21 — rescore the 20% pilot with 3 ideas per seed (idx 1/2/3).

Same fixed sample of 20 subdomains as E19 (4 per domain, evenly spaced), same
lit8d critic, but every (model, track, subdomain) seed cell now contributes ALL
its ideas (idea_index 1,2,3 from E20 replication) instead of just one. Each idea
gets its own Semantic Scholar prior-art evidence and 3-critic score; the seed
score is the mean of its ideas' weighted scores → lower per-seed sampling noise.

Raw-data rule: does NOT touch E19's `lit8d_scores`. New table `lit8d_scores_3seed`
carries an extra idea_index column. idx=1 evidence reuses E19's cached item_ids
(no idx suffix); idx 2,3 build fresh evidence under `{short}|{tr}|{sub}|{idx}`.

Usage (staged, each idempotent):
  /usr/bin/python3 experiments/e21_pilot20_3seed.py --prep    # SS evidence (slow)
  /usr/bin/python3 experiments/e21_pilot20_3seed.py --score
  /usr/bin/python3 experiments/e21_pilot20_3seed.py --analyze
  /usr/bin/python3 experiments/e21_pilot20_3seed.py --audit
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
from experiments.e19_pilot20_lit8d import pick_subdomains

WSUM = sum(W.values())
DOMAINS = ["Biology", "CS", "Chemistry", "Medicine", "Physics"]
REPS = (1, 2, 3)


def _iid(short, tr, sub, idx):
    """idx=1 reuses E19's evidence item_id (no idx suffix); idx>1 gets a suffix."""
    base = f"{short}|{tr}|{sub[:40]}"
    return base if idx == 1 else f"{base}|{idx}"


def sample_items(conn):
    """(item_id, model, track, subdomain, domain, idx, idea_text) for the 20
    subdomains, ALL models/tracks, ALL idea_index in {1,2,3} that have text."""
    picks = pick_subdomains(conn)
    models = [r[0] for r in conn.execute(
        "SELECT idea_model FROM subdomain_ideas GROUP BY idea_model HAVING COUNT(*)>50")]
    items, seen = [], set()
    for dom, sub in picks:
        for m in models:
            short = m.split("/")[-1]
            for tr in ("B", "C"):
                for idx in REPS:
                    r = conn.execute(
                        "SELECT idea_text, domain FROM subdomain_ideas WHERE idea_model=? "
                        "AND track=? AND subdomain=? AND idea_index=? AND TRIM(idea_text)!='' LIMIT 1",
                        (m, tr, sub, idx)).fetchone()
                    if not r:
                        continue
                    iid = _iid(short, tr, sub, idx)
                    if iid in seen:
                        continue
                    seen.add(iid)
                    items.append((iid, m, tr, sub, r[1] or dom, idx, r[0]))
    return items


def ensure_score_tbl(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS lit8d_scores_3seed (
        item_id TEXT, idea_model TEXT, track TEXT, subdomain TEXT, domain TEXT,
        idea_index INTEGER, critic_model TEXT, scores_json TEXT, error TEXT, created_at TEXT,
        PRIMARY KEY (idea_model, track, subdomain, idea_index, critic_model));""")
    conn.commit()


def prep(workers=4):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60); conn.row_factory = sqlite3.Row
    ensure_tables(conn)
    items = sample_items(conn)
    todo = [it for it in items
            if not conn.execute("SELECT 1 FROM e13_evidence WHERE item_id=?", (it[0],)).fetchone()]
    print(f"{len(items)} sampled ideas; {len(todo)} need evidence", flush=True)
    ts = datetime.now(timezone.utc).isoformat()

    def _ev(it):
        iid, m, tr, sub, dom, idx, txt = it
        q, ev, ns = build_evidence_for_item(iid, "pilot20_3seed", txt, V4_IDEA_CUTOFF, None, None)
        return iid, q, ev, ns

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_ev, it) for it in todo]):
            iid, q, ev, ns = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                (iid, "pilot20_3seed", V4_IDEA_CUTOFF, json.dumps(q), json.dumps(ev), len(ev), ns, ts))
            conn.commit(); done += 1
            if done % 25 == 0 or done == len(todo): print(f"  [{done}/{len(todo)}]", flush=True)
    conn.close(); print("prep done")


def score(n_critics=3, workers=6, only_critic=None):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60); conn.row_factory = sqlite3.Row
    ensure_score_tbl(conn)
    critics = list(cfg.CRITIC_MODELS)[:n_critics]
    if only_critic is not None:  # shard by a single critic to avoid redundant parallel work
        critics = [critics[only_critic]]
    items = sample_items(conn)
    refs, miss = {}, 0
    for it in items:
        iid = it[0]
        r = conn.execute("SELECT evidence_json, cutoff_date FROM e13_evidence WHERE item_id=?", (iid,)).fetchone()
        if not r: miss += 1; continue
        refs[iid] = format_evidence_block(json.loads(r[0] or "[]"), r[1])
    if miss:
        print(f"WARN: {miss} items lack evidence (run --prep); scoring the rest")
    tasks = [(iid, m, tr, sub, dom, idx, txt, cr)
             for iid, m, tr, sub, dom, idx, txt in items if iid in refs
             for cr in critics
             if not conn.execute(
                 "SELECT 1 FROM lit8d_scores_3seed WHERE idea_model=? AND track=? AND subdomain=? "
                 "AND idea_index=? AND critic_model=?", (m, tr, sub, idx, cr)).fetchone()]
    print(f"score tasks: {len(tasks)}", flush=True)

    def _sc(t):
        iid, m, tr, sub, dom, idx, txt, cr = t
        s, _ = score_one(txt, cr, dom, LIT8D_SYSTEM, refs[iid])
        return iid, m, tr, sub, dom, idx, cr, s

    ts = datetime.now(timezone.utc).isoformat(); ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_sc, t) for t in tasks]), 1):
            iid, m, tr, sub, dom, idx, cr, s = f.result()
            conn.execute("INSERT OR IGNORE INTO lit8d_scores_3seed VALUES (?,?,?,?,?,?,?,?,?,?)",
                (iid, m, tr, sub, dom, idx, cr, json.dumps(s) if s else None,
                 None if s else "parse_fail", ts))
            conn.commit(); ok += 1 if s else 0; err += 0 if s else 1
            if i % 100 == 0 or i == len(tasks): print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close(); print(f"done ok={ok} err={err}")


def _load():
    """Return seed_w[(m,tr,sub)] = mean over ideas of weighted trimmed-critic score,
    plus n_ideas[(m,tr,sub)], and meta[(m,tr,sub)]=(m,tr,dom)."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    # per idea: dim -> list of critic scores
    idea = defaultdict(lambda: defaultdict(list)); imeta = {}
    for r in conn.execute("SELECT idea_model,track,subdomain,domain,idea_index,scores_json "
                          "FROM lit8d_scores_3seed WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"]); k = (r["idea_model"], r["track"], r["subdomain"], r["idea_index"])
        imeta[k] = (r["idea_model"], r["track"], r["domain"])
        for d in DIMS:
            idea[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()

    def tm(vs): return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)
    idea_w = {k: sum(tm(idea[k][d]) * W[d] for d in DIMS) / WSUM for k in idea}
    # aggregate ideas -> seed
    seed_vals = defaultdict(list); meta = {}
    for (m, tr, sub, idx), w in idea_w.items():
        seed = (m, tr, sub); seed_vals[seed].append(w)
        meta[seed] = imeta[(m, tr, sub, idx)][:2] + (imeta[(m, tr, sub, idx)][2],)
    seed_w = {k: float(np.mean(v)) for k, v in seed_vals.items()}
    n_ideas = {k: len(v) for k, v in seed_vals.items()}
    return seed_w, n_ideas, meta


def analyze():
    from scipy import stats
    seed_w, n_ideas, meta = _load()
    tot_ideas = sum(n_ideas.values())
    avg_ideas = np.mean(list(n_ideas.values()))
    print(f"seeds scored: {len(seed_w)} | ideas: {tot_ideas} | mean ideas/seed: {avg_ideas:.2f}\n")

    bym = defaultdict(lambda: {"B": [], "C": []})
    for k, w in seed_w.items():
        m, tr, dom = meta[k]; bym[m][tr].append(w)
    print(f"{'model':<40}{'nB':>4}{'static':>8}{'nC':>4}{'active':>8}{'boost':>8}")
    rows = []
    for m in sorted(bym):
        b = bym[m]["B"]; c = bym[m]["C"]
        sb = np.mean(b) if b else float('nan'); sc = np.mean(c) if c else float('nan')
        boost = (sc - sb) if (b and c) else float('nan')
        rows.append((m, len(b), sb, len(c), sc, boost))
    for m, nb, sb, nc, sc, boost in sorted(rows, key=lambda x: -(x[4] if not np.isnan(x[4]) else -9)):
        print(f"{m.split('/')[-1]:<40}{nb:>4}{sb:>8.2f}{nc:>4}{sc:>8.2f}{boost:>+8.2f}")

    allB = [w for k, w in seed_w.items() if meta[k][1] == "B"]
    allC = [w for k, w in seed_w.items() if meta[k][1] == "C"]
    print(f"\noverall Static n={len(allB)} mean={np.mean(allB):.2f} | "
          f"Active n={len(allC)} mean={np.mean(allC):.2f} | boost={np.mean(allC)-np.mean(allB):+.2f}")

    # matched-model boost stats
    paired = [(r[2], r[5]) for r in rows if not np.isnan(r[5])]
    boosts = [b for _, b in paired]
    statics = [s for s, _ in paired]
    npos = sum(1 for x in boosts if x > 0)
    print(f"per-model boost: {npos}/{len(boosts)} positive, mean={np.mean(boosts):+.2f}")
    if len(boosts) >= 3:
        sign_p = stats.binomtest(npos, len(boosts)).pvalue
        pear_r, pear_p = stats.pearsonr(statics, boosts)
        spear_r, spear_p = stats.spearmanr(statics, boosts)
        try:
            w_p = stats.wilcoxon(boosts).pvalue
        except Exception:
            w_p = float('nan')
        t_p = stats.ttest_1samp(boosts, 0).pvalue
        print(f"sign-test p={sign_p:.3f} | Pearson r(static,boost)={pear_r:+.2f} p={pear_p:.3f} | "
              f"Spearman r={spear_r:+.2f} p={spear_p:.3f} | Wilcoxon p={w_p:.3f} | t p={t_p:.3f}")
        order = sorted(paired, key=lambda x: -x[0])
        top = [b for _, b in order[:10]]; bot = [b for _, b in order[-10:]]
        print(f"top-10 strong models mean boost={np.mean(top):+.2f} | bottom-10 weak mean boost={np.mean(bot):+.2f}")

    # per-domain pooled boost
    print("\nper-domain pooled boost (matched seeds):")
    dompair = defaultdict(lambda: {"B": {}, "C": {}})
    for k, w in seed_w.items():
        m, tr, dom = meta[k]; dompair[dom][tr][(m, k[2])] = w
    for dom in DOMAINS:
        bs = dompair[dom]["B"]; cs = dompair[dom]["C"]
        # match on (model, subdomain)
        pairs = [(bs[key], cs[key]) for key in bs if key in cs]
        if pairs:
            d = [c - b for b, c in pairs]
            print(f"  {dom:<10} n={len(pairs):>3} static={np.mean([b for b,_ in pairs]):.2f} "
                  f"active={np.mean([c for _,c in pairs]):.2f} boost={np.mean(d):+.2f}")


def audit():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    ensure_score_tbl(conn)
    items = sample_items(conn)
    ev = conn.execute("SELECT COUNT(*) FROM e13_evidence WHERE grp='pilot20_3seed'").fetchone()[0]
    ev1 = sum(1 for it in items if it[5] == 1)  # idx1 reuse E19 evidence
    scored_ideas = conn.execute(
        "SELECT COUNT(DISTINCT idea_model||track||subdomain||idea_index) "
        "FROM lit8d_scores_3seed WHERE scores_json IS NOT NULL").fetchone()[0]
    by_idx = conn.execute(
        "SELECT idea_index, COUNT(DISTINCT idea_model||track||subdomain) FROM lit8d_scores_3seed "
        "WHERE scores_json IS NOT NULL GROUP BY idea_index ORDER BY idea_index").fetchall()
    print(f"sampled ideas: {len(items)} (idx1={ev1}) | evidence(pilot20_3seed, idx2/3): {ev} | "
          f"scored idea-units: {scored_ideas}")
    print("  scored seeds by idx: " + " ".join(f"idx{r[0]}={r[1]}" for r in by_idx))
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    for f in ("prep", "score", "analyze", "audit"): ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.prep: prep(min(a.workers, 4))
    if a.score: score(workers=a.workers)
    if a.analyze: analyze()
    if a.audit: audit()


if __name__ == "__main__":
    main()
