"""E7 — Critic rigor: do real high-impact paper hypotheses score above model gen?

Goal (plan E7): validate the modern5 critic pool has discrimination + isn't
saturated. Feed REAL published hypotheses from high-citation papers (positive
control) through the production critic prompt and compare where they land vs
(a) model-generated Static ideas and (b) the copy baseline (weak control).

If critics are rigorous: real high-impact hypotheses should land clearly above
average model generations, and ideally break the ~6.x "saturation band" — which
also answers E3 (is 6.x a model ceiling or a scoring ceiling?).

Positive control = top-citation filtered papers' gt_hypothesis (real, peer
work). Picked by citation_count, balanced across domains. We already store
gt_hypothesis per paper; no new sourcing needed.

SAFETY: writes ONLY to new table `e7_bestpaper_scores`. results/papers untouched.
Critics are all open-weight -> default OPENROUTER_API_KEY. Idempotent (PK skip).

Usage:
  python experiments/e7_critic_rigor_bestpaper.py --score          # run critics
  python experiments/e7_critic_rigor_bestpaper.py --analyze        # aggregate + compare
  python experiments/e7_critic_rigor_bestpaper.py --score --analyze
  python experiments/e7_critic_rigor_bestpaper.py --top-per-domain 8
"""
import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from evaluation.absolute_scorer import score_idea
from evaluation.critic_manager import _format_refs_for_judge
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as WEIGHTS

MODERN_CRITICS = list(cfg.CRITIC_MODELS)
WSUM = sum(WEIGHTS.values())


def weighted_norm(s):
    return sum(s.get(d, 0) * WEIGHTS[d] for d in DIMS if d in s) / WSUM


def trimmed_mean(vs):
    if not vs:
        return None
    if len(vs) < 2:
        return vs[0]
    return statistics.mean(sorted(vs)[:-1])


def ensure_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e7_bestpaper_scores (
        paper_id     TEXT NOT NULL,
        domain       TEXT,
        citation_count INTEGER,
        critic_model TEXT NOT NULL,
        scores_json  TEXT,
        reasoning_json TEXT,
        raw_response TEXT,
        error        TEXT,
        created_at   TEXT NOT NULL,
        PRIMARY KEY (paper_id, critic_model)
    );
    """)
    conn.commit()


def pick_best_papers(top_per_domain):
    cp = sqlite3.connect(str(cfg.PAPERS_DB)); cp.row_factory = sqlite3.Row
    rows = [dict(r) for r in cp.execute(
        "SELECT * FROM papers WHERE status='filtered' "
        "AND gt_hypothesis IS NOT NULL AND gt_hypothesis!=''")]
    cp.close()
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    picked = []
    for dom, ps in by_dom.items():
        ps.sort(key=lambda r: (r.get("citation_count") or 0), reverse=True)
        picked.extend(ps[:top_per_domain])
    return picked


def do_score(top_per_domain, workers):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    ensure_table(conn)
    papers = pick_best_papers(top_per_domain)
    print(f"best papers (top {top_per_domain}/domain): {len(papers)}")

    tasks = []
    for p in papers:
        for cr in MODERN_CRITICS:
            exists = conn.execute(
                "SELECT 1 FROM e7_bestpaper_scores WHERE paper_id=? AND critic_model=?",
                (p["paper_id"], cr)).fetchone()
            if not exists:
                tasks.append((p, cr))
    print(f"critic tasks: {len(tasks)}")
    if not tasks:
        return

    def _w(t):
        p, cr = t
        refs = _format_refs_for_judge(p)
        try:
            scores, raw, _ = score_idea(p["gt_hypothesis"], cr, domain=p.get("domain", ""),
                                        references=refs, judge_mode="static",
                                        db_papers=str(cfg.PAPERS_DB))
            return (p, cr, scores, raw, None)
        except Exception as e:
            return (p, cr, None, None, str(e)[:300])

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            p, cr, scores, raw, e = f.result()
            if scores:
                conn.execute("""INSERT OR IGNORE INTO e7_bestpaper_scores
                    (paper_id, domain, citation_count, critic_model, scores_json,
                     reasoning_json, raw_response, error, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (p["paper_id"], p.get("domain"), p.get("citation_count"), cr,
                     json.dumps({d: v["score"] for d, v in scores.items()}),
                     json.dumps({d: v["reasoning"] for d, v in scores.items()}),
                     (raw or "")[:3000], None, ts))
                conn.commit(); ok += 1
            else:
                err += 1
            if i % 20 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}")
    conn.close()
    print(f"done score: ok={ok} err={err}")


def do_analyze():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    # best-paper gt scores per paper (trimmed mean over critics)
    cell = defaultdict(list)
    meta = {}
    for r in conn.execute("SELECT * FROM e7_bestpaper_scores WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"])
        cell[r["paper_id"]].append(weighted_norm(s))
        meta[r["paper_id"]] = (r["domain"], r["citation_count"])
    best_scores = {p: trimmed_mean(v) for p, v in cell.items() if trimmed_mean(v) is not None}

    # model-gen Static distribution from uniform pool (per model mean)
    ucell = defaultdict(list)
    for r in conn.execute("SELECT idea_model, track, paper_id, idea_index, scores_json "
                          "FROM uniform_critic_scores WHERE track='B' AND scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"])
        ucell[(r["idea_model"], r["paper_id"], r["idea_index"])].append(weighted_norm(s))
    # per (model,paper) best idx
    mp = {}
    for (m, p, i), v in ucell.items():
        tm = trimmed_mean(v)
        if tm is None:
            continue
        if (m, p) not in mp or tm > mp[(m, p)]:
            mp[(m, p)] = tm
    model_means = defaultdict(list)
    for (m, p), sc in mp.items():
        if not m.startswith("baseline/"):
            model_means[m].append(sc)
    model_gen = {m: float(np.mean(v)) for m, v in model_means.items()}

    # copy baseline (weak control) — uniform pool baseline/copy
    copy_vals = [trimmed_mean(v) for (m, p, i), v in ucell.items() if m == "baseline/copy"]
    copy_vals = [x for x in copy_vals if x is not None]

    bp = list(best_scores.values())
    conn.close()

    def summ(xs):
        if not xs:
            return {"n": 0}
        a = np.array(xs)
        return {"n": len(xs), "mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
                "std": round(float(a.std(ddof=1)), 3) if len(xs) > 1 else 0.0,
                "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
                "p90": round(float(np.percentile(a, 90)), 3)}

    mg = list(model_gen.values())
    result = {
        "best_paper_gt": summ(bp),
        "model_generation_static_per_model": summ(mg),
        "copy_baseline": summ(copy_vals),
        "discrimination": {
            "best_minus_model_gen_mean": (round(np.mean(bp) - np.mean(mg), 3)
                                          if bp and mg else None),
            "best_above_model_max": (round(max(bp) - max(mg), 3) if bp and mg else None),
            "best_breaks_6.5_band_frac": (round(sum(1 for x in bp if x > 6.5) / len(bp), 3)
                                          if bp else None),
            "model_gen_breaks_6.5_band_frac": (round(sum(1 for x in mg if x > 6.5) / len(mg), 3)
                                               if mg else None),
        },
        "top_models_by_static": sorted(model_gen.items(), key=lambda x: -x[1])[:5],
    }
    out = ROOT / "reports" / "e7_bestpaper"; out.mkdir(parents=True, exist_ok=True)
    (out / "e7_critic_rigor.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n✓ wrote {out}/e7_critic_rigor.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--top-per-domain", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not (args.score or args.analyze):
        args.score = args.analyze = True
    if args.score:
        do_score(args.top_per_domain, args.workers)
    if args.analyze:
        do_analyze()


if __name__ == "__main__":
    main()
