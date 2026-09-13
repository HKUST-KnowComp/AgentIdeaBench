"""v3 — score subdomain Static vs Active ideas (current production rubric).

Scores subdomain_ideas (track B = Static, track C = Active) for the 27-model open
roster with the modern5 critics under the current production SYSTEM_PROMPT (the
promoted fix_idea rubric), NO references — same apples-to-apples setting used for
the anchor/deepseek comparisons. Compares Static vs Active per model + overall.

SAFETY: new table subdomain_idea_scores. modern5 open -> default key. Idempotent.

Usage:
  /usr/bin/python3 experiments/v3_static_vs_active_score.py --score --workers 8
  /usr/bin/python3 experiments/v3_static_vs_active_score.py --analyze
  # optional smaller pass: --critics 3  (use first 3 critics)  --max-subdomains 40
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
from evaluation import absolute_scorer as asc
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as WEIGHTS
from experiments.e10_idea_anchor_calibration import score_one, weighted_norm, trimmed_mean
from experiments.v3_subdomain_ideation import roster


def ensure_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS subdomain_idea_scores (
        idea_model TEXT NOT NULL, subdomain TEXT NOT NULL, track TEXT NOT NULL,
        domain TEXT, critic_model TEXT NOT NULL, scores_json TEXT, error TEXT, created_at TEXT NOT NULL,
        PRIMARY KEY (idea_model, subdomain, track, critic_model)
    );""")
    conn.commit()


def do_score(workers, n_critics, max_sub):
    critics = list(cfg.CRITIC_MODELS)[:n_critics] if n_critics else list(cfg.CRITIC_MODELS)
    models = set(roster())
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_table(conn)
    rows = [r for r in conn.execute(
        "SELECT idea_model, subdomain, domain, track, idea_text FROM subdomain_ideas "
        "WHERE TRIM(idea_text)!=''") if r[0] in models]
    if max_sub:
        keep = sorted({r[1] for r in rows})[:max_sub]
        rows = [r for r in rows if r[1] in keep]
    tasks = []
    for m, sub, dom, tr, txt in rows:
        for cr in critics:
            if not conn.execute("SELECT 1 FROM subdomain_idea_scores WHERE idea_model=? AND subdomain=? "
                                "AND track=? AND critic_model=?", (m, sub, tr, cr)).fetchone():
                tasks.append((m, sub, dom, tr, txt, cr))
    print(f"ideas={len(rows)} critics={len(critics)} score_tasks={len(tasks)} workers={workers}")
    if not tasks:
        conn.close(); return

    def _w(t):
        m, sub, dom, tr, txt, cr = t
        scores, _ = score_one(txt, cr, dom, asc.SYSTEM_PROMPT, "")
        return (m, sub, dom, tr, cr, scores)

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            m, sub, dom, tr, cr, scores = f.result()
            conn.execute("INSERT OR IGNORE INTO subdomain_idea_scores VALUES (?,?,?,?,?,?,?,?)",
                         (m, sub, tr, dom, cr,
                          json.dumps({d: v["score"] for d, v in scores.items()}) if scores else None,
                          None if scores else "parse_fail", ts))
            conn.commit()
            ok += 1 if scores else 0; err += 0 if scores else 1
            if i % 100 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close()
    print(f"done score: ok={ok} err={err}")


def do_analyze():
    from scipy import stats as st
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(list)  # (model,sub,track) -> [weighted per critic]
    for r in conn.execute("SELECT * FROM subdomain_idea_scores WHERE scores_json IS NOT NULL"):
        cell[(r["idea_model"], r["subdomain"], r["track"])].append(weighted_norm(json.loads(r["scores_json"])))
    per_idea = {k: trimmed_mean(v) for k, v in cell.items() if trimmed_mean(v) is not None}
    conn.close()

    # per-model mean for B and C; paired boost
    bym = defaultdict(lambda: {"B": [], "C": []})
    for (m, sub, tr), w in per_idea.items():
        bym[m][tr].append(w)
    rows = []
    boosts = []
    for m in sorted(bym):
        B = bym[m]["B"]; C = bym[m]["C"]
        if B and C:
            mb, mc = float(np.mean(B)), float(np.mean(C))
            rows.append((m, len(B), round(mb, 3), len(C), round(mc, 3), round(mc - mb, 3)))
            boosts.append(mc - mb)
    allB = [w for (m, s, tr), w in per_idea.items() if tr == "B"]
    allC = [w for (m, s, tr), w in per_idea.items() if tr == "C"]
    summary = {
        "static_overall_mean": round(float(np.mean(allB)), 3) if allB else None,
        "active_overall_mean": round(float(np.mean(allC)), 3) if allC else None,
        "active_minus_static_mean_boost": round(float(np.mean(boosts)), 3) if boosts else None,
        "n_models_paired": len(boosts),
        "models_active_gt_static": sum(1 for b in boosts if b > 0),
        "wilcoxon_p": (round(float(st.wilcoxon(boosts).pvalue), 4) if len(boosts) > 5 else None),
        "n_static_ideas": len(allB), "n_active_ideas": len(allC),
    }
    out = {"rubric": "production (fix_idea), no refs", "summary": summary,
           "per_model": [{"model": r[0], "nB": r[1], "static": r[2], "nC": r[3],
                          "active": r[4], "boost": r[5]} for r in rows]}
    odir = ROOT / "reports" / "e11_ref_overlap"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "v3_static_vs_active.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n{'model':<40}{'nB':>4}{'static':>8}{'nC':>4}{'active':>8}{'boost':>8}")
    for r in rows:
        print(f"{r[0]:<40}{r[1]:>4}{r[2]:>8}{r[3]:>4}{r[4]:>8}{r[5]:>8}")
    print(f"\n✓ wrote {odir}/v3_static_vs_active.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--critics", type=int, default=0, help="use first N critics (0=all 5)")
    ap.add_argument("--max-subdomains", type=int, default=0, help="cap subdomains (0=all)")
    a = ap.parse_args()
    if not (a.score or a.analyze):
        a.analyze = True
    if a.score:
        do_score(a.workers, a.critics, a.max_subdomains)
    if a.analyze:
        do_analyze()


if __name__ == "__main__":
    main()
