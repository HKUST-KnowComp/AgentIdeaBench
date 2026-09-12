"""E10-deepseek — score ALL ideas of the two deepseek models with the NEW
(production) rubric and compare against the best-paper anchors.

User: 用新评分标准评判两个 deepseek 的所有 idea,和 bestpaper 对比看是否有明显差距。

- Models: deepseek/deepseek-r1-0528, deepseek/deepseek-v4-pro
- Ideas: ALL their non-blank ideas, BOTH tracks (B=Static, C=Active-Blind), v1.
- Rubric: current production absolute_scorer.SYSTEM_PROMPT (= the fixed fix_idea
  rubric, already promoted). Scored with modern5 critics, NO references — the
  SAME apples-to-apples setting the 15 anchors were scored under (variant=fix_idea
  in e10_idea_anchor_scores), so the comparison is valid.
- weighted =(O×2+F+C×0.5+I×1.5+S×0.5)/5.5; per-idea = trimmed_mean over 5 critics.

SAFETY: writes ONLY to new table e10_deepseek_scores. modern5 = open-weight →
default OPENROUTER_API_KEY. Idempotent (PK incl. track).

Usage:
  /usr/bin/python3 experiments/e10_deepseek_vs_anchor.py --score --analyze
  /usr/bin/python3 experiments/e10_deepseek_vs_anchor.py --analyze
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
from evaluation import absolute_scorer as asc
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as WEIGHTS
from experiments.e10_idea_anchor_calibration import score_one, weighted_norm, trimmed_mean

MODELS = ["deepseek/deepseek-r1-0528", "deepseek/deepseek-v4-pro"]
CRITICS = list(cfg.CRITIC_MODELS)


def ensure_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e10_deepseek_scores (
        idea_model   TEXT NOT NULL,
        track        TEXT NOT NULL,
        paper_id     TEXT NOT NULL,
        idea_index   INTEGER NOT NULL,
        critic_model TEXT NOT NULL,
        scores_json  TEXT,
        error        TEXT,
        created_at   TEXT NOT NULL,
        PRIMARY KEY (idea_model, track, paper_id, idea_index, critic_model)
    );
    """)
    conn.commit()


def pull_ideas():
    cr = sqlite3.connect(str(cfg.RESULTS_DB)); cr.row_factory = sqlite3.Row
    cp = sqlite3.connect(str(cfg.PAPERS_DB))
    dom = {r[0]: r[1] for r in cp.execute("SELECT paper_id, domain FROM papers")}
    cp.close()
    rows = []
    for m in MODELS:
        for r in cr.execute(
                "SELECT idea_model, track, paper_id, idea_index, idea_text FROM results "
                "WHERE idea_model=? AND prompt_version='v1_paper_refs' AND critic_model='' "
                "AND track IN ('B','C') AND TRIM(idea_text)!=''", (m,)):
            rows.append({"model": r["idea_model"], "track": r["track"],
                         "paper_id": r["paper_id"], "idx": r["idea_index"],
                         "domain": dom.get(r["paper_id"], "General Science"),
                         "text": r["idea_text"]})
    cr.close()
    return rows


def do_score(workers):
    ideas = pull_ideas()
    print(f"deepseek ideas: {len(ideas)} "
          f"(B={sum(1 for i in ideas if i['track']=='B')}, "
          f"C={sum(1 for i in ideas if i['track']=='C')})")
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_table(conn)
    tasks = []
    for it in ideas:
        for cr in CRITICS:
            ex = conn.execute("SELECT 1 FROM e10_deepseek_scores WHERE idea_model=? AND track=? "
                              "AND paper_id=? AND idea_index=? AND critic_model=?",
                              (it["model"], it["track"], it["paper_id"], it["idx"], cr)).fetchone()
            if not ex:
                tasks.append((it, cr))
    print(f"critic tasks: {len(tasks)} (workers={workers})")
    if not tasks:
        return

    # current production rubric (= promoted fix_idea), no references
    sysprompt = asc.SYSTEM_PROMPT

    def _w(t):
        it, cr = t
        scores, _ = score_one(it["text"], cr, it["domain"], sysprompt, "")
        return (it, cr, scores)

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            it, cr, scores = f.result()
            conn.execute("""INSERT OR IGNORE INTO e10_deepseek_scores
                (idea_model, track, paper_id, idea_index, critic_model, scores_json, error, created_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (it["model"], it["track"], it["paper_id"], it["idx"], cr,
                 json.dumps({d: v["score"] for d, v in scores.items()}) if scores else None,
                 None if scores else "parse_fail", ts))
            conn.commit()
            ok += 1 if scores else 0
            err += 0 if scores else 1
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close()
    print(f"done score: ok={ok} err={err}")


def summ(xs):
    if not xs:
        return {"n": 0}
    a = np.array(xs)
    return {"n": len(xs), "mean": round(float(a.mean()), 3),
            "median": round(float(np.median(a)), 3),
            "std": round(float(a.std(ddof=1)), 3) if len(xs) > 1 else 0.0,
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
            "p90": round(float(np.percentile(a, 90)), 3)}


def do_analyze():
    from scipy import stats as st
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    # deepseek per-idea weighted (trimmed mean over critics), grouped by (model,track)
    cell = defaultdict(list)  # (model,track,paper,idx) -> [weighted per critic]
    for r in conn.execute("SELECT * FROM e10_deepseek_scores WHERE scores_json IS NOT NULL"):
        cell[(r["idea_model"], r["track"], r["paper_id"], r["idea_index"])].append(
            weighted_norm(json.loads(r["scores_json"])))
    per_idea = {k: trimmed_mean(v) for k, v in cell.items() if trimmed_mean(v) is not None}
    groups = defaultdict(list)          # (model,track) -> per-idea weighted
    dimcell = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT * FROM e10_deepseek_scores WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"])
        for d in DIMS:
            if d in s:
                dimcell[(r["idea_model"], r["track"])][d].append(s[d])
    for (m, tr, p, i), w in per_idea.items():
        groups[(m, tr)].append(w)

    # anchors under fix_idea (= same rubric, same no-refs)
    acell = defaultdict(list)
    for r in conn.execute("SELECT anchor_id, scores_json FROM e10_idea_anchor_scores "
                          "WHERE variant='fix_idea' AND scores_json IS NOT NULL"):
        if r["anchor_id"].startswith("ctrl_"):
            continue
        acell[r["anchor_id"]].append(weighted_norm(json.loads(r["scores_json"])))
    anchors = [trimmed_mean(v) for v in acell.values() if trimmed_mean(v) is not None]
    conn.close()

    TR = {"B": "Static", "C": "Active"}
    result = {"rubric": "production (fix_idea), no references",
              "anchors_bestpaper": summ(anchors), "groups": {}}
    for (m, tr), ws in sorted(groups.items()):
        key = f"{m} [{TR.get(tr, tr)}]"
        dims = {d: round(float(np.mean(dimcell[(m, tr)][d])), 2)
                for d in DIMS if dimcell[(m, tr)][d]}
        g = summ(ws)
        # gap vs anchors
        if ws and anchors:
            t = st.ttest_ind(anchors, ws, equal_var=False)
            mwu = st.mannwhitneyu(anchors, ws, alternative="greater")
            g["anchor_minus_group_mean"] = round(np.mean(anchors) - np.mean(ws), 3)
            g["welch_p_anchor_greater"] = round(float(st.ttest_ind(anchors, ws, equal_var=False).pvalue) / 2, 4)
            g["mannwhitney_p_anchor_greater"] = round(float(mwu.pvalue), 4)
            g["frac_idea_above_anchor_mean"] = round(sum(1 for x in ws if x > np.mean(anchors)) / len(ws), 3)
        g["dims_OFCIS"] = dims
        result["groups"][key] = g

    out = ROOT / "reports" / "e10_idea_anchor"; out.mkdir(parents=True, exist_ok=True)
    (out / "e10_deepseek_vs_anchor.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n✓ wrote {out}/e10_deepseek_vs_anchor.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not (args.score or args.analyze):
        args.score = args.analyze = True
    if args.score:
        do_score(args.workers)
    if args.analyze:
        do_analyze()


if __name__ == "__main__":
    main()
