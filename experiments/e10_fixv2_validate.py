"""E10 fix_v2 validation — test rubric changes A (Originality calibration) +
B (Clarity/Specificity = idea-not-writing) WITHOUT touching production.

Problem (diagnosed): under current production rubric the best-paper anchors do
NOT clearly beat normal model ideas. Per-dim:
  - Originality (×2) fails to separate (anchors ~6.5 ≈ strong-model Active).
  - Clarity & Specificity REWARD verbosity/polish → verbose model proposals beat
    terse curated idea-abstracts.

fix_v2 = current production SYSTEM_PROMPT, plus:
  A) ORIGINALITY CALIBRATION: explicit few-shot anchors + "cluster-busting"
     (competent/fluent combos = 5-6; reserve 8-10 for genuine reframing).
  B) IDEA-NOT-WRITING note on Clarity & Specificity: judge the idea, not length/
     polish/jargon; concise precise idea scores same as a verbose one.

Validation: re-score the 15 best-paper anchors AND all deepseek ideas (B+C, both
models) under fix_v2, no references, modern5. Check anchors clearly > deepseek.

SAFETY: writes ONLY to new table e10_fixv2_scores. Does NOT modify production
absolute_scorer.py. modern5 = open-weight → default key. Idempotent.

Usage:
  /usr/bin/python3 experiments/e10_fixv2_validate.py --score --analyze
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
from utils.constants import SCORING_DIMS as DIMS
from experiments.e10_idea_anchor_calibration import (
    parse_bestpapers, score_one, weighted_norm, trimmed_mean)
from experiments.e10_deepseek_vs_anchor import pull_ideas

CRITICS = list(cfg.CRITIC_MODELS)

# ---- Build fix_v2 from CURRENT PRODUCTION prompt (A + B) -------------------
_ANCHOR = ("A clean problem reframing that an expert would call 'I haven't seen "
           "it framed this way' is a 8-9 even if every component is known.\n\n"
           "Score the research proposal on FIVE dimensions, each from 1 to 10:")

_AB = ("A clean problem reframing that an expert would call 'I haven't seen it "
       "framed this way' is a 8-9 even if every component is known.\n\n"
       "ORIGINALITY CALIBRATION (use the FULL range; do NOT cluster scores at 6-7): "
       "Most proposals that merely sound plausible, are well-motivated, or competently "
       "combine known methods are ORIGINALITY 5-6, NOT 7+. A fluent, well-written "
       "proposal with no genuinely new angle is a 5. Reserve 8-10 ONLY for a genuine "
       "problem REFRAMING that would change how researchers in the field think — an "
       "insight an expert would call non-obvious and field-shifting. Examples: "
       "(a) \"Reframe multi-turn LLM evaluation as a control problem and measure "
       "aptitude loss / recovery under single-vs-multi-turn transforms\" = 9 (new "
       "framing); (b) \"Combine retrieval-augmented generation with chain-of-thought "
       "to improve QA accuracy\" = 5 (competent combination, no new angle); "
       "(c) \"Fine-tune model X on dataset Y and expect higher accuracy\" = 3.\n\n"
       "IDEA-NOT-WRITING (applies to Clarity and Specificity): This is an idea-quality "
       "benchmark. Judge the IDEA itself, not writing polish, length, formatting, or "
       "jargon density. A concise idea with a precise mechanism must score the SAME on "
       "Clarity and Specificity as a longer, more elaborately written proposal with the "
       "same underlying mechanism. Do NOT reward added detail or terminology that does "
       "not sharpen the core claim — verbosity is not specificity, and polish is not "
       "clarity.\n\n"
       "Score the research proposal on FIVE dimensions, each from 1 to 10:")

if _ANCHOR not in asc.SYSTEM_PROMPT:
    raise SystemExit("FIX_V2 anchor string not found in production SYSTEM_PROMPT — "
                     "the prompt changed; update _ANCHOR.")
FIX_V2_SYSTEM = asc.SYSTEM_PROMPT.replace(_ANCHOR, _AB)


def ensure_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e10_fixv2_scores (
        kind         TEXT NOT NULL,         -- 'anchor' | 'deepseek'
        item_id      TEXT NOT NULL,         -- anchor_id  OR  model|track|paper|idx
        idea_model   TEXT, track TEXT, domain TEXT,
        critic_model TEXT NOT NULL,
        scores_json  TEXT, error TEXT, created_at TEXT NOT NULL,
        PRIMARY KEY (kind, item_id, critic_model)
    );
    """)
    conn.commit()


def build_items():
    items = []
    for a in parse_bestpapers():
        items.append({"kind": "anchor", "item_id": a["anchor_id"], "model": "bestpaper",
                      "track": "-", "domain": a["domain"], "text": a["abstract"]})
    for it in pull_ideas():
        iid = f"{it['model']}|{it['track']}|{it['paper_id']}|{it['idx']}"
        items.append({"kind": "deepseek", "item_id": iid, "model": it["model"],
                      "track": it["track"], "domain": it["domain"], "text": it["text"]})
    return items


def do_score(workers):
    items = build_items()
    print(f"items: {len(items)} (anchors={sum(1 for i in items if i['kind']=='anchor')}, "
          f"deepseek={sum(1 for i in items if i['kind']=='deepseek')})")
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_table(conn)
    tasks = []
    for it in items:
        for cr in CRITICS:
            ex = conn.execute("SELECT 1 FROM e10_fixv2_scores WHERE kind=? AND item_id=? AND critic_model=?",
                              (it["kind"], it["item_id"], cr)).fetchone()
            if not ex:
                tasks.append((it, cr))
    print(f"critic tasks: {len(tasks)} (workers={workers})")
    if not tasks:
        return

    def _w(t):
        it, cr = t
        scores, _ = score_one(it["text"], cr, it["domain"], FIX_V2_SYSTEM, "")
        return (it, cr, scores)

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            it, cr, scores = f.result()
            conn.execute("""INSERT OR IGNORE INTO e10_fixv2_scores
                (kind,item_id,idea_model,track,domain,critic_model,scores_json,error,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (it["kind"], it["item_id"], it["model"], it["track"], it["domain"], cr,
                 json.dumps({d: v["score"] for d, v in scores.items()}) if scores else None,
                 None if scores else "parse_fail", ts))
            conn.commit()
            ok += 1 if scores else 0; err += 0 if scores else 1
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close()
    print(f"done score: ok={ok} err={err}")


def summ(xs):
    if not xs:
        return {"n": 0}
    a = np.array(xs)
    return {"n": len(xs), "mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
            "std": round(float(a.std(ddof=1)), 3) if len(xs) > 1 else 0.0,
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
            "p90": round(float(np.percentile(a, 90)), 3)}


def do_analyze():
    from scipy import stats as st
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(list)      # item_id -> [weighted per critic]
    dimcell = defaultdict(lambda: defaultdict(list))  # group -> dim -> [raw]
    meta = {}
    for r in conn.execute("SELECT * FROM e10_fixv2_scores WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"])
        cell[r["item_id"]].append(weighted_norm(s))
        if r["kind"] == "anchor":
            grp = "ANCHOR"
        else:
            grp = f"{r['idea_model'].split('/')[-1]} [{ {'B':'Static','C':'Active'}.get(r['track'], r['track']) }]"
        meta[r["item_id"]] = grp
        for d in DIMS:
            if d in s:
                dimcell[grp][d].append(s[d])
    per_item = {k: trimmed_mean(v) for k, v in cell.items() if trimmed_mean(v) is not None}
    groups = defaultdict(list)
    for iid, w in per_item.items():
        groups[meta[iid]].append(w)
    conn.close()

    anchors = groups.get("ANCHOR", [])
    res = {"rubric": "fix_v2 (A: originality calibration + B: idea-not-writing), no refs",
           "ANCHOR": summ(anchors),
           "anchor_dims": {d: round(float(np.mean(dimcell["ANCHOR"][d])), 2) for d in DIMS if dimcell["ANCHOR"][d]},
           "groups": {}}
    for grp, ws in sorted(groups.items()):
        if grp == "ANCHOR":
            continue
        g = summ(ws)
        if ws and anchors:
            mwu = st.mannwhitneyu(anchors, ws, alternative="greater")
            g["anchor_minus_group"] = round(np.mean(anchors) - np.mean(ws), 3)
            g["mannwhitney_p_anchor_greater"] = round(float(mwu.pvalue), 4)
            g["frac_above_anchor_mean"] = round(sum(1 for x in ws if x > np.mean(anchors)) / len(ws), 3)
        g["dims_OFCIS"] = {d: round(float(np.mean(dimcell[grp][d])), 2) for d in DIMS if dimcell[grp][d]}
        res["groups"][grp] = g

    out = ROOT / "reports" / "e10_idea_anchor"; out.mkdir(parents=True, exist_ok=True)
    (out / "e10_fixv2_validate.json").write_text(json.dumps(res, indent=2))
    # compact print
    print(json.dumps({"ANCHOR": res["ANCHOR"], "anchor_dims": res["anchor_dims"]}, indent=2))
    print(f"\n{'group':<28}{'n':>4}{'mean':>7}{'anc-grp':>8}{'p(anc>)':>9}{'>ancμ':>7}  dims O/F/C/I/S")
    ad = res["anchor_dims"]
    print(f"{'ANCHOR':<28}{res['ANCHOR']['n']:>4}{res['ANCHOR']['mean']:>7}{'--':>8}{'--':>9}{'--':>7}  "
          + "/".join(f"{ad[d]:.1f}" for d in DIMS))
    for grp, g in res["groups"].items():
        dd = g["dims_OFCIS"]
        print(f"{grp:<28}{g['n']:>4}{g['mean']:>7}{g.get('anchor_minus_group',0):>8}"
              f"{g.get('mannwhitney_p_anchor_greater',0):>9}{g.get('frac_above_anchor_mean',0):>7}  "
              + "/".join(f"{dd.get(d,0):.1f}" for d in DIMS))
    print(f"\n✓ wrote {out}/e10_fixv2_validate.json")


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
