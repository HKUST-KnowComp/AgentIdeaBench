"""E10 fix_v3 — recalibrate the ABSOLUTE scale so best-paper-caliber ideas land
at 8-9 overall (user expectation), while keeping separation from model ideas.

Background: even with fix_v2 (A originality calibration + B idea-not-writing),
the anchors sit at ~6.2 because the rubric reserves 8-10 for extreme cases
(paradigm-shift / Nobel-level / pre-registration protocol). The user expects a
genuine top-venue idea to score 8-9 overall.

fix_v3 = fix_v2 PLUS a SCALE CALIBRATION block mapping quality tiers to scores
(top-venue/ICLR-Oral caliber = 8-9; solid-but-unexciting = 6-7; derivative = 4-5;
weak = 1-3) and instructing the judge to use 8-9 for genuinely excellent ideas
and NOT inflate competent-but-unexciting ones above 7.

Validation re-uses the fix_v2 harness (15 anchors + 303 deepseek ideas, no refs,
modern5). SAFETY: new table e10_fixv3_scores; production untouched.

Usage: /usr/bin/python3 experiments/e10_fixv3_validate.py --score --analyze
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
from utils.constants import SCORING_DIMS as DIMS
from experiments.e10_idea_anchor_calibration import score_one, weighted_norm, trimmed_mean
from experiments.e10_fixv2_validate import FIX_V2_SYSTEM, build_items, summ

CRITICS = list(cfg.CRITIC_MODELS)

# Insert the scale-calibration block right after the CRITICAL: line.
_CRIT = ("CRITICAL: Use the FULL 1-10 range and differentiate clearly. A weak "
         "proposal and a strong proposal must NOT receive similar scores.")
_SCALE = _CRIT + (
    "\n\nSCALE CALIBRATION (anchor every dimension AND your overall impression to "
    "these quality tiers — this OVERRIDES any tendency to compress scores into 5-7):\n"
    "  9-10: landmark, best-paper-at-a-top-venue caliber — a genuinely field-"
    "advancing idea (e.g. ICLR Oral / Nature-level). Such an idea should score 9-10 "
    "on the dimensions that make it great (typically Originality and Impact) and 8+ "
    "overall.\n"
    "  8 : clearly excellent — a strong, novel, publishable idea an expert is "
    "enthusiastic about.\n"
    "  6-7: solid and competent but not exciting; an obvious, safe, or incremental "
    "contribution.\n"
    "  4-5: weak or derivative; mostly recombines known methods with no new angle.\n"
    "  1-3: vague, trivial, or flawed.\n"
    "A genuinely excellent research idea of top-venue caliber MUST land at 8-9 "
    "OVERALL — do NOT compress excellent ideas into 6-7. Conversely, do NOT inflate "
    "competent-but-unexciting ideas above 7. Most ordinary proposals are 5-6.")

if _CRIT not in FIX_V2_SYSTEM:
    raise SystemExit("CRITICAL anchor not found; prompt changed.")
FIX_V3_SYSTEM = FIX_V2_SYSTEM.replace(_CRIT, _SCALE)


def ensure_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e10_fixv3_scores (
        kind TEXT NOT NULL, item_id TEXT NOT NULL,
        idea_model TEXT, track TEXT, domain TEXT,
        critic_model TEXT NOT NULL, scores_json TEXT, error TEXT, created_at TEXT NOT NULL,
        PRIMARY KEY (kind, item_id, critic_model)
    );""")
    conn.commit()


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
            if not conn.execute("SELECT 1 FROM e10_fixv3_scores WHERE kind=? AND item_id=? AND critic_model=?",
                                (it["kind"], it["item_id"], cr)).fetchone():
                tasks.append((it, cr))
    print(f"critic tasks: {len(tasks)} (workers={workers})")
    if not tasks:
        return

    def _w(t):
        it, cr = t
        scores, _ = score_one(it["text"], cr, it["domain"], FIX_V3_SYSTEM, "")
        return (it, cr, scores)

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            it, cr, scores = f.result()
            conn.execute("""INSERT OR IGNORE INTO e10_fixv3_scores
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


def do_analyze():
    from scipy import stats as st
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(list); dimcell = defaultdict(lambda: defaultdict(list)); meta = {}
    for r in conn.execute("SELECT * FROM e10_fixv3_scores WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"]); cell[r["item_id"]].append(weighted_norm(s))
        grp = "ANCHOR" if r["kind"] == "anchor" else \
            f"{r['idea_model'].split('/')[-1]} [{ {'B':'Static','C':'Active'}.get(r['track'], r['track']) }]"
        meta[r["item_id"]] = grp
        for d in DIMS:
            if d in s:
                dimcell[grp][d].append(s[d])
    per = {k: trimmed_mean(v) for k, v in cell.items() if trimmed_mean(v) is not None}
    groups = defaultdict(list)
    for iid, w in per.items():
        groups[meta[iid]].append(w)
    conn.close()
    anchors = groups.get("ANCHOR", [])
    ad = {d: round(float(np.mean(dimcell["ANCHOR"][d])), 2) for d in DIMS if dimcell["ANCHOR"][d]}
    print(f"ANCHOR: {summ(anchors)}")
    print(f"anchor dims O/F/C/I/S: " + "/".join(f"{ad[d]:.1f}" for d in DIMS))
    print(f"\n{'group':<28}{'n':>4}{'mean':>7}{'anc-grp':>8}{'p(anc>)':>9}{'>ancμ':>7}  O/F/C/I/S")
    print(f"{'ANCHOR':<28}{len(anchors):>4}{np.mean(anchors):>7.2f}{'--':>8}{'--':>9}{'--':>7}  "
          + "/".join(f"{ad[d]:.1f}" for d in DIMS))
    order = sorted((g for g in groups if g != "ANCHOR"), key=lambda g: -np.mean(groups[g]))
    res = {"ANCHOR": summ(anchors), "anchor_dims": ad, "groups": {}}
    for g in order:
        ws = groups[g]; dd = {d: round(float(np.mean(dimcell[g][d])), 2) for d in DIMS if dimcell[g][d]}
        mwu = st.mannwhitneyu(anchors, ws, alternative="greater")
        gg = summ(ws)
        gg.update({"anchor_minus_group": round(np.mean(anchors) - np.mean(ws), 3),
                   "p_anchor_greater": round(float(mwu.pvalue), 4),
                   "frac_above_anchor_mean": round(sum(1 for x in ws if x > np.mean(anchors)) / len(ws), 3),
                   "dims": dd})
        res["groups"][g] = gg
        print(f"{g:<28}{gg['n']:>4}{gg['mean']:>7.2f}{gg['anchor_minus_group']:>8}"
              f"{gg['p_anchor_greater']:>9}{gg['frac_above_anchor_mean']:>7}  "
              + "/".join(f"{dd.get(d,0):.1f}" for d in DIMS))
    out = ROOT / "reports" / "e10_idea_anchor"; out.mkdir(parents=True, exist_ok=True)
    (out / "e10_fixv3_validate.json").write_text(json.dumps(res, indent=2))
    print(f"\n✓ wrote {out}/e10_fixv3_validate.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not (a.score or a.analyze):
        a.score = a.analyze = True
    if a.score:
        do_score(a.workers)
    if a.analyze:
        do_analyze()


if __name__ == "__main__":
    main()
