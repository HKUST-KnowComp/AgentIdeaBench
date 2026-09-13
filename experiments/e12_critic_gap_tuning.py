"""E12 — tune the critic until bestpaper anchors clearly beat v4-flash & v4-pro.

Scoring set: bestpaper anchors (docs/bestpapers.md) vs deepseek-v4-flash and
deepseek-v4-pro subdomain ideas (Static track B + Active track C). All scored
under a chosen SYSTEM-prompt variant, NO references (isolates the idea ceiling),
with the modern critics. Raw 5 dims are stored so aggregation weights can be
explored in-memory without re-scoring.

Goal: anchors_mean - max(v4flash_mean, v4pro_mean) is clearly positive, while the
weak CONTROL_IDEAS stay low (a fix must separate, not just inflate everything).

Tables: e12_gap_scores (variant-tagged). Idempotent.

Usage:
  /usr/bin/python3 experiments/e12_critic_gap_tuning.py --variant fix_idea --score --per-group 25 --critics 3
  /usr/bin/python3 experiments/e12_critic_gap_tuning.py --variant fix_idea --analyze
  /usr/bin/python3 experiments/e12_critic_gap_tuning.py --variant fix_idea --analyze --weights orig_heavy
"""
import argparse
import json
import random
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
    score_one, trimmed_mean, parse_bestpapers, CONTROL_IDEAS,
    FIX_ORIG_SYSTEM, FIX_IDEA_SYSTEM,
)

V4 = ["deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"]

# ---- candidate SYSTEM-prompt variants (extend here while iterating) ----
# fix_calib: fix_idea + an explicit ceiling-discipline paragraph pushing back on
# polished-but-incremental proposals (which strong models produce fluently).
_CEILING = """

CRITICAL CALIBRATION FOR ORIGINALITY AND IMPACT (read carefully):
Most proposals you see are competent but INCREMENTAL — they recombine known
methods or apply an established approach to an adjacent problem. Such proposals
must land at 5-6 on Originality no matter how fluently or confidently they are
written. Polish, buzzwords, and a confident tone are NOT originality.
Reserve Originality 8-10 for a genuine problem REFRAMING or a new mechanism that
an expert would call non-obvious and field-shifting. Reserve Impact 8-10 for work
that would change how the field operates, not merely be "useful". When uncertain
between two adjacent scores, choose the LOWER one. A well-written proposal that
recombines existing ideas is a 5-6 overall, not a 7-8."""
FIX_CALIB_SYSTEM = FIX_IDEA_SYSTEM + _CEILING

# fix_calib2: attacks the three drivers of the anchor<v4 asymmetry found in E12.
_CALIB2 = """

HOW TO JUDGE IDEA QUALITY (apply strictly and symmetrically):
1. NEW-MECHANISM CLAIMS ARE CHEAP. A proposal that names a mechanism, tool, or
   effect and calls it novel is usually recombining known components. Unless the
   proposal makes a NON-OBVIOUS conceptual leap (not just "apply known method A to
   problem B" or "combine A+B+C"), Originality is 5-6. Fluent, detailed,
   confident, or heavily-cited writing is NOT evidence of originality — strong
   writers dress up incremental ideas convincingly. Be MORE skeptical, not less,
   when a proposal sounds impressively novel but rests on standard tools.
2. AVOID HINDSIGHT BIAS. Do NOT down-score a proposal just because its core idea
   resembles something now considered standard or "well known". Many landmark
   ideas look obvious in retrospect. Judge whether the CENTRAL insight or
   reframing was non-obvious at the moment it was proposed — a clean reframing
   that reorganizes how a problem is seen is highly original (8-9) even if the
   phrasing now feels familiar.
3. LENGTH AND DETAIL ARE NOT CLARITY OR SPECIFICITY. A long, jargon-dense,
   citation-heavy proposal is not clearer or more specific than a terse one. Score
   Clarity and Specificity on whether the CENTRAL idea, mechanism, and testable
   prediction are precisely pinned down — a two-sentence idea can be a 8, a
   paragraph of buzzwords can be a 4."""
FIX_CALIB2_SYSTEM = FIX_CALIB_SYSTEM + _CALIB2

# fix_calib3: adds a novelty burden-of-proof clause on top of calib2.
_CALIB3 = """

ORIGINALITY — BURDEN OF PROOF (decisive):
To earn Originality >= 7, the proposal must EXPLICITLY articulate WHY its central
insight is non-obvious relative to standard practice — i.e. it names the prevailing
assumption or approach and states what it changes or overturns. A proposal that
merely ASSERTS a mechanism, technique, or combination (however specific or
technical) WITHOUT arguing why it is a non-obvious departure is capped at 5. Do not
grant novelty credit for confident tone, technical vocabulary, or a detailed method
description. Reward the proposal that makes its conceptual departure explicit;
withhold credit from the proposal that only sounds sophisticated."""
FIX_CALIB3_SYSTEM = FIX_CALIB2_SYSTEM + _CALIB3

SYS_BY_VARIANT = {
    "fix_orig": FIX_ORIG_SYSTEM,
    "fix_idea": FIX_IDEA_SYSTEM,
    "fix_calib": FIX_CALIB_SYSTEM,
    "fix_calib2": FIX_CALIB2_SYSTEM,
    "fix_calib3": FIX_CALIB3_SYSTEM,
    "prod": asc.SYSTEM_PROMPT,
}

# ---- candidate aggregation weight schemes (explored in-memory) ----
WEIGHT_SCHEMES = {
    "prod": {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5, "impact": 1.5, "specificity": 0.5},
    "orig_heavy": {"originality": 3.0, "feasibility": 1.0, "clarity": 0.0, "impact": 1.0, "specificity": 0.0},
    "orig_impact": {"originality": 3.0, "feasibility": 0.5, "clarity": 0.0, "impact": 2.0, "specificity": 0.5},
    "no_writing": {"originality": 2.0, "feasibility": 1.0, "clarity": 0.0, "impact": 1.5, "specificity": 0.0},
    "orig_only": {"originality": 1.0, "feasibility": 0.0, "clarity": 0.0, "impact": 0.0, "specificity": 0.0},
}


def wmean(s, w):
    tot = sum(w[d] for d in DIMS)
    return sum(s.get(d, 0) * w[d] for d in DIMS) / tot if tot else 0


def ensure_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e12_gap_scores (
        item_id TEXT NOT NULL, grp TEXT NOT NULL, model TEXT, track TEXT, domain TEXT,
        variant TEXT NOT NULL, critic_model TEXT NOT NULL,
        scores_json TEXT, error TEXT, created_at TEXT NOT NULL,
        PRIMARY KEY (item_id, variant, critic_model)
    );""")
    conn.commit()


def build_items(per_group, seed=0):
    """Returns list of (item_id, grp, model, track, domain, text)."""
    rng = random.Random(seed)
    items = []
    # anchors
    for a in parse_bestpapers():
        items.append((a["anchor_id"], "anchor", "anchor", "-", a["domain"], a["abstract"]))
    # controls
    for cid, _name, dom, txt in CONTROL_IDEAS:
        items.append((cid, "control", "control", "-", dom, txt))
    # v4 ideas
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    for m in V4:
        for tr in ("B", "C"):
            rows = [r for r in conn.execute(
                "SELECT subdomain, domain, idea_text FROM subdomain_ideas "
                "WHERE idea_model=? AND track=? AND TRIM(idea_text)!=''", (m, tr))]
            rng.shuffle(rows)
            for r in rows[:per_group]:
                short = m.split("/")[-1]
                iid = f"{short}|{tr}|{r['subdomain'][:40]}"
                items.append((iid, f"{short}-{tr}", m, tr, r["domain"], r["idea_text"]))
    conn.close()
    return items


def do_score(variant, per_group, n_critics, workers, seed):
    critics = list(cfg.CRITIC_MODELS)[:n_critics] if n_critics else list(cfg.CRITIC_MODELS)
    system = SYS_BY_VARIANT[variant]
    items = build_items(per_group, seed)
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000"); ensure_table(conn)
    tasks = []
    for iid, grp, model, tr, dom, txt in items:
        for cr in critics:
            if not conn.execute("SELECT 1 FROM e12_gap_scores WHERE item_id=? AND variant=? AND critic_model=?",
                                (iid, variant, cr)).fetchone():
                tasks.append((iid, grp, model, tr, dom, txt, cr))
    print(f"variant={variant} items={len(items)} critics={len(critics)} tasks={len(tasks)}", flush=True)
    if not tasks:
        conn.close(); return

    def _w(t):
        iid, grp, model, tr, dom, txt, cr = t
        scores, _ = score_one(txt, cr, dom, system, "")
        return (iid, grp, model, tr, dom, cr, scores)

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            iid, grp, model, tr, dom, cr, scores = f.result()
            conn.execute("INSERT OR IGNORE INTO e12_gap_scores VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (iid, grp, model, tr, dom, variant, cr,
                          json.dumps(scores) if scores else None,
                          None if scores else "parse_fail", ts))
            conn.commit()
            ok += 1 if scores else 0; err += 0 if scores else 1
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close()
    print(f"done: ok={ok} err={err}")


def do_analyze(variant, weight_key):
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(list)   # (item_id,grp) -> [dim-dict per critic]
    for r in conn.execute("SELECT * FROM e12_gap_scores WHERE variant=? AND scores_json IS NOT NULL", (variant,)):
        sj = json.loads(r["scores_json"])
        def _num(v):
            return v.get("score", 0) if isinstance(v, dict) else (v or 0)
        cell[(r["item_id"], r["grp"])].append({d: _num(sj.get(d)) for d in DIMS})
    conn.close()

    w = WEIGHT_SCHEMES[weight_key]
    per_group_w = defaultdict(list)      # grp -> [weighted per item]
    per_group_dim = defaultdict(lambda: defaultdict(list))  # grp -> dim -> [trimmed per item]
    for (iid, grp), critic_dims in cell.items():
        # trimmed mean per dimension across critics, then weighted
        dim_tm = {}
        for d in DIMS:
            vals = [cd[d] for cd in critic_dims]
            tm = trimmed_mean(vals)
            if tm is not None:
                dim_tm[d] = tm
                per_group_dim[grp][d].append(tm)
        per_group_w[grp].append(wmean(dim_tm, w))

    print(f"\n=== variant={variant}  weights={weight_key} {w} ===")
    hdr = f"{'group':<22}{'n':>4}{'weighted':>10}" + "".join(f"{d[:4]:>7}" for d in DIMS)
    print(hdr)
    means = {}
    for grp in sorted(per_group_w):
        n = len(per_group_w[grp]); wm = float(np.mean(per_group_w[grp])); means[grp] = wm
        dims = "".join(f"{np.mean(per_group_dim[grp][d]):>7.2f}" for d in DIMS)
        print(f"{grp:<22}{n:>4}{wm:>10.3f}{dims}")

    anchor = means.get("anchor")
    v4 = [means[g] for g in means if g.startswith("deepseek")]
    ctrl = means.get("control")
    print("\n--- gap summary ---")
    if anchor is not None and v4:
        worst_gap = anchor - max(v4)
        print(f"anchor={anchor:.3f}  max(v4 group)={max(v4):.3f}  gap(anchor - worst v4)={worst_gap:+.3f}")
        print(f"  clear separation (>=0.5)? {'YES' if worst_gap >= 0.5 else 'NO'}")
    if ctrl is not None and anchor is not None:
        print(f"control={ctrl:.3f}  (anchor should stay >> control; anchor-control={anchor-ctrl:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fix_idea", choices=list(SYS_BY_VARIANT))
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--per-group", type=int, default=25)
    ap.add_argument("--critics", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--weights", default="prod", choices=list(WEIGHT_SCHEMES))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not (a.score or a.analyze):
        a.analyze = True
    if a.score:
        do_score(a.variant, a.per_group, a.critics, a.workers, a.seed)
    if a.analyze:
        do_analyze(a.variant, a.weights)


if __name__ == "__main__":
    main()
