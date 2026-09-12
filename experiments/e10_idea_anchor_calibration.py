"""E10 — Idea-anchor calibration: do hand-picked full-score ideas score high?

User supplied docs/bestpapers.md: 15 manually-selected ICLR 2026 Oral/Outstanding
idea-abstracts, explicitly chosen as "full-score reference anchors" (idea quality,
not paper quality; abstracts rewritten as clean idea abstracts).

Goal: feed these through the PRODUCTION critic (modern5 pool) and check whether
they land clearly ABOVE model-generated ideas. If a benchmark critic is viable,
human-curated top-tier ideas must score near the top of the scale. If they sit in
the same 5–6 band as average model gen, the rubric/anchors are compressed and the
critic is NOT viable as-is.

This extends E7 (real high-citation gt_hypothesis scored 4.78 < model-gen 5.20).
E7's anchors were tangled with each paper's own refs (Originality self-penalty).
These anchors are standalone clean ideas, so they isolate the rubric ceiling.

Variants (--variant):
  prod      : production SYSTEM_PROMPT, NO references  (default, isolates ceiling)
  prod_refs : production SYSTEM_PROMPT, domain-only context line (no real refs)
  fix_orig  : patched system prompt — Originality judged on absolute novelty
              (NOT "derivable from provided refs"), best-idea anchors recalibrated.
              For investigating a fix WITHOUT touching production absolute_scorer.

SAFETY: writes ONLY to new table `e10_idea_anchor_scores` (keyed by variant).
results/papers untouched. modern5 = open-weight -> default OPENROUTER_API_KEY.
Idempotent (PK = anchor_id, critic_model, variant).

Usage:
  /usr/bin/python3 experiments/e10_idea_anchor_calibration.py --score --analyze
  /usr/bin/python3 experiments/e10_idea_anchor_calibration.py --variant fix_orig --score --analyze
"""
import argparse
import json
import re
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

MODERN_CRITICS = list(cfg.CRITIC_MODELS)
WSUM = sum(WEIGHTS.values())
BESTPAPERS_MD = ROOT / "docs" / "bestpapers.md"

# ---------------------------------------------------------------------------
# fix_orig: patched system prompt. Only the ORIGINALITY framing + the
# self-reference penalty are changed; all other dims/checks stay identical so
# the experiment isolates the Originality self-penalty hypothesis.
# ---------------------------------------------------------------------------
FIX_ORIG_SYSTEM = asc.SYSTEM_PROMPT.replace(
    "When scoring ORIGINALITY, compare the proposal against the provided background \
literature. If the proposal's core idea is already covered by or easily \
derivable from the references, score 4 or below.",
    "When scoring ORIGINALITY, judge whether the proposal REFRAMES a problem or \
introduces a genuinely new mechanism/insight relative to common practice in the \
field. A strong idea that builds on prior work is NOT unoriginal: foundational \
ideas legitimately rest on existing literature. Do NOT penalize an idea merely \
because its components appear in the background; penalize only if the proposal \
adds no new conceptual angle. A clean problem reframing that an expert would call \
'I haven't seen it framed this way' is a 8-9 even if every component is known."
).replace(
    "  7 : Proposes a specific mechanism or approach NOT mentioned or implied by ANY \
of the background papers, AND explains why existing approaches are insufficient; \
identifies specific gaps in the literature rather than just summarizing what exists",
    "  7 : Proposes a specific mechanism or reframing that an expert would find \
non-obvious and explains why existing approaches are insufficient"
)

# ---------------------------------------------------------------------------
# fix_idea: fix_orig PLUS an idea-level Specificity ladder. The production
# Specificity rubric demands pre-registration protocol detail (numbers,
# datasets, thresholds) which a pure IDEA abstract structurally cannot have,
# so it crushes idea-quality anchors (~4.0). For an IDEA benchmark, Specificity
# should measure conceptual precision (is the mechanism + testable prediction
# stated precisely?), not protocol completeness.
# ---------------------------------------------------------------------------
_SPEC_PROD = """SPECIFICITY
  1 : No concrete details whatsoever ("study X using AI")
  2 : Names a broad area but no specific mechanism or target
  3 : Mentions a general technique but not how it applies here
  4 : Names specific tools/methods but not the exact experimental setup
  5 : Identifies key variables but missing controls or sample sizes
  6 : Specifies method, target, and expected direction of effect
  7 : Includes controls, metrics, and testable predictions with specific numerical values
  8 : Detailed enough to reproduce: exact methods, datasets, and thresholds
  9 : Formal hypothesis with all variables precisely operationalized
  10: Complete pre-registration-quality protocol"""
_SPEC_IDEA = """SPECIFICITY (conceptual precision of the IDEA — this is an idea-level \
benchmark, do NOT require a full experimental protocol or numerical thresholds)
  1 : No concrete content whatsoever ("study X using AI")
  2 : Names a broad area but no specific mechanism, target, or claim
  3 : States a direction but the central mechanism is left vague
  4 : Mechanism gestured at but not clearly articulated
  5 : Core mechanism is identifiable; the specific change vs current practice is \
implied but not sharp
  6 : Mechanism is clearly stated and one can see what would be done differently
  7 : Mechanism + a concrete testable prediction or clear evaluation axis are \
both precisely stated (numbers NOT required for an idea)
  8 : Mechanism, the precise problem reframing, and how it would be tested are \
all unambiguous; an expert could design the study from this idea alone
  9 : All of the above plus the key variables / conditions that would decide the \
hypothesis are named
  10: Crisp, complete idea: mechanism, prediction, and decisive test all explicit"""
FIX_IDEA_SYSTEM = FIX_ORIG_SYSTEM.replace(_SPEC_PROD, _SPEC_IDEA)

# Weak control ideas — must STAY low under any fix, else the fix just inflates.
CONTROL_IDEAS = [
    ("ctrl_vague", "Vague boilerplate", "Machine Learning",
     "We propose to use deep learning and large language models to improve "
     "performance on important AI tasks. By leveraging neural networks and "
     "advanced training techniques, our approach will achieve better results "
     "across a range of benchmarks and advance the state of the art."),
    ("ctrl_kwstuff", "Keyword stuffing", "Machine Learning",
     "We combine Transformers, GANs, contrastive learning, LoRA, RLHF, GNNs, "
     "diffusion models, and mixture-of-experts into a unified framework that "
     "applies chain-of-thought reasoning with retrieval-augmented generation "
     "to solve multimodal reasoning, achieving synergy across all components."),
    ("ctrl_restate", "Restated common practice", "Machine Learning",
     "We propose to fine-tune a pretrained language model on a downstream "
     "dataset using supervised learning with cross-entropy loss, then evaluate "
     "accuracy on a held-out test set, expecting fine-tuning to outperform the "
     "zero-shot baseline."),
]


def parse_bestpapers():
    """Parse docs/bestpapers.md into anchor records."""
    text = BESTPAPERS_MD.read_text()
    anchors = []
    # split on "## NN. Title" (two-digit numbered headers), stop before "# Suggested"
    body = text.split("# Suggested Metadata")[0]
    blocks = re.split(r'\n## (\d+)\.\s+(.+)\n', body)
    # blocks: [pre, num1, title1, content1, num2, title2, content2, ...]
    for i in range(1, len(blocks), 3):
        num = blocks[i].strip()
        title = blocks[i + 1].strip()
        content = blocks[i + 2]
        sub = re.search(r'\*\*Subdomain:\*\*\s*(.+)', content)
        subdomain = sub.group(1).strip() if sub else ""
        # abstract = text after "### Abstract" up to "### Why"
        ab = re.search(r'### Abstract\s*\n(.+?)\n### ', content, re.DOTALL)
        abstract = ab.group(1).strip() if ab else ""
        if abstract:
            anchors.append({
                "anchor_id": f"anchor_{num.zfill(2)}",
                "title": title,
                "subdomain": subdomain,
                "domain": subdomain.split("/")[0].strip() if subdomain else "Machine Learning",
                "abstract": abstract,
            })
    return anchors


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
    CREATE TABLE IF NOT EXISTS e10_idea_anchor_scores (
        anchor_id    TEXT NOT NULL,
        title        TEXT,
        domain       TEXT,
        variant      TEXT NOT NULL,
        critic_model TEXT NOT NULL,
        scores_json  TEXT,
        reasoning_json TEXT,
        raw_response TEXT,
        error        TEXT,
        created_at   TEXT NOT NULL,
        PRIMARY KEY (anchor_id, critic_model, variant)
    );
    """)
    conn.commit()


def score_one(idea, critic, domain, system_prompt, references):
    """Thin scorer that lets us swap the system prompt (production score_idea
    hard-codes SYSTEM_PROMPT). Reuses absolute_scorer's parser/normaliser."""
    from utils.LLM import CriticLLM
    llm = CriticLLM(model_name=critic)
    prompt = asc.USER_TEMPLATE.format(
        idea=idea.strip(),
        domain=domain or "General Science",
        references=references or "(no background literature available)",
    )
    raw = ""
    for attempt in range(3):
        try:
            resp = llm.score_idea(
                prompt + (asc.JSON_RETRY_SUFFIX if attempt > 0 else ""),
                system_prompt=system_prompt,
            )
            if isinstance(resp, tuple):
                raw = f"<reasoning>{resp[1]}</reasoning>\n{resp[0] or ''}"
                resp = resp[0]
            else:
                raw = resp or ""
            if not resp:
                continue
            parsed = asc._extract_json(resp)
            if parsed:
                return asc._normalise(parsed), raw
        except Exception as e:
            raw = f"ERR: {e}"
    return None, raw


SYS_BY_VARIANT = {
    "prod": asc.SYSTEM_PROMPT,
    "prod_refs": asc.SYSTEM_PROMPT,
    "fix_orig": FIX_ORIG_SYSTEM,
    "fix_idea": FIX_IDEA_SYSTEM,
}


def do_score(variant, workers):
    system_prompt = SYS_BY_VARIANT[variant]
    anchors = parse_bestpapers()
    # append weak controls (same scoring path) so we can prove a fix opens the
    # gap rather than inflating everything.
    for cid, ctitle, cdom, ctext in CONTROL_IDEAS:
        anchors.append({"anchor_id": cid, "title": ctitle, "subdomain": cdom,
                        "domain": cdom, "abstract": ctext})
    print(f"parsed {len(anchors)} items ({len(CONTROL_IDEAS)} controls) (variant={variant})")
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    ensure_table(conn)

    tasks = []
    for a in anchors:
        for cr in MODERN_CRITICS:
            exists = conn.execute(
                "SELECT 1 FROM e10_idea_anchor_scores WHERE anchor_id=? AND critic_model=? AND variant=?",
                (a["anchor_id"], cr, variant)).fetchone()
            if not exists:
                tasks.append((a, cr))
    print(f"critic tasks: {len(tasks)}")
    if not tasks:
        return

    def _w(t):
        a, cr = t
        scores, raw = score_one(a["abstract"], cr, a["domain"], system_prompt, "")
        return (a, cr, scores, raw)

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            a, cr, scores, raw = f.result()
            if scores:
                conn.execute("""INSERT OR IGNORE INTO e10_idea_anchor_scores
                    (anchor_id, title, domain, variant, critic_model, scores_json,
                     reasoning_json, raw_response, error, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (a["anchor_id"], a["title"], a["domain"], variant, cr,
                     json.dumps({d: v["score"] for d, v in scores.items()}),
                     json.dumps({d: v["reasoning"] for d, v in scores.items()}),
                     (raw or "")[:3000], None, ts))
                conn.commit(); ok += 1
            else:
                conn.execute("""INSERT OR IGNORE INTO e10_idea_anchor_scores
                    (anchor_id, title, domain, variant, critic_model, scores_json,
                     reasoning_json, raw_response, error, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (a["anchor_id"], a["title"], a["domain"], variant, cr,
                     None, None, (raw or "")[:3000], "parse_fail", ts))
                conn.commit(); err += 1
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}")
    conn.close()
    print(f"done score variant={variant}: ok={ok} err={err}")


def ensure_model_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e10_model_idea_scores (
        idea_model   TEXT NOT NULL,
        paper_id     TEXT NOT NULL,
        idea_index   INTEGER NOT NULL,
        variant      TEXT NOT NULL,
        critic_model TEXT NOT NULL,
        scores_json  TEXT,
        reasoning_json TEXT,
        error        TEXT,
        created_at   TEXT NOT NULL,
        PRIMARY KEY (idea_model, paper_id, idea_index, critic_model, variant)
    );
    """)
    conn.commit()


def pick_model_ideas(n_per_model):
    """For each Static (track B) idea_model, deterministically pick n distinct
    ideas (lowest paper_id, then idea_index). Returns list of
    {idea_model, paper_id, idea_index, domain, idea_text}."""
    cr = sqlite3.connect(str(cfg.RESULTS_DB)); cr.row_factory = sqlite3.Row
    cp = sqlite3.connect(str(cfg.PAPERS_DB)); cp.row_factory = sqlite3.Row
    dom = {r["paper_id"]: r["domain"] for r in cp.execute("SELECT paper_id, domain FROM papers")}
    cp.close()
    rows = cr.execute(
        "SELECT DISTINCT idea_model, paper_id, idea_index FROM results "
        "WHERE track='B' AND idea_text IS NOT NULL AND idea_text!='' "
        "AND idea_model NOT IN ('baseline/gt') "
        "ORDER BY idea_model, paper_id, idea_index").fetchall()
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["idea_model"]].append((r["paper_id"], r["idea_index"]))
    picks = []
    for m, lst in by_model.items():
        for pid, idx in lst[:n_per_model]:
            t = cr.execute("SELECT idea_text FROM results WHERE idea_model=? AND paper_id=? "
                           "AND idea_index=? AND idea_text!='' LIMIT 1",
                           (m, pid, idx)).fetchone()
            if t and t["idea_text"]:
                picks.append({"idea_model": m, "paper_id": pid, "idea_index": idx,
                              "domain": dom.get(pid, "General Science"),
                              "idea_text": t["idea_text"]})
    cr.close()
    return picks


def do_score_models(variant, n_per_model, workers):
    """Score n ideas/model under the SAME variant + no-refs as the anchors, so
    anchors vs model ideas is apples-to-apples under one rubric."""
    system_prompt = SYS_BY_VARIANT[variant]
    picks = pick_model_ideas(n_per_model)
    nmodels = len(set(p["idea_model"] for p in picks))
    print(f"model ideas: {len(picks)} ({n_per_model}/model, {nmodels} models) variant={variant}")
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    ensure_model_table(conn)

    tasks = []
    for p in picks:
        for cr in MODERN_CRITICS:
            exists = conn.execute(
                "SELECT 1 FROM e10_model_idea_scores WHERE idea_model=? AND paper_id=? "
                "AND idea_index=? AND critic_model=? AND variant=?",
                (p["idea_model"], p["paper_id"], p["idea_index"], cr, variant)).fetchone()
            if not exists:
                tasks.append((p, cr))
    print(f"critic tasks: {len(tasks)}")
    if not tasks:
        return

    def _w(t):
        p, cr = t
        scores, raw = score_one(p["idea_text"], cr, p["domain"], system_prompt, "")
        return (p, cr, scores)

    ts = datetime.now(timezone.utc).isoformat()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_w, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            p, cr, scores = f.result()
            if scores:
                conn.execute("""INSERT OR IGNORE INTO e10_model_idea_scores
                    (idea_model, paper_id, idea_index, variant, critic_model,
                     scores_json, reasoning_json, error, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (p["idea_model"], p["paper_id"], p["idea_index"], variant, cr,
                     json.dumps({d: v["score"] for d, v in scores.items()}),
                     json.dumps({d: v["reasoning"] for d, v in scores.items()}),
                     None, ts))
                conn.commit(); ok += 1
            else:
                conn.execute("""INSERT OR IGNORE INTO e10_model_idea_scores
                    (idea_model, paper_id, idea_index, variant, critic_model,
                     scores_json, reasoning_json, error, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (p["idea_model"], p["paper_id"], p["idea_index"], variant, cr,
                     None, None, "parse_fail", ts))
                conn.commit(); err += 1
            if i % 25 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}")
    conn.close()
    print(f"done score-models variant={variant}: ok={ok} err={err}")


def do_compare(variant):
    """Compare anchors vs model ideas, BOTH under the same variant + no-refs."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    # anchors (exclude controls)
    acell = defaultdict(list)
    for r in conn.execute("SELECT anchor_id, scores_json FROM e10_idea_anchor_scores "
                          "WHERE variant=? AND scores_json IS NOT NULL", (variant,)):
        if r["anchor_id"].startswith("ctrl_"):
            continue
        acell[r["anchor_id"]].append(weighted_norm(json.loads(r["scores_json"])))
    anchors = {a: trimmed_mean(v) for a, v in acell.items() if trimmed_mean(v) is not None}

    # model ideas: per (model, paper, idea) trimmed_mean over critics, then per-model mean
    mcell = defaultdict(list)
    for r in conn.execute("SELECT idea_model, paper_id, idea_index, scores_json "
                          "FROM e10_model_idea_scores WHERE variant=? AND scores_json IS NOT NULL",
                          (variant,)):
        mcell[(r["idea_model"], r["paper_id"], r["idea_index"])].append(
            weighted_norm(json.loads(r["scores_json"])))
    per_idea = {k: trimmed_mean(v) for k, v in mcell.items() if trimmed_mean(v) is not None}
    model_means = defaultdict(list)
    for (m, p, i), sc in per_idea.items():
        model_means[m].append(sc)
    # exclude baseline/copy from the "model" distribution; report it separately
    model_gen = {m: float(np.mean(v)) for m, v in model_means.items() if not m.startswith("baseline/")}
    copy_vals = [sc for (m, p, i), sc in per_idea.items() if m == "baseline/copy"]

    def summ(xs):
        if not xs:
            return {"n": 0}
        a = np.array(xs)
        return {"n": len(xs), "mean": round(float(a.mean()), 3),
                "median": round(float(np.median(a)), 3),
                "std": round(float(a.std(ddof=1)), 3) if len(xs) > 1 else 0.0,
                "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
                "p90": round(float(np.percentile(a, 90)), 3)}

    bp = list(anchors.values())
    mg = list(model_gen.values())
    pi = list(per_idea.values())  # all individual model ideas (not per-model averaged)
    pi = [v for (m, p, i), v in per_idea.items() if not m.startswith("baseline/")]
    # Welch t-test anchors vs per-model-mean
    from scipy import stats as _st
    tt = _st.ttest_ind(bp, mg, equal_var=False) if bp and mg else None
    mwu = _st.mannwhitneyu(bp, mg, alternative="greater") if bp and mg else None

    result = {
        "variant": variant,
        "note": "anchors vs model ideas BOTH scored under this variant + no references (apples-to-apples)",
        "anchors": summ(bp),
        "model_ideas_per_model_mean": summ(mg),
        "model_ideas_individual": summ(pi),
        "copy_baseline": summ(copy_vals),
        "gap": {
            "anchor_mean_minus_model_mean": (round(np.mean(bp) - np.mean(mg), 3) if bp and mg else None),
            "anchor_max_minus_model_max": (round(max(bp) - max(mg), 3) if bp and mg else None),
            "anchor_min_minus_model_max": (round(min(bp) - max(mg), 3) if bp and mg else None),
            "frac_model_above_anchor_mean": (round(sum(1 for x in mg if x > np.mean(bp)) / len(mg), 3) if bp and mg else None),
            "welch_t": (round(float(tt.statistic), 3) if tt else None),
            "welch_p": (round(float(tt.pvalue), 4) if tt else None),
            "mannwhitney_p_anchor_greater": (round(float(mwu.pvalue), 4) if mwu else None),
        },
        "top5_models": sorted(model_gen.items(), key=lambda x: -x[1])[:5],
        "bottom5_models": sorted(model_gen.items(), key=lambda x: x[1])[:5],
    }
    conn.close()
    out = ROOT / "reports" / "e10_idea_anchor"; out.mkdir(parents=True, exist_ok=True)
    (out / f"e10_compare_{variant}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n✓ wrote {out}/e10_compare_{variant}.json")


def do_analyze(variant):
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    cell = defaultdict(list)
    dimcell = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in conn.execute(
            "SELECT * FROM e10_idea_anchor_scores WHERE variant=? AND scores_json IS NOT NULL",
            (variant,)):
        s = json.loads(r["scores_json"])
        cell[r["anchor_id"]].append(weighted_norm(s))
        for d in DIMS:
            if d in s:
                dimcell[r["anchor_id"]][d].append(s[d])
        meta[r["anchor_id"]] = r["title"]
    all_scores = {a: trimmed_mean(v) for a, v in cell.items() if trimmed_mean(v) is not None}
    anchor_scores = {a: s for a, s in all_scores.items() if not a.startswith("ctrl_")}
    control_scores = {a: s for a, s in all_scores.items() if a.startswith("ctrl_")}

    # per-anchor table
    per_anchor = []
    for a, sc in sorted(all_scores.items(), key=lambda x: -x[1]):
        dims = {d: round(float(np.mean(dimcell[a][d])), 2) for d in DIMS if dimcell[a][d]}
        per_anchor.append({"anchor": a, "title": meta[a][:55], "weighted": round(sc, 3), "dims": dims})

    # per-dimension overall mean (raw 1-10) — anchors only, excludes controls
    dim_overall = {}
    for d in DIMS:
        allv = [v for a in dimcell if not a.startswith("ctrl_") for v in dimcell[a][d]]
        dim_overall[d] = round(float(np.mean(allv)), 3) if allv else None

    # model-gen Static baseline (from uniform pool, per-model mean) — recompute
    ucell = defaultdict(list)
    for r in conn.execute("SELECT idea_model, paper_id, idea_index, scores_json "
                          "FROM uniform_critic_scores WHERE track='B' AND scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"])
        ucell[(r["idea_model"], r["paper_id"], r["idea_index"])].append(weighted_norm(s))
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
    mg = list(model_gen.values())
    conn.close()

    def summ(xs):
        if not xs:
            return {"n": 0}
        a = np.array(xs)
        return {"n": len(xs), "mean": round(float(a.mean()), 3),
                "median": round(float(np.median(a)), 3),
                "std": round(float(a.std(ddof=1)), 3) if len(xs) > 1 else 0.0,
                "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
                "p90": round(float(np.percentile(a, 90)), 3)}

    bp = list(anchor_scores.values())
    ct = list(control_scores.values())
    result = {
        "variant": variant,
        "idea_anchors": summ(bp),
        "weak_controls": summ(ct),
        "control_detail": {a: round(s, 3) for a, s in sorted(control_scores.items())},
        "model_generation_static_per_model": summ(mg),
        "discrimination": {
            "anchor_minus_model_gen_mean": (round(np.mean(bp) - np.mean(mg), 3) if bp and mg else None),
            "anchor_minus_control_mean": (round(np.mean(bp) - np.mean(ct), 3) if bp and ct else None),
            "anchor_max_minus_model_max": (round(max(bp) - max(mg), 3) if bp and mg else None),
            "anchor_breaks_6.5_frac": (round(sum(1 for x in bp if x > 6.5) / len(bp), 3) if bp else None),
            "anchor_breaks_7.0_frac": (round(sum(1 for x in bp if x > 7.0) / len(bp), 3) if bp else None),
            "model_gen_breaks_6.5_frac": (round(sum(1 for x in mg if x > 6.5) / len(mg), 3) if mg else None),
        },
        "dim_overall_raw_1to10": dim_overall,
        "per_anchor": per_anchor,
        "top_models_by_static": sorted(model_gen.items(), key=lambda x: -x[1])[:5],
    }
    out = ROOT / "reports" / "e10_idea_anchor"; out.mkdir(parents=True, exist_ok=True)
    (out / f"e10_{variant}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n✓ wrote {out}/e10_{variant}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--variant", default="prod", choices=["prod", "prod_refs", "fix_orig", "fix_idea"])
    ap.add_argument("--score-models", action="store_true",
                    help="score N ideas/model under --variant (no refs) for apples-to-apples compare")
    ap.add_argument("--compare", action="store_true",
                    help="compare anchors vs model ideas, both under --variant")
    ap.add_argument("--n-per-model", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not (args.score or args.analyze or args.score_models or args.compare):
        args.score = args.analyze = True
    if args.score:
        do_score(args.variant, args.workers)
    if args.analyze:
        do_analyze(args.variant)
    if args.score_models:
        do_score_models(args.variant, args.n_per_model, args.workers)
    if args.compare:
        do_compare(args.variant)


if __name__ == "__main__":
    main()
