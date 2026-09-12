"""E11 — Active vs Static reference overlap + idea↔refs embedding similarity.

User request:
  - For each Active idea, find which refs it actually SEARCHED/FETCHed; for each
    Static idea, the refs it was GIVEN. Quantify their OVERLAP.
  - Embedding cosine similarity:
    (1) Static idea  ↔ Static refs (the refs that paper's Static prompt gave)
    (2) Active idea  ↔ Static refs (same anchor paper's given refs)
    (3) Active idea  ↔ Active refs (the refs that Active idea actually retrieved)

Data sources:
  - Static refs given: papers.db ranked_refs_json (all ~20/paper, title+abstract);
    _format_refs puts ALL of them in the Track-B prompt.
  - Active refs used: results.raw_response telemetry → trace[].result_preview
    (JSON list of {paperId,title,abstract,...} from search_papers + get_paper_references).
  - Idea texts: results.idea_text (track B = Static, track C = Active).

Embeddings: sentence-transformers all-MiniLM-L6-v2, offline, normalized → cosine.
For "idea ↔ ref set" we report MEAN cosine over the set (primary) and MAX (closest).

Scope: models that have BOTH Active and Static (so 1/2/3 are comparable), on the
smoke25 anchor paper set.

SAFETY: read-only on results/papers; writes only reports/e11_*. No API calls.

Usage:
  /usr/bin/python3 experiments/e11_ref_overlap_embedding.py
  /usr/bin/python3 experiments/e11_ref_overlap_embedding.py --model all-MiniLM-L6-v2
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

PV = "v1_paper_refs"
_PID = re.compile(r'"paperId":\s*"([0-9a-f]{40})"')


def ref_text(d):
    t = (d.get("title") or "").strip()
    a = (d.get("abstract") or "").strip()
    return (t + ". " + a).strip() if a else t


def load_static_refs():
    """paper_id -> {ref_id: text, ...}  (the ~20 refs the Static prompt gave)."""
    cp = sqlite3.connect(str(cfg.PAPERS_DB)); cp.row_factory = sqlite3.Row
    out = {}
    for r in cp.execute("SELECT paper_id, ranked_refs_json FROM papers "
                        "WHERE status='filtered' AND ranked_refs_json IS NOT NULL"):
        try:
            refs = json.loads(r["ranked_refs_json"])
        except Exception:
            continue
        d = {}
        for ref in refs:
            if isinstance(ref, dict) and ref.get("paperId"):
                d[ref["paperId"]] = ref_text(ref)
        if d:
            out[r["paper_id"]] = d
    cp.close()
    return out


def load_active(open_only=True):
    """Returns active_refs[(m,p,idx)] = {ref_id: text}, active_text[(m,p,idx)]."""
    cr = sqlite3.connect(str(cfg.RESULTS_DB)); cr.row_factory = sqlite3.Row
    arefs = {}; atext = {}
    for r in cr.execute("SELECT idea_model,paper_id,idea_index,idea_text,raw_response "
                        "FROM results WHERE track='C' AND prompt_version=? AND critic_model='' "
                        "AND TRIM(idea_text)!='' AND raw_response IS NOT NULL", (PV,)):
        m = r["idea_model"]
        if open_only and (m.startswith("baseline/") or cfg.is_us_key_model(m)):
            continue
        try:
            tel = json.loads(r["raw_response"])
        except Exception:
            continue
        if not isinstance(tel, dict):
            continue
        refs = {}
        for t in tel.get("trace", []):
            if not isinstance(t, dict):
                continue
            rp = t.get("result_preview", "") or ""
            # parse JSON list if possible, else regex paperIds
            try:
                items = json.loads(rp)
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict) and it.get("paperId"):
                            refs[it["paperId"]] = ref_text(it)
                    continue
            except Exception:
                pass
            for pid in _PID.findall(rp):
                refs.setdefault(pid, "")
        key = (m, r["paper_id"], r["idea_index"])
        arefs[key] = refs
        atext[key] = r["idea_text"]
    cr.close()
    return arefs, atext


def load_static_ideas(models):
    cr = sqlite3.connect(str(cfg.RESULTS_DB)); cr.row_factory = sqlite3.Row
    out = {}
    qs = ",".join("?" * len(models))
    for r in cr.execute(f"SELECT idea_model,paper_id,idea_index,idea_text FROM results "
                        f"WHERE track='B' AND prompt_version='{PV}' AND critic_model='' "
                        f"AND TRIM(idea_text)!='' AND idea_model IN ({qs})", tuple(models)):
        out[(r["idea_model"], r["paper_id"], r["idea_index"])] = r["idea_text"]
    cr.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    print("loading data ...", flush=True)
    static_refs = load_static_refs()
    active_refs, active_text = load_active(open_only=True)
    models = sorted({k[0] for k in active_refs})
    static_ideas = load_static_ideas(models)
    print(f"models={len(models)} active_ideas={len(active_text)} static_ideas={len(static_ideas)} "
          f"papers_with_static_refs={len(static_refs)}", flush=True)

    # ---- build global text tables to embed (dedup) ----
    ref_text_by_id = {}          # ref_id -> text (static refs preferred for text)
    for p, d in static_refs.items():
        for rid, txt in d.items():
            if txt:
                ref_text_by_id[rid] = txt
    for key, d in active_refs.items():
        for rid, txt in d.items():
            if rid not in ref_text_by_id and txt:
                ref_text_by_id[rid] = txt
    ref_ids = [rid for rid, t in ref_text_by_id.items() if t]
    # NOTE: Active and Static share the same (model,paper,idx) keys, so we must
    # build texts/vectors by TRACK explicitly — never look up by bare key (that
    # would let an active text leak into a static slot).
    active_keys = list(active_text.keys())
    static_keys = list(static_ideas.keys())
    idea_keys = active_keys + static_keys
    idea_texts = ([active_text[k] for k in active_keys]
                  + [static_ideas[k] for k in static_keys])

    print(f"embedding {len(ref_ids)} unique refs + {len(idea_texts)} ideas ...", flush=True)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.model)
    refmat = enc.encode([ref_text_by_id[r] for r in ref_ids], batch_size=256,
                        show_progress_bar=True, normalize_embeddings=True)
    ideamat = enc.encode(idea_texts, batch_size=256, show_progress_bar=True,
                         normalize_embeddings=True)
    ref_vec = {rid: refmat[i] for i, rid in enumerate(ref_ids)}
    idea_vec = {}
    na = len(active_text)
    for i, k in enumerate(idea_keys):
        idea_vec[("C" if i < na else "B", k)] = ideamat[i]

    def setsim(vec, ref_id_list):
        ids = [r for r in ref_id_list if r in ref_vec]
        if vec is None or not ids:
            return None, None
        M = np.vstack([ref_vec[r] for r in ids])
        cs = M @ vec
        return float(cs.mean()), float(cs.max())

    # ---- per-idea metrics ----
    rows_overlap = []   # active ideas
    sim1 = defaultdict(list)  # model -> [mean cos] static idea vs static refs
    sim2 = defaultdict(list)  # active idea vs static refs
    sim3 = defaultdict(list)  # active idea vs active refs
    ov_shared = defaultdict(list); ov_jac = defaultdict(list); ov_nactive = defaultdict(list)

    # (1) static idea vs static refs
    for (m, p, idx), txt in static_ideas.items():
        if p not in static_refs:
            continue
        v = idea_vec.get(("B", (m, p, idx)))
        mc, _ = setsim(v, list(static_refs[p].keys()))
        if mc is not None:
            sim1[m].append(mc)

    # (2)(3) + overlap for active ideas
    for (m, p, idx), arefs in active_refs.items():
        v = idea_vec.get(("C", (m, p, idx)))
        sref_ids = set(static_refs.get(p, {}).keys())
        aref_ids = set(arefs.keys())
        # overlap
        shared = len(aref_ids & sref_ids)
        union = len(aref_ids | sref_ids)
        ov_shared[m].append(shared)
        ov_jac[m].append(shared / union if union else 0.0)
        ov_nactive[m].append(len(aref_ids))
        # (2) active idea vs static refs
        if sref_ids:
            mc2, _ = setsim(v, list(sref_ids))
            if mc2 is not None:
                sim2[m].append(mc2)
        # (3) active idea vs active refs
        if aref_ids:
            mc3, _ = setsim(v, list(aref_ids))
            if mc3 is not None:
                sim3[m].append(mc3)

    def agg(d):
        allv = [x for m in d for x in d[m]]
        return round(float(np.mean(allv)), 4) if allv else None

    overall = {
        "n_active_ideas": len(active_text),
        "n_static_ideas": len(static_ideas),
        "overlap_shared_refs_mean": agg(ov_shared),
        "overlap_jaccard_mean": agg(ov_jac),
        "active_refs_per_idea_mean": agg(ov_nactive),
        "sim1_static_idea_vs_static_refs": agg(sim1),
        "sim2_active_idea_vs_static_refs": agg(sim2),
        "sim3_active_idea_vs_active_refs": agg(sim3),
    }
    per_model = {}
    for m in models:
        per_model[m] = {
            "overlap_shared_mean": round(np.mean(ov_shared[m]), 3) if ov_shared[m] else None,
            "overlap_jaccard_mean": round(np.mean(ov_jac[m]), 4) if ov_jac[m] else None,
            "active_refs_per_idea": round(np.mean(ov_nactive[m]), 1) if ov_nactive[m] else None,
            "sim1_static_idea_vs_static_refs": round(np.mean(sim1[m]), 4) if sim1[m] else None,
            "sim2_active_idea_vs_static_refs": round(np.mean(sim2[m]), 4) if sim2[m] else None,
            "sim3_active_idea_vs_active_refs": round(np.mean(sim3[m]), 4) if sim3[m] else None,
        }

    out = {"embed_model": args.model, "overall": overall, "per_model": per_model}
    odir = ROOT / "reports" / "e11_ref_overlap"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "e11_ref_overlap_embedding.json").write_text(json.dumps(out, indent=2))

    print("\n=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    print("\n=== PER MODEL ===")
    print(f"{'model':<40}{'shared':>7}{'jac':>7}{'a_refs':>7}{'sim1':>7}{'sim2':>7}{'sim3':>7}")
    for m in models:
        d = per_model[m]
        def f(x): return f"{x:.3f}" if isinstance(x, float) else "-"
        print(f"{m:<40}{str(d['overlap_shared_mean']):>7}{f(d['overlap_jaccard_mean']):>7}"
              f"{str(d['active_refs_per_idea']):>7}{f(d['sim1_static_idea_vs_static_refs']):>7}"
              f"{f(d['sim2_active_idea_vs_static_refs']):>7}{f(d['sim3_active_idea_vs_active_refs']):>7}")
    print(f"\n✓ wrote {odir}/e11_ref_overlap_embedding.json")


if __name__ == "__main__":
    main()
