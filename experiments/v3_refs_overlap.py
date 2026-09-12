"""v3 refs overlap + idea↔refs embedding similarity (subdomain design).

Same analysis as E11 but on the v3 subdomain data, where BOTH Static and Active
are rooted in the SAME subdomain:
  - Static refs  = subdomain_refs (SS top-10 for the subdomain query).
  - Active refs  = refs the agent retrieved (subdomain_ideas.telemetry trace),
                   now searching the SAME subdomain it was given.
Hypothesis (from E11): giving Active the subdomain (not the broad domain) should
RAISE overlap vs the old broad-domain Active.

Metrics (per the 27-model open roster):
  - overlap: Active refs vs Static refs (per subdomain) — Jaccard + shared count
  - sim1 Static idea ↔ Static refs ; sim2 Active idea ↔ Static refs ;
    sim3 Active idea ↔ Active refs   (mean cosine, MiniLM-L6-v2, normalized)

Read-only; writes reports/e11_ref_overlap/v3_refs_overlap.json. No API.
Run AFTER active generation. /usr/bin/python3 experiments/v3_refs_overlap.py
"""
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

_PID = re.compile(r'"paperId":\s*"([0-9a-f]{40})"')


def ref_text(d):
    t = (d.get("title") or "").strip()
    a = (d.get("abstract") or "").strip()
    return (t + ". " + a).strip() if a else t


def main():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    # static refs per subdomain
    sref = {}
    for r in conn.execute("SELECT subdomain, refs_json FROM subdomain_refs"):
        d = {}
        for ref in json.loads(r["refs_json"]):
            if isinstance(ref, dict) and ref.get("paperId"):
                d[ref["paperId"]] = ref_text(ref)
        sref[r["subdomain"]] = d

    # static + active ideas; active refs from telemetry
    static_idea = {}   # (model,sub) -> text
    active_idea = {}   # (model,sub) -> text
    active_refs = {}   # (model,sub) -> {ref_id:text}
    for r in conn.execute("SELECT idea_model,subdomain,track,idea_text,telemetry FROM subdomain_ideas "
                          "WHERE TRIM(idea_text)!=''"):
        key = (r["idea_model"], r["subdomain"])
        if r["track"] == "B":
            static_idea[key] = r["idea_text"]
        else:
            active_idea[key] = r["idea_text"]
            refs = {}
            if r["telemetry"]:
                try:
                    tel = json.loads(r["telemetry"])
                    for t in tel.get("trace", []):
                        rp = t.get("result_preview", "") or "" if isinstance(t, dict) else ""
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
                except Exception:
                    pass
            active_refs[key] = refs
    conn.close()

    # ---- embeddings ----
    ref_text_by_id = {}
    for d in sref.values():
        for rid, t in d.items():
            if t:
                ref_text_by_id[rid] = t
    for d in active_refs.values():
        for rid, t in d.items():
            if t and rid not in ref_text_by_id:
                ref_text_by_id[rid] = t
    ref_ids = [r for r, t in ref_text_by_id.items() if t]

    s_keys = list(static_idea.keys())
    a_keys = list(active_idea.keys())
    print(f"static_ideas={len(s_keys)} active_ideas={len(a_keys)} unique_refs={len(ref_ids)}", flush=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    rmat = enc.encode([ref_text_by_id[r] for r in ref_ids], batch_size=256,
                      show_progress_bar=True, normalize_embeddings=True)
    rvec = {rid: rmat[i] for i, rid in enumerate(ref_ids)}
    smat = enc.encode([static_idea[k] for k in s_keys], batch_size=256,
                      show_progress_bar=True, normalize_embeddings=True) if s_keys else np.zeros((0, 384))
    amat = enc.encode([active_idea[k] for k in a_keys], batch_size=256,
                      show_progress_bar=True, normalize_embeddings=True) if a_keys else np.zeros((0, 384))
    svec = {k: smat[i] for i, k in enumerate(s_keys)}
    avec = {k: amat[i] for i, k in enumerate(a_keys)}

    def setmean(v, ids):
        ids = [r for r in ids if r in rvec]
        if v is None or not ids:
            return None
        return float((np.vstack([rvec[r] for r in ids]) @ v).mean())

    sim1 = defaultdict(list); sim2 = defaultdict(list); sim3 = defaultdict(list)
    ov_jac = defaultdict(list); ov_shared = defaultdict(list); ov_n = defaultdict(list)

    for (m, sub), txt in static_idea.items():
        mc = setmean(svec.get((m, sub)), list(sref.get(sub, {}).keys()))
        if mc is not None:
            sim1[m].append(mc)
    for (m, sub), arefs in active_refs.items():
        sids = set(sref.get(sub, {}).keys()); aids = set(arefs.keys())
        if sids:
            sh = len(aids & sids); un = len(aids | sids)
            ov_jac[m].append(sh / un if un else 0); ov_shared[m].append(sh)
        ov_n[m].append(len(aids))
        v = avec.get((m, sub))
        if sids:
            mc = setmean(v, list(sids))
            if mc is not None:
                sim2[m].append(mc)
        if aids:
            mc = setmean(v, list(aids))
            if mc is not None:
                sim3[m].append(mc)

    def agg(d):
        allv = [x for m in d for x in d[m]]
        return round(float(np.mean(allv)), 4) if allv else None

    out = {
        "design": "v3 subdomain (Active given the SAME subdomain as Static)",
        "overall": {
            "overlap_jaccard_mean": agg(ov_jac),
            "overlap_shared_mean": agg(ov_shared),
            "active_refs_per_idea_mean": agg(ov_n),
            "sim1_static_idea_vs_static_refs": agg(sim1),
            "sim2_active_idea_vs_static_refs": agg(sim2),
            "sim3_active_idea_vs_active_refs": agg(sim3),
        },
        "compare_to_E11_broad_domain": {
            "E11_overlap_jaccard": 0.0008, "E11_sim1": 0.395, "E11_sim2": 0.183, "E11_sim3": 0.472,
        },
        "per_model": {m: {
            "overlap_jaccard": round(np.mean(ov_jac[m]), 4) if ov_jac[m] else None,
            "overlap_shared": round(np.mean(ov_shared[m]), 3) if ov_shared[m] else None,
            "sim1": round(np.mean(sim1[m]), 4) if sim1[m] else None,
            "sim2": round(np.mean(sim2[m]), 4) if sim2[m] else None,
            "sim3": round(np.mean(sim3[m]), 4) if sim3[m] else None,
        } for m in sorted(set(list(sim1) + list(sim3)))},
    }
    odir = ROOT / "reports" / "e11_ref_overlap"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "v3_refs_overlap.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["overall"], indent=2))
    print("\ncompare to E11 (broad-domain active):", json.dumps(out["compare_to_E11_broad_domain"]))
    print(f"\n✓ wrote {odir}/v3_refs_overlap.json")


if __name__ == "__main__":
    main()
