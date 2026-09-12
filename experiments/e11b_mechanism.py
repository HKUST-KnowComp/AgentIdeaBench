"""E11b — WHY do Active's self-searched refs differ from Static's refs?

Tests the user's intuition ("static is just the top-n of a subdomain search, so it
should overlap with what Active searches"). Two facts re-frame it:
  - Static refs come in TWO flavors:
      v1_paper_refs : the anchor paper's OWN bibliography (/references).  [MAIN]
      v2_topic_refs : SS top-10 for the paper's TOPIC query (1 of 92).
  - Active is given only the BROAD domain (1 of 5: CS/Biology/Physics/Chemistry/
    Medicine) and invents its own sub-areas + query phrasings.

So we quantify, for the SUBDOMAIN framing too (v2 topic refs vs Active refs):
  - overlap (paperId intersection)
  - and the mechanism evidence: year & citationCount distributions of
    (a) static v1 bibliography refs, (b) static v2 topic-search refs,
    (c) active self-searched refs.

Outputs reports/e11_ref_overlap/e11b_mechanism.json (+ arrays for plotting).
Read-only; no API. Run with /usr/bin/python3.
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

PV = "v1_paper_refs"
_PID = re.compile(r'"paperId":\s*"([0-9a-f]{40})"')


def load_active_refs_full():
    """(model,paper,idx) -> {ref_id: {year,cite}} from telemetry traces (open models)."""
    cr = sqlite3.connect(str(cfg.RESULTS_DB)); cr.row_factory = sqlite3.Row
    out = {}
    for r in cr.execute("SELECT idea_model,paper_id,idea_index,raw_response FROM results "
                        "WHERE track='C' AND prompt_version=? AND critic_model='' "
                        "AND TRIM(idea_text)!='' AND raw_response IS NOT NULL", (PV,)):
        m = r["idea_model"]
        if m.startswith("baseline/") or cfg.is_us_key_model(m):
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
            try:
                items = json.loads(rp)
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict) and it.get("paperId"):
                            refs[it["paperId"]] = {"year": it.get("year"),
                                                   "cite": it.get("citationCount")}
                    continue
            except Exception:
                pass
            for pid in _PID.findall(rp):
                refs.setdefault(pid, {"year": None, "cite": None})
        out[(m, r["paper_id"], r["idea_index"])] = refs
    cr.close()
    return out


def main():
    cp = sqlite3.connect(str(cfg.PAPERS_DB)); cp.row_factory = sqlite3.Row
    cr = sqlite3.connect(str(cfg.RESULTS_DB)); cr.row_factory = sqlite3.Row

    # paper_id -> query (topic) and domain
    pmeta = {r["paper_id"]: (r["query"], r["domain"]) for r in
             cp.execute("SELECT paper_id, query, domain FROM papers WHERE status='filtered'")}

    # v1 bibliography refs per paper: {paper_id: {ref_id:{year,cite}}}
    v1 = {}
    for r in cp.execute("SELECT paper_id, ranked_refs_json FROM papers "
                        "WHERE status='filtered' AND ranked_refs_json IS NOT NULL"):
        try:
            refs = json.loads(r["ranked_refs_json"])
        except Exception:
            continue
        d = {}
        for ref in refs:
            if isinstance(ref, dict) and ref.get("paperId"):
                yr = ref.get("year")
                try:
                    yr = int(yr) if yr not in (None, "") else None
                except Exception:
                    yr = None
                ct = ref.get("citationCount")
                try:
                    ct = int(ct) if ct not in (None, "") else None
                except Exception:
                    ct = None
                d[ref["paperId"]] = {"year": yr, "cite": ct}
        v1[r["paper_id"]] = d

    # v2 topic refs per query: {query: {ref_id:{year,cite}}}
    v2q = {}
    for r in cr.execute("SELECT query, refs_json FROM domain_topic_refs"):
        try:
            refs = json.loads(r["refs_json"])
        except Exception:
            continue
        d = {}
        for ref in refs:
            if isinstance(ref, dict) and ref.get("paperId"):
                d[ref["paperId"]] = {"year": ref.get("year"), "cite": ref.get("citationCount")}
        v2q[r["query"]] = d

    active = load_active_refs_full()

    # ---- overlap: active vs v1 (bibliography) and active vs v2 (topic search) ----
    ov_v1, ov_v2 = [], []
    for (m, p, idx), arefs in active.items():
        aids = set(arefs.keys())
        s1 = set(v1.get(p, {}).keys())
        if s1:
            ov_v1.append(len(aids & s1) / len(aids | s1) if (aids | s1) else 0)
        q = pmeta.get(p, (None, None))[0]
        s2 = set(v2q.get(q, {}).keys()) if q else set()
        if s2:
            ov_v2.append(len(aids & s2) / len(aids | s2) if (aids | s2) else 0)

    # ---- year + citation distributions ----
    def yrs(dd):  # dd: dict ref_id->{year,cite}
        out = []
        for v in dd.values():
            y = v.get("year")
            try:
                y = int(y) if y not in (None, "") else None
            except Exception:
                y = None
            if y and 1950 < y <= 2026:
                out.append(y)
        return out

    def cites(dd):
        out = []
        for v in dd.values():
            c = v.get("cite")
            try:
                c = int(c) if c not in (None, "") else None
            except Exception:
                c = None
            if c is not None:
                out.append(c)
        return out

    v1_years, v1_cites = [], []
    for d in v1.values():
        v1_years += yrs(d); v1_cites += cites(d)
    v2_years, v2_cites = [], []
    for d in v2q.values():
        v2_years += yrs(d); v2_cites += cites(d)
    act_years, act_cites = [], []
    for d in active.values():
        act_years += yrs(d); act_cites += cites(d)

    def st(xs):
        a = np.array(xs, float)
        return {"n": len(xs), "mean": round(float(a.mean()), 2), "median": float(np.median(a)),
                "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75))} if len(xs) else {"n": 0}

    res = {
        "overlap_active_vs_v1_bibliography_jaccard_mean": round(float(np.mean(ov_v1)), 4) if ov_v1 else None,
        "overlap_active_vs_v2_topicsearch_jaccard_mean": round(float(np.mean(ov_v2)), 4) if ov_v2 else None,
        "n_active_ideas_v1cmp": len(ov_v1), "n_active_ideas_v2cmp": len(ov_v2),
        "year": {"static_v1_biblio": st(v1_years), "static_v2_topic": st(v2_years), "active": st(act_years)},
        "cite": {"static_v1_biblio": st(v1_cites), "static_v2_topic": st(v2_cites), "active": st(act_cites)},
        "n_topic_queries": len(v2q), "n_broad_domains": 5,
    }
    odir = ROOT / "reports" / "e11_ref_overlap"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "e11b_mechanism.json").write_text(json.dumps(res, indent=2))
    # also dump arrays for plotting
    np.savez(odir / "e11b_arrays.npz",
             v1_years=v1_years, v2_years=v2_years, act_years=act_years,
             v1_cites=v1_cites, v2_cites=v2_cites, act_cites=act_cites)
    print(json.dumps(res, indent=2))
    print(f"\n✓ wrote {odir}/e11b_mechanism.json (+ e11b_arrays.npz)")


if __name__ == "__main__":
    main()
