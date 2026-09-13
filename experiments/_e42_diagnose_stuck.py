"""Diagnose the 39 E42 cells that three generation rounds could not fill.

Two distinct failure shapes came out of the miss-distribution:

  Track B: 9 cells = claude-sonnet-4-5 x 3 biomedical subdomains x all 3 idx.
           Track B never calls Semantic Scholar, so its only failure path is
           `not txt or len(txt.split()) < MIN_WORDS` in e41_frontier_extend.
           Deterministic across all idx and confined to one vendor -> suspect a
           provider-side content filter.

  Track C: 18 of 30 sit on the single subdomain "antiviral resistance
           infectious disease mechanism", spread over 9 models from two vendors.
           Cross-vendor -> suspect the SS query returns nothing, tripping the
           NORETR gate, rather than a model refusal.

This probes both directly and prints the raw provider response / raw SS payload.
Read-only: makes live API calls but writes nothing to any DB.
"""
import sys, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STUCK_B = ("azure/anthropic/claude-sonnet-4-5", [
    "antibody engineering affinity maturation therapeutic",
    "antiviral resistance infectious disease mechanism",
    "lipid nanoparticle mRNA delivery transfection",
])
STUCK_C_SUB = "antiviral resistance infectious disease mechanism"


def probe_track_b():
    import sqlite3
    import config as cfg
    from experiments.v3_subdomain_ideation import (
        _synthetic_paper, _build_generation_payload, _clean_idea_text)
    from experiments.e41_frontier_extend import _refs_map
    from utils.LLM import IdeaLLM
    model, subs = STUCK_B
    conn = sqlite3.connect("file:%s?mode=ro" % cfg.RESULTS_DB, uri=True)
    refs = _refs_map(conn)
    conn.close()
    print(f"=== Track B probe: {model} ===")
    for sub in subs:
        if sub not in refs:
            print(f"  {sub[:45]:45s}  NO REFS ENTRY"); continue
        dom, rf = refs[sub][0], refs[sub][1]
        paper = _synthetic_paper(dom, sub, rf)
        prompt, fb, system = _build_generation_payload(paper, "B")
        try:
            out = IdeaLLM(model_name=model).generate_idea(
                prompt, fallback_prompt=fb, system_prompt=system)
            raw = out.get("idea") or ""
            txt = _clean_idea_text(raw)
            print(f"  {sub[:45]:45s}  raw_len={len(raw):5d} words={len(txt.split()):4d} "
                  f"err={out.get('error')!r}")
            if len(txt.split()) < 40:
                print(f"      RAW: {raw[:400]!r}")
        except Exception as e:
            print(f"  {sub[:45]:45s}  EXC {type(e).__name__}: {str(e)[:200]}")


def probe_track_c_retrieval():
    from generation.active_agent import _search_papers as search_papers
    print(f"\n=== Track C retrieval probe: {STUCK_C_SUB!r} ===")
    queries = [
        STUCK_C_SUB,
        "antiviral resistance mechanism",
        "antiviral drug resistance mutation",
        "influenza antiviral resistance neuraminidase",
        "HIV drug resistance mutation mechanism",
    ]
    for q in queries:
        try:
            r = search_papers(q, limit=10)
            n = len(r) if isinstance(r, list) else -1
            prev = json.dumps(r, ensure_ascii=False)[:120] if r else "(empty)"
            print(f"  n={n:3d}  {q[:55]:55s}  {prev}")
        except Exception as e:
            print(f"  EXC  {q[:55]:55s}  {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "b"):
        probe_track_b()
    if which in ("both", "c"):
        probe_track_c_retrieval()
