"""E13 — push bestpaper anchors toward ~8 via (1) idea-level ceiling unlock +
(2) literature-verified Originality with a date-filtered prior-art search.

Follows E12's finding (state.md F10): the anchor<v4-active gap is driven by
knowledge asymmetry / hindsight (critics recognize anchor ideas as "known" and
can't find prior art for v4's novel-sounding recombinations). Prompt-only
iterations plateaued at +0.5 avg / +0.32 vs strongest v4 active, and anchors
never approach 8 because IMPACT/FEASIBILITY 9-10 rungs demand executed /
Nobel-level work.

Two new variants (scored into e12_gap_scores, same item set as E12, seed=0):
  rubric8 : fix_calib2 + idea-level recalibration of the 9-10 rungs
            (direction 1 alone, no evidence — isolates the ceiling unlock)
  lit8    : rubric8 prompt + EVIDENCE-BASED ORIGINALITY. Each item gets a real
            Semantic Scholar prior-art search restricted to papers published
            BEFORE the idea was conceived (anchors: the anchor paper's own
            publicationDate, resolved via SS, self-paper excluded; v4 ideas:
            2026-05-31). The critic must judge Originality against this
            evidence and may NOT use its own memory to overrule it.

Pipeline:
  --resolve-anchors : SS-resolve each anchor's own paper -> publicationDate
                      (table e13_anchor_meta)
  --prep-evidence   : LLM query extraction (3 queries/item) + date-filtered SS
                      search, top-8 prior-art papers cached (table e13_evidence)
  --score --variant rubric8|lit8 : score the E12 item set (3 critics)
  --analyze --variant X [--weights prod|no_writing] : reuse E12 group analysis

SAFETY: INSERT-only into e12_gap_scores (new variant keys) + new tables
e13_anchor_meta / e13_evidence. No production code touched, no rows deleted.
All critics open-weight (default OPENROUTER_API_KEY).

Usage:
  /usr/bin/python3 experiments/e13_litverify_rubric8.py --resolve-anchors
  /usr/bin/python3 experiments/e13_litverify_rubric8.py --prep-evidence [--limit 3]
  /usr/bin/python3 experiments/e13_litverify_rubric8.py --score --variant rubric8
  /usr/bin/python3 experiments/e13_litverify_rubric8.py --score --variant lit8
  /usr/bin/python3 experiments/e13_litverify_rubric8.py --analyze --variant lit8 --weights no_writing
"""
import argparse
import json
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from experiments.e10_idea_anchor_calibration import score_one, parse_bestpapers
from experiments.e12_critic_gap_tuning import (
    build_items, do_analyze, FIX_CALIB2_SYSTEM,
)
from generation.active_agent import _ss_call, SS_BASE
from data_collection.fetch_ss_search import _get

V4_IDEA_CUTOFF = "2026-05-31"     # v4 subdomain ideas were generated 2026-06/07
ANCHOR_FALLBACK_CUTOFF = "2025-06-30"  # if an anchor's own paper can't be resolved
SS_BACKOFFS = [3, 6, 12]          # short schedule, matches active agent
EVIDENCE_TOP_N = 8
N_QUERIES = 3

# ---------------------------------------------------------------------------
# Variant prompts
# ---------------------------------------------------------------------------
_RUBRIC8 = """

IDEA-LEVEL SCALE RECALIBRATION (this is an IDEA benchmark — overrides the \
ladders above wherever they conflict):
The 9-10 rungs of IMPACT and FEASIBILITY above were written for completed \
research programs, which makes 9-10 structurally unreachable for a pure idea. \
Recalibrate as follows:
1. IMPACT 9-10 does NOT require Nobel/Turing-level or already-realized \
influence. An idea whose success would redefine the research agenda of its \
field — the caliber of an award-winning or oral paper at a top venue — \
deserves Impact 9-10. Impact 8 = would clearly influence work beyond its \
immediate sub-field.
2. FEASIBILITY 9-10 does NOT require that the study be executable in a week \
with minimal resources. Feasibility measures whether the DECISIVE test is \
realistically executable: 9-10 = the decisive experiment can clearly be run \
with existing tools, data, and ordinary lab/compute budgets as stated, even if \
the overall project is ambitious.
3. USE THE TOP OF THE SCALE. A genuinely field-defining idea should receive 9s \
and 10s on its strong dimensions. If you find yourself capping every proposal \
at 7 regardless of quality, you are compressing the scale and failing this \
calibration. Keep 5-6 for competent-but-incremental proposals — but do not let \
that band absorb the truly exceptional ones.
4. Calibration anchors: a landmark idea of award-paper caliber = Originality \
~9, Impact ~9; a competent incremental recombination = Originality ~5, Impact \
~5; vague boilerplate = 2-3. Score boldly at BOTH ends."""

_EVIDENCE_RULES = """

EVIDENCE-BASED ORIGINALITY (decisive — overrides memory-based judgment):
The "Background literature" below is NOT generic context. It is the output of \
a REAL prior-art search (Semantic Scholar), restricted to papers published \
BEFORE this proposal was conceived. Treat it as the factual record of what \
existed at proposal time:
1. If a retrieved paper already contains the proposal's CENTRAL mechanism or \
claim, score Originality 1-4 (restatement / minor twist) unless the proposal \
articulates a clear increment beyond that paper.
2. If the retrieved papers are merely ADJACENT (same area or shared \
components, but none contains the central move), then the central move was NOT \
in the prior art: score Originality 8-10 according to the size of the \
conceptual leap.
3. DO NOT let your own memory of the literature overrule this evidence. If the \
idea "feels familiar" or "is now standard practice" but NO retrieved paper \
contains it, that familiarity is hindsight from work published AFTER the \
proposal date — it must NOT lower Originality. Reviewers systematically \
under-score landmark ideas they recognize; this rule exists to stop that.
4. In your originality reasoning, cite the evidence items ([1], [2], ...) you \
compared the proposal against.
If the search returned no relevant prior art, say so and judge Originality by \
the size of the conceptual leap from standard practice in the field."""

_EVIDENCE_RULES_B = """

EVIDENCE-BASED ORIGINALITY (decisive — overrides memory-based judgment):
The "Background literature" below is NOT generic context. It is the output of \
a REAL prior-art search (Semantic Scholar), restricted to papers published \
BEFORE this proposal was conceived. Treat it as the factual record of the \
field at proposal time, and score Originality as the SIZE OF THE STEP the \
proposal takes beyond this evidence. Classify the relationship first:
1. CONTAINED (score 1-4): ONLY if you can QUOTE a passage from one evidence \
item that itself states the proposal's central claim or mechanism — not merely \
the same topic, task, components, or research line — AND the proposal adds no \
articulated claim beyond that passage. You MUST include the quote in your \
originality reasoning. If you cannot produce such a quote, you may NOT score \
below 5. Working on the same problem is NOT containment.
2. SAME LINE, NEW CLAIM (score 6-8): the evidence belongs to the same research \
line, but the proposal asserts a distinct central claim, unification, or \
capability that NO evidence item asserts — e.g. it generalizes a mechanism to \
a new regime, demonstrates or proves something the line only assumed, inverts \
the line's standard assumption, or unifies several evidence items under one \
mechanism. Score by step size; a step that would genuinely surprise \
researchers working in that line = 8.
3. ADJACENT ONLY (score 8-10): no evidence item shares the central move (they \
share only components, methods, or application area). Score by the size of \
the conceptual leap.
4. DO NOT let your own memory of the literature overrule this evidence. If the \
idea "feels familiar" or "is now standard practice" but no retrieved passage \
states it, that familiarity is hindsight from work published AFTER the \
proposal date — it must NOT lower Originality.
5. In your originality reasoning, name the classification (CONTAINED / SAME \
LINE / ADJACENT) and cite the evidence items ([1], [2], ...) you used.

EVIDENCE-BASED IMPACT: if validating the proposal would redirect or unify the \
research line visible in the evidence — i.e. the groups producing those papers \
would change what they work on or how — Impact 8-10 is warranted. \
Award-caliber ideas typically sit here; do not hold Impact at 6-7 for an idea \
whose success would reorganize its line of research.
If the search returned no relevant prior art, say so and judge Originality by \
the size of the conceptual leap from standard practice in the field."""

RUBRIC8_SYSTEM = FIX_CALIB2_SYSTEM + _RUBRIC8
LIT8_SYSTEM = RUBRIC8_SYSTEM + _EVIDENCE_RULES
LIT8B_SYSTEM = RUBRIC8_SYSTEM + _EVIDENCE_RULES_B

# lit8c: lit8b + SPECIFICITY/CLARITY measure the NEW CONTRIBUTION, not overall
# detail density. Diagnosis (E14/2026-07-02): under lit8b, v4 biomedical ideas
# beat human award anchors on Clarity (+1.0) and Specificity (+1.5) by packing
# named entities (enzymes, PTM sites, cohorts, cell lines) — often precisely
# RESTATING a mechanism that the prior-art evidence already contains. The rubric
# rewards text detail, which LLMs produce fluently, diluting the real O/I gap.
_SPEC_CLARITY_FIX = """

SPECIFICITY AND CLARITY MEASURE THE NEW CONTRIBUTION, NOT DETAIL DENSITY \
(decisive — apply using the prior-art evidence above):
A proposal can be dense with named entities (specific enzymes, modification \
sites, cohorts, cell lines, datasets, hyper-parameters) and still contribute \
almost nothing new — because those precise details RESTATE a mechanism the \
prior-art evidence already established. Named-entity density and protocol-style \
phrasing are NOT specificity or clarity.
- SPECIFICITY: score the conceptual precision of the proposal's NEW claim — the \
part that goes beyond the prior-art evidence. If the precise, technical portion \
of the proposal is a restatement of a mechanism already present in the evidence \
(e.g. an evidence item already names the same enzyme/site/effect), that \
precision earns NO specificity credit; specificity then rests only on how \
sharply the INCREMENT (what is new) is operationalized. A proposal whose new \
claim is vague ("this will reveal novel regulatory mechanisms") scores 3-4 on \
Specificity even if it is otherwise packed with known technical detail.
- CLARITY: score whether the NEW idea — the central insight and what would be \
done differently — is unambiguous. A long, jargon- and citation-dense proposal \
whose actual novel move is buried or hand-waved is NOT clear; a terse proposal \
whose new claim is crisp can be a 9. Do not reward length or fluency.
When the evidence shows the proposal is SAME LINE or CONTAINED, be especially \
strict on Specificity and Clarity: ask "how precisely is the NEW part pinned \
down?", not "how much technical detail is present?"."""
LIT8C_SYSTEM = LIT8B_SYSTEM + _SPEC_CLARITY_FIX

# lit8d: lit8c + two fixes so the remaining ceiling is removed symmetrically.
# (1) FEASIBILITY for theory/algorithmic ideas — the executed-lab-project ladder
#     under-scores award-caliber theory whose decisive test is a proof or a
#     standard benchmark, not a wet-lab program. (2) IMPACT is tied to the same
#     prior-art evidence as Originality: a SAME-LINE/CONTAINED idea cannot
#     reorganize a field it merely extends, so its Impact is capped; only an
#     ADJACENT (genuinely new) move earns Impact 8-10. This lowers the residual
#     Impact of v4's "restate a known mechanism" ideas while letting clean-move
#     award ideas reach the top.
_LIT8D = """

FEASIBILITY FOR THEORETICAL AND ALGORITHMIC IDEAS (recalibrate):
The feasibility ladder is written for empirical lab programs. For a theoretical, \
mathematical, or algorithmic proposal, the DECISIVE test is a proof, a \
derivation, or evaluation on standard public benchmarks/simulations — NOT a \
multi-year wet-lab or industrial-scale training run. Do not cap Feasibility at \
5-6 merely because the idea is theoretical or lacks large-scale experiments. \
Score 8-9 when the decisive test (the proof or the benchmark that would settle \
the claim) is clearly executable with ordinary academic resources — which is \
typical for a well-posed theory or algorithm proposal.

IMPACT IS CONSTRAINED BY THE PRIOR-ART EVIDENCE (apply with the Originality \
classification):
Impact measures whether validating the proposal would reorganize a field. An \
idea that only EXTENDS a research line it belongs to cannot reorganize that \
line. Therefore:
- CONTAINED: Impact at most 4 (it adds nothing the evidence did not already have).
- SAME LINE, NEW CLAIM: Impact at most 6-7 (a real but within-line contribution).
- ADJACENT / genuinely new move: Impact 8-10 IF validating it would change what \
researchers in the neighbouring lines work on. Reserve 9-10 for a move that \
would redirect or unify a field.
Do not grant Impact 8-10 to a fluently written proposal whose central claim the \
evidence shows is already established or merely extended."""
LIT8D_SYSTEM = LIT8C_SYSTEM + _LIT8D

E13_VARIANTS = {"rubric8": RUBRIC8_SYSTEM, "lit8": LIT8_SYSTEM,
                "lit8b": LIT8B_SYSTEM, "lit8c": LIT8C_SYSTEM,
                "lit8d": LIT8D_SYSTEM}

EXTRACT_SYSTEM = "You extract literature-search queries. Return ONLY JSON."
EXTRACT_TEMPLATE = """Given the research idea below, write {n} DIFFERENT keyword \
queries (each 4-10 words, plain keywords, no quotes or boolean operators) for \
finding PRIOR WORK that contains the idea's CENTRAL mechanism or claim, if such \
work exists.
Query 1: the central mechanism/claim stated as directly as possible.
Query 2: the same mechanism phrased with alternative standard terminology.
Query 3: the closest established research line this idea builds on.

Idea:
---
{idea}
---

Return ONLY: {{"queries": ["...", "...", "..."]}}"""


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def ensure_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS e13_anchor_meta (
        anchor_id TEXT PRIMARY KEY, title TEXT, ss_paper_id TEXT,
        publication_date TEXT, cutoff_date TEXT NOT NULL,
        resolved INTEGER NOT NULL, created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS e13_evidence (
        item_id TEXT PRIMARY KEY, grp TEXT, cutoff_date TEXT NOT NULL,
        queries_json TEXT, evidence_json TEXT,
        n_hits INTEGER, self_excluded INTEGER, created_at TEXT NOT NULL
    );""")
    conn.commit()


# ---------------------------------------------------------------------------
# SS helpers
# ---------------------------------------------------------------------------
def _norm_title(t):
    return set(re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split())


def _title_overlap(a, b):
    sa, sb = _norm_title(a), _norm_title(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ss_search(query, before_date=None, limit=10):
    """Relevance search with an optional 'published before' date filter."""
    api_key = getattr(cfg, "SEMANTIC_SCHOLAR_API_KEY", None)
    params = {
        "query": query,
        "fields": "paperId,title,abstract,year,citationCount,publicationDate",
        "limit": max(1, min(limit, 20)),
    }
    if before_date:
        params["publicationDateOrYear"] = f":{before_date}"
    data = _ss_call(lambda: _get(f"{SS_BASE}/paper/search", params, api_key,
                                 0.2, backoffs=SS_BACKOFFS))
    return [p for p in (data.get("data") or []) if p] if data else []


def resolve_anchors():
    """Find each anchor's own SS record -> publicationDate; cutoff = that date."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_tables(conn)
    ts = datetime.now(timezone.utc).isoformat()
    for a in parse_bestpapers():
        if conn.execute("SELECT 1 FROM e13_anchor_meta WHERE anchor_id=?",
                        (a["anchor_id"],)).fetchone():
            continue
        hits = ss_search(a["title"], before_date=None, limit=5)
        best, best_ov = None, 0.0
        for h in hits:
            ov = _title_overlap(a["title"], h.get("title"))
            if ov > best_ov:
                best, best_ov = h, ov
        if best and best_ov >= 0.6 and best.get("publicationDate"):
            cutoff = best["publicationDate"]
            conn.execute("INSERT OR IGNORE INTO e13_anchor_meta VALUES (?,?,?,?,?,?,?)",
                         (a["anchor_id"], a["title"], best.get("paperId"),
                          best["publicationDate"], cutoff, 1, ts))
            print(f"  {a['anchor_id']} resolved: {best['publicationDate']}  "
                  f"(overlap {best_ov:.2f}) {a['title'][:60]}")
        else:
            conn.execute("INSERT OR IGNORE INTO e13_anchor_meta VALUES (?,?,?,?,?,?,?)",
                         (a["anchor_id"], a["title"], None, None,
                          ANCHOR_FALLBACK_CUTOFF, 0, ts))
            print(f"  {a['anchor_id']} UNRESOLVED -> fallback cutoff "
                  f"{ANCHOR_FALLBACK_CUTOFF}  {a['title'][:60]}")
        conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Evidence prep
# ---------------------------------------------------------------------------
def extract_queries(idea_text):
    from utils.LLM import CriticLLM
    llm = CriticLLM(model_name=cfg.CRITIC_MODELS[0])
    prompt = EXTRACT_TEMPLATE.format(n=N_QUERIES, idea=idea_text.strip()[:4000])
    for _ in range(2):
        try:
            resp = llm.completion(prompt, system_prompt=EXTRACT_SYSTEM)
            if isinstance(resp, tuple):
                resp = resp[0]
            m = re.search(r"\{.*\}", resp or "", re.DOTALL)
            if m:
                qs = json.loads(m.group(0)).get("queries") or []
                qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
                if qs:
                    return qs[:N_QUERIES]
        except Exception:
            pass
    # fallback: first sentence as a single query
    first = re.split(r"[.\n]", idea_text.strip())[0]
    return [" ".join(first.split()[:10])]


def _item_cutoffs(conn):
    """anchor_id -> cutoff_date from e13_anchor_meta."""
    return {r[0]: r[1] for r in conn.execute(
        "SELECT anchor_id, cutoff_date FROM e13_anchor_meta")}


def _anchor_exclusion(conn):
    """anchor_id -> (own ss_paper_id, title) for self-exclusion."""
    return {r[0]: (r[1], r[2]) for r in conn.execute(
        "SELECT anchor_id, ss_paper_id, title FROM e13_anchor_meta")}


def build_evidence_for_item(item_id, grp, idea_text, cutoff, self_pid, self_title):
    queries = extract_queries(idea_text)
    per_query = [ss_search(q, before_date=cutoff, limit=10) for q in queries]
    seen, evidence, n_self = set(), [], 0
    # round-robin across queries, keeping SS relevance order within each
    for rank in range(10):
        for qi, hits in enumerate(per_query):
            if rank >= len(hits) or len(evidence) >= EVIDENCE_TOP_N:
                continue
            h = hits[rank]
            pid = h.get("paperId")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            if self_pid and pid == self_pid:
                n_self += 1
                continue
            if self_title and _title_overlap(self_title, h.get("title")) >= 0.55:
                n_self += 1
                continue
            evidence.append({
                "paperId": pid, "title": h.get("title"),
                "year": h.get("year"),
                "publicationDate": h.get("publicationDate"),
                "citationCount": h.get("citationCount") or 0,
                "abstract": (h.get("abstract") or "")[:450],
                "query": queries[qi],
            })
    return queries, evidence, n_self


def format_evidence_block(evidence, cutoff):
    if not evidence:
        return (f"PRIOR-ART SEARCH RESULTS (Semantic Scholar, restricted to "
                f"papers published before {cutoff}):\n"
                "(the prior-art search returned no relevant papers)")
    lines = [f"PRIOR-ART SEARCH RESULTS (Semantic Scholar, restricted to "
             f"papers published before {cutoff}):"]
    for i, e in enumerate(evidence, 1):
        ab = e["abstract"] or "(no abstract available)"
        lines.append(f"[{i}] ({e['year']}, cited {e['citationCount']}) "
                     f"{e['title']}\n    {ab}")
    return "\n".join(lines)


def prep_evidence(limit=None, verbose=False):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_tables(conn)
    cutoffs = _item_cutoffs(conn)
    excl = _anchor_exclusion(conn)
    if not cutoffs:
        print("no anchor meta — run --resolve-anchors first")
        conn.close()
        return
    items = build_items(per_group=25, seed=0)
    todo = [it for it in items if not conn.execute(
        "SELECT 1 FROM e13_evidence WHERE item_id=?", (it[0],)).fetchone()]
    if limit:
        todo = todo[:limit]
    print(f"evidence prep: {len(todo)} items to do (of {len(items)})")
    ts = datetime.now(timezone.utc).isoformat()
    for k, (iid, grp, model, tr, dom, txt) in enumerate(todo, 1):
        if grp == "anchor":
            cutoff = cutoffs.get(iid, ANCHOR_FALLBACK_CUTOFF)
            self_pid, self_title = excl.get(iid, (None, None))
        elif grp == "control":
            cutoff, self_pid, self_title = ANCHOR_FALLBACK_CUTOFF, None, None
        else:
            cutoff, self_pid, self_title = V4_IDEA_CUTOFF, None, None
        queries, evidence, n_self = build_evidence_for_item(
            iid, grp, txt, cutoff, self_pid, self_title)
        conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                     (iid, grp, cutoff, json.dumps(queries),
                      json.dumps(evidence), len(evidence), n_self, ts))
        conn.commit()
        print(f"  [{k}/{len(todo)}] {iid[:50]:<52} cutoff={cutoff} "
              f"hits={len(evidence)} self_excluded={n_self}", flush=True)
        if verbose:
            print("    queries:", queries)
            print(format_evidence_block(evidence, cutoff)[:1500])
    conn.close()


# ---------------------------------------------------------------------------
# Scoring (mirrors e12.do_score but with per-item references for lit8)
# ---------------------------------------------------------------------------
def do_score(variant, n_critics, workers):
    critics = list(cfg.CRITIC_MODELS)[:n_critics]
    system = E13_VARIANTS[variant]
    items = build_items(per_group=25, seed=0)
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_tables(conn)

    refs_by_item = {}
    if variant.startswith("lit"):
        rows = conn.execute(
            "SELECT item_id, evidence_json, cutoff_date FROM e13_evidence").fetchall()
        refs_by_item = {r[0]: format_evidence_block(json.loads(r[1] or "[]"), r[2])
                        for r in rows}
        missing = [it[0] for it in items if it[0] not in refs_by_item]
        if missing:
            print(f"ABORT: {len(missing)} items lack evidence "
                  f"(run --prep-evidence). e.g. {missing[:3]}")
            conn.close()
            return

    tasks = []
    for iid, grp, model, tr, dom, txt in items:
        for cr in critics:
            if not conn.execute(
                    "SELECT 1 FROM e12_gap_scores WHERE item_id=? AND variant=? AND critic_model=?",
                    (iid, variant, cr)).fetchone():
                tasks.append((iid, grp, model, tr, dom, txt, cr))
    print(f"variant={variant} items={len(items)} critics={len(critics)} tasks={len(tasks)}",
          flush=True)
    if not tasks:
        conn.close()
        return

    def _w(t):
        iid, grp, model, tr, dom, txt, cr = t
        scores, _ = score_one(txt, cr, dom, system, refs_by_item.get(iid, ""))
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
            ok += 1 if scores else 0
            err += 0 if scores else 1
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close()
    print(f"done: ok={ok} err={err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve-anchors", action="store_true")
    ap.add_argument("--prep-evidence", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--variant", default="lit8", choices=list(E13_VARIANTS))
    ap.add_argument("--weights", default="prod")
    ap.add_argument("--critics", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items in --prep-evidence (smoke)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.resolve_anchors:
        resolve_anchors()
    if a.prep_evidence:
        prep_evidence(limit=a.limit, verbose=a.verbose)
    if a.score:
        do_score(a.variant, a.critics, a.workers)
    if a.analyze:
        do_analyze(a.variant, a.weights)


if __name__ == "__main__":
    main()
