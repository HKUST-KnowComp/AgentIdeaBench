"""E26 — does a lightweight "Domain World-Model" scaffold boost hypothesis quality?

Isolation design: BOTH variants receive the SAME retrieved literature (a subdomain's
static reference set: title + abstract). Only the synthesis prompt differs, so any
score delta is attributable to the scaffold, not to different evidence.

  base : one-shot synthesis — "given these papers, propose one novel hypothesis".
  wm   : two-step (WorldLLM-lite, Levy et al. 2025 arXiv:2506.06725):
         (1) extract an explicit domain world-model from the papers —
             3-5 established mechanisms + 2-3 open tensions/gaps;
         (2) generate a hypothesis that EXPLOITS an open tension without
             contradicting an established mechanism, then self-check & revise.

Scoring reuses the exact lit8d critic pipeline (real-time Semantic Scholar prior-art
evidence + LIT8D_SYSTEM + 3 critics + weighted trimmed mean) so scores are comparable
to the main benchmark.

Raw-data rule: own tables `wm_ideas` / `wm_scores`; evidence cached in `e13_evidence`
under grp='wm'. Touches NO production idea/score tables. All stages idempotent.

Generators + critics are OPEN-SOURCE only (default OPENROUTER key), per the
smoke-test rule. Usage (staged):
  /usr/bin/python3 experiments/e26_worldmodel.py --gen [--models a,b] [--n-sub 6]
  /usr/bin/python3 experiments/e26_worldmodel.py --prep [--workers 4]
  /usr/bin/python3 experiments/e26_worldmodel.py --score [--critic-idx 0|1|2]
  /usr/bin/python3 experiments/e26_worldmodel.py --analyze
"""
import argparse, json, sqlite3, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as W
from experiments.e10_idea_anchor_calibration import score_one, trimmed_mean
from experiments.e13_litverify_rubric8 import (
    ensure_tables, build_evidence_for_item, format_evidence_block,
    LIT8D_SYSTEM, V4_IDEA_CUTOFF)

WSUM = sum(W.values())
DOMAINS = ["Biology", "CS", "Chemistry", "Medicine", "Physics"]
# open-source generators (default OPENROUTER key), mid-capability => room to improve
DEFAULT_GEN = ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b"]
VARIANTS = ("base", "wm", "analogy")
REF_TOPK = 8

SYS_SCI = ("You are a domain scientist proposing a single novel, testable scientific "
           "hypothesis. Output exactly one paragraph of 80-150 words, first-person "
           "future tense, starting with 'Hypothesis:'. No lists, no headers, no preamble.")


def _short(m):
    return m.split("/")[-1]


def _iid(short, variant, sub):
    return f"wm|{short}|{variant}|{sub[:40]}"


def refs_block(refs):
    lines = []
    for i, r in enumerate(refs[:REF_TOPK], 1):
        ab = (r.get("abstract") or "").strip().replace("\n", " ")
        if len(ab) > 700:
            ab = ab[:700] + "..."
        lines.append(f"[{i}] {r.get('title','(untitled)')} ({r.get('year','?')})\n{ab}")
    return "\n\n".join(lines) if lines else "(no background literature available)"


def pick_subs(conn, n_per_domain):
    """n_per_domain subdomains from each domain that have >=5 embedded refs, stable order."""
    out = []
    for dom in DOMAINS:
        rows = conn.execute(
            "SELECT subdomain, refs_json FROM subdomain_refs WHERE domain=? "
            "ORDER BY subdomain", (dom,)).fetchall()
        got = 0
        for sub, rj in rows:
            refs = json.loads(rj)
            if len([x for x in refs if x.get("abstract")]) >= 5:
                out.append((dom, sub, refs))
                got += 1
                if got >= n_per_domain:
                    break
    return out


def ensure_tbls(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS wm_ideas (
        gen_model TEXT, subdomain TEXT, domain TEXT, variant TEXT,
        idea_text TEXT, aux_json TEXT, created_at TEXT,
        PRIMARY KEY (gen_model, subdomain, variant));
    CREATE TABLE IF NOT EXISTS wm_scores (
        item_id TEXT, gen_model TEXT, subdomain TEXT, domain TEXT, variant TEXT,
        critic_model TEXT, scores_json TEXT, error TEXT, created_at TEXT,
        PRIMARY KEY (gen_model, subdomain, variant, critic_model));""")
    conn.commit()


# ---------------------------------------------------------------- generation
def _gen_base(llm, domain, rblock):
    p = (f"Research subfield: {domain}.\nBackground literature:\n\n{rblock}\n\n"
         "Propose ONE novel, testable hypothesis that goes BEYOND what these papers "
         "already establish. Output only the hypothesis paragraph.")
    return llm.generate_idea(p, system_prompt=SYS_SCI)["idea"].strip(), None


def _gen_wm(llm, domain, rblock):
    # step 1: extract the explicit domain world-model
    wm_prompt = (
        f"Research subfield: {domain}.\nLiterature:\n\n{rblock}\n\n"
        "Extract the current DOMAIN WORLD-MODEL as concise bullet points:\n"
        "A. ESTABLISHED (3-5 mechanisms/regularities the literature treats as true)\n"
        "B. OPEN TENSIONS (2-3 contradictions, gaps, or unresolved questions)\n"
        "Be specific and terse. Output only the two lists.")
    world = llm.completion(wm_prompt, system_prompt=(
        "You are a scientific analyst who distills a field into its core mechanisms "
        "and open problems."))
    if isinstance(world, tuple):
        world = world[0]
    world = (world or "").strip()
    # step 2: hypothesis conditioned on the world-model + self-check
    syn = (
        f"Research subfield: {domain}.\nLiterature:\n\n{rblock}\n\n"
        f"DOMAIN WORLD-MODEL you just built:\n{world}\n\n"
        "Now propose ONE novel, testable hypothesis that (i) EXPLOITS one OPEN TENSION "
        "and (ii) does NOT contradict any ESTABLISHED mechanism. Before answering, "
        "silently self-check the hypothesis against each item and revise if it fails. "
        "Output only the final hypothesis paragraph.")
    idea = llm.generate_idea(syn, system_prompt=SYS_SCI)["idea"].strip()
    return idea, {"world_model": world}


def _gen_analogy(llm, domain, rblock):
    # step 1: retrieve distant structural analogies (targets Originality — the
    # dimension the mechanism analysis shows grounding does NOT help). MOOSE-Chem /
    # combinatorial-creativity style (Yang et al. 2024; Chen et al. 2024).
    an_prompt = (
        f"Research subfield: {domain}.\nLiterature:\n\n{rblock}\n\n"
        "Identify the single core unsolved mechanism/problem in this literature. "
        "Then name 2-3 concepts, mechanisms, or mathematical structures from DISTANT, "
        "unrelated fields that are STRUCTURALLY analogous to that core problem. For "
        "each, state the analogy in one sentence. Output only the core problem + the "
        "2-3 cross-domain analogies.")
    an = llm.completion(an_prompt, system_prompt=(
        "You are a scientist skilled at analogical transfer across disciplines."))
    if isinstance(an, tuple):
        an = an[0]
    an = (an or "").strip()
    syn = (
        f"Research subfield: {domain}.\nLiterature:\n\n{rblock}\n\n"
        f"Cross-domain analogies you just found:\n{an}\n\n"
        "Now propose ONE novel, testable hypothesis that TRANSFERS one distant "
        "analogy into this subfield to attack its core problem in a way the existing "
        "papers do not. Make the transferred mechanism concrete and falsifiable. "
        "Output only the final hypothesis paragraph.")
    idea = llm.generate_idea(syn, system_prompt=SYS_SCI)["idea"].strip()
    return idea, {"analogies": an}


def gen(models, n_per_domain, variants=VARIANTS):
    from utils.LLM import IdeaLLM
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    ensure_tbls(conn)
    subs = pick_subs(conn, n_per_domain)
    print(f"{len(subs)} subdomains x {len(models)} models x {len(variants)} variants "
          f"{variants}", flush=True)
    ts = datetime.now(timezone.utc).isoformat()
    tasks = []
    for m in models:
        for dom, sub, refs in subs:
            for v in variants:
                if conn.execute("SELECT 1 FROM wm_ideas WHERE gen_model=? AND subdomain=? "
                                "AND variant=?", (m, sub, v)).fetchone():
                    continue
                tasks.append((m, dom, sub, refs, v))
    print(f"gen tasks: {len(tasks)}", flush=True)

    def _one(t):
        m, dom, sub, refs, v = t
        llm = IdeaLLM(model_name=m)
        rblock = refs_block(refs)
        try:
            if v == "base":
                idea, aux = _gen_base(llm, dom, rblock)
            elif v == "wm":
                idea, aux = _gen_wm(llm, dom, rblock)
            else:
                idea, aux = _gen_analogy(llm, dom, rblock)
            return m, dom, sub, v, idea, aux, None
        except Exception as e:
            return m, dom, sub, v, "", None, str(e)[:200]

    ok = err = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, f in enumerate(as_completed([ex.submit(_one, t) for t in tasks]), 1):
            m, dom, sub, v, idea, aux, e = f.result()
            conn.execute("INSERT OR IGNORE INTO wm_ideas VALUES (?,?,?,?,?,?,?)",
                         (m, sub, dom, v, idea, json.dumps(aux) if aux else None, ts))
            conn.commit()
            ok += 1 if idea else 0; err += 0 if idea else 1
            if i % 5 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close(); print(f"gen done ok={ok} err={err}")


# ---------------------------------------------------------------- evidence
def _items(conn):
    return conn.execute("SELECT gen_model, subdomain, domain, variant, idea_text "
                        "FROM wm_ideas WHERE TRIM(idea_text)!=''").fetchall()


def prep(workers=4):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    ensure_tables(conn); ensure_tbls(conn)
    items = _items(conn)
    todo = [it for it in items
            if not conn.execute("SELECT 1 FROM e13_evidence WHERE item_id=?",
                                (_iid(_short(it[0]), it[3], it[1]),)).fetchone()]
    print(f"{len(items)} ideas; {len(todo)} need evidence", flush=True)
    ts = datetime.now(timezone.utc).isoformat()

    def _ev(it):
        m, sub, dom, v, txt = it
        iid = _iid(_short(m), v, sub)
        q, ev, ns = build_evidence_for_item(iid, "wm", txt, V4_IDEA_CUTOFF, None, None)
        return iid, q, ev, ns

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_ev, it) for it in todo]):
            iid, q, ev, ns = f.result()
            conn.execute("INSERT OR IGNORE INTO e13_evidence VALUES (?,?,?,?,?,?,?,?)",
                         (iid, "wm", V4_IDEA_CUTOFF, json.dumps(q), json.dumps(ev), len(ev), ns, ts))
            conn.commit(); done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}]", flush=True)
    conn.close(); print("prep done")


# ---------------------------------------------------------------- scoring
def score(n_critics=3, workers=6, only_critic=None):
    conn = sqlite3.connect(str(cfg.RESULTS_DB), timeout=60)
    ensure_tbls(conn)
    critics = list(cfg.CRITIC_MODELS)[:n_critics]
    if only_critic is not None:
        critics = [critics[only_critic]]
    items = _items(conn)
    refs, miss = {}, 0
    for m, sub, dom, v, txt in items:
        iid = _iid(_short(m), v, sub)
        r = conn.execute("SELECT evidence_json, cutoff_date FROM e13_evidence WHERE item_id=?",
                         (iid,)).fetchone()
        if not r:
            miss += 1; continue
        refs[iid] = format_evidence_block(json.loads(r[0] or "[]"), r[1])
    if miss:
        print(f"WARN: {miss} ideas lack evidence (run --prep); scoring the rest")
    tasks = [(m, sub, dom, v, txt, cr)
             for m, sub, dom, v, txt in items if _iid(_short(m), v, sub) in refs
             for cr in critics
             if not conn.execute("SELECT 1 FROM wm_scores WHERE gen_model=? AND subdomain=? "
                                 "AND variant=? AND critic_model=?", (m, sub, v, cr)).fetchone()]
    print(f"score tasks: {len(tasks)}", flush=True)

    def _sc(t):
        m, sub, dom, v, txt, cr = t
        iid = _iid(_short(m), v, sub)
        s, _ = score_one(txt, cr, dom, LIT8D_SYSTEM, refs[iid])
        return iid, m, sub, dom, v, cr, s

    ts = datetime.now(timezone.utc).isoformat(); ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_sc, t) for t in tasks]), 1):
            iid, m, sub, dom, v, cr, s = f.result()
            conn.execute("INSERT OR IGNORE INTO wm_scores VALUES (?,?,?,?,?,?,?,?,?)",
                         (iid, m, sub, dom, v, cr, json.dumps(s) if s else None,
                          None if s else "parse_fail", ts))
            conn.commit(); ok += 1 if s else 0; err += 0 if s else 1
            if i % 20 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} err={err}", flush=True)
    conn.close(); print(f"score done ok={ok} err={err}")


# ---------------------------------------------------------------- analysis
def _weighted(scores):
    return sum(scores[d] * W[d] for d in DIMS) / WSUM


def analyze():
    from scipy import stats
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    # per (model, sub, variant): dim -> [critic scores]
    cell = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT gen_model,subdomain,domain,variant,scores_json "
                          "FROM wm_scores WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"])
        k = (r["gen_model"], r["subdomain"], r["variant"])
        for d in DIMS:
            cell[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()

    def tm(vs): return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)
    dim_cell = {k: {d: tm(cell[k][d]) for d in DIMS} for k in cell}
    w_cell = {k: _weighted(v) for k, v in dim_cell.items()}

    # group weighted + per-dim by (model, sub) -> variant
    pairs = defaultdict(dict)   # (m,sub) -> variant -> weighted
    dpair = defaultdict(dict)   # (m,sub) -> variant -> {dim: score}
    for (m, sub, v), w in w_cell.items():
        pairs[(m, sub)][v] = w
        dpair[(m, sub)][v] = dim_cell[(m, sub, v)]

    def compare(variant):
        both = [(k, d["base"], d[variant]) for k, d in pairs.items()
                if "base" in d and variant in d]
        if not both:
            return {"n_pairs": 0}
        base = np.array([b for _, b, _ in both]); alt = np.array([a for _, _, a in both])
        delta = alt - base
        r = {"n_pairs": len(both),
             "base_mean": round(float(base.mean()), 4),
             f"{variant}_mean": round(float(alt.mean()), 4),
             "delta_mean": round(float(delta.mean()), 4),
             "delta_win_rate": round(float((delta > 0).mean()), 4)}
        if len(both) >= 6:
            _, p = stats.wilcoxon(alt, base)
            r["wilcoxon_p"] = round(float(p), 4)
            lo, hi = np.percentile(
                [delta[np.random.default_rng(s).integers(0, len(delta), len(delta))].mean()
                 for s in range(2000)], [2.5, 97.5])
            r["delta_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
        pd = {}
        for d in DIMS:
            ds = [dd[variant][d] - dd["base"][d] for dd in dpair.values()
                  if "base" in dd and variant in dd
                  and dd["base"][d] is not None and dd[variant][d] is not None]
            pd[d] = round(float(np.mean(ds)), 4) if ds else None
        r["per_dim_delta"] = pd
        pm = defaultdict(list)
        for (k, b, a) in both:
            pm[k[0]].append(a - b)
        r["per_model_delta"] = {m: round(float(np.mean(v)), 4) for m, v in sorted(pm.items())}
        return r

    out = {"base_mean_overall": round(float(np.mean([pairs[k]["base"]
           for k in pairs if "base" in pairs[k]])), 4)}
    for v in VARIANTS:
        if v != "base":
            out[v] = compare(v)

    (ROOT / "reports" / "e26_worldmodel.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--models", default=",".join(DEFAULT_GEN))
    ap.add_argument("--n-sub", type=int, default=2, help="subdomains per domain")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma list from base,wm,analogy")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--critic-idx", type=int, default=None)
    a = ap.parse_args()
    models = [x.strip() for x in a.models.split(",") if x.strip()]
    variants = tuple(x.strip() for x in a.variants.split(",") if x.strip())
    if a.gen:
        gen(models, a.n_sub, variants=variants)
    if a.prep:
        prep(workers=a.workers)
    if a.score:
        score(workers=a.workers, only_critic=a.critic_idx)
    if a.analyze:
        analyze()
    if not any([a.gen, a.prep, a.score, a.analyze]):
        ap.print_help()


if __name__ == "__main__":
    main()
