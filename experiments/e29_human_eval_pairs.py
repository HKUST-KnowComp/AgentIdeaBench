"""E29 — Build 100 CS/AI hypothesis PAIRS for a human pairwise-evaluation study.

Goal: validate the production lit8d critic against human judgment on CS/AI ideas.
We select 100 within-subfield idea pairs where the critic sees a clear, discriminative
difference (>=2 of 5 dimensions differ by >=0.75), spanning 8 CS/AI subfields and a mix
of Static/Active pairings. Humans will later judge left-vs-right (see reports/e29_*),
and we compute (a) human-human Fleiss kappa and (b) human-vs-critic agreement.

READ-ONLY on data/results.db (no writes to any experiment table). Deterministic
(fixed seed). Outputs to reports/ only (regenerable). /usr/bin/python3.

Aggregation matches the pipeline exactly:
  per-dim score = trimmed_mean over the 3 lit8d critics  (drop max, mean of rest)
  weighted total = sum(dim * W[dim]) / sum(W),  W = O2 F1 C0.5 I1.5 S0.5

Usage:
  /usr/bin/python3 experiments/e29_human_eval_pairs.py            # select + write JSON
  /usr/bin/python3 experiments/e29_human_eval_pairs.py --audit    # print distribution only
"""
import argparse, json, sqlite3, statistics, random, sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "results.db"
OUT = ROOT / "reports"
DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5, "impact": 1.5, "specificity": 0.5}
WSUM = sum(W.values())

# selection knobs
N_TARGET = 100
N_SUBFIELDS_CAP = None          # use all CS subfields available
PER_SUBFIELD_CAP = 13           # <= this many pairs per subfield (8 subfields * ~12-13 ~ 100)
MIN_DIMS_GE = 2                 # >= this many dims differ by >= DIM_GAP
DIM_GAP = 0.75                 # user's discrimination threshold
TOTAL_MIN = 0.40               # clear-ish weighted-total winner (so a directional critic pref exists)
IDEA_REUSE_CAP = 3             # an idea may appear in at most this many selected pairs
WORD_MIN, WORD_MAX = 25, 260   # keep well-formed hypotheses
SEED = 20260722


def trimmed_mean(vs):
    if not vs:
        return None
    if len(vs) < 2:
        return vs[0]
    return statistics.mean(sorted(vs)[:-1])   # drop the most generous critic


# NLP-focused subfields (annotators are NLP researchers). LLM-CoT reasoning is
# already in lit8d_scores_3seed; the other 5 were scored into e29_nlp_scores by
# experiments/e29_score_nlp.py (same lit8d pipeline). See --nlp mode.
NLP_SUBS_EXTRA = [
    "LLM agent planning tool use autonomy",
    "code generation large language model program synthesis",
    "retrieval augmented generation knowledge grounding",
    "transformer efficiency long context attention",
    "multimodal vision language model alignment",
]
NLP_SUB_EXISTING = "large language model reasoning chain of thought"  # in lit8d_scores_3seed


def _rows_query(conn, sql, params=()):
    return conn.execute(sql, params)


def load_nlp_ideas():
    """Like load_cs_ideas but for NLP subfields: 5 subs from e29_nlp_scores + the
    LLM-CoT sub from lit8d_scores_3seed. Returns the same idea[key] dict shape."""
    conn = sqlite3.connect(str(DB)); conn.row_factory = sqlite3.Row
    raw = defaultdict(lambda: defaultdict(list)); ncrit = defaultdict(set)

    def ingest(cursor):
        for r in cursor:
            key = (r["idea_model"], r["track"], r["subdomain"], r["idea_index"])
            ok = True
            try:
                s = json.loads(r["scores_json"])
            except Exception:
                continue
            for d in DIMS:
                v = s.get(d); v = v["score"] if isinstance(v, dict) else v
                if v is None:
                    ok = False; break
                raw[key][d].append(float(v))
            if ok:
                ncrit[key].add(r["critic_model"])

    ph = ",".join("?" * len(NLP_SUBS_EXTRA))
    ingest(conn.execute(
        "SELECT idea_model,track,subdomain,idea_index,critic_model,scores_json "
        "FROM e29_nlp_scores WHERE scores_json IS NOT NULL AND error IS NULL "
        f"AND subdomain IN ({ph})", NLP_SUBS_EXTRA))
    ingest(conn.execute(
        "SELECT idea_model,track,subdomain,idea_index,critic_model,scores_json "
        "FROM lit8d_scores_3seed WHERE scores_json IS NOT NULL AND error IS NULL "
        "AND subdomain=?", (NLP_SUB_EXISTING,)))
    return _finish_ideas(conn, raw, ncrit)


def load_cs_ideas():
    """idea[key] = dict(model,track,sub,idx, dims{d:score}, wtotal, text, n_crit)."""
    conn = sqlite3.connect(str(DB)); conn.row_factory = sqlite3.Row
    # 1) per-idea critic scores from lit8d_scores_3seed (CS only)
    raw = defaultdict(lambda: defaultdict(list))  # key -> dim -> [critic scores]
    ncrit = defaultdict(set)
    for r in conn.execute(
            "SELECT idea_model,track,subdomain,idea_index,critic_model,scores_json "
            "FROM lit8d_scores_3seed WHERE domain='CS' AND scores_json IS NOT NULL AND error IS NULL"):
        try:
            s = json.loads(r["scores_json"])
        except Exception:
            continue
        key = (r["idea_model"], r["track"], r["subdomain"], r["idea_index"])
        ok = True
        for d in DIMS:
            v = s.get(d)
            v = v["score"] if isinstance(v, dict) else v
            if v is None:
                ok = False; break
            raw[key][d].append(float(v))
        if ok:
            ncrit[key].add(r["critic_model"])
    return _finish_ideas(conn, raw, ncrit)


def _finish_ideas(conn, raw, ncrit):
    """Recover idea_text (exact query the scorer used), require full 3-critic
    coverage, aggregate to per-dim trimmed means + weighted total."""
    ideas = {}
    for key, dimmap in raw.items():
        m, tr, sub, idx = key
        if len(ncrit[key]) < 3:                       # require full 3-critic coverage
            continue
        if any(len(dimmap[d]) < 3 for d in DIMS):
            continue
        row = conn.execute(
            "SELECT idea_text FROM subdomain_ideas WHERE idea_model=? AND track=? "
            "AND subdomain=? AND idea_index=? AND TRIM(idea_text)!='' LIMIT 1",
            (m, tr, sub, idx)).fetchone()
        if not row:
            continue
        text = (row["idea_text"] or "").strip()
        nw = len(text.split())
        if not (WORD_MIN <= nw <= WORD_MAX):
            continue
        dims = {d: round(trimmed_mean(dimmap[d]), 4) for d in DIMS}
        wtot = round(sum(dims[d] * W[d] for d in DIMS) / WSUM, 4)
        ideas[key] = dict(model=m, track=tr, subfield=sub, idea_index=idx,
                          dims=dims, wtotal=wtot, text=text, words=nw)
    conn.close()
    return ideas


def candidate_pairs(ideas):
    """All within-subfield pairs meeting the discrimination filter."""
    by_sub = defaultdict(list)
    for key, it in ideas.items():
        by_sub[it["subfield"]].append(key)
    cands = []
    for sub, keys in by_sub.items():
        keys = sorted(keys)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = ideas[keys[i]], ideas[keys[j]]
                gaps = {d: abs(a["dims"][d] - b["dims"][d]) for d in DIMS}
                n_ge = sum(1 for d in DIMS if gaps[d] >= DIM_GAP)
                tgap = abs(a["wtotal"] - b["wtotal"])
                if n_ge >= MIN_DIMS_GE and tgap >= TOTAL_MIN:
                    cands.append(dict(sub=sub, ka=keys[i], kb=keys[j],
                                      n_dims_ge=n_ge, dim_gaps=gaps, total_gap=round(tgap, 4),
                                      pref="A" if a["wtotal"] > b["wtotal"] else "B",
                                      track_combo="".join(sorted([a["track"], b["track"]])),
                                      same_model=(a["model"] == b["model"])))
    return cands


def bucket(tg):
    if tg < 1.0: return "moderate"      # 0.40-1.0
    if tg < 2.0: return "clear"         # 1.0-2.0
    return "large"                      # >=2.0


DIFFS = ["moderate", "clear", "large"]
COMBOS = ["BB", "BC", "CC"]     # Static-Static, Static-Active, Active-Active


def select(cands, ideas):
    """Stratified selection balancing difficulty x track-combo x subfield; cap idea
    reuse; deterministic. 9 strata (3 difficulty x 3 combo) get ~equal quotas, and
    within each stratum we round-robin across the 8 subfields."""
    rnd = random.Random(SEED)
    for c in cands:
        c["_diff"] = bucket(c["total_gap"])
    rnd.shuffle(cands)  # break ties randomly but reproducibly

    # organize: stratum -> subfield -> [candidates]
    strata = defaultdict(lambda: defaultdict(list))
    for c in cands:
        strata[(c["_diff"], c["track_combo"])][c["sub"]].append(c)
    # within each (stratum, subfield), prefer more-discriminative pairs first
    for st in strata:
        for s in strata[st]:
            strata[st][s].sort(key=lambda c: (-c["n_dims_ge"], -c["total_gap"]))

    subs = sorted({c["sub"] for c in cands})
    # base quota per stratum, remainder spread over first strata deterministically
    base = N_TARGET // 9
    quotas = {}
    order = [(d, k) for d in DIFFS for k in COMBOS]
    for i, st in enumerate(order):
        quotas[st] = base + (1 if i < (N_TARGET - base * 9) else 0)

    per_sub = Counter(); reuse = Counter(); by_combo = Counter(); by_bucket = Counter()
    chosen = []
    ptr = defaultdict(int)  # (stratum, sub) -> next index

    def try_take(st, s):
        lst = strata[st].get(s, [])
        while ptr[(st, s)] < len(lst):
            c = lst[ptr[(st, s)]]; ptr[(st, s)] += 1
            if per_sub[s] >= PER_SUBFIELD_CAP: return False
            if reuse[c["ka"]] >= IDEA_REUSE_CAP or reuse[c["kb"]] >= IDEA_REUSE_CAP: continue
            chosen.append(c); per_sub[s] += 1
            reuse[c["ka"]] += 1; reuse[c["kb"]] += 1
            by_combo[c["track_combo"]] += 1; by_bucket[c["_diff"]] += 1
            return True
        return False

    # Pass 1: fill each stratum to its quota, round-robin across subfields
    for st in order:
        got = 0
        progressing = True
        while got < quotas[st] and progressing:
            progressing = False
            for s in subs:
                if got >= quotas[st]: break
                if try_take(st, s):
                    got += 1; progressing = True
    # Pass 2: top up to N_TARGET from any leftover, round-robin subfield then stratum
    progressing = True
    while len(chosen) < N_TARGET and progressing:
        progressing = False
        for s in subs:
            for st in order:
                if len(chosen) >= N_TARGET: break
                if try_take(st, s):
                    progressing = True
    return chosen, dict(per_sub=dict(per_sub), by_combo=dict(by_combo),
                        by_bucket=dict(by_bucket), reuse_max=max(reuse.values()) if reuse else 0)


def build_records(chosen, ideas, prefix="E29"):
    """Attach texts/scores, randomize left/right deterministically, blind annotator view."""
    rnd = random.Random(SEED + 1)
    recs = []
    for n, c in enumerate(sorted(chosen, key=lambda x: (x["sub"], x["total_gap"])), 1):
        a, b = ideas[c["ka"]], ideas[c["kb"]]
        # randomize which is Left
        left_is_a = rnd.random() < 0.5
        L, R = (a, b) if left_is_a else (b, a)
        pref_side = c["pref"]  # 'A' or 'B' (critic's higher-wtotal idea)
        critic_left_wins = (pref_side == "A") == left_is_a
        recs.append(dict(
            pair_id=f"{prefix}-{n:03d}",
            subfield=c["sub"],
            n_dims_ge_075=c["n_dims_ge"],
            total_gap=c["total_gap"],
            difficulty=bucket(c["total_gap"]),
            track_combo=c["track_combo"],
            same_model=c["same_model"],
            left=dict(model=L["model"], track=L["track"], idea_index=L["idea_index"],
                      wtotal=L["wtotal"], dims=L["dims"], words=L["words"], text=L["text"]),
            right=dict(model=R["model"], track=R["track"], idea_index=R["idea_index"],
                       wtotal=R["wtotal"], dims=R["dims"], words=R["words"], text=R["text"]),
            critic_pref="LEFT" if critic_left_wins else "RIGHT",   # ground-truth-by-critic
            dim_gaps={d: round(c["dim_gaps"][d], 3) for d in DIMS},
        ))
    return recs


def assign_annotators(recs):
    """Split 100 -> two sets of 50; 6 annotators (A1..A6); 3 annotators per pair.
    Set S1 = pairs 1..50 -> annotators A1,A2,A3 ; Set S2 = pairs 51..100 -> A4,A5,A6."""
    for i, r in enumerate(recs):
        if i < len(recs) // 2:
            r["annot_set"] = "S1"; r["annotators"] = ["A1", "A2", "A3"]
        else:
            r["annot_set"] = "S2"; r["annotators"] = ["A4", "A5", "A6"]
    return recs


def main():
    global PER_SUBFIELD_CAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true", help="print distributions only, no write")
    ap.add_argument("--nlp", action="store_true",
                    help="select from NLP subfields (e29_nlp_scores + LLM-CoT); "
                         "pair_id prefix N29; writes e29_human_eval_pairs_nlp.json")
    args = ap.parse_args()

    if args.nlp:
        ideas = load_nlp_ideas()
        prefix, outfile = "N29", "e29_human_eval_pairs_nlp.json"
        source = "e29_nlp_scores (5 NLP subs) + lit8d_scores_3seed (LLM-CoT)"
        PER_SUBFIELD_CAP = 18          # ~6 NLP subfields x 17 ~ 100
    else:
        ideas = load_cs_ideas()
        prefix, outfile = "E29", "e29_human_eval_pairs.json"
        source = "lit8d_scores_3seed (domain=CS)"
    subs = sorted({it["subfield"] for it in ideas.values()})
    print(f"{'NLP' if args.nlp else 'CS'} ideas (3-critic, well-formed): {len(ideas)} across {len(subs)} subfields")
    for s in subs:
        print(f"  - {s}: {sum(1 for it in ideas.values() if it['subfield']==s)} ideas")

    cands = candidate_pairs(ideas)
    print(f"\ncandidate pairs (>= {MIN_DIMS_GE} dims differ >= {DIM_GAP} & total_gap >= {TOTAL_MIN}): {len(cands)}")
    bc = Counter(bucket(c["total_gap"]) for c in cands)
    tc = Counter(c["track_combo"] for c in cands)
    print("  by difficulty:", dict(bc), "| by track combo:", dict(tc))
    per_sub_c = Counter(c["sub"] for c in cands)
    print("  candidates per subfield:", {s: per_sub_c[s] for s in subs})

    if args.audit:
        return

    chosen, stats = select(cands, ideas)
    recs = assign_annotators(build_records(chosen, ideas, prefix=prefix))
    print(f"\nselected {len(recs)} pairs")
    print("  per subfield:", stats["per_sub"])
    print("  by difficulty:", stats["by_bucket"])
    print("  by track combo (BB=Static-Static, BC=Static-Active, CC=Active-Active):", stats["by_combo"])
    print("  max idea reuse:", stats["reuse_max"])
    print("  critic prefers LEFT/RIGHT:", dict(Counter(r["critic_pref"] for r in recs)))

    OUT.mkdir(exist_ok=True)
    master = dict(
        meta=dict(n_pairs=len(recs), source=source, domain_focus=("NLP" if args.nlp else "CS/AI"),
                  critic_ensemble=["qwen/qwen3.6-plus", "moonshotai/kimi-k2.6", "z-ai/glm-5.1"],
                  aggregation="per-dim trimmed_mean(3 critics, drop max); weighted total O2/F1/C0.5/I1.5/S0.5 / 5.5",
                  filter=dict(min_dims_ge_075=MIN_DIMS_GE, dim_gap=DIM_GAP, total_min=TOTAL_MIN,
                              per_subfield_cap=PER_SUBFIELD_CAP, idea_reuse_cap=IDEA_REUSE_CAP),
                  seed=SEED, subfields=subs),
        pairs=recs)
    (OUT / outfile).write_text(json.dumps(master, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT / outfile}")


if __name__ == "__main__":
    main()
