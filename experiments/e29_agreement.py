"""E29 — Agreement analysis for the CS/AI pairwise human study.

Consumes the annotators' exported CSVs (from the blind HTML tool) plus the master
reports/e29_human_eval_pairs.json, and reports:
  (1) human-human agreement: Fleiss' kappa over 3 annotators/pair (ternary L/R/T and
      binary L/R with ties dropped), plus raw % agreement;
  (2) human-vs-critic agreement: majority human vote vs the lit8d critic's preference
      (critic_pref), as % agreement and Cohen's kappa, overall and by difficulty /
      confidence subset.

READ-ONLY. Writes reports/e29_agreement.json. /usr/bin/python3.

Usage:
  /usr/bin/python3 experiments/e29_agreement.py --labels reports/e29_labels/   # real CSVs
  /usr/bin/python3 experiments/e29_agreement.py --simulate                      # dry-run self-test
"""
import argparse, csv, json, glob, statistics
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
MASTER = OUT / "e29_human_eval_pairs_nlp.json"  # active NLP human-eval set (N29 ids); pass --master for the older CS set (E29)
CH = {"L": "LEFT", "R": "RIGHT", "T": "TIE"}


def fleiss_kappa(item_cat_counts, cats):
    """item_cat_counts: list of dict cat->count (same rater-count R per item). cats: list."""
    items = [ic for ic in item_cat_counts if sum(ic.values()) >= 2]
    if not items:
        return None
    R = sum(items[0].values())
    if any(sum(ic.values()) != R for ic in items) or R < 2:
        # allow variable R via per-item normalization (generalized Fleiss)
        pass
    N = len(items)
    # category marginals
    total = sum(sum(ic.values()) for ic in items)
    p = {c: sum(ic.get(c, 0) for ic in items) / total for c in cats}
    Pe = sum(v * v for v in p.values())
    Pis = []
    for ic in items:
        r = sum(ic.values())
        if r < 2:
            continue
        Pi = (sum(ic.get(c, 0) ** 2 for c in cats) - r) / (r * (r - 1))
        Pis.append(Pi)
    Pbar = statistics.mean(Pis)
    if abs(1 - Pe) < 1e-12:
        return None
    return (Pbar - Pe) / (1 - Pe)


def cohen_kappa(pairs, cats):
    """pairs: list of (a,b) categorical labels."""
    pairs = [(a, b) for a, b in pairs if a in cats and b in cats]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs); cb = Counter(b for _, b in pairs)
    pe = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in cats)
    if abs(1 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def majority(votes):
    """votes: list of 'L'/'R'/'T'. Return majority; ties in the vote -> 'T'."""
    c = Counter(votes)
    top = c.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return "T"


def load_labels_from_csv(paths):
    """returns votes[pair_id] = {annotator: {'choice','confidence'}}"""
    votes = defaultdict(dict)
    for pth in paths:
        with open(pth, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ch = (row.get("overall") or row.get("choice") or "").strip().upper()[:1]
                if ch in ("L", "R", "T"):
                    votes[row["pair_id"]][row.get("annotator", pth)] = {
                        "choice": ch, "confidence": (row.get("confidence") or "").strip()}
    return votes


def simulate(master, p_agree=0.82, p_tie=0.05, seed=7):
    """Synthetic 3-annotator votes correlated with critic_pref. NOT REAL DATA."""
    import random
    rnd = random.Random(seed)
    votes = defaultdict(dict)
    for p in master["pairs"]:
        truth = "L" if p["critic_pref"] == "LEFT" else "R"
        raters = p.get("annotators", ["A1", "A2", "A3"])
        for a in raters:
            u = rnd.random()
            if u < p_tie:
                ch = "T"
            elif u < p_tie + p_agree:
                ch = truth
            else:
                ch = "R" if truth == "L" else "L"
            conf = rnd.choice(["low", "med", "med", "high"])
            votes[p["pair_id"]][a] = {"choice": ch, "confidence": conf}
    return votes


def analyze(master, votes, simulated=False):
    pairs = {p["pair_id"]: p for p in master["pairs"]}
    # human-human
    tern_counts, bin_counts = [], []
    raw_agree = []
    for pid, av in votes.items():
        chs = [v["choice"] for v in av.values()]
        if len(chs) < 2:
            continue
        tern_counts.append(Counter(chs))
        bin_chs = [c for c in chs if c in ("L", "R")]
        if len(bin_chs) >= 2:
            bin_counts.append(Counter(bin_chs))
        # raw pairwise agreement among the raters of this item
        agr = sum(1 for i in range(len(chs)) for j in range(i + 1, len(chs)) if chs[i] == chs[j])
        tot = len(chs) * (len(chs) - 1) / 2
        raw_agree.append(agr / tot if tot else None)
    hh = dict(
        n_items=len(tern_counts),
        fleiss_ternary=fleiss_kappa(tern_counts, ["L", "R", "T"]),
        fleiss_binary_ties_dropped=fleiss_kappa(bin_counts, ["L", "R"]),
        raw_pct_agreement=statistics.mean([r for r in raw_agree if r is not None]) if raw_agree else None,
        tie_rate=sum(sum(1 for v in av.values() if v["choice"] == "T") for av in votes.values())
                 / max(1, sum(len(av) for av in votes.values())),
    )
    # human-vs-critic
    def hc(subset_filter=None, conf_min=None):
        hv, cv = [], []
        for pid, av in votes.items():
            if pid not in pairs:
                continue
            p = pairs[pid]
            if subset_filter and not subset_filter(p):
                continue
            chs = [v["choice"] for v in av.values()
                   if (conf_min is None or v.get("confidence") in conf_min)]
            if len(chs) < 1:
                continue
            mj = majority(chs)
            if mj == "T":
                continue  # exclude human-ties from binary critic comparison
            crit = "L" if p["critic_pref"] == "LEFT" else "R"
            hv.append(mj); cv.append(crit)
        n = len(hv)
        acc = sum(1 for a, b in zip(hv, cv) if a == b) / n if n else None
        return dict(n=n, pct_agreement=acc, cohen_kappa=cohen_kappa(list(zip(hv, cv)), ["L", "R"]))
    hvc = dict(
        overall=hc(),
        by_difficulty={d: hc(lambda p, d=d: p["difficulty"] == d) for d in ["moderate", "clear", "large"]},
        by_track_combo={k: hc(lambda p, k=k: p["track_combo"] == k) for k in ["BB", "BC", "CC"]},
        high_confidence_only=hc(conf_min={"high"}),
    )
    return dict(simulated=simulated, human_human=hh, human_vs_critic=hvc)


def load_llm_verdicts():
    """{pair_id: {judge: overall in LEFT/RIGHT/split}} from e29_pairwise_llm, if present."""
    import sqlite3
    db = ROOT / "data" / "results.db"
    try:
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT pair_id, judge_model, overall FROM e29_pairwise_llm "
                            "WHERE overall IS NOT NULL").fetchall()
        conn.close()
    except Exception:
        return {}
    out = defaultdict(dict)
    for pid, j, ov in rows:
        out[pid][j] = ov
    return out


def _llm_majority(verd):
    """majority over non-split judge verdicts; None if undecidable."""
    bin_v = [v for v in verd.values() if v in ("LEFT", "RIGHT")]
    if not bin_v:
        return None
    c = Counter(bin_v); top = c.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return None


def analyze_llm(master, llm, votes):
    pairs = {p["pair_id"]: p for p in master["pairs"]}
    if not llm:
        return {"note": "e29_pairwise_llm not populated yet (run e29_pairwise_llm.py --run)"}
    # LLM inter-judge agreement (ternary L/R/split, and binary dropping split)
    tern, binc = [], []
    split_n = tot = 0
    for pid, verd in llm.items():
        vs = list(verd.values())
        tot += len(vs); split_n += sum(1 for v in vs if v == "split")
        if len(vs) >= 2:
            tern.append(Counter({"LEFT": "L", "RIGHT": "R", "split": "T"}[v] for v in vs))
            b = [v for v in vs if v in ("LEFT", "RIGHT")]
            if len(b) >= 2:
                binc.append(Counter("L" if v == "LEFT" else "R" for v in b))
    # LLM-majority vs critic
    hv, cv = [], []
    for pid, verd in llm.items():
        mj = _llm_majority(verd)
        if mj and pid in pairs:
            hv.append("L" if mj == "LEFT" else "R")
            cv.append("L" if pairs[pid]["critic_pref"] == "LEFT" else "R")
    n = len(hv); acc = sum(1 for a, b in zip(hv, cv) if a == b) / n if n else None
    res = dict(
        n_pairs=len(llm), position_split_rate=(split_n / tot if tot else None),
        llm_inter_judge_fleiss_ternary=fleiss_kappa(tern, ["L", "R", "T"]),
        llm_inter_judge_fleiss_binary=fleiss_kappa(binc, ["L", "R"]),
        llm_majority_vs_critic=dict(n=n, pct_agreement=acc,
                                    cohen_kappa=cohen_kappa(list(zip(hv, cv)), ["L", "R"])),
    )
    # human-majority vs LLM-majority (only if human votes exist)
    if votes:
        h, l = [], []
        for pid, av in votes.items():
            if pid not in llm:
                continue
            hm = majority([v["choice"] for v in av.values()])
            lm = _llm_majority(llm[pid])
            if hm in ("L", "R") and lm:
                h.append(hm); l.append("L" if lm == "LEFT" else "R")
        n2 = len(h); a2 = sum(1 for a, b in zip(h, l) if a == b) / n2 if n2 else None
        res["human_majority_vs_llm_majority"] = dict(
            n=n2, pct_agreement=a2, cohen_kappa=cohen_kappa(list(zip(h, l)), ["L", "R"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="dir or glob of exported annotator CSVs")
    ap.add_argument("--simulate", action="store_true", help="dry-run on synthetic votes (NOT real data)")
    ap.add_argument("--master", default=str(MASTER),
                    help="pair master JSON (default: NLP set e29_human_eval_pairs_nlp.json / N29 ids; "
                         "pass e29_human_eval_pairs.json for the older CS set / E29 ids)")
    args = ap.parse_args()
    master = json.loads(Path(args.master).read_text())

    if args.simulate:
        votes = simulate(master)
        res = analyze(master, votes, simulated=True)
        print("### SIMULATED (synthetic annotators, p_agree=0.82) — NOT real human data ###")
    elif args.labels:
        paths = glob.glob(args.labels + "/*.csv") if Path(args.labels).is_dir() else glob.glob(args.labels)
        if not paths:
            print("no CSVs found at", args.labels); return
        votes = load_labels_from_csv(paths)
        if not votes:
            print("ERROR: 0 votes parsed. The CSV needs an 'overall' (or 'choice') column with L/R/T values."); return
        master_ids = {p["pair_id"] for p in master["pairs"]}
        overlap = len(set(votes) & master_ids)
        if overlap == 0:
            print(f"ERROR: 0/{len(votes)} labeled pair_ids match master {Path(args.master).name}.")
            print(f"  labels look like {sorted(votes)[:1]}, master like {sorted(master_ids)[:1]} -> wrong master.")
            print("  Pass --master reports/e29_human_eval_pairs_nlp.json (N29 ids) or e29_human_eval_pairs.json (E29 ids)."); return
        if overlap < len(votes):
            print(f"WARNING: {len(votes)-overlap}/{len(votes)} labeled pairs absent from this master (partial set or wrong master?).")
        res = analyze(master, votes, simulated=False)
        print(f"### REAL labels from {len(paths)} CSV files ({overlap}/{len(votes)} pairs matched master) ###")
    else:
        print("give --labels <dir> or --simulate"); return

    res["llm_judge"] = analyze_llm(master, load_llm_verdicts(), votes)
    print(json.dumps(res, indent=2))
    if not args.simulate:
        (OUT / "e29_agreement.json").write_text(json.dumps(res, indent=2))
        print("\nwrote", OUT / "e29_agreement.json")


if __name__ == "__main__":
    main()
