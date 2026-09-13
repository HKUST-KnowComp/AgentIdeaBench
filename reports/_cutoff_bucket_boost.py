"""Per-cutoff-half-year bucket: mean Static, mean Active, paradigm advantage.

Buckets the 13 Active-paradigm models (v1_paper_refs) by knowledge cutoff
half-year, then reports mean Static (Track B), mean Active (Track C), and
the paradigm-level advantage (C-B) per bucket.

Uses raw weighted scores from `results` (not aggregated model_scores) so we
can keep paired (model, paper) structure for boost.
"""
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from reports._make_cross_year_plot import KNOWLEDGE_CUTOFFS, to_decimal_year

W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())

def half_year_bucket(decimal_year: float) -> str:
    """Map e.g. 2024.45 -> '2024 H1', 2024.75 -> '2024 H2'."""
    if decimal_year is None:
        return "unknown"
    year = int(decimal_year)
    frac = decimal_year - year
    half = "H1" if frac < 0.5 else "H2"
    return f"{year} {half}"

def weighted(s: dict) -> float:
    return sum(W[d] * s.get(d, 0) for d in W) / WS

def load_scores(track: str, prompt_version: str = "v1_paper_refs") -> dict:
    """Return {(model, paper_id): mean_weighted_score} averaged over critics."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    rows = conn.execute(
        "SELECT idea_model, paper_id, scores_json FROM results "
        "WHERE track=? AND prompt_version=? "
        "  AND critic_model != '' AND scores_json IS NOT NULL",
        (track, prompt_version),
    ).fetchall()
    conn.close()
    by_key = defaultdict(list)
    for m, pid, sj in rows:
        try:
            s = json.loads(sj)
        except Exception:
            continue
        by_key[(m, pid)].append(weighted(s))
    return {k: statistics.mean(v) for k, v in by_key.items() if v}

def main():
    static = load_scores("B")
    active = load_scores("C")

    # Build per-model paired data
    models_active = sorted({m for (m, _) in active})
    print(f"Active-paradigm models (v1_paper_refs): {len(models_active)}\n")

    # Per-bucket aggregation
    buckets = defaultdict(lambda: {
        "static_per_paper": [],  # all (model, paper) static weighted scores in bucket
        "active_per_paper": [],
        "paired_boost": [],      # per-(model, paper) Active - Static where both exist
        "models": set(),
    })

    # Aggregate over models that have BOTH Static and Active coverage
    for m in models_active:
        cutoff = KNOWLEDGE_CUTOFFS.get(m)
        if cutoff is None:
            continue
        b = half_year_bucket(cutoff)
        m_static_papers = {p: w for (mm, p), w in static.items() if mm == m}
        m_active_papers = {p: w for (mm, p), w in active.items() if mm == m}

        buckets[b]["models"].add(m)

        # Per-paper unpaired means
        for w in m_static_papers.values():
            buckets[b]["static_per_paper"].append(w)
        for w in m_active_papers.values():
            buckets[b]["active_per_paper"].append(w)

        # Paired boost (same model, same paper)
        for p, ws in m_static_papers.items():
            if p in m_active_papers:
                buckets[b]["paired_boost"].append(m_active_papers[p] - ws)

    # Display
    bucket_order = sorted(buckets.keys(),
                          key=lambda k: (int(k.split()[0]), k.split()[1]))

    header = (
        f"{'Bucket':<10} {'n_mod':>5} {'n_S':>5} {'n_A':>5} "
        f"{'meanStatic':>11} {'meanActive':>11} {'boost(paired)':>15} {'n_pairs':>8}"
    )
    print(header)
    print("-" * len(header))
    for b in bucket_order:
        d = buckets[b]
        n_mod = len(d["models"])
        n_S = len(d["static_per_paper"])
        n_A = len(d["active_per_paper"])
        mS = statistics.mean(d["static_per_paper"]) if d["static_per_paper"] else float("nan")
        mA = statistics.mean(d["active_per_paper"]) if d["active_per_paper"] else float("nan")
        if d["paired_boost"]:
            mB = statistics.mean(d["paired_boost"])
            n_pairs = len(d["paired_boost"])
            sd = statistics.stdev(d["paired_boost"]) if len(d["paired_boost"]) > 1 else 0.0
            sem = sd / (len(d["paired_boost"]) ** 0.5)
            ci95 = 1.96 * sem
            boost_str = f"{mB:+.2f} (±{ci95:.2f})"
        else:
            boost_str = "  n/a"
            n_pairs = 0
        print(f"{b:<10} {n_mod:>5} {n_S:>5} {n_A:>5} "
              f"{mS:>11.2f} {mA:>11.2f} {boost_str:>15} {n_pairs:>8}")

    print("\nPer-model paired boost (Active - Static) within each bucket:")
    # Need to recompute per-model
    for b in bucket_order:
        print(f"\n  {b} ({len(buckets[b]['models'])} models):")
        for m in sorted(buckets[b]["models"]):
            m_static = {p: w for (mm, p), w in static.items() if mm == m}
            m_active = {p: w for (mm, p), w in active.items() if mm == m}
            paired = [m_active[p] - m_static[p] for p in m_static if p in m_active]
            if not paired:
                continue
            cutoff = KNOWLEDGE_CUTOFFS[m]
            mean_b = statistics.mean(paired)
            mean_s = statistics.mean(m_static.values())
            mean_a = statistics.mean(m_active.values())
            print(f"    {m:<42} cutoff={cutoff:.2f}  "
                  f"Static={mean_s:.2f}  Active={mean_a:.2f}  "
                  f"boost={mean_b:+.2f}  (n_pairs={len(paired)})")

    # Also write a JSON for downstream use
    out = {}
    for b in bucket_order:
        d = buckets[b]
        out[b] = {
            "models": sorted(d["models"]),
            "n_models": len(d["models"]),
            "mean_static": (statistics.mean(d["static_per_paper"])
                            if d["static_per_paper"] else None),
            "mean_active": (statistics.mean(d["active_per_paper"])
                            if d["active_per_paper"] else None),
            "boost_paired_mean": (statistics.mean(d["paired_boost"])
                                  if d["paired_boost"] else None),
            "boost_paired_ci95_halfwidth": (
                1.96 * (statistics.stdev(d["paired_boost"])
                        / (len(d["paired_boost"]) ** 0.5))
                if len(d["paired_boost"]) > 1 else None
            ),
            "n_pairs": len(d["paired_boost"]),
            "n_static_paper_obs": len(d["static_per_paper"]),
            "n_active_paper_obs": len(d["active_per_paper"]),
        }
    out_path = ROOT / "reports" / "cutoff_bucket_boost.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
