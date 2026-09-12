"""Bradley-Terry pairwise ranking validation against absolute weighted-score ranking.

If pairwise BT and absolute rankings agree, both methods converge on the same truth.
If they diverge, one (or both) is biased.

Track C (Active mode) only — that's where pairwise data exists.

Output: prints BT ranking + Spearman ρ between BT and absolute rankings.
"""
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in DIMS) / WS


def load_track_c_absolute():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    per_paper = defaultdict(lambda: defaultdict(list))
    for m, pid, sj in conn.execute("""
        SELECT idea_model, paper_id, scores_json FROM results
        WHERE prompt_version='v1_paper_refs' AND track='C' AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        per_paper[m][pid].append(weighted(s))
    conn.close()
    out = {}
    for m, pp in per_paper.items():
        vals = [statistics.mean(v) for v in pp.values()]
        if len(vals) >= 5:
            out[m] = statistics.mean(vals)
    return out


def load_pairwise_C():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    out = defaultdict(lambda: [0, 0, 0])  # (a, b) -> [a_wins, b_wins, ties]
    for ma, mb, w in conn.execute("""
        SELECT model_a, model_b, overall_winner FROM pairwise_results
        WHERE track='C' AND overall_winner IS NOT NULL
    """):
        if w == "A":
            out[(ma, mb)][0] += 1
        elif w == "B":
            out[(ma, mb)][1] += 1
        else:
            out[(ma, mb)][2] += 1
    conn.close()
    return out


def fit_bradley_terry(pairwise, n_iter=200):
    """Iteratively fit BT log-scores."""
    models = set()
    for (a, b) in pairwise:
        models.add(a); models.add(b)
    models = sorted(models)
    skill = {m: 0.0 for m in models}  # log-scale BT score; will be scaled at end

    # Build wins matrix W[a][b] = wins of a over b
    Wmat = defaultdict(lambda: defaultdict(float))
    for (a, b), (wa, wb, t) in pairwise.items():
        Wmat[a][b] += wa + t * 0.5
        Wmat[b][a] += wb + t * 0.5

    # MM algorithm
    for _ in range(n_iter):
        new_skill = {}
        for m in models:
            num = sum(Wmat[m][o] for o in models if o != m)
            denom = sum(
                (Wmat[m][o] + Wmat[o][m]) /
                (math.exp(skill[m]) + math.exp(skill[o]))
                for o in models if o != m and (Wmat[m][o] + Wmat[o][m]) > 0
            )
            if denom > 0 and num > 0:
                new_skill[m] = math.log(num / denom)
            else:
                new_skill[m] = skill[m]
        # Center
        c = sum(new_skill.values()) / len(new_skill)
        skill = {m: v - c for m, v in new_skill.items()}
    return skill


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    def ranks(vs):
        n = len(vs)
        si = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[si[j + 1]] == vs[si[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[si[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((r - mx) ** 2 for r in rx)
    dy = sum((r - my) ** 2 for r in ry)
    return num / (dx * dy) ** 0.5 if dx * dy > 0 else None


def main():
    print("=" * 76)
    print("BRADLEY-TERRY vs ABSOLUTE — Track C (Active) ranking consistency")
    print("=" * 76)

    absolute = load_track_c_absolute()
    pairwise = load_pairwise_C()
    print(f"Absolute scores: {len(absolute)} models")
    print(f"Pairwise games:  {sum(sum(v) for v in pairwise.values())} games across {len(pairwise)} ordered pairs")

    bt_skill = fit_bradley_terry(pairwise)
    common = sorted(set(absolute) & set(bt_skill))
    print(f"\nModels with both absolute + BT: {len(common)}")

    print(f"\n{'Model':<42} {'Abs':>7} {'BT':>7}")
    for m in sorted(common, key=lambda x: -absolute[x]):
        print(f"  {m:<40} {absolute[m]:>7.3f} {bt_skill[m]:>+7.3f}")

    abs_vals = [absolute[m] for m in common]
    bt_vals = [bt_skill[m] for m in common]
    rho = spearman(abs_vals, bt_vals)
    print(f"\nSpearman ρ(Absolute_C, BT_C) = {rho:+.3f}")
    if abs(rho) > 0.8:
        print("→ Strong agreement: both methods converge on the same ranking.")
    elif abs(rho) > 0.5:
        print("→ Moderate agreement.")
    else:
        print("→ Weak agreement: methods disagree.")


if __name__ == "__main__":
    main()
