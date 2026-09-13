"""
Pairwise Aggregation — win-rate matrix + Bradley-Terry ranking

Reads pairwise_results table and produces:
  - Win-rate matrix (pairs between models)
  - Overall win-rate per model
  - Bradley-Terry log-likelihood score per model (unnormalized ranking)

Output: reports/pairwise_active.md + pairwise_active.json
"""

import json
import logging
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _load_verdicts(db_results: str, track: str = "C") -> List[Tuple[str, str, str]]:
    """Return list of (model_a, model_b, winner) per pairwise decision.

    Ignores 'split' (position-bias disagreement).
    """
    conn = sqlite3.connect(db_results)
    rows = conn.execute("""
        SELECT model_a, model_b, overall_winner
        FROM pairwise_results
        WHERE track = ? AND overall_winner IN ('A', 'B')
    """, (track,)).fetchall()
    conn.close()
    out = []
    for ma, mb, w in rows:
        winner = ma if w == "A" else mb
        out.append((ma, mb, winner))
    return out


def win_rate_matrix(verdicts) -> Tuple[Dict, Dict]:
    """Compute win-rate matrix[model_a][model_b] = P(a beats b)."""
    wins = defaultdict(lambda: defaultdict(int))
    plays = defaultdict(lambda: defaultdict(int))
    for ma, mb, winner in verdicts:
        plays[ma][mb] += 1
        plays[mb][ma] += 1
        if winner == ma:
            wins[ma][mb] += 1
        else:
            wins[mb][ma] += 1

    rate = defaultdict(dict)
    models = sorted(set(plays.keys()) | set(wins.keys()))
    for m in models:
        for o in models:
            if m == o:
                rate[m][o] = None
                continue
            n = plays[m][o]
            rate[m][o] = wins[m][o] / n if n > 0 else None
    return rate, plays


def overall_win_rate(verdicts) -> Dict[str, float]:
    wins = defaultdict(int)
    plays = defaultdict(int)
    for ma, mb, winner in verdicts:
        plays[ma] += 1
        plays[mb] += 1
        wins[winner] += 1
    return {m: (wins[m] / plays[m] if plays[m] > 0 else 0.0) for m in plays}


def bradley_terry(verdicts, n_iters: int = 200, eps: float = 1e-6) -> Dict[str, float]:
    """Simple MM algorithm for Bradley-Terry strengths (log-scale)."""
    models = sorted({m for ma, mb, _ in verdicts for m in (ma, mb)})
    if not models:
        return {}
    strength = {m: 1.0 for m in models}

    wins = defaultdict(lambda: defaultdict(int))
    plays = defaultdict(lambda: defaultdict(int))
    for ma, mb, winner in verdicts:
        plays[ma][mb] += 1
        plays[mb][ma] += 1
        if winner == ma:
            wins[ma][mb] += 1
        else:
            wins[mb][ma] += 1

    for _ in range(n_iters):
        new_s = {}
        for m in models:
            num = sum(wins[m][o] for o in models if o != m)
            den = 0.0
            for o in models:
                if o == m: continue
                p = plays[m][o]
                if p > 0:
                    den += p / (strength[m] + strength[o])
            new_s[m] = (num / den) if den > 0 else strength[m]
        # Normalize (geometric mean = 1)
        geo = math.exp(sum(math.log(v + eps) for v in new_s.values()) / len(new_s))
        strength = {m: new_s[m] / geo for m in models}

    # Convert to log-scale (standard BT ranking metric)
    return {m: math.log(v + eps) for m, v in strength.items()}


def render_report(win_rate: Dict, overall: Dict, bt: Dict,
                  plays: Dict) -> str:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    models = sorted(overall.keys(), key=lambda m: bt.get(m, 0), reverse=True)

    lines = [
        "# Pairwise Leaderboard — Active Mode (Track C)",
        f"\n_Generated: {ts}_\n",
        "## Overall Ranking (by Bradley-Terry log-strength)",
        "",
        "| # | Model | BT score | Win-rate | N games |",
        "|---|-------|----------|----------|---------|",
    ]
    for rank, m in enumerate(models, 1):
        n_games = sum(plays[m].values())
        lines.append(
            f"| {rank} | `{m.split('/')[-1]}` "
            f"| {bt.get(m, 0):+.3f} "
            f"| {overall.get(m, 0):.1%} "
            f"| {n_games} |"
        )

    lines += [
        "",
        "## Win-rate Matrix — P(row beats column)",
        "",
        "| | " + " | ".join(f"`{m.split('/')[-1][:12]}`" for m in models) + " |",
        "|---|" + "|".join("---" for _ in models) + "|",
    ]
    for m in models:
        row = [f"`{m.split('/')[-1][:12]}`"]
        for o in models:
            if m == o:
                row.append("—")
            elif win_rate[m][o] is None:
                row.append("·")
            else:
                row.append(f"{win_rate[m][o]:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Methodology",
        "",
        "- Each ordered pair (A, B) evaluated twice: once A-first, once B-first (swap).",
        "- 'split' = A-first says A wins but B-first says B wins (position-bias disagreement) — excluded.",
        "- Winner = model that wins the `overall_winner` dimension (critic considers originality + feasibility + specificity).",
        "- Bradley-Terry: log-strength from pairwise outcomes; higher = stronger model.",
    ]
    return "\n".join(lines)


def aggregate_pairwise(db_results: str, out_dir: str, track: str = "C") -> None:
    verdicts = _load_verdicts(db_results, track)
    if not verdicts:
        logger.warning(f"No pairwise verdicts for track={track}")
        return

    win_rate, plays = win_rate_matrix(verdicts)
    overall = overall_win_rate(verdicts)
    bt = bradley_terry(verdicts)

    report = render_report(win_rate, overall, bt, plays)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"pairwise_{track.lower()}.md").write_text(report, encoding="utf-8")
    (out / f"pairwise_{track.lower()}.json").write_text(
        json.dumps({
            "track": track,
            "models": list(overall.keys()),
            "bradley_terry": bt,
            "overall_win_rate": overall,
            "n_verdicts": len(verdicts),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info(f"Pairwise report → {out}/pairwise_{track.lower()}.md")
    print(report)


if __name__ == "__main__":
    import sys, argparse
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="C")
    parser.add_argument("--output", default=str(Path(__file__).parent.parent / "reports"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    aggregate_pairwise(str(cfg.RESULTS_DB), args.output, args.track)
