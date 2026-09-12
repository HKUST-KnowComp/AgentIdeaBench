"""
Leaderboard Generator  (Phase 4-B)

Reads model_scores from results.db, aggregates per model, and outputs:
  - leaderboard.md   (Markdown table, human-readable)
  - leaderboard.json (machine-readable, for downstream use)

Metrics shown:
  | Model | Track A (abs) | Track B (abs) | Context Gain | Overlap |

Usage:
    python analysis/leaderboard.py
    python analysis/leaderboard.py --output reports/
"""

import argparse
import json
import logging
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _std(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return round(var ** 0.5, 3)


def _fmt(val: Optional[float], decimals: int = 3) -> str:
    return f"{val:.{decimals}f}" if val is not None else "—"


def _fmt_with_std(mean_val: Optional[float], std_val: Optional[float]) -> str:
    if mean_val is None:
        return "—"
    if std_val is not None:
        return f"{mean_val:.3f}±{std_val:.2f}"
    return f"{mean_val:.3f}"


# ---------------------------------------------------------------------------
# Build leaderboard data
# ---------------------------------------------------------------------------

def build_leaderboard(db_results: str) -> List[dict]:
    """Return a list of dicts, one per model, with Static / Active / combined scores.

    Primary metric = Static Mode (Track B). Active Mode (Track C) shown as
    supplementary — it measures tool-use benefit but does NOT cleanly separate
    same-family models (small models gain more from search).
    """
    conn = sqlite3.connect(db_results)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM model_scores").fetchall()]
    conn.close()

    if not rows:
        logger.warning("model_scores table is empty — run compute_scores first")
        return []

    per_model: Dict[str, Dict] = defaultdict(lambda: {
        "static": [],   # Track B = Static Mode (given refs)
        "active": [],   # Track C = Active Mode (agent w/ search tools)
        "n_papers": set(),
    })

    for r in rows:
        m = r["idea_model"]
        t = r["track"]
        d = per_model[m]
        d["n_papers"].add(r["paper_id"])
        if r.get("mean_absolute_score") is not None:
            if t == "C":
                d["active"].append(r["mean_absolute_score"])
            else:
                # Treat any non-C track as Static (covers 'B' and any legacy rows)
                d["static"].append(r["mean_absolute_score"])

    leaderboard = []
    for model, d in per_model.items():
        static_mean = _mean(d["static"])
        active_mean = _mean(d["active"])
        gain = None
        if static_mean is not None and active_mean is not None:
            gain = round(active_mean - static_mean, 3)
        entry = {
            "model":         model,
            "model_short":   model.split("/")[-1],
            "n_papers":      len(d["n_papers"]),
            "static_score":  static_mean,
            "static_std":    _std(d["static"]),
            "active_score":  active_mean,
            "active_std":    _std(d["active"]),
            "tool_gain":     gain,
        }
        leaderboard.append(entry)

    # Sort by Static score (primary metric)
    leaderboard.sort(
        key=lambda x: x["static_score"] if x["static_score"] is not None else -99,
        reverse=True
    )
    return leaderboard


# ---------------------------------------------------------------------------
# Render Markdown
# ---------------------------------------------------------------------------

def render_markdown(leaderboard: List[dict]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# SciSynthBench — Leaderboard",
        f"\n_Generated: {ts}_\n",
        "## Static Mode — Primary Metric",
        "",
        "Models see the **survey refs** for their domain and propose a hypothesis. "
        "This is the canonical metric for model capability: it measures synthesis "
        "ability on a fixed literature context. **Ranked by Static score.**",
        "",
        "| # | Model | Papers | Static ↑ | Active | Δ tool |",
        "|---|-------|--------|----------|--------|--------|",
    ]

    for rank, entry in enumerate(leaderboard, 1):
        static = _fmt_with_std(entry["static_score"], entry.get("static_std"))
        active = _fmt_with_std(entry["active_score"], entry.get("active_std"))
        gain = entry.get("tool_gain")
        if gain is None:
            gain_str = "—"
        elif gain > 0:
            gain_str = f"+{gain:.2f}"
        else:
            gain_str = f"{gain:.2f}"
        lines.append(
            f"| {rank} | `{entry['model_short']}` "
            f"| {entry['n_papers']} "
            f"| {static} "
            f"| {active} "
            f"| {gain_str} |"
        )

    lines += [
        "",
        "## Metric Definitions",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| **Static ↑** | Primary metric. Model sees 20 survey refs for the domain, outputs a single 80-150 word hypothesis. Weighted by O×2+F+C×0.5+I×1.5+S×0.5. |",
        "| Active | Supplementary. Model gets only domain name + Semantic Scholar search tools; must discover refs itself. |",
        "| Δ tool | Active − Static. Positive = the model benefits from tool use (common in smaller models). |",
        "",
        "**Why Static is primary**: Active Mode scores can be inflated by smaller models that "
        "use search to retrieve specific terminology; this compresses same-family rankings "
        "(e.g. qwen 9b vs 397b). Static Mode isolates pure synthesis capability.",
        "",
        "**Weighted score formula**: Originality×2 + Feasibility×1 + Clarity×0.5 + Impact×1.5 + Specificity×0.5 "
        "(scientific innovation 64%, feasibility 18%, writing quality 18%).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_leaderboard(db_results: str, output_dir: str) -> None:
    leaderboard = build_leaderboard(db_results)
    if not leaderboard:
        logger.warning("No data to display")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = out / "leaderboard.md"
    md_path.write_text(render_markdown(leaderboard), encoding="utf-8")
    logger.info(f"Written: {md_path}")

    # JSON
    json_path = out / "leaderboard.json"
    json_path.write_text(
        json.dumps(leaderboard, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info(f"Written: {json_path}")

    # Print table to stdout
    print(render_markdown(leaderboard))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    import config as cfg

    parser = argparse.ArgumentParser(description="Generate SciSynthBench leaderboard")
    parser.add_argument("--results-db", default=str(cfg.RESULTS_DB))
    parser.add_argument("--output",     default=str(ROOT / "reports"))
    args = parser.parse_args()

    generate_leaderboard(args.results_db, args.output)
