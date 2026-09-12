"""Aggregate per-model latency / usage stats from results.telemetry.

Reads `results.db.results` rows (phase-2 ideation + phase-3 critic), extracts
the `telemetry` JSON column, and outputs:
  - reports/latency_<tag>.json   — full structured stats
  - reports/latency_<tag>.md     — markdown summary

Usage:
    python scripts/extract_telemetry_report.py
    python scripts/extract_telemetry_report.py --tag run-2026-05-17
    python scripts/extract_telemetry_report.py --since "2026-05-17 12:00"
"""
import argparse
import json
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg


def _percentile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))
    return s[idx]


def aggregate(db_path: str, since: str = None) -> dict:
    """Return {role: {model: stats}} where role ∈ {'idea','critic'}.

    Aggregates telemetry from THREE tables:
      results                       — main pipeline (phase 2 idea + phase 3 critic)
      dynamic_critic_sweep          — Sweep A (critic only, 3 modes)
      dynamic_critic_sweep_ideas    — Sweep A idea regen
      budget_sweep_scores           — Sweep B critic
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    buckets = {"idea": {}, "critic": {}}

    def add_row(model, role, telemetry_json):
        if not telemetry_json:
            return
        try:
            tel = json.loads(telemetry_json)
        except Exception:
            return
        lat = tel.get("latency_ms")
        if lat is None:
            return
        usage = tel.get("usage") or {}
        rec = (
            lat,
            usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0,
            usage.get("reasoning_tokens") or 0,
            tel.get("attempts", 1),
        )
        buckets[role].setdefault(model, []).append(rec)

    where_extra = "AND created_at >= ?" if since else ""
    params = [since] if since else []

    # results: idea (critic_model='') + critic (critic_model != '')
    for r in conn.execute(f"""
        SELECT idea_model, critic_model, telemetry FROM results
        WHERE telemetry IS NOT NULL {where_extra}
    """, params).fetchall():
        role = "critic" if r["critic_model"] else "idea"
        model = r["critic_model"] or r["idea_model"]
        add_row(model, role, r["telemetry"])

    # Sweep A: idea regen
    try:
        for r in conn.execute(f"""
            SELECT idea_model, telemetry FROM dynamic_critic_sweep_ideas
            WHERE telemetry IS NOT NULL {where_extra}
        """, params).fetchall():
            add_row(r["idea_model"], "idea", r["telemetry"])
    except sqlite3.OperationalError:
        pass

    # Sweep A: critic
    try:
        for r in conn.execute(f"""
            SELECT critic_model, telemetry FROM dynamic_critic_sweep
            WHERE telemetry IS NOT NULL {where_extra}
        """, params).fetchall():
            add_row(r["critic_model"], "critic", r["telemetry"])
    except sqlite3.OperationalError:
        pass

    # Sweep B: critic
    try:
        for r in conn.execute(f"""
            SELECT critic_model, telemetry FROM budget_sweep_scores
            WHERE telemetry IS NOT NULL {where_extra}
        """, params).fetchall():
            add_row(r["critic_model"], "critic", r["telemetry"])
    except sqlite3.OperationalError:
        pass

    # Phase 5 pairwise: telemetry is {latency_ms (sum of 2 calls), forward, swap}
    try:
        for r in conn.execute(f"""
            SELECT critic_model, telemetry FROM pairwise_results
            WHERE telemetry IS NOT NULL {where_extra}
        """, params).fetchall():
            add_row(r["critic_model"], "critic", r["telemetry"])
    except sqlite3.OperationalError:
        pass

    def summarise(records):
        lats = [r[0] for r in records]
        comp = [r[2] for r in records]
        reason = [r[3] for r in records]
        attempts = [r[4] for r in records]
        return {
            "n": len(records),
            "latency_ms_p50": int(statistics.median(lats)),
            "latency_ms_p95": int(_percentile(lats, 0.95)),
            "latency_ms_max": max(lats),
            "comp_tokens_mean": int(statistics.mean(comp)) if comp else 0,
            "reasoning_tokens_mean": int(statistics.mean(reason)) if reason else 0,
            "reasoning_tokens_max": max(reason) if reason else 0,
            "retry_rate": round(sum(1 for a in attempts if a > 1) / len(attempts), 3),
        }

    summary = {"idea": {}, "critic": {}}
    for role, models in buckets.items():
        for model, recs in models.items():
            summary[role][model] = summarise(recs)
    return summary


def render_markdown(summary: dict, tag: str) -> str:
    lines = [f"# Latency report — {tag}", ""]
    for role in ("idea", "critic"):
        if not summary[role]:
            continue
        lines.append(f"## {role.capitalize()} models")
        lines.append("")
        lines.append("| Model | n | p50 (ms) | p95 (ms) | max (ms) | comp_tok_mean | reasoning_tok_mean | retry_rate |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        rows = sorted(summary[role].items(), key=lambda kv: kv[1]["latency_ms_p50"])
        for m, s in rows:
            lines.append(
                f"| {m} | {s['n']} | {s['latency_ms_p50']} | {s['latency_ms_p95']} | "
                f"{s['latency_ms_max']} | {s['comp_tokens_mean']} | "
                f"{s['reasoning_tokens_mean']} | {s['retry_rate']:.1%} |"
            )
        lines.append("")

        lines.append("### Column Definitions")
        lines.append("")
        lines.append(
            "- `Model`: model identifier (idea generator or critic).\n"
            "- `n`: number of successful LLM calls with telemetry recorded.\n"
            "- `p50 (ms)`: median wall-clock latency of `chat.completions.create` (network + server). Lower is better.\n"
            "- `p95 (ms)`: 95th percentile latency.\n"
            "- `max (ms)`: worst-case latency observed.\n"
            "- `comp_tok_mean`: mean of `usage.completion_tokens`.\n"
            "- `reasoning_tok_mean`: mean of `usage.completion_tokens_details.reasoning_tokens` (>0 → model is running internal thinking).\n"
            "- `retry_rate`: fraction of calls that needed ≥2 attempts (parse failure or transient error)."
        )
        lines.append("")
        lines.append(f"Source: `data/results.db.results.telemetry` (JSON column). "
                     f"Filtered by `idea_model='' XOR critic_model=''`.")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(cfg.RESULTS_DB))
    ap.add_argument("--tag", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--since", default=None, help="ISO timestamp; only include rows created_at >= this")
    args = ap.parse_args()

    out_json = ROOT / "reports" / f"latency_{args.tag}.json"
    out_md = ROOT / "reports" / f"latency_{args.tag}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    summary = aggregate(args.db, since=args.since)
    out_json.write_text(json.dumps(summary, indent=2))
    out_md.write_text(render_markdown(summary, args.tag))

    print(f"✓ {out_json}")
    print(f"✓ {out_md}")

    # Print quick stdout summary
    for role in ("idea", "critic"):
        if not summary[role]:
            continue
        print(f"\n## {role} ({len(summary[role])} models)")
        for m, s in sorted(summary[role].items(), key=lambda kv: kv[1]["latency_ms_p50"]):
            print(f"  {m:<40} n={s['n']:<4} p50={s['latency_ms_p50']:>6}ms p95={s['latency_ms_p95']:>6}ms")


if __name__ == "__main__":
    main()
