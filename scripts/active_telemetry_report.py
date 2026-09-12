"""Active mode telemetry report.

Reads track='C' rows from results.db and produces per-idea-model statistics:

  1. Total context length distribution (prompt_tokens of the FINAL LLM call)
  2. Tool call count distribution
  3. Iteration count distribution
  4. Reasoning token usage per turn
  5. Budget-nudge frequency

Compatible with both legacy raw_response format (trace truncated to 4000 char
plus tool list) and the new full-telemetry format introduced 2026-05-05.

Output: prints a markdown report and saves JSON snapshot to
reports/active_telemetry.json.
"""
import sys, json, sqlite3, statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg


def parse_legacy_trace(raw):
    """Legacy raw_response: a JSON list of {iter, tool, args, result_preview}.
    No token info; only tool-call count and iter set are recoverable."""
    if not raw:
        return None
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        # Truncated; recover what we can
        last = raw.rfind('}')
        if last <= 0:
            return None
        try:
            arr = json.loads(raw[:last + 1] + ']')
        except Exception:
            return None
    if not isinstance(arr, list):
        return None
    iters = {t.get('iter') for t in arr if isinstance(t, dict)}
    return {
        "format": "legacy",
        "n_tool_calls": len(arr),
        "iters_used": len(iters),
        "tools": [t.get('tool') for t in arr if isinstance(t, dict)],
    }


def parse_new_telemetry(raw):
    """New raw_response: dict with trace + turns + iters_used + ..."""
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "turns" not in d:
        return None
    return {
        "format": "new",
        "n_tool_calls": d.get("n_tool_calls"),
        "iters_used": d.get("iters_used"),
        "n_turns": d.get("n_turns"),
        "final_prompt_tokens": d.get("final_prompt_tokens"),
        "final_total_tokens": d.get("final_total_tokens"),
        "budget_nudge_used": d.get("budget_nudge_used"),
        "turns": d.get("turns", []),
        "tools": [t.get("tool") for t in (d.get("trace") or []) if isinstance(t, dict)],
    }


def parse(raw):
    new = parse_new_telemetry(raw)
    if new:
        return new
    return parse_legacy_trace(raw)


def fmt_dist(vals, label):
    if not vals:
        return f"{label}: (no data)"
    return (f"{label:30s}  n={len(vals):3d}  "
            f"min={min(vals):>5}  p25={int(statistics.quantiles(vals, n=4)[0]) if len(vals) >= 4 else min(vals):>5}  "
            f"median={int(statistics.median(vals)):>5}  "
            f"p75={int(statistics.quantiles(vals, n=4)[2]) if len(vals) >= 4 else max(vals):>5}  "
            f"max={max(vals):>5}  mean={statistics.mean(vals):.1f}")


def main():
    conn = sqlite3.connect(cfg.RESULTS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT idea_model, paper_id, idea_index, raw_response "
        "FROM results WHERE track='C' AND critic_model='' "
        "AND idea_model NOT LIKE 'baseline/%'"
    ).fetchall()
    conn.close()

    by_model = defaultdict(list)
    for r in rows:
        parsed = parse(r['raw_response'])
        if parsed:
            by_model[r['idea_model']].append(parsed)

    print("# Active Mode Telemetry Report")
    print()
    print(f"Loaded {len(rows)} rows; {sum(len(v) for v in by_model.values())} parsed")
    fmt_counts = defaultdict(int)
    for v in by_model.values():
        for r in v:
            fmt_counts[r['format']] += 1
    print(f"Formats: {dict(fmt_counts)}")
    print()

    # Per-model statistics
    print("## Per-model summary")
    print()
    print(f"{'model':40s}  {'n':>3s}  {'tools(mean)':>11s}  {'iters(mean)':>11s}  "
          f"{'final_ctx(mean)':>15s}  {'nudge%':>6s}  {'fmt':>6s}")
    print("-" * 110)
    summary = {}
    for m in sorted(by_model):
        items = by_model[m]
        n = len(items)
        tools = [r['n_tool_calls'] for r in items if r.get('n_tool_calls') is not None]
        iters = [r['iters_used'] for r in items if r.get('iters_used') is not None]
        ctx = [r['final_prompt_tokens'] for r in items if r.get('final_prompt_tokens') is not None]
        nudge = [r.get('budget_nudge_used') for r in items if r.get('budget_nudge_used') is not None]
        nudge_pct = (sum(1 for x in nudge if x) / len(nudge) * 100) if nudge else None
        fmt = items[0]['format'] if items else "?"
        tool_mean = f"{statistics.mean(tools):.2f}" if tools else "-"
        iter_mean = f"{statistics.mean(iters):.2f}" if iters else "-"
        ctx_mean = f"{int(statistics.mean(ctx))}" if ctx else "(legacy)"
        nudge_str = f"{nudge_pct:.0f}%" if nudge_pct is not None else "-"
        print(f"{m:40s}  {n:>3d}  {tool_mean:>11s}  {iter_mean:>11s}  "
              f"{ctx_mean:>15s}  {nudge_str:>6s}  {fmt:>6s}")
        summary[m] = {
            "n": n, "format": fmt,
            "tools": tools, "iters": iters,
            "final_ctx": ctx, "nudge_pct": nudge_pct,
        }

    # Per-model distribution detail (only for new-format data)
    print()
    print("## Per-model distributions (new-format data only)")
    for m in sorted(by_model):
        items = by_model[m]
        if not items or items[0]['format'] != 'new':
            continue
        print()
        print(f"### {m}  (n={len(items)})")
        tools = [r['n_tool_calls'] for r in items if r.get('n_tool_calls') is not None]
        iters = [r['iters_used'] for r in items if r.get('iters_used') is not None]
        ctx = [r['final_prompt_tokens'] for r in items if r.get('final_prompt_tokens') is not None]
        if tools: print("  ", fmt_dist(tools, "tool calls"))
        if iters: print("  ", fmt_dist(iters, "iterations"))
        if ctx:   print("  ", fmt_dist(ctx,   "final ctx (prompt_tokens)"))
        # Per-turn token usage across all turns
        all_prompt = []
        all_reason = []
        for r in items:
            for t in r.get('turns', []) or []:
                pt = t.get('prompt_tokens')
                rt = t.get('reasoning_tokens')
                if pt is not None: all_prompt.append(pt)
                if rt is not None: all_reason.append(rt)
        if all_prompt: print("  ", fmt_dist(all_prompt, "all-turns prompt_tokens"))
        if all_reason: print("  ", fmt_dist(all_reason, "all-turns reasoning_tokens"))

    # Save snapshot
    out = ROOT / "reports" / "active_telemetry.json"
    out.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
