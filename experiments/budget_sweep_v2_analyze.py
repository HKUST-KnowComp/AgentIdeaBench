"""Aggregate sweep B (tool budget saturation) results.

For each (idea_model, budget):
  - mean weighted score (across paper × idea × critic, trimmed mean over critics)
  - mean actual n_tool_calls used
  - mean iters_used
  - budget utilisation = n_tool_calls / budget

Saturation judgement:
  score_delta(b1, b2) = mean_score(b2) - mean_score(b1)
  If score_delta(10, 15) < 0.5 * score_delta(5, 10), declare saturated at b=10.

Outputs: reports/sweep_budget.{json, md, zh.md}
"""
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
WEIGHTS = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
           "impact": 1.5, "specificity": 0.5}
WSUM = sum(WEIGHTS.values())

IDEA_MODELS = ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "qwen/qwen3.5-397b-a17b"]
BUDGETS = [1, 5, 10, 15, 20]   # 20 added 2026-06-23 per user E9 sweep request


def weighted_score(scores: dict) -> float:
    return sum(WEIGHTS[d] * scores.get(d, 0) for d in DIMS) / WSUM


def trimmed_mean(vs):
    if len(vs) < 2:
        return statistics.mean(vs) if vs else 0.0
    s = sorted(vs, reverse=True)
    return statistics.mean(s[1:])


def aggregate(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Pull ideas (for tool/iter metrics)
    ideas = conn.execute("""
        SELECT paper_id, idea_model, budget, idea_index, idea_text,
               n_tool_calls, iters_used, n_turns, budget_nudge
        FROM budget_sweep_ideas
    """).fetchall()
    ideas = [dict(r) for r in ideas]

    # Pull scores
    score_rows = conn.execute("""
        SELECT paper_id, idea_model, budget, idea_index, critic_model, scores_json
        FROM budget_sweep_scores
        WHERE scores_json IS NOT NULL
    """).fetchall()
    score_rows = [dict(r) for r in score_rows]

    # Group critic scores by (model, budget, paper_id, idea_index)
    score_by_combo = {}
    for r in score_rows:
        scores = json.loads(r["scores_json"])
        w = weighted_score(scores)
        key = (r["idea_model"], r["budget"], r["paper_id"], r["idea_index"])
        score_by_combo.setdefault(key, []).append(w)

    # Trimmed mean per (combo) → list of scores per (model, budget)
    per_mb = {m: {b: {"scores": [], "tool_calls": [], "iters": [],
                      "nudges": 0, "n_ideas": 0} for b in BUDGETS}
              for m in IDEA_MODELS}
    for key, critic_scores in score_by_combo.items():
        m, b, _, _ = key
        per_mb[m][b]["scores"].append(trimmed_mean(critic_scores))
    for idea in ideas:
        m, b = idea["idea_model"], idea["budget"]
        per_mb[m][b]["tool_calls"].append(idea["n_tool_calls"] or 0)
        per_mb[m][b]["iters"].append(idea["iters_used"] or 0)
        per_mb[m][b]["n_ideas"] += 1
        if idea["budget_nudge"]:
            per_mb[m][b]["nudges"] += 1

    summary = {}
    for m in IDEA_MODELS:
        summary[m] = {}
        for b in BUDGETS:
            d = per_mb[m][b]
            summary[m][b] = {
                "n_ideas": d["n_ideas"],
                "mean_score": (round(statistics.mean(d["scores"]), 4)
                               if d["scores"] else None),
                "mean_tool_calls": (round(statistics.mean(d["tool_calls"]), 2)
                                    if d["tool_calls"] else 0),
                "mean_iters": (round(statistics.mean(d["iters"]), 2)
                               if d["iters"] else 0),
                "budget_util": (round(statistics.mean(d["tool_calls"]) / b, 2)
                                if d["tool_calls"] and b else 0),
                "nudge_rate": (round(d["nudges"] / d["n_ideas"], 2)
                               if d["n_ideas"] else 0),
            }
        # Saturation judgement (generalised over the full budget ladder).
        # Walk consecutive (b_prev -> b) steps; the saturation point is the first
        # budget whose marginal gain drops below max(0.5 * previous gain, 0.05).
        series = [(b, summary[m][b]["mean_score"]) for b in BUDGETS
                  if summary[m][b]["mean_score"] is not None]
        deltas = []           # [(b_prev, b, gain)]
        for (bp, sp), (b, s) in zip(series, series[1:]):
            deltas.append((bp, b, round(s - sp, 4)))
        sat = None
        for i in range(1, len(deltas)):
            prev_gain = deltas[i - 1][2]
            cur_gain = deltas[i][2]
            if cur_gain < max(0.5 * prev_gain, 0.05):
                sat = f"saturated@{deltas[i][0]}"   # b_prev of the dropping step
                break
        if sat is None and deltas:
            sat = "still-growing"
        summary[m]["_saturation"] = sat
        summary[m]["_deltas"] = deltas

    return summary


def render_md(summary: dict, lang: str = "en") -> str:
    L = lang == "zh"
    lines = []
    title = ("Sweep B — Tool budget 饱和点测试 (报告)" if L
             else "Sweep B — Tool budget saturation")
    lines += [f"# {title}", ""]
    lines += ["## Scope" if not L else "## 范围",
              f"10 papers × 3 idea models (Qwen 3.5 9B/27B/397B) × {len(BUDGETS)} budgets {BUDGETS} × 3 ideas × 3 critics"]
    lines += ["", "## Data Sources",
              "- `data/results.db.budget_sweep_ideas` — active-agent runs with budget ∈ {1,5,10,15,20}",
              "- `data/results.db.budget_sweep_scores` — critic verdicts (static judge mode)",
              "- Note: qwen3.5-397b at budget=20 had a high malformed-output rate (only 15/30 valid ideas); its budget=20 cell is low-reliability."]
    lines += ["", ("## 一级结果（每模型每 budget 平均加权分数）"
                   if L else
                   "## Primary Results (mean weighted score per model × budget)")]
    lines += [""]

    for m in IDEA_MODELS:
        lines += [f"### {m}", ""]
        lines += ["| Budget | n_ideas | mean_score | mean_tool_calls | budget_util | nudge_rate |"]
        lines += ["|---:|---:|---:|---:|---:|---:|"]
        for b in BUDGETS:
            s = summary[m][b]
            lines.append(
                f"| {b} | {s['n_ideas']} | {s['mean_score']} | "
                f"{s['mean_tool_calls']} | {s['budget_util']:.0%} | {s['nudge_rate']:.0%} |"
            )
        sat = summary[m]["_saturation"]
        deltas = summary[m].get("_deltas", [])
        delta_str = ", ".join(f"{bp}->{b}:{g:+.3f}" for bp, b, g in deltas)
        lines += [f"\n**Saturation**: `{sat}`  | marginal Δscore: {delta_str}", ""]

    lines += ["### Column Definitions" if not L else "### 字段定义"]
    if L:
        defs = [
            "- `Budget`：active agent 的 max_iters（一轮最多 emit 多少 tool call）。范围 {1,5,10,15,20}。",
            "- `n_ideas`：该 (model, budget) 下生成的 idea 数量（10 paper × 3 idea/combo = 30）。",
            "- `mean_score`：该 (model, budget) 下 idea 的平均加权分数。计算：每 idea × 3 critic 加权打分 = O×2 + F + C×0.5 + I×1.5 + S×0.5，per-idea critic 间去最高分取均值，再在 idea 上取均值。范围 [0, 10]，越大越好。",
            "- `mean_tool_calls`：该 (model, budget) 下 idea 实际用的 tool call 次数均值（从 active_agent.trace 计）。",
            "- `budget_util`：mean_tool_calls / budget，反映模型对预算的实际利用率。",
            "- `nudge_rate`：触发 'you've reached budget' 提示的 idea 占比；接近 0 说明模型自己早停。",
            "- `Saturation`：沿完整 budget 阶梯逐步比较边际增益的启发式。从第二段起，若某段边际 Δscore < max(0.5×上一段增益, 0.05)，判该段起点为饱和点（`saturated@B`）；否则 `still-growing`。`marginal Δscore` 给出每相邻段的实际增量。n=30/cell，方向性结论。",
        ]
    else:
        defs = [
            "- `Budget`: active agent's `max_iters` parameter. Range {1,5,10,15,20}.",
            "- `n_ideas`: number of ideas generated in this (model, budget) cell (10 paper × 3 ideas = 30).",
            "- `mean_score`: mean weighted score for ideas in this cell. Per idea, 3 critics each emit 5-dim scores; weighted score = O×2+F+C×0.5+I×1.5+S×0.5; trimmed mean over critics (drop highest); then mean over ideas. Range [0, 10], higher better.",
            "- `mean_tool_calls`: mean of `n_tool_calls` (total tool calls across all agent loop iterations) for ideas in this cell.",
            "- `budget_util`: mean_tool_calls / budget. Reflects how much of the allocated budget the model actually consumed.",
            "- `nudge_rate`: fraction of ideas that triggered the 'you've reached budget' nudge. Near 0 = model self-terminated early.",
            "- `Saturation`: heuristic walking the full budget ladder. From the 2nd step on, if a step's marginal Δscore < max(0.5×previous gain, 0.05), its start budget is the saturation point (`saturated@B`); else `still-growing`. `marginal Δscore` lists each consecutive step's gain. n=30/cell; directional.",
        ]
    lines += defs

    lines += ["", "### Minimal Interpretation" if not L else "### 简短解读"]
    if L:
        lines += [
            "*Hypothesis*：弱模型（9B）应在更小 budget 就饱和（不擅长用工具）；强模型 397B 应充分利用更大 budget。",
            "*Hypothesis*：budget_util 在 budget=15 时 < 50% 说明模型在 ~7 个 tool call 时已经早停（架构而非预算才是瓶颈）。",
            "实际结论以上数字为准（10 paper × 3 idea × 3 critic / cell）。",
        ]
    else:
        lines += [
            "*Hypothesis*: weak model (9B) should saturate at smaller budget (poor tool use); strong 397B should leverage larger budget.",
            "*Hypothesis*: budget_util < 50% at b=15 means model self-terminates around ~7 tool calls (architecture, not budget, is the ceiling).",
            "Actual conclusions follow the numbers (10 paper × 3 idea × 3 critic / cell).",
        ]

    return "\n".join(lines)


def main():
    db = str(cfg.RESULTS_DB)
    summary = aggregate(db)
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_budget.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "sweep_budget.md").write_text(render_md(summary, lang="en"))
    (out_dir / "sweep_budget_zh.md").write_text(render_md(summary, lang="zh"))
    print(f"✓ reports/sweep_budget.{{json, md, zh.md}}")
    print()
    print(render_md(summary, lang="en"))


if __name__ == "__main__":
    main()
