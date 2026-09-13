"""Sweep B per-dimension diagnostic.

Hypothesis: larger budget → more tool calls → more unrelated papers in context
→ dilutes hypothesis focus → lower Originality / Clarity / Specificity.

For each (idea_model, budget, dim ∈ {O,F,C,I,S}), compute mean raw score
across (paper × idea × critic). Then per model compute the per-dim delta
between budget=1 and budget=15 to identify which dimension drives the drop.

Outputs: reports/sweep_budget_per_dim.{json, md, zh.md}
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
IDEA_MODELS = ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "qwen/qwen3.5-397b-a17b"]
BUDGETS = [1, 5, 10, 15]


def aggregate(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT idea_model, budget, scores_json
        FROM budget_sweep_scores
        WHERE scores_json IS NOT NULL
    """).fetchall()

    # (model, budget) -> dim -> [scores]
    buckets = {m: {b: {d: [] for d in DIMS} for b in BUDGETS} for m in IDEA_MODELS}
    for r in rows:
        if r["idea_model"] not in IDEA_MODELS or r["budget"] not in BUDGETS:
            continue
        sc = json.loads(r["scores_json"])
        for d in DIMS:
            v = sc.get(d)
            if v is not None:
                buckets[r["idea_model"]][r["budget"]][d].append(float(v))

    summary = {}
    for m in IDEA_MODELS:
        summary[m] = {"by_budget": {}, "delta_1_to_15": {}}
        for b in BUDGETS:
            per_dim = {}
            for d in DIMS:
                xs = buckets[m][b][d]
                per_dim[d] = {
                    "n": len(xs),
                    "mean": round(statistics.mean(xs), 3) if xs else None,
                }
            summary[m]["by_budget"][b] = per_dim
        for d in DIMS:
            b1 = summary[m]["by_budget"][1][d]["mean"]
            b15 = summary[m]["by_budget"][15][d]["mean"]
            if b1 is not None and b15 is not None:
                summary[m]["delta_1_to_15"][d] = round(b15 - b1, 3)
            else:
                summary[m]["delta_1_to_15"][d] = None
    return summary


def render_md(summary: dict, lang: str = "en") -> str:
    L = lang == "zh"
    lines = []
    title = ("Sweep B 单维度根因分析（per-dimension diagnostic）"
             if L else "Sweep B per-dimension diagnostic")
    lines += [f"# {title}", ""]
    lines += ["## Scope" if not L else "## 范围",
              "Source data: `budget_sweep_scores` (2 paper × 3 model × 4 budget × 1 idea × 3 critics = 72 score rows).",
              ""]
    lines += ["## Per-budget mean raw scores (each dim, 1-10 scale)"
              if not L else
              "## 每 budget 下每 idea_model 的单维度均值（1-10 raw）"]
    lines += [""]
    for m in IDEA_MODELS:
        lines += [f"### {m}", ""]
        lines += ["| Budget | Originality | Feasibility | Clarity | Impact | Specificity |"]
        lines += ["|---:|---:|---:|---:|---:|---:|"]
        for b in BUDGETS:
            d = summary[m]["by_budget"][b]
            row = f"| {b}"
            for dim in DIMS:
                v = d[dim]["mean"]
                row += f" | {v if v is not None else '—'}"
            row += " |"
            lines.append(row)
        deltas = summary[m]["delta_1_to_15"]
        lines += [
            "",
            "**Δ (budget=15 − budget=1)** "
            f"O={deltas['originality']}  "
            f"F={deltas['feasibility']}  "
            f"C={deltas['clarity']}  "
            f"I={deltas['impact']}  "
            f"S={deltas['specificity']}",
            "",
        ]
    lines += ["### Column Definitions" if not L else "### 字段定义"]
    if L:
        defs = [
            "- `Budget`：active agent max_iters。",
            "- `Originality / Feasibility / Clarity / Impact / Specificity`：5 维评分均值（per critic 直接打分 1-10，未加权）。每个 cell = mean over (2 paper × 1 idea × 3 critic) = up to 6 数据点。",
            "- `Δ (budget=15 − budget=1)`：从最小 budget 到最大 budget，每维度的均值变化。负值 = budget 越大该维度越差；正值 = 越好。",
        ]
    else:
        defs = [
            "- `Budget`: active agent max_iters parameter.",
            "- `Originality / Feasibility / Clarity / Impact / Specificity`: per-dim mean (raw 1-10, unweighted). Each cell = mean over (2 paper × 1 idea × 3 critic) ≤ 6 obs.",
            "- `Δ (budget=15 − budget=1)`: change from min to max budget per dim. Negative = dim degrades with budget; positive = improves.",
        ]
    lines += defs

    # Cross-model dim winners
    lines += ["", "### Which dimension drives the budget-induced drop?" if not L else "### 哪个维度主导了 budget 增加导致的下降？"]
    losers = {d: 0 for d in DIMS}
    for m in IDEA_MODELS:
        deltas = summary[m]["delta_1_to_15"]
        valid = [(d, v) for d, v in deltas.items() if v is not None]
        if not valid:
            continue
        worst = min(valid, key=lambda kv: kv[1])
        losers[worst[0]] += 1
    lines += [""]
    lines += ["| Dim | # models where it's the largest dropper |"]
    lines += ["|---|---:|"]
    for d in DIMS:
        lines.append(f"| {d} | {losers[d]} |")
    lines += ["",
              ("*Interpretation*: 上表统计在 3 模型中，每个 dim 作为 b=1→b=15 最大跌幅的次数。"
               if L else
               "*Interpretation*: count of how often each dim is the single largest drop across the 3 models from b=1 to b=15.")]
    return "\n".join(lines)


def main():
    db = str(cfg.RESULTS_DB)
    s = aggregate(db)
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_budget_per_dim.json").write_text(json.dumps(s, indent=2))
    (out_dir / "sweep_budget_per_dim.md").write_text(render_md(s, "en"))
    (out_dir / "sweep_budget_per_dim_zh.md").write_text(render_md(s, "zh"))
    print(f"✓ reports/sweep_budget_per_dim.{{json, md, zh.md}}\n")
    print(render_md(s, "en"))


if __name__ == "__main__":
    main()
