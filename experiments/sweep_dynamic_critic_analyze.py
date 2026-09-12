"""Aggregate sweep A results: compare static / dynamic_search / dynamic_cited.

Outputs (under reports/):
  sweep_dynamic_critic.json
  sweep_dynamic_critic.md           — primary numbers + column definitions
  sweep_dynamic_critic_zh.md        — Chinese version

For each (idea_model, judge_mode):
  - mean weighted score across (paper × idea × critic)
  - spread (max - min weighted score) across models within the mode
  - Spearman ρ between mode ranking vs static ranking
  - same-family monotonicity (9B < 27B < 397B?)

Weighted score = O*2 + F + C*0.5 + I*1.5 + S*0.5  (matches main pipeline).
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
WEIGHT_SUM = sum(WEIGHTS.values())

IDEA_MODELS = [
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.5-397b-a17b",
    "xiaomi/mimo-v2.5",
    "xiaomi/mimo-v2.5-pro",
]
# Family groupings — checked separately for monotonic size-scaling.
# Each list is ordered small → large.
FAMILIES = {
    "Qwen3.5": ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "qwen/qwen3.5-397b-a17b"],
    "MiMo-v2.5": ["xiaomi/mimo-v2.5", "xiaomi/mimo-v2.5-pro"],
}
# Display labels (column headers in markdown reports)
MODEL_LABELS = {
    "qwen/qwen3.5-9b": "Qwen3.5-9B",
    "qwen/qwen3.5-27b": "Qwen3.5-27B",
    "qwen/qwen3.5-397b-a17b": "Qwen3.5-397B",
    "xiaomi/mimo-v2.5": "MiMo-v2.5",
    "xiaomi/mimo-v2.5-pro": "MiMo-v2.5-pro",
}
JUDGE_MODES = ["static", "dynamic_search", "dynamic_cited"]


def weighted_score(scores: dict) -> float:
    """Weighted mean of 5-dim scores."""
    s = 0.0
    for d in DIMS:
        s += WEIGHTS[d] * scores.get(d, 0)
    return s / WEIGHT_SUM


def trimmed_mean(values: list) -> float:
    """Mean of values after dropping the maximum (per-critic trim)."""
    if len(values) < 2:
        return statistics.mean(values) if values else 0.0
    s = sorted(values, reverse=True)
    return statistics.mean(s[1:])


def spearman(x: list, y: list) -> float:
    """Spearman rank correlation."""
    if len(x) < 2:
        return None
    def rank(arr):
        s = sorted(range(len(arr)), key=lambda i: arr[i])
        r = [0] * len(arr)
        for rk, idx in enumerate(s):
            r[idx] = rk + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d_sq / (n * (n * n - 1))


def aggregate(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT paper_id, idea_model, track, idea_index, critic_model,
               judge_mode, scores_json
        FROM dynamic_critic_sweep
        WHERE scores_json IS NOT NULL
    """).fetchall()

    # (mode, idea_model, paper_id, idea_index) -> {critic_model: weighted_score}
    by_combo = {}
    for r in rows:
        scores = json.loads(r["scores_json"])
        ws = weighted_score(scores)
        key = (r["judge_mode"], r["idea_model"], r["paper_id"], r["idea_index"])
        by_combo.setdefault(key, {})[r["critic_model"]] = ws

    # Aggregate per (mode, idea_model): trimmed mean across critics, then mean across (paper, idea)
    per_combo_trimmed = {}
    for key, by_critic in by_combo.items():
        per_combo_trimmed[key] = trimmed_mean(list(by_critic.values()))

    # mode → idea_model → list of trimmed scores (one per paper × idea_index)
    by_mode_model = {m: {im: [] for im in IDEA_MODELS} for m in JUDGE_MODES}
    for (mode, im, _, _), s in per_combo_trimmed.items():
        if im in IDEA_MODELS and mode in JUDGE_MODES:
            by_mode_model[mode][im].append(s)

    # Summary per mode
    summary = {}
    for mode in JUDGE_MODES:
        per_model_mean = {}
        for im in IDEA_MODELS:
            vals = by_mode_model[mode][im]
            per_model_mean[im] = round(statistics.mean(vals), 4) if vals else None
        valid = [v for v in per_model_mean.values() if v is not None]
        spread = round(max(valid) - min(valid), 4) if valid else None
        # Per-family monotonicity (small → large within each family list)
        family_mono = {}
        for fam, members in FAMILIES.items():
            vals = [per_model_mean[m] for m in members]
            if any(v is None for v in vals):
                family_mono[fam] = None
            else:
                family_mono[fam] = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
        summary[mode] = {
            "per_model_mean": per_model_mean,
            "spread": spread,
            "family_monotonic": family_mono,
            "n_per_model": {im: len(by_mode_model[mode][im]) for im in IDEA_MODELS},
        }

    # Spearman vs static
    static_means = [summary["static"]["per_model_mean"][im] for im in IDEA_MODELS]
    for mode in JUDGE_MODES:
        if mode == "static":
            summary[mode]["spearman_vs_static"] = 1.0
        else:
            other = [summary[mode]["per_model_mean"][im] for im in IDEA_MODELS]
            if None in other:
                summary[mode]["spearman_vs_static"] = None
            else:
                summary[mode]["spearman_vs_static"] = round(
                    spearman(static_means, other), 4)

    return summary


def render_md(summary: dict, lang: str = "en") -> str:
    L = lang == "zh"
    lines = []
    title = "Sweep A — Dynamic critic mode 三选一 (报告)" if L else "Sweep A — Dynamic critic mode three-way comparison"
    lines += [f"# {title}", ""]
    lines += ["## Scope" if not L else "## 范围"]
    # n_per_model = paper × idea (trimmed-mean across critics, so critics dim collapsed).
    # Default 3 ideas per paper.
    n_per_model_max = max(
        summary[m]["n_per_model"].get(im, 0)
        for m in JUDGE_MODES for im in IDEA_MODELS
    )
    n_papers = n_per_model_max // 3 if n_per_model_max else "?"
    family_str = " + ".join(
        f"{fam} ({len(members)} sizes)" for fam, members in FAMILIES.items()
    )
    lines += [f"{n_papers} papers × {len(IDEA_MODELS)} idea models ({family_str}) × 3 ideas × 3 critics × 3 judge_modes"]
    lines += ["", "## Data Sources"]
    lines += ["- `data/results.db.dynamic_critic_sweep_ideas` — fresh ideas with REQUIRE_CITES=True"]
    lines += ["- `data/results.db.dynamic_critic_sweep` — critic verdicts per (idea × critic × mode)"]
    lines += ["", "## Primary Results (Weighted Score = O×2 + F + C×0.5 + I×1.5 + S×0.5, trimmed mean over critics)" if not L else "## 一级结果（加权分数；critic 内去最高分后取均值）"]
    lines += [""]

    # Main table: one column per model + spread + Spearman
    headers = ["Mode"] + [MODEL_LABELS[m] for m in IDEA_MODELS] + ["spread", "Spearman vs static"]
    aligns = ["|---"] + ["|---:"] * len(IDEA_MODELS) + ["|---:", "|---:"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("".join(aligns) + "|")
    for mode in JUDGE_MODES:
        s = summary[mode]
        cells = [mode]
        for m in IDEA_MODELS:
            v = s["per_model_mean"].get(m)
            cells.append(f"{v}" if v is not None else "—")
        cells.append(str(s["spread"]) if s["spread"] is not None else "—")
        sp = s.get("spearman_vs_static")
        cells.append(f"{sp:.3f}" if isinstance(sp, (int, float)) else "—")
        lines.append("| " + " | ".join(cells) + " |")

    # Family monotonicity table
    lines += ["", "### Family monotonicity (size-scaling within each family)" if not L else "### 同家族单调性（按 size 从小到大）"]
    fam_headers = ["Mode"] + [f"{fam} (small→large)" for fam in FAMILIES]
    fam_aligns = ["|---"] + ["|:---:"] * len(FAMILIES)
    lines.append("| " + " | ".join(fam_headers) + " |")
    lines.append("".join(fam_aligns) + "|")
    for mode in JUDGE_MODES:
        fm = summary[mode]["family_monotonic"]
        cells = [mode]
        for fam in FAMILIES:
            v = fm.get(fam)
            cells.append("✓" if v else ("✗" if v is False else "—"))
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "### Sample sizes (n per model per mode)" if not L else "### 样本量"]
    lines.append("| Mode | " + " | ".join(MODEL_LABELS[m] for m in IDEA_MODELS) + " |")
    lines.append("|---" + "|---:" * len(IDEA_MODELS) + "|")
    for mode in JUDGE_MODES:
        n = summary[mode]["n_per_model"]
        lines.append(f"| {mode} | " + " | ".join(str(n[m]) for m in IDEA_MODELS) + " |")

    lines += ["", "### Column Definitions" if not L else "### 字段定义"]
    model_labels_str = " / ".join(MODEL_LABELS[m] for m in IDEA_MODELS)
    if L:
        defs = [
            "- `Mode`：critic 输入策略。`static` = papers.db 里的 survey refs；`dynamic_search` = critic 用 hypothesis 当 query 搜 SS；`dynamic_cited` = critic 读 idea_text 末尾的 Cited footer。",
            f"- `{model_labels_str}`：该 idea_model 在该 mode 下的平均加权分数（每 cell = 5 paper × 3 idea × 3 critic = 45 row 上 trimmed-mean）。范围 [0, 10]，越大越好。来源 `dynamic_critic_sweep.scores_json` 的 5 维度按 O×2+F+C×0.5+I×1.5+S×0.5 加权。",
            "- `spread`：mode 内所有 5 个模型的加权分数最大值与最小值之差。越大表示该 mode 区分模型能力越强。单位与分数相同。",
            "- `Spearman vs static`：该 mode 下 5 个模型排名与 static mode 下排名的 Spearman ρ。1.0 = 完全一致；-1.0 = 完全反转。仅 5 个数据点，统计噪声较大，仅供方向性参考。",
            "- `Family monotonicity` 列：每家族内 size 从小到大对应的加权分数是否单调上升。✓ = 同家族梯度保持；✗ = 反常。Qwen3.5 = (9B, 27B, 397B)；MiMo-v2.5 = (v2.5, v2.5-pro)。",
            "- `Sample sizes`：每 cell 进入聚合的 trimmed-mean record 数（每个 record 是一组 critic 评分 trimmed-mean 后得到的 weighted score）。",
        ]
    else:
        defs = [
            "- `Mode`: critic input strategy. `static` = survey refs from papers.db; `dynamic_search` = critic searches SS using the hypothesis; `dynamic_cited` = critic reads the trailing Cited: footer.",
            f"- `{model_labels_str}`: mean weighted score for that idea_model in that mode (each cell = 5 paper × 3 idea × 3 critic = 45 rows, with per-(paper, idea) trimmed mean over critics). Range [0, 10], higher better. Source: `dynamic_critic_sweep.scores_json` aggregated as O×2 + F + C×0.5 + I×1.5 + S×0.5.",
            "- `spread`: max − min of all 5 models' weighted scores within the mode. Larger = mode discriminates models better. Same units as score.",
            "- `Spearman vs static`: Spearman ρ between this mode's 5-model ranking and the static ranking. 1.0 = identical, -1.0 = inverted. With only n=5 data points this is noisy; treat as directional only.",
            "- `Family monotonicity`: whether weighted score increases monotonically with model size within each family. ✓ = same-family gradient holds; ✗ = inverted. Qwen3.5 = (9B, 27B, 397B); MiMo-v2.5 = (v2.5, v2.5-pro).",
            "- `Sample sizes`: number of trimmed-mean records aggregated per cell (each record is one weighted score after trimming the highest critic per (paper, idea)).",
        ]
    lines += defs

    lines += ["", "### Minimal Interpretation" if not L else "### 简短解读"]
    if L:
        lines += [
            "*Hypothesis*：dynamic_cited 让 critic 看模型自己说引用了什么，应该跟 ideation 信号最对齐。",
            "*Hypothesis*：dynamic_search 引入 SS search 噪声，但可能在 spread 上更敏锐（smoke n=3 时已观察到）。",
            "*Hypothesis*：跨家族（Qwen vs MiMo）同时观察 family-internal monotonicity，可以排除 dynamic_search 是否 family-specific。",
            "实际结论以上表数字为准；这只是 candidate findings。",
        ]
    else:
        lines += [
            "*Hypothesis*: `dynamic_cited` shows the critic what the ideation said it used, so should align best with the ideation signal.",
            "*Hypothesis*: `dynamic_search` introduces SS retrieval noise but may yield sharper spread (observed in 3-model smoke).",
            "*Hypothesis*: Cross-family check (Qwen vs MiMo) on family-internal monotonicity falsifies the family-specific confound.",
            "Actual conclusions follow the numbers above; treat these as candidate interpretations only.",
        ]
    return "\n".join(lines)


def main():
    db = str(cfg.RESULTS_DB)
    summary = aggregate(db)

    out_json = ROOT / "reports" / "sweep_dynamic_critic.json"
    out_md_en = ROOT / "reports" / "sweep_dynamic_critic.md"
    out_md_zh = ROOT / "reports" / "sweep_dynamic_critic_zh.md"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    out_md_en.write_text(render_md(summary, lang="en"))
    out_md_zh.write_text(render_md(summary, lang="zh"))

    print(f"✓ {out_json}")
    print(f"✓ {out_md_en}")
    print(f"✓ {out_md_zh}")
    print()
    print(render_md(summary, lang="en"))


if __name__ == "__main__":
    main()
