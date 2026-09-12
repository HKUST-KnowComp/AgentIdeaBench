"""Cross-year Static performance plots.

Two visualizations of the same Static-Mode leaderboard:
  1) Box plot (discrete x = release year): vertical box per year spans
     [min, max] of Static scores; matches the "strong/weak gap shrinks"
     framing.
  2) Scatter plot (continuous x = decimal release date): each model is a
     single dot at its actual OpenRouter listing date converted to a
     fractional year (e.g. 2024-09-19 → 2024.719). Family-colored.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import sqlite3
import sys

try:
    from adjustText import adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ────────────────────────────────────────────────────────────────────────────
# Date → decimal year formula (FIXED, do not change without updating callers).
#
#   decimal_year = year + ((month - 1) + (day - 1) / days_in_month) / 12
#
# Properties:
#   - Jan  1 → year + 0                       (whole-year integer)
#   - Jan 15 → year + 14/(31*12)   ≈ year + 0.038
#   - Jul  1 → year + 6/12 = year + 0.500     (start of July, mid-year)
#   - Oct  1 → year + 9/12 = year + 0.750     (start of October, 3/4-year)
#   - Dec 31 → year + (11 + 30/31)/12 ≈ year + 0.997
#
# i.e. each month gets exactly 1/12 of the y-axis distance, and day-1 sits at
# the month's left edge; day 16 of a 30-day month sits at half (0.5).
# ────────────────────────────────────────────────────────────────────────────
def to_decimal_year(year: int, month: int, day: int) -> float:
    import calendar
    days_in_month = calendar.monthrange(year, month)[1]
    return year + ((month - 1) + (day - 1) / days_in_month) / 12

# Year → list of (model_id, static_score)
# Pulled from current leaderboard / model_scores table.
YEAR_GROUPS = {
    "2023": [
        ("mistralai/mistral-7b-instruct-v0.1", "Mistral-7B-v0.1"),
        ("openai/gpt-4", "GPT-4"),
    ],
    "2024": [
        ("meta-llama/llama-3.1-8b-instruct", "Llama-3.1-8B"),
        ("google/gemma-2-27b-it", "Gemma-2-27B"),
        ("qwen/qwen-2.5-7b-instruct", "Qwen2.5-7B"),
        ("openai/gpt-4o", "GPT-4o"),
        ("qwen/qwen-2.5-72b-instruct", "Qwen2.5-72B"),
        ("openai/gpt-4o-mini", "GPT-4o-mini"),
        ("anthropic/claude-3.7-sonnet", "Claude-3.7-Sonnet"),
        # o1 (full) released 2024-12-05; belongs in 2024 group, not 2025
        ("openai/o1", "o1"),
    ],
    "2025": [
        # 2025 group: real-date 2025 releases
        ("google/gemini-2.5-pro", "Gemini-2.5-Pro"),       # real 2025-06-17
        ("deepseek/deepseek-r1-0528", "DeepSeek-R1-0528"), # real 2025-05-28
        ("meta-llama/llama-4-maverick", "Llama-4-Maverick"),
        ("mistralai/mistral-small-24b-instruct-2501", "Mistral-Small-3"),
        ("qwen/qwen3-8b", "Qwen3-8B"),
        ("google/gemma-3-27b-it", "Gemma-3-27B"),
        ("openai/gpt-4.1", "GPT-4.1"),
        ("anthropic/claude-sonnet-4", "Claude-Sonnet-4"),
        ("openai/gpt-5", "GPT-5"),
    ],
    "2026": [
        ("moonshotai/kimi-k2.5", "Kimi-K2.5"),             # real 2026-01-27 (user)
        ("qwen/qwen3.5-9b", "Qwen3.5-9B"),
        ("anthropic/claude-sonnet-4.6", "Claude-Sonnet-4.6"),
        ("google/gemma-4-31b-it", "Gemma-4-31B"),
        ("qwen/qwen3.5-27b", "Qwen3.5-27B"),
        ("qwen/qwen3.5-397b-a17b", "Qwen3.5-397B"),
        ("openai/gpt-5.5", "GPT-5.5"),
        ("z-ai/glm-5.1", "GLM-5.1"),                      # real 2026-03
        ("deepseek/deepseek-v4-pro", "DeepSeek-V4-Pro"),  # real 2026-04-24
    ],
}


# Decimal release date per model_id (OpenRouter `created` field where
# available; manual fallback for models with no OR timestamp).
# All values computed via to_decimal_year() above. Exceptions are tagged
# "(pinned)" — meaning the displayed position is a narrative grouping rather
# than the real release date.
RELEASE_DATES = {
    # ─── Static + Active idea models (corrected per user 2026-05-19 table) ───
    "openai/gpt-4":                              to_decimal_year(2023, 3, 14),
    "mistralai/mistral-7b-instruct-v0.1":        to_decimal_year(2023, 9, 27),
    "openai/gpt-4o":                             to_decimal_year(2024, 5, 13),
    "google/gemma-2-27b-it":                     to_decimal_year(2024, 6, 27),
    "openai/gpt-4o-mini":                        to_decimal_year(2024, 7, 18),
    "meta-llama/llama-3.1-8b-instruct":          to_decimal_year(2024, 7, 23),
    "openai/o1":                                 to_decimal_year(2024, 9, 12),
    "qwen/qwen-2.5-72b-instruct":                to_decimal_year(2024, 9, 19),
    "qwen/qwen-2.5-7b-instruct":                 to_decimal_year(2024, 9, 19),
    "mistralai/mistral-small-24b-instruct-2501": to_decimal_year(2025, 1, 30),
    "anthropic/claude-3.7-sonnet":               to_decimal_year(2025, 2, 24),
    "google/gemma-3-27b-it":                     to_decimal_year(2025, 3, 12),
    "meta-llama/llama-4-maverick":               to_decimal_year(2025, 4, 5),
    "openai/gpt-4.1":                            to_decimal_year(2025, 4, 14),
    "qwen/qwen3-8b":                             to_decimal_year(2025, 4, 29),
    "qwen/qwen3-32b":                            to_decimal_year(2025, 4, 29),
    "anthropic/claude-sonnet-4":                 to_decimal_year(2025, 5, 22),
    "deepseek/deepseek-r1-0528":                 to_decimal_year(2025, 5, 28),
    "google/gemini-2.5-pro":                     to_decimal_year(2025, 6, 17),
    "qwen/qwen3-235b-a22b-thinking-2507":        to_decimal_year(2025, 7, 25),
    "openai/gpt-5":                              to_decimal_year(2025, 8, 7),
    "qwen/qwen3-vl-8b-thinking":                 to_decimal_year(2025, 10, 15),
    "moonshotai/kimi-k2.5":                      to_decimal_year(2026, 1, 27),
    "anthropic/claude-sonnet-4.6":               to_decimal_year(2026, 2, 17),
    "qwen/qwen3.5-397b-a17b":                    to_decimal_year(2026, 2, 16),
    "qwen/qwen3.5-27b":                          to_decimal_year(2026, 2, 24),
    "qwen/qwen3.5-9b":                           to_decimal_year(2026, 3, 2),
    "openai/gpt-5.4-mini":                       to_decimal_year(2026, 3, 17),
    "openai/gpt-5.4-nano":                       to_decimal_year(2026, 3, 17),
    "mistralai/mistral-small-2603":              to_decimal_year(2026, 3, 16),
    "google/gemma-4-31b-it":                     to_decimal_year(2026, 4, 2),
    "z-ai/glm-5.1":                              to_decimal_year(2026, 4, 7),
    "moonshotai/kimi-k2.6":                      to_decimal_year(2026, 4, 20),
    "xiaomi/mimo-v2.5":                          to_decimal_year(2026, 4, 22),
    "openai/gpt-5.5":                            to_decimal_year(2026, 4, 23),
    "deepseek/deepseek-v4-pro":                  to_decimal_year(2026, 4, 24),
    # ─── 2026-05-19 plateau-extension batch (9 open-source models) ───
    "qwen/qwen3-coder":                          to_decimal_year(2025, 7, 22),  # 480B A35B per OR catalog
    "qwen/qwen3-30b-a3b-instruct-2507":          to_decimal_year(2025, 7, 28),
    # DROPPED 2026-05-19: qwen3-30b-a3b-thinking-2507 (reasoning mandate, 100% gen fail)
    # NOTE: moonshotai/kimi-dev-72b dropped — not in OpenRouter catalog
    "minimax/minimax-m1":                        to_decimal_year(2025, 6, 17),
    "mistralai/devstral-medium":                 to_decimal_year(2025, 7, 10),
    "mistralai/mistral-medium-3.1":              to_decimal_year(2025, 8, 12),
    "z-ai/glm-4.5-air":                          to_decimal_year(2025, 7, 28),
    "z-ai/glm-4.6":                              to_decimal_year(2025, 9, 30),
    # ─── Critic-only models (no idea generation; for appendix robustness) ───
    "deepseek/deepseek-r1":                      to_decimal_year(2025, 1, 20),
    "moonshotai/kimi-k2-0905":                   to_decimal_year(2025, 9, 5),
    "tencent/hunyuan-a13b-instruct":             to_decimal_year(2025, 6, 27),
    "x-ai/grok-4":                               to_decimal_year(2025, 7, 9),
    "qwen/qwen3-max":                            to_decimal_year(2025, 9, 23),
    "minimax/minimax-m2.7":                      to_decimal_year(2026, 3, 18),
    "qwen/qwen3.6-plus":                         to_decimal_year(2026, 4, 2),
    "deepseek/deepseek-v4-flash":                to_decimal_year(2026, 4, 24),
    "x-ai/grok-4.3":                             to_decimal_year(2026, 5, 1),
}


# ────────────────────────────────────────────────────────────────────────────
# Knowledge cutoff dates (corrected per user 2026-05-19 table).
# Day = 15 (mid-month) unless user supplies explicit day.
# `None` = unknown / not officially disclosed.
# ────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_CUTOFFS = {
    # ─── E42 closed-source extension (NVIDIA gateway ids) ───
    # Cutoffs and provenance come from experiments/e42_cutoff_safe_closed.py ROSTER.
    # 'undisclosed' group has no recorded cutoff and is intentionally absent, so
    # those models are dropped from cutoff regressions rather than guessed at.
    # -- group: safe --
    "azure/openai/gpt-4o-mini":                   to_decimal_year(2023, 10, 1),  # disclosed
    "azure/openai/gpt-4o":                        to_decimal_year(2023, 10, 1),  # disclosed
    "azure/openai/o1":                            to_decimal_year(2023, 10, 1),  # disclosed
    "azure/openai/o3-mini":                       to_decimal_year(2023, 10, 1),  # disclosed
    "us/azure/openai/gpt-4.1-nano":               to_decimal_year(2024, 6, 15),  # family
    "azure/openai/gpt-4.1-mini":                  to_decimal_year(2024, 6, 15),  # family
    "azure/openai/gpt-4.1":                       to_decimal_year(2024, 6, 15),  # disclosed
    "azure/openai/o3":                            to_decimal_year(2024, 6, 1),  # disclosed
    "azure/openai/o4-mini":                       to_decimal_year(2024, 6, 1),  # disclosed
    "azure/openai/gpt-5-nano":                    to_decimal_year(2024, 5, 30),  # disclosed
    "azure/openai/gpt-5-mini":                    to_decimal_year(2024, 5, 30),  # disclosed
    "azure/openai/gpt-5":                         to_decimal_year(2024, 9, 30),  # disclosed
    # -- group: partial --
    "azure/openai/gpt-5.4-nano":                  to_decimal_year(2025, 8, 31),  # disclosed
    "azure/openai/gpt-5.4-mini":                  to_decimal_year(2025, 8, 31),  # disclosed
    "azure/openai/gpt-5.4":                       to_decimal_year(2025, 8, 31),  # family
    "azure/openai/gpt-5.5":                       to_decimal_year(2025, 12, 1),  # disclosed
    # -- group: boundary --
    "azure/anthropic/claude-haiku-4-5":           to_decimal_year(2025, 4, 1),  # config
    "azure/anthropic/claude-sonnet-4-5":          to_decimal_year(2025, 4, 1),  # config
    "azure/anthropic/claude-opus-4-5":            to_decimal_year(2025, 4, 1),  # family
    # -- group: partial --
    "azure/anthropic/claude-sonnet-4-6":          to_decimal_year(2025, 5, 15),  # disclosed
    "azure/anthropic/claude-opus-4-6":            to_decimal_year(2025, 5, 15),  # family
    # ─── Static + Active idea models ───
    "openai/gpt-4":                              to_decimal_year(2021, 9, 15),
    "mistralai/mistral-7b-instruct-v0.1":        to_decimal_year(2023, 9, 15),
    "openai/gpt-4o":                             to_decimal_year(2023, 10, 1),
    "openai/gpt-4o-mini":                        to_decimal_year(2023, 10, 1),
    "openai/o1":                                 to_decimal_year(2023, 10, 1),
    "mistralai/mistral-small-24b-instruct-2501": to_decimal_year(2023, 10, 15),
    "meta-llama/llama-3.1-8b-instruct":          to_decimal_year(2023, 12, 15),
    "google/gemma-2-27b-it":                     to_decimal_year(2024, 6, 15),
    "openai/gpt-4.1":                            to_decimal_year(2024, 6, 15),
    "qwen/qwen-2.5-72b-instruct":                to_decimal_year(2024, 6, 30),
    "qwen/qwen-2.5-7b-instruct":                 to_decimal_year(2024, 6, 30),
    "deepseek/deepseek-r1-0528":                 to_decimal_year(2024, 7, 15),
    "google/gemma-3-27b-it":                     to_decimal_year(2024, 8, 15),
    "meta-llama/llama-4-maverick":               to_decimal_year(2024, 8, 15),
    "openai/gpt-5":                              to_decimal_year(2024, 9, 30),
    "anthropic/claude-3.7-sonnet":               to_decimal_year(2024, 10, 15),
    "qwen/qwen3-8b":                             to_decimal_year(2024, 10, 15),
    "qwen/qwen3-32b":                            to_decimal_year(2024, 10, 15),
    "google/gemini-2.5-pro":                     to_decimal_year(2025, 1, 15),
    # ─── E33 cross-family Gemini extension (native endpoint) ───
    # All generations share a ~2025-01 knowledge cutoff (best-effort; Google does
    # not disclose per-model cutoffs beyond "early 2025"). Same-cutoff cluster →
    # strengthens family robustness (F2), NOT a within-family cutoff ladder (F3).
    "google/gemini-2.5-flash-lite":              to_decimal_year(2025, 1, 15),  # best-effort
    "google/gemini-2.5-flash":                   to_decimal_year(2025, 1, 15),  # best-effort
    "google/gemini-3-flash-preview":             to_decimal_year(2025, 1, 15),  # best-effort
    "google/gemini-3.5-flash":                   to_decimal_year(2025, 1, 15),  # best-effort
    "google/gemini-3.1-pro-preview":             to_decimal_year(2025, 1, 15),  # best-effort
    "anthropic/claude-sonnet-4":                 to_decimal_year(2025, 1, 15),  # reliable cutoff per Anthropic
    "moonshotai/kimi-k2.5":                      to_decimal_year(2025, 1, 15),
    "google/gemma-4-31b-it":                     to_decimal_year(2025, 1, 15),
    "moonshotai/kimi-k2.6":                      to_decimal_year(2025, 1, 15),
    "qwen/qwen3-vl-8b-thinking":                 to_decimal_year(2025, 3, 31),
    "qwen/qwen3-235b-a22b-thinking-2507":        to_decimal_year(2025, 3, 31),
    "qwen/qwen3.5-9b":                           to_decimal_year(2025, 4, 15),
    "qwen/qwen3.5-27b":                          to_decimal_year(2025, 4, 15),
    "qwen/qwen3.5-397b-a17b":                    to_decimal_year(2025, 4, 15),
    "anthropic/claude-sonnet-4.6":               to_decimal_year(2025, 5, 15),  # reliable cutoff per Anthropic
    "xiaomi/mimo-v2.5":                          to_decimal_year(2025, 5, 15),
    "mistralai/mistral-small-2603":              to_decimal_year(2025, 6, 15),
    "openai/gpt-5.4-mini":                       to_decimal_year(2025, 8, 31),
    "openai/gpt-5.4-nano":                       to_decimal_year(2025, 8, 31),
    "openai/gpt-5.5":                            to_decimal_year(2025, 12, 1),
    "z-ai/glm-5.1":                              to_decimal_year(2025, 12, 15),
    "deepseek/deepseek-v4-pro":                  to_decimal_year(2026, 1, 15),
    # ─── 2026-05-19 plateau-extension batch (only models with disclosed cutoff) ───
    "qwen/qwen3-30b-a3b-instruct-2507":          to_decimal_year(2025, 4, 15),  # best-effort
    "mistralai/devstral-medium":                 to_decimal_year(2025, 5, 15),  # best-effort
    "mistralai/mistral-medium-3.1":              to_decimal_year(2025, 5, 15),  # best-effort
    "z-ai/glm-4.5-air":                          to_decimal_year(2025, 3, 15),  # best-effort
    # NOTE: cutoff "Unknown / not officially disclosed" for:
    #   qwen3-coder-480b-a35b-instruct, moonshotai/kimi-dev-72b,
    #   minimax/minimax-m1, z-ai/glm-4.6 — omitted from cutoff axis.
    # ─── Critic-only models ───
    "deepseek/deepseek-r1":                      to_decimal_year(2024, 7, 15),
    "moonshotai/kimi-k2-0905":                   to_decimal_year(2024, 10, 15),
    "x-ai/grok-4":                               to_decimal_year(2024, 11, 15),
    "qwen/qwen3-max":                            to_decimal_year(2025, 4, 15),
    "x-ai/grok-4.3":                             to_decimal_year(2025, 12, 15),
    "deepseek/deepseek-v4-flash":                to_decimal_year(2026, 1, 15),
    # Unknown cutoffs (user marked "not disclosed") — omitted intentionally:
    #   tencent/hunyuan-a13b-instruct, minimax/minimax-m2.7, qwen/qwen3.6-plus
}


# Family → (display name, color). Picked for colorblind-friendlier palette.
FAMILY_STYLE = {
    "openai":    ("OpenAI",     "#10a37f"),  # teal-green
    "anthropic": ("Anthropic",  "#d97706"),  # amber
    "qwen":      ("Qwen",       "#c0392b"),  # red
    "google":    ("Gemma",      "#3b82f6"),  # blue
    "meta":      ("Llama",      "#7c3aed"),  # purple
    "mistral":   ("Mistral",    "#475569"),  # slate
}


def family_for(model_id: str) -> str:
    head = model_id.split("/")[0].lower()
    if head.startswith("meta"):
        return "meta"
    if head.startswith("mistral"):
        return "mistral"
    return head


def load_static_scores(db_results: Path) -> dict:
    """idea_model -> mean Static score across all papers."""
    conn = sqlite3.connect(db_results)
    rows = conn.execute(
        "SELECT idea_model, AVG(mean_absolute_score) "
        "FROM model_scores WHERE track='B' GROUP BY idea_model"
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows if r[1] is not None}


def make_box_plot(year_data, out_pdf, out_png):
    """Original visualization: discrete year on x, [min,max] box per year."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    box_color = "#a8d0e6"
    edge_color = "#3a6b8a"
    point_color = "#1a3a5c"

    years = list(year_data.keys())
    x_positions = list(range(len(years)))

    box_half_width = 0.22
    for i, year in enumerate(years):
        vals = [v for _, v in year_data[year]]
        labels = [lab for lab, _ in year_data[year]]
        if not vals:
            continue
        ymin, ymax = min(vals), max(vals)
        rect = patches.Rectangle(
            (i - box_half_width, ymin),
            2 * box_half_width,
            ymax - ymin,
            facecolor=box_color, edgecolor=edge_color, linewidth=1.6, alpha=0.55,
        )
        ax.add_patch(rect)

        # Individual model dots
        for v in vals:
            ax.plot(i, v, "o", color=point_color, markersize=5, zorder=3)

        # Spread annotation (top center)
        spread = ymax - ymin
        ax.text(i, ymax + 0.08, f"spread = {spread:.3f}",
                ha="center", va="bottom", fontsize=9, color=edge_color)

        # Min/Max labels (small, outside)
        ax.text(i - box_half_width - 0.04, ymax, f"{ymax:.2f}",
                ha="right", va="center", fontsize=8, color=edge_color)
        ax.text(i - box_half_width - 0.04, ymin, f"{ymin:.2f}",
                ha="right", va="center", fontsize=8, color=edge_color)

    # Dashed reference line: an arbitrary "ceiling target"
    ax.axhline(y=7.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(len(years) - 0.5, 7.05, "hypothetical ceiling = 7.0",
            ha="right", va="bottom", fontsize=8, color="gray")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(years, fontsize=12)
    ax.set_xlabel("Model release year", fontsize=12)
    ax.set_ylabel("Static-Mode score (weighted, out of 10)", fontsize=12)
    ax.set_xlim(-0.6, len(years) - 0.4)
    ax.set_ylim(3.8, 7.4)
    ax.set_title(
        "Strong/weak gap across model release years\n"
        "Box = [min, max] of Static scores across that year's idea models",
        fontsize=12, pad=14,
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def make_scatter_plot(scores, out_pdf, out_png):
    """Continuous x = decimal release date; one dot per model, family-colored."""
    points = []  # (decimal_year, score, model_id, label, family)
    label_for = {mid: lab for year_list in YEAR_GROUPS.values() for mid, lab in year_list}
    for mid, score in scores.items():
        if mid not in RELEASE_DATES or mid not in label_for:
            continue
        points.append((RELEASE_DATES[mid], score, mid, label_for[mid], family_for(mid)))
    points.sort(key=lambda p: p[0])

    # Larger figure + automatic label-overlap avoidance via adjustText
    fig, ax = plt.subplots(figsize=(14, 8.0))

    # Per-family scatter (single legend handle per family)
    drawn = set()
    texts = []
    for x, y, mid, lab, fam in points:
        disp, color = FAMILY_STYLE.get(fam, (fam, "#444"))
        ax.scatter(
            x, y, s=75, color=color, edgecolor="white", linewidth=1.0,
            zorder=3, label=disp if fam not in drawn else None,
        )
        drawn.add(fam)
        # Collect text objects for adjustText to lay out without overlap
        texts.append(ax.text(x, y, lab, fontsize=8.0, color="#222", zorder=4))

    # Per-quarter rolling envelope (max/min within ±0.25 year window) —
    # purely visual cue for "ceiling rising, floor lifting".
    if points:
        xs = [p[0] for p in points]
        x_min, x_max = min(xs), max(xs)
        span = x_max - x_min
        sample_xs = [x_min + span * i / 80 for i in range(81)]
        win = 0.4
        roll_top, roll_bot = [], []
        for sx in sample_xs:
            vals = [p[1] for p in points if abs(p[0] - sx) <= win]
            if vals:
                roll_top.append(max(vals))
                roll_bot.append(min(vals))
            else:
                roll_top.append(None)
                roll_bot.append(None)

        def _draw(line_vals, color, label):
            seg_x, seg_y = [], []
            for sx, v in zip(sample_xs, line_vals):
                if v is None:
                    if seg_x:
                        ax.plot(seg_x, seg_y, color=color, linewidth=1.4,
                                alpha=0.5, linestyle="--",
                                label=label if seg_x and label else None,
                                zorder=1)
                        label = None
                        seg_x, seg_y = [], []
                else:
                    seg_x.append(sx)
                    seg_y.append(v)
            if seg_x:
                ax.plot(seg_x, seg_y, color=color, linewidth=1.4, alpha=0.5,
                        linestyle="--", label=label, zorder=1)

        _draw(roll_top, "#16a34a", "Rolling max (±0.4 yr)")
        _draw(roll_bot, "#dc2626", "Rolling min (±0.4 yr)")

    # Year boundary lines (subtle)
    for yr in range(2023, 2027):
        ax.axvline(x=yr, color="gray", linestyle=":", linewidth=0.6, alpha=0.35)

    ax.set_xlabel("Model release date", fontsize=12)
    ax.set_ylabel("Static-Mode score (weighted, out of 10)", fontsize=12)
    ax.set_xlim(2023.0, 2026.6)
    # X-axis ticks: whole years as major (labeled), quarterly as minor (unlabeled)
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x))}"))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="x", which="minor", length=3)
    # Tighter y range now that all 6 new models include credit-fix rerun;
    # widen automatically if any score falls outside.
    score_vals = [p[1] for p in points] if points else [4.0, 7.5]
    y_lo = min(3.5, min(score_vals) - 0.2)
    y_hi = max(7.6, max(score_vals) + 0.2)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(
        f"Static-Mode performance vs. model release year\n"
        f"Each dot = one of {len(points)} cross-year idea models",
        fontsize=12, pad=14,
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)

    # Apply automatic label-overlap avoidance now that all artists are drawn.
    # only_move='y' restricts label motion to vertical: prevents labels from
    # drifting horizontally into the wrong year (observed: o1 dot at 2025.025
    # had its label pushed to ~2024.95 before this fix).
    if _HAS_ADJUST_TEXT and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.5, alpha=0.6),
            expand=(1.2, 1.4),
            force_text=(0.0, 0.6),
            force_static=(0.0, 0.4),
            force_pull=(1.0, 0.05),
            only_move={"text": "y", "static": "y", "explode": "y"},
        )

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def write_summary_json(scores, year_data, out_path):
    summary = {
        year: {
            "models": [
                {
                    "label": lab,
                    "model_id": mid,
                    "score": scores.get(mid),
                    "release_date_decimal": RELEASE_DATES.get(mid),
                }
                for mid, lab in YEAR_GROUPS[year]
            ],
            "min": min((v for _, v in year_data[year]), default=None),
            "max": max((v for _, v in year_data[year]), default=None),
            "spread": (max(v for _, v in year_data[year]) -
                       min(v for _, v in year_data[year]))
                      if year_data[year] else None,
            "n": len(year_data[year]),
        }
        for year in YEAR_GROUPS
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


def main():
    scores = load_static_scores(ROOT / "data" / "results.db")

    # Build per-year data
    year_data = {}
    for year, models in YEAR_GROUPS.items():
        vals = []
        for mid, label in models:
            if mid in scores:
                vals.append((label, scores[mid]))
        year_data[year] = vals
        if vals:
            print(f"{year}: {len(vals)} models, "
                  f"range [{min(v for _,v in vals):.3f}, {max(v for _,v in vals):.3f}], "
                  f"spread {max(v for _,v in vals) - min(v for _,v in vals):.3f}")
        else:
            print(f"{year}: 0 models")

    out_dir = ROOT / "reports"
    make_box_plot(year_data,
                  out_dir / "cross_year_static.pdf",
                  out_dir / "cross_year_static.png")
    make_scatter_plot(scores,
                      out_dir / "cross_year_scatter.pdf",
                      out_dir / "cross_year_scatter.png")
    write_summary_json(scores, year_data, out_dir / "cross_year_static.json")


if __name__ == "__main__":
    main()
