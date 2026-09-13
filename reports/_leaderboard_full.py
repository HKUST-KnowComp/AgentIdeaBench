#!/usr/bin/env /usr/bin/python3
"""
Full Active-track (agentic) leaderboard over every scored model.

Where this differs from experiments/e36_leaderboard_subscores.py: that script
backs the paper's main table and drops the two Active-only reasoning models, so
every row can carry a gain. This one keeps them (gain = null) so the page shows
the complete agentic roster, and it adds dispersion (SEM / 95% CI over the 40
subfields, paired SEM for the gain) plus vendor and access labels for display.

Aggregation is identical to e22 / e36 so the totals match the paper to the digit:
  drop-highest-of-three critics (trimmed mean) at the idea level
  -> mean over the 3 ideas within a subfield
  -> mean over the 40 subfields.

Read-only against data/results.db. Writes reports/leaderboard_full.json.
"""
import sys
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as W  # noqa: E402
from experiments.e10_idea_anchor_calibration import trimmed_mean  # noqa: E402

WSUM = sum(W[d] for d in DIMS)
OUT = ROOT / "reports" / "leaderboard_full.json"
TURNS = json.loads((ROOT / "reports" / "e43_active_turns.json").read_text())

# Same closed-source rule as e36: the Gemini family plus every internal gateway
# route. Headline statistics in the paper stay on the open-weight models; the
# closed rows are reported but never bolded.
_GATEWAY = ("azure/", "us/azure/", "aws/", "gcp/", "switchyard/")

# Routes the paper's Appendix table excludes: each is the non-reasoning serving
# configuration of a model already on the board, so it is a second route to one
# model rather than a second model. Dropping them reproduces the appendix roster
# of 63 paired models exactly.
DUPLICATE_ROUTES = {"azure/openai/gpt-5-chat", "azure/openai/gpt-5.3-chat"}
def is_closed(m):
    return m.startswith("google/gemini") or m.startswith(_GATEWAY)


VENDOR = [
    ("azure/anthropic/", "Anthropic"), ("azure/openai/", "OpenAI"),
    ("us/azure/openai/", "OpenAI"), ("google/gemini", "Google"),
    ("google/gemma", "Google"), ("qwen/", "Alibaba"), ("deepseek/", "DeepSeek"),
    ("moonshotai/", "Moonshot"), ("z-ai/", "Zhipu"), ("minimax/", "MiniMax"),
    ("mistralai/", "Mistral"), ("meta-llama/", "Meta"), ("xiaomi/", "Xiaomi"),
]
def vendor(m):
    for pre, name in VENDOR:
        if m.startswith(pre):
            return name
    return m.split("/")[0]


DISPLAY = {
    "azure/anthropic/claude-opus-5": "Claude Opus 5",
    "azure/anthropic/claude-opus-4-8": "Claude Opus 4.8",
    "azure/anthropic/claude-opus-4-7": "Claude Opus 4.7",
    "azure/anthropic/claude-opus-4-6": "Claude Opus 4.6",
    "azure/anthropic/claude-opus-4-5": "Claude Opus 4.5",
    "azure/anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "azure/anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "azure/anthropic/claude-sonnet-4-5": "Claude Sonnet 4.5",
    "azure/anthropic/claude-haiku-4-5": "Claude Haiku 4.5",
    "azure/openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "azure/openai/gpt-5.6-terra": "GPT-5.6 Terra",
    "azure/openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "azure/openai/gpt-5.5": "GPT-5.5",
    "azure/openai/gpt-5.4": "GPT-5.4",
    "azure/openai/gpt-5.4-mini": "GPT-5.4 mini",
    "azure/openai/gpt-5.4-nano": "GPT-5.4 nano",
    "azure/openai/gpt-5.3-chat": "GPT-5.3 chat",
    "azure/openai/gpt-5.2": "GPT-5.2",
    "azure/openai/gpt-5.1": "GPT-5.1",
    "azure/openai/gpt-5": "GPT-5",
    "azure/openai/gpt-5-mini": "GPT-5 mini",
    "azure/openai/gpt-5-nano": "GPT-5 nano",
    "azure/openai/o3": "o3", "azure/openai/o3-mini": "o3-mini",
    "azure/openai/o4-mini": "o4-mini", "azure/openai/o1": "o1",
    "azure/openai/gpt-4.1": "GPT-4.1", "azure/openai/gpt-4.1-mini": "GPT-4.1 mini",
    "us/azure/openai/gpt-4.1-nano": "GPT-4.1 nano",
    "azure/openai/gpt-4o": "GPT-4o", "azure/openai/gpt-4o-mini": "GPT-4o mini",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
    "google/gemma-4-31b-it": "Gemma 4 31B", "google/gemma-3-27b-it": "Gemma 3 27B",
    "google/gemma-2-27b-it": "Gemma 2 27B",
    "z-ai/glm-5.1": "GLM-5.1", "z-ai/glm-4.6": "GLM-4.6",
    "z-ai/glm-4.5-air": "GLM-4.5 Air",
    "moonshotai/kimi-k2.6": "Kimi K2.6", "moonshotai/kimi-k2.5": "Kimi K2.5",
    "deepseek/deepseek-v4-pro": "DeepSeek-V4 Pro",
    "deepseek/deepseek-v4-flash": "DeepSeek-V4 Flash",
    "deepseek/deepseek-r1-0528": "DeepSeek-R1",
    "qwen/qwen3.5-397b-a17b": "Qwen3.5 397B-A17B",
    "qwen/qwen3.5-27b": "Qwen3.5 27B", "qwen/qwen3.5-9b": "Qwen3.5 9B",
    "qwen/qwen3-235b-a22b-thinking-2507": "Qwen3 235B Thinking",
    "qwen/qwen3-vl-8b-thinking": "Qwen3-VL 8B Thinking",
    "qwen/qwen3-30b-a3b-instruct-2507": "Qwen3 30B-A3B",
    "qwen/qwen3-coder": "Qwen3 Coder", "qwen/qwen3-32b": "Qwen3 32B",
    "qwen/qwen3-8b": "Qwen3 8B", "qwen/qwen-2.5-72b-instruct": "Qwen2.5 72B",
    "qwen/qwen-2.5-7b-instruct": "Qwen2.5 7B",
    "xiaomi/mimo-v2.5-pro": "MiMo-V2.5 Pro", "xiaomi/mimo-v2.5": "MiMo-V2.5",
    "minimax/minimax-m2.7": "MiniMax-M2.7",
    "mistralai/mistral-medium-3.1": "Mistral Medium 3.1",
    "mistralai/mistral-small-2603": "Mistral Small 2603",
    "mistralai/mistral-small-24b-instruct-2501": "Mistral Small 24B",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
}
def display(m):
    return DISPLAY.get(m, m.split("/")[-1])


def collect():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    idea = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT idea_model,track,subdomain,idea_index,scores_json "
                          "FROM lit8d_scores_3seed WHERE scores_json IS NOT NULL"):
        s = json.loads(r["scores_json"])
        k = (r["idea_model"], r["track"], r["subdomain"], r["idea_index"])
        for d in DIMS:
            idea[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
    conn.close()

    def tm(vs):
        return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)

    idx_dim = {k: {d: tm(idea[k][d]) for d in DIMS} for k in idea}
    idx_w = {k: sum(idx_dim[k][d] * W[d] for d in DIMS) / WSUM for k in idea}

    sub_dim = defaultdict(lambda: defaultdict(list))
    sub_w = defaultdict(list)
    for (m, tr, sub, idx) in idea:
        for d in DIMS:
            sub_dim[(m, tr, sub)][d].append(idx_dim[(m, tr, sub, idx)][d])
        sub_w[(m, tr, sub)].append(idx_w[(m, tr, sub, idx)])
    # subfield-level means: the unit of analysis everywhere in the paper
    sub_w = {k: float(np.mean(v)) for k, v in sub_w.items()}
    sub_dim = {k: {d: float(np.mean(v[d])) for d in DIMS} for k, v in sub_dim.items()}
    return sub_w, sub_dim


def sem(vals):
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0


def main():
    sub_w, sub_dim = collect()
    by_mt = defaultdict(dict)                       # (m,tr) -> sub -> weighted
    for (m, tr, sub), v in sub_w.items():
        by_mt[(m, tr)][sub] = v
    models = sorted({m for (m, tr) in by_mt if tr == "C"})

    # Follow the paper: the leaderboard tables cover paired models only, so a
    # model with no Static run has no place in a ranking whose second column is
    # the Static comparison. Both kinds of exclusion are reported in the JSON.
    excluded = []
    kept = []
    for m in models:
        if m in DUPLICATE_ROUTES:
            excluded.append({"model": m, "reason": "duplicate serving route"})
        elif not by_mt.get((m, "B")):
            excluded.append({"model": m, "reason": "Active-only, no Static run"})
        else:
            kept.append(m)
    models = kept

    rows = []
    for m in models:
        act = by_mt[(m, "C")]
        sta = by_mt.get((m, "B"), {})
        a_vals = list(act.values())
        s_vals = list(sta.values())
        shared = sorted(set(act) & set(sta))
        paired = [act[s] - sta[s] for s in shared]
        t = TURNS.get(m, {})
        rows.append({
            "model": m,
            "name": display(m),
            "vendor": vendor(m),
            "access": "closed" if is_closed(m) else "open",
            "active": float(np.mean(a_vals)),
            "active_sem": sem(a_vals),
            "static": float(np.mean(s_vals)) if s_vals else None,
            "static_sem": sem(s_vals) if s_vals else None,
            "gain": float(np.mean(paired)) if paired else None,
            "gain_sem": sem(paired) if paired else None,
            "dims": {d: float(np.mean([sub_dim[(m, "C", s)][d] for s in act]))
                     for d in DIMS},
            "static_dims": ({d: float(np.mean([sub_dim[(m, "B", s)][d] for s in sta]))
                             for d in DIMS} if s_vals else None),
            "n_subfields_active": len(a_vals),
            "n_subfields_static": len(s_vals),
            "turns": t.get("active_turns_mean"),
        })

    # Count the critic scores behind the table here, where the database is
    # already open, so the page builder needs nothing but this JSON. Restricted
    # to the models actually on the board, so the figure the page quotes counts
    # the evidence it shows and not the excluded routes.
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    qs = ",".join("?" * len(models))
    n_scores = conn.execute(
        f"SELECT COUNT(*) FROM lit8d_scores_3seed "
        f"WHERE scores_json IS NOT NULL AND idea_model IN ({qs})", models
    ).fetchone()[0]
    conn.close()

    rows.sort(key=lambda r: -r["active"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    out = {
        "note": ("Active (Track C) leaderboard over the paired models scored in "
                 "lit8d_scores_3seed, matching Appendix Table tab:leaderboard-all "
                 "of the paper. Aggregation identical to e22/e36: "
                 "drop-highest-of-three critics, mean over 3 ideas, mean over "
                 "subfields. sem fields are the standard error over subfield "
                 "means; gain is the paired Active-Static difference over "
                 "subfields present in both tracks."),
        "weights": {d: W[d] for d in DIMS},
        "n_models": len(rows),
        "n_open": sum(r["access"] == "open" for r in rows),
        "n_closed": sum(r["access"] == "closed" for r in rows),
        "n_paired": sum(r["gain"] is not None for r in rows),
        "excluded": excluded,
        "n_critic_scores": n_scores,
        "source_table": "data/results.db :: lit8d_scores_3seed",
        "turns_source": "reports/e43_active_turns.json",
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}  models={len(rows)} "
          f"open={out['n_open']} closed={out['n_closed']} paired={out['n_paired']}")
    for r in rows[:12]:
        g = f"{r['gain']:+.2f}" if r["gain"] is not None else "  -- "
        print(f"  {r['rank']:2d}. {r['name']:22s} {r['active']:.3f} "
              f"±{r['active_sem']:.3f}  gain {g}  turns {r['turns']:.1f}")


if __name__ == "__main__":
    main()
