"""Compare reasoning vs non-reasoning models' Active mode behavior.

Models in our active pool (v1):
  Reasoning:    qwen/qwen3-235b-a22b-thinking-2507, qwen/qwen3-vl-8b-thinking,
                deepseek/deepseek-r1-0528, openai/gpt-5 (reasoning by default)
  Non-reasoning: qwen/qwen3.5-9b, qwen/qwen3.5-27b, qwen/qwen3.5-397b-a17b,
                google/gemma-4-31b-it, z-ai/glm-5.1, moonshotai/kimi-k2.6,
                qwen/qwen-2.5-72b-instruct, meta-llama/llama-3.1-8b-instruct,
                mistralai/mistral-7b-instruct-v0.1, google/gemma-3-27b-it

Comparison:
  - boost
  - per-dim breakdown
  - tool-call counts
"""
import json
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

REASONING_MODELS = {
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-vl-8b-thinking",
    "deepseek/deepseek-r1-0528",
}

NON_REASONING_MODELS = {
    "qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "qwen/qwen3.5-397b-a17b",
    "google/gemma-4-31b-it", "z-ai/glm-5.1", "moonshotai/kimi-k2.6",
    "qwen/qwen-2.5-72b-instruct", "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.1", "google/gemma-3-27b-it",
}


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in DIMS) / WS


def load_v1():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    per_paper = defaultdict(lambda: defaultdict(list))
    for m, t, pid, sj in conn.execute("""
        SELECT idea_model, track, paper_id, scores_json FROM results
        WHERE prompt_version='v1_paper_refs' AND track IN ('B','C') AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        per_paper[(m, t)][pid].append(weighted(s))
    conn.close()
    return per_paper


def load_perdim():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    out = defaultdict(list)
    for m, t, sj in conn.execute("""
        SELECT idea_model, track, scores_json FROM results
        WHERE prompt_version='v1_paper_refs' AND track IN ('B','C') AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
    """):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        for d in DIMS:
            if d in s:
                out[(m, t, d)].append(float(s[d]))
    conn.close()
    return out


def load_tool_counts():
    """For Active mode v1 data, get tool calls per model."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    import re
    counts_per_model = defaultdict(list)
    for m, raw in conn.execute("""
        SELECT idea_model, raw_response FROM results
        WHERE prompt_version='v1_paper_refs' AND track='C' AND idea_index=1
          AND critic_model='' AND raw_response IS NOT NULL
    """):
        if not raw:
            continue
        # Try to parse, fall back to regex tool count
        try:
            d = json.loads(raw)
            if isinstance(d, dict) and "n_tool_calls" in d:
                counts_per_model[m].append(d["n_tool_calls"])
            elif isinstance(d, list):
                counts_per_model[m].append(len([s for s in d if isinstance(s, dict) and s.get("tool")]))
        except Exception:
            tc = len(re.findall(r'"tool"\s*:\s*"', raw))
            if tc > 0:
                counts_per_model[m].append(tc)
    conn.close()
    return counts_per_model


def main():
    print("=" * 80)
    print("REASONING vs NON-REASONING — Active mode behavior")
    print("=" * 80)

    v1 = load_v1()
    perdim = load_perdim()
    tcounts = load_tool_counts()

    # Per-group boost
    group_boost = {"reasoning": [], "non-reasoning": []}
    group_perdim = {"reasoning": defaultdict(list), "non-reasoning": defaultdict(list)}
    group_calls = {"reasoning": [], "non-reasoning": []}

    print(f"\n{'Group':<18} {'Model':<42} {'B':>7} {'C':>7} {'Boost':>7} {'#calls':>7} {'nC':>4}")
    for group, models in [("reasoning", REASONING_MODELS),
                           ("non-reasoning", NON_REASONING_MODELS)]:
        for m in sorted(models):
            b = v1.get((m, "B"), {})
            c = v1.get((m, "C"), {})
            if not b or not c or len(c) < 5:
                continue
            b_vals = [statistics.mean(v) for v in b.values()]
            c_vals = [statistics.mean(v) for v in c.values()]
            b_mean = statistics.mean(b_vals)
            c_mean = statistics.mean(c_vals)
            boost = c_mean - b_mean
            calls = tcounts.get(m, [])
            mc = statistics.mean(calls) if calls else 0
            group_boost[group].append(boost)
            group_calls[group].append(mc)
            for d in DIMS:
                b_d = perdim.get((m, "B", d), [])
                c_d = perdim.get((m, "C", d), [])
                if b_d and c_d:
                    group_perdim[group][d].append(statistics.mean(c_d) - statistics.mean(b_d))
            print(f"  {group:<16} {m:<40} {b_mean:>7.2f} {c_mean:>7.2f} {boost:>+7.2f} {mc:>7.2f} {len(c_vals):>4}")

    print()
    print("Group-level summary:")
    for g in ["reasoning", "non-reasoning"]:
        boosts = group_boost[g]
        calls = group_calls[g]
        if not boosts:
            continue
        print(f"\n  {g.upper()} (n={len(boosts)} models):")
        print(f"    mean boost: {statistics.mean(boosts):+.3f}")
        print(f"    mean #calls: {statistics.mean(calls):.2f}")
        print(f"    per-dim boost:")
        for d in DIMS:
            vals = group_perdim[g][d]
            if vals:
                print(f"      {d:<14}: {statistics.mean(vals):+.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
