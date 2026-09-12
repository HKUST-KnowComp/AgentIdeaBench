#!/usr/bin/env /usr/bin/python3
"""
E36 — Active-track per-dimension leaderboard for the main results table.

Reuses the exact aggregation of experiments/e22_new_axis_stats.py (trimmed-mean
over critics = drop-highest-of-three, then per-(model,track,subdomain) idx-mean,
then mean over subdomains), so the weighted Active total here matches the
leaderboard/e22 numbers. Adds the five per-dimension Active sub-scores.

Roster: 28 open-weight paired + 5 held-out Gemini paired + 5 held-out 2026 frontier
paired (E41: gpt-5.6-sol/terra/luna, claude-opus-5, claude-sonnet-5) = 38 (the two
Active-only reasoning models have no Static and are excluded from a gain table).
Any model with both tracks in lit8d_scores_3seed is picked up automatically.

Read-only. Writes reports/e36_leaderboard_subscores.json and prints a LaTeX
table body (sorted by Active total ascending; per-column top-1 bold, top-2/3
underlined). Nothing runs on import.
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
OUT = ROOT / "reports" / "e36_leaderboard_subscores.json"

# Active-only reasoning models (no Static path) -> excluded from a gain table.
ACTIVE_ONLY = {"qwen/qwen3-235b-a22b-thinking-2507", "qwen/qwen3-vl-8b-thinking"}
# Held-out closed frontier models: the Gemini family (E33) plus the 2026 frontier
# routes reached through the internal NVIDIA gateway (E41 — gpt-5.6-*, claude-*).
# Both groups are reported but never bolded; headline statistics stay on the 28
# open-weight paired models.
# NOTE: "us/azure/" is the US-region variant of the same gateway route; without it
# us/azure/openai/gpt-4.1-nano (closed OpenAI) would be mis-labelled open-weight.
_CLOSED_GATEWAY_PREFIXES = ("azure/", "us/azure/", "aws/", "gcp/", "switchyard/")
CLOSED = lambda m: (m.startswith("google/gemini")
                    or m.startswith(_CLOSED_GATEWAY_PREFIXES))


def shorten(m):
    s = m.split("/")[-1]
    repl = {
        "qwen3.5-397b-a17b": "qwen3.5-397b", "qwen3-30b-a3b-instruct-2507": "qwen3-30b-instruct",
        "qwen-2.5-72b-instruct": "qwen-2.5-72b", "qwen-2.5-7b-instruct": "qwen-2.5-7b",
        "mistral-small-24b-instruct-2501": "mistral-small-24b", "llama-3.1-8b-instruct": "llama-3.1-8b",
        "gemma-2-27b-it": "gemma-2-27b", "gemma-3-27b-it": "gemma-3-27b", "gemma-4-31b-it": "gemma-4-31b",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite", "gemini-3.1-pro-preview": "gemini-3.1-pro",
        "gemini-3-flash-preview": "gemini-3-flash",
    }
    return repl.get(s, s)


def aggregate():
    """Return per-model dict with Active sub-scores + weighted totals + gain."""
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

    # idx-level trimmed per-dim values + weighted total
    idx_dim = {k: {d: tm(idea[k][d]) for d in DIMS} for k in idea}
    idx_w = {k: sum(idx_dim[k][d] * W[d] for d in DIMS) / WSUM for k in idea}

    # collapse idx -> subdomain mean, then mean over subdomains, per (model,track)
    sub_dim = defaultdict(lambda: defaultdict(list))     # (m,tr,sub) -> dim -> [idx vals]
    sub_w = defaultdict(list)                            # (m,tr,sub) -> [idx weighted]
    for (m, tr, sub, idx) in idea:
        for d in DIMS:
            sub_dim[(m, tr, sub)][d].append(idx_dim[(m, tr, sub, idx)][d])
        sub_w[(m, tr, sub)].append(idx_w[(m, tr, sub, idx)])

    mt_dim = defaultdict(lambda: defaultdict(list))      # (m,tr) -> dim -> [sub means]
    mt_w = defaultdict(list)
    for (m, tr, sub) in sub_w:
        for d in DIMS:
            mt_dim[(m, tr)][d].append(float(np.mean(sub_dim[(m, tr, sub)][d])))
        mt_w[(m, tr)].append(float(np.mean(sub_w[(m, tr, sub)])))

    models = sorted({m for (m, tr) in mt_w})
    out = {}
    for m in models:
        if m in ACTIVE_ONLY:
            continue
        if ("B") not in [tr for (mm, tr) in mt_w if mm == m] or \
           ("C") not in [tr for (mm, tr) in mt_w if mm == m]:
            continue
        b = float(np.mean(mt_w[(m, "B")])); c = float(np.mean(mt_w[(m, "C")]))
        active_dims = {d: float(np.mean(mt_dim[(m, "C")][d])) for d in DIMS}
        out[m] = {
            "short": shorten(m), "closed": CLOSED(m),
            "static_total": b, "active_total": c, "gain": c - b,
            "active_dims": active_dims,
        }
    return out


def latex_table(agg):
    """Sorted by Active total ascending; per-column top1 bold, top2/3 underline."""
    rows = sorted(agg.values(), key=lambda r: r["active_total"])
    cols = {d: [r["active_dims"][d] for r in rows] for d in DIMS}
    cols["active_total"] = [r["active_total"] for r in rows]
    cols["gain"] = [r["gain"] for r in rows]

    def ranks(vals):   # index-> style; top1 bold, top2/3 underline (ties share)
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        style = {}
        for rank, i in enumerate(order):
            style[i] = "b" if rank == 0 else ("u" if rank in (1, 2) else "")
        return style

    styles = {col: ranks(v) for col, v in cols.items()}

    def fmt(col, i):
        v = cols[col][i]
        txt = f"{v:+.2f}" if col == "gain" else f"{v:.2f}"
        st = styles[col][i]
        if st == "b":
            return f"\\textbf{{{txt}}}"
        if st == "u":
            return f"\\underline{{{txt}}}"
        return txt

    lines = []
    for i, r in enumerate(rows):
        name = r["short"] + (r"$^\dagger$" if r["closed"] else "")
        cells = " & ".join(fmt(d, i) for d in DIMS)
        lines.append(f"{name} & {cells} & {fmt('active_total', i)} & {fmt('gain', i)} \\\\")
    return "\n".join(lines), len(rows)


def main():
    agg = aggregate()
    OUT.write_text(json.dumps(agg, indent=2))
    body, n = latex_table(agg)
    print(f"% e36 leaderboard: {n} paired models "
          f"({sum(r['closed'] for r in agg.values())} held-out closed-source, marked †)")
    print(body)
    # sanity: print a few active totals to cross-check vs e22 / tab:leaderboard
    print("\n% cross-check Active totals (should match e22 per_model 'active'):")
    for m in ("z-ai/glm-5.1", "deepseek/deepseek-v4-pro", "google/gemma-2-27b-it"):
        if m in agg:
            print(f"%   {agg[m]['short']:20s} static={agg[m]['static_total']:.2f} "
                  f"active={agg[m]['active_total']:.2f} gain={agg[m]['gain']:+.2f}")


if __name__ == "__main__":
    main()
