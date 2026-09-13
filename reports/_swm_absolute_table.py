#!/usr/bin/env python3
"""Absolute-value ablation table for tab:swm (baseline -> methods, per backbone).

Recomputes per-backbone absolute weighted scores for the closed-loop SWM designs
on the COMMON paired cells (subdomains where active_base AND all three shown
designs are present for that backbone), so the single baseline row is internally
consistent with every method row (method - baseline = paired delta on the same
cells). Uses the exact scoring aggregation from e27_swm (per-critic trimmed mean
-> originality-weighted total).

Read-only against data/results.db. Prints the LaTeX-ready numbers.
"""
import os, sys, sqlite3, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import defaultdict
from utils.constants import SCORING_DIMS as DIMS, SCORING_WEIGHTS as W
from experiments.e10_idea_anchor_calibration import trimmed_mean

WSUM = sum(W[k] for k in DIMS)
def weighted(d):  # originality-weighted total, /5.5, matching e27_swm._weighted
    return sum(d[k] * W[k] for k in DIMS) / WSUM
def tm(vs):
    return trimmed_mean(vs) if len(vs) >= 2 else (vs[0] if vs else None)

BACKBONES = [("qwen/qwen3.5-9b", "9b"), ("qwen/qwen3.5-27b", "27b"),
             ("deepseek/deepseek-v4-flash", "v4-fl"), ("deepseek/deepseek-v4-pro", "v4-pr")]
CONDS = ["active_base", "swm_S2", "swm_S4", "swm_S4b"]
LABEL = {"active_base": "Active (no SWM)", "swm_S2": "+ multi-role",
         "swm_S4": "+ dynamic panel", "swm_S4b": "+ dynamic + hard"}

conn = sqlite3.connect("data/results.db"); conn.row_factory = sqlite3.Row
cell = defaultdict(lambda: defaultdict(list))
for r in conn.execute("SELECT gen_model,subdomain,condition,scores_json "
                      "FROM swm_scores WHERE scores_json IS NOT NULL"):
    s = json.loads(r["scores_json"]); k = (r["gen_model"], r["subdomain"], r["condition"])
    for d in DIMS:
        cell[k][d].append(s[d]["score"] if isinstance(s[d], dict) else s[d])
wcell = {k: weighted({d: tm(cell[k][d]) for d in DIMS}) for k in cell}
byms = defaultdict(dict)
for (m, sub, c), w in wcell.items():
    byms[(m, sub)][c] = w

absr, ns = {}, {}
for full, sh in BACKBONES:
    common = [sub for (m, sub) in byms if m == full
              and all(c in byms[(full, sub)] for c in CONDS)]
    ns[sh] = len(common)
    for c in CONDS:
        absr[(sh, c)] = float(np.mean([byms[(full, sub)][c] for sub in common]))

print("n per backbone:", ns)
print(f"{'Design':20}" + "".join(f"{sh:>8}" for _, sh in BACKBONES))
for c in CONDS:
    print(f"{LABEL[c]:20}" + "".join(f"{absr[(sh, c)]:>8.2f}" for _, sh in BACKBONES))
print("\ncolumn max (bold in table):")
for _, sh in BACKBONES:
    best = max(CONDS, key=lambda c: absr[(sh, c)])
    print(f"  {sh}: {LABEL[best]} ({absr[(sh, best)]:.2f})")
print("\nper-backbone S4b delta vs base (should match intro +0.61/+0.14/-0.06):")
for _, sh in BACKBONES:
    print(f"  {sh}: {absr[(sh,'swm_S4b')] - absr[(sh,'active_base')]:+.2f}")
