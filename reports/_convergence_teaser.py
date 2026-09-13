#!/usr/bin/env python3
"""Page-1 teaser: Static compresses strong models, Active re-separates them.

Numbers are taken verbatim from the Active-track leaderboard (tab:leaderboard)
in docs/paper/acl_latex.tex so the figure is guaranteed consistent with the
paper. Static total = Active total - Delta. No database access.

Output: docs/paper/convergence_separation.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (model, Active total, Delta = Active - Static) -- 28 open-weight paired models
OPEN = [
    ("gemma-2-27b", 3.20, -0.82),
    ("llama-3.1-8b", 3.54, -0.26),
    ("qwen-2.5-7b", 3.67, -0.30),
    ("llama-4-maverick", 3.84, -0.50),
    ("mistral-small-24b", 3.93, -0.26),
    ("qwen-2.5-72b", 3.97, -0.23),
    ("qwen3-8b", 4.21, -0.03),
    ("qwen3-32b", 4.44, -0.10),
    ("qwen3-coder", 4.72, -0.15),
    ("glm-4.5-air", 4.82, +0.51),
    ("gemma-3-27b", 4.82, +0.14),
    ("qwen3.5-9b", 4.87, +0.64),
    ("qwen3-30b-instruct", 4.94, +0.01),
    ("mistral-small-2603", 5.16, +0.55),
    ("deepseek-r1-0528", 5.30, +0.44),
    ("minimax-m2.7", 5.33, +0.62),
    ("mistral-medium-3.1", 5.37, +0.62),
    ("mimo-v2.5", 5.46, +0.56),
    ("gemma-4-31b", 5.49, +0.35),
    ("deepseek-v4-flash", 5.52, +0.42),
    ("qwen3.5-27b", 5.53, +0.82),
    ("glm-4.6", 5.78, +0.95),
    ("mimo-v2.5-pro", 5.91, +0.81),
    ("deepseek-v4-pro", 5.92, +0.60),
    ("kimi-k2.5", 5.92, +1.02),
    ("qwen3.5-397b", 5.95, +0.89),
    ("kimi-k2.6", 6.17, +1.08),
    ("glm-5.1", 6.33, +1.21),
]
# Held-out Gemini family (dagger) -- shown lightly, not in headline stats
GEMINI = [
    ("gemini-2.5-flash-lite", 4.27, -0.37),
    ("gemini-2.5-flash", 5.05, -0.05),
    ("gemini-3.1-pro", 6.20, +0.57),
    ("gemini-3.5-flash", 6.22, +0.35),
    ("gemini-3-flash", 6.29, +0.70),
]

def static_of(rows):
    return np.array([a - d for (_, a, d) in rows])

def active_of(rows):
    return np.array([a for (_, a, d) in rows])

os_static = static_of(OPEN)
os_active = active_of(OPEN)
os_delta = np.array([d for (_, _, d) in OPEN])

# ---- verification stats (printed, cross-checked against S5.1) ----
def rng(x):
    return float(x.min()), float(x.max()), float(x.max() - x.min())

print("== open-weight (n=%d) ==" % len(OPEN))
print("Static range  min/max/spread: %.2f %.2f %.2f" % rng(os_static))
print("Active range  min/max/spread: %.2f %.2f %.2f" % rng(os_active))
print("var(Active)/var(Static)     : %.2f" % (os_active.var(ddof=1) / os_static.var(ddof=1)))
print("std(Active)/std(Static)     : %.2f" % (os_active.std(ddof=1) / os_static.std(ddof=1)))
print("Active > Static count       : %d/%d" % (int((os_active > os_static).sum()), len(OPEN)))
# top-8 by Static
order = np.argsort(os_static)
top8 = order[-8:]
print("top-8-by-Static Static spread: %.2f" % (os_static[top8].max() - os_static[top8].min()))
print("top-8-by-Static Active spread: %.2f" % (os_active[top8].max() - os_active[top8].min()))

# ---- plot ----
plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans",
})
fig, ax = plt.subplots(figsize=(3.4, 3.1), dpi=200)

lo, hi = 3.0, 6.7
ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#9aa0a6", zorder=1,
        label="$y=x$ (no change)")

# open-weight: green helped / red hurt
helped = os_delta > 0
ax.scatter(os_static[helped], os_active[helped], s=42, c="#1a9c74",
           edgecolors="white", linewidths=0.6, zorder=3)
ax.scatter(os_static[~helped], os_active[~helped], s=42, c="#d23b3b",
           edgecolors="white", linewidths=0.6, zorder=3)

# annotate the two endpoints of the story
ann = {"glm-5.1": (4, 6), "gemma-2-27b": (6, -10)}
for (name, a, d) in OPEN:
    if name in ann:
        s = a - d
        dx, dy = ann[name]
        ax.annotate(name, (s, a), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.5, color="#333333")

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Static score (curated refs)")
ax.set_ylabel("Active score (agent retrieval)")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, lw=0.4, color="#e8e8e8", zorder=0)
ax.tick_params(length=2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# compact stat box
ax.text(0.03, 0.97,
        "Static span 1.53\n4.4$\\times$ variance under Active\n19/28 above $y{=}x$",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.6))
ax.legend(loc="lower right", fontsize=6.2, frameon=False, handlelength=1.6)

fig.tight_layout(pad=0.4)
out = os.path.join(os.path.dirname(__file__), "..", "docs", "paper",
                   "convergence_separation.png")
out = os.path.normpath(out)
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
