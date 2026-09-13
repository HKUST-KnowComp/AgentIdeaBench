#!/usr/bin/env python
"""Two-panel summary figure for Experiment 20 (E41, frontier closed-source extension).

Panel A: Static -> Active dumbbells for the top 15 models by Active score, with the
five 2026 frontier models highlighted.
Panel B: the capability gate (gain vs static ability) with two fits -- the paper's
open-weight roster (n=28) and the full extended roster (n=38) -- showing the collapse.

Read-only w.r.t. the DB: reads reports/e36_leaderboard_subscores.json only.
Writes reports/figures/summary/e41_frontier.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "reports" / "e36_leaderboard_subscores.json"
OUT = REPO / "reports" / "figures" / "summary" / "e41_frontier.png"

# Categorical hues assigned by group identity, fixed order, never cycled.
C_FRONTIER = "#c1440e"   # frontier 2026 closed
C_GEMINI = "#8a6d1f"     # held-out closed
C_OPEN = "#1f4e79"       # open-weight roster
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

GATEWAY_PREFIXES = ("azure/", "aws/", "gcp/", "switchyard/")


def group_of(model_id: str) -> str:
    if model_id.startswith(GATEWAY_PREFIXES):
        return "frontier"
    if model_id.startswith("google/gemini"):
        return "gemini"
    return "open"


COLOR = {"frontier": C_FRONTIER, "gemini": C_GEMINI, "open": C_OPEN}
LABEL = {
    "frontier": "2026 frontier closed (new)",
    "gemini": "held-out closed (Gemini)",
    "open": "open-weight roster",
}


def main() -> None:
    data = json.loads(SRC.read_text())
    rows = [
        {
            "id": mid,
            "short": v["short"],
            "group": group_of(mid),
            "static": v["static_total"],
            "active": v["active_total"],
            "gain": v["gain"],
        }
        for mid, v in data.items()
    ]
    rows.sort(key=lambda r: -r["active"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.6))

    # ---------- Panel A: top-15 dumbbells ----------
    top = rows[:15][::-1]  # bottom-up so rank 1 sits at the top
    y = np.arange(len(top))
    for i, r in enumerate(top):
        c = COLOR[r["group"]]
        axL.plot([r["static"], r["active"]], [i, i], color=c, lw=2, alpha=0.55,
                 solid_capstyle="round", zorder=1)
        axL.scatter(r["static"], i, s=46, facecolor="white", edgecolor=c,
                    lw=1.8, zorder=3)
        axL.scatter(r["active"], i, s=54, color=c, edgecolor="white", lw=1.2, zorder=4)

    axL.set_yticks(y)
    axL.set_yticklabels([r["short"] for r in top], fontsize=9, color=INK)
    axL.set_xlabel("weighted lit8d score", fontsize=10, color=INK)
    axL.set_title("A. Top 15 by Active score: Static → Active",
                  fontsize=11, color=INK, loc="left", pad=10)
    axL.grid(axis="x", color=GRID, lw=0.7)
    axL.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axL.spines[s].set_visible(False)
    axL.spines["bottom"].set_color(GRID)
    axL.tick_params(length=0, colors=MUTED)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COLOR[g], label=LABEL[g], markersize=7)
        for g in ("frontier", "gemini", "open")
    ]
    handles += [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor=MUTED, label="Static (hollow)", markersize=7),
        plt.Line2D([], [], marker="o", ls="", color=MUTED, label="Active (filled)",
                   markersize=7),
    ]
    axL.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")

    # ---------- Panel B: the capability gate, n=28 vs n=38 ----------
    def fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
        r, p = stats.pearsonr(xs, ys)
        slope, intercept = np.polyfit(xs, ys, 1)
        return r, p, slope, intercept

    open_rows = [r for r in rows if r["group"] == "open"]
    x_open = np.array([r["static"] for r in open_rows])
    y_open = np.array([r["gain"] for r in open_rows])
    x_all = np.array([r["static"] for r in rows])
    y_all = np.array([r["gain"] for r in rows])

    r28, p28, s28, i28 = fit(x_open, y_open)
    r38, p38, s38, i38 = fit(x_all, y_all)

    axR.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for g in ("open", "gemini", "frontier"):
        pts = [r for r in rows if r["group"] == g]
        axR.scatter([r["static"] for r in pts], [r["gain"] for r in pts],
                    s=58, color=COLOR[g], edgecolor="white", lw=1.2,
                    label=LABEL[g], zorder=3)

    xs28 = np.linspace(x_open.min(), x_open.max(), 100)
    axR.plot(xs28, s28 * xs28 + i28, color=C_OPEN, lw=2,
             label=f"fit, open-weight n=28 (r={r28:+.2f}, p<1e-4)", zorder=2)
    xs38 = np.linspace(x_all.min(), x_all.max(), 100)
    axR.plot(xs38, s38 * xs38 + i38, color=INK, lw=2, ls=(0, (5, 2)),
             label=f"fit, all n=38 (r={r38:+.2f}, p={p38:.2f} n.s.)", zorder=2)

    # sol and terra land almost on top of each other; nudge their labels apart.
    label_offset = {"gpt-5.6-sol": (7, 7), "gpt-5.6-terra": (7, -10)}
    for r in rows:
        if r["group"] == "frontier":
            axR.annotate(r["short"], (r["static"], r["gain"]),
                         textcoords="offset points",
                         xytext=label_offset.get(r["short"], (6, 5)),
                         fontsize=7.5, color=C_FRONTIER)
    best_open = max(open_rows, key=lambda r: r["gain"])
    axR.annotate(best_open["short"], (best_open["static"], best_open["gain"]),
                 textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=C_OPEN)

    # Headroom above the cloud for the legend, and room at the right for the
    # frontier labels, which otherwise run off the axis.
    axR.set_ylim(y_all.min() - 0.18, y_all.max() + 0.62)
    axR.set_xlim(x_all.min() - 0.22, x_all.max() + 0.95)

    axR.set_xlabel("Static ability (weighted lit8d score)", fontsize=10, color=INK)
    axR.set_ylabel("Active − Static gain", fontsize=10, color=INK)
    axR.set_title("B. The capability gate collapses once frontier models enter",
                  fontsize=11, color=INK, loc="left", pad=10)
    axR.grid(color=GRID, lw=0.7)
    axR.set_axisbelow(True)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axR.spines[s].set_color(GRID)
    axR.tick_params(length=0, colors=MUTED)
    axR.legend(fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190, facecolor="white")
    print(f"wrote {OUT}")
    print(f"n=28 r={r28:+.4f} p={p28:.3g} | n=38 r={r38:+.4f} p={p38:.3g}")


if __name__ == "__main__":
    main()
