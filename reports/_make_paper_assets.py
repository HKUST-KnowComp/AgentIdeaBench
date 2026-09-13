"""Generate paper figures + LaTeX tables from v1 data in results.db.

Outputs:
  docs/paper/figures/finding1_size_vs_boost.{pdf,png}
  docs/paper/figures/finding1_scatter_with_active.{pdf,png}
  docs/paper/figures/finding2_perdim_boost.{pdf,png}
  docs/paper/figures/finding3_perdomain_boost.{pdf,png}
  docs/paper/tables/equalizer_table.tex
  docs/paper/tables/perdim_table.tex
  docs/paper/tables/domain_table.tex
"""
import json
import sqlite3
import statistics
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

PAPER = ROOT / "docs" / "paper"
FIG = PAPER / "figures"
TAB = PAPER / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())

# Rough parameter counts (B = billions). Best-effort; use closest published number.
# For MoE: a_xxxb = active params. Conservatively use total params for scale.
PARAMS_B = {
    "qwen/qwen3.5-9b": 9,
    "qwen/qwen3.5-27b": 27,
    "qwen/qwen3.5-397b-a17b": 397,  # MoE, 17B active
    "qwen/qwen3-235b-a22b-thinking-2507": 235,  # MoE, 22B active
    "qwen/qwen3-vl-8b-thinking": 8,
    "qwen/qwen-2.5-72b-instruct": 72,
    "qwen/qwen-2.5-7b-instruct": 7,
    "qwen/qwen3-8b": 8,
    "meta-llama/llama-3.1-8b-instruct": 8,
    "meta-llama/llama-4-maverick": 400,  # MoE
    "mistralai/mistral-7b-instruct-v0.1": 7,
    "mistralai/mistral-small-24b-instruct-2501": 24,
    "google/gemma-2-27b-it": 27,
    "google/gemma-3-27b-it": 27,
    "google/gemma-4-31b-it": 31,
    "deepseek/deepseek-r1-0528": 671,  # MoE
    "deepseek/deepseek-v4-pro": 671,
    "moonshotai/kimi-k2.5": 1000,  # MoE
    "moonshotai/kimi-k2.6": 1000,
    "z-ai/glm-5.1": 110,
    "openai/gpt-4": 1700,
    "openai/gpt-4o": 200,
    "openai/gpt-4o-mini": 8,
    "openai/gpt-4.1": 200,
    "openai/gpt-5": 1000,
    "openai/gpt-5.5": 1500,
    "openai/o1": 200,
    "anthropic/claude-3.7-sonnet": 200,
    "anthropic/claude-sonnet-4": 200,
    "anthropic/claude-sonnet-4.6": 200,
    "google/gemini-2.5-pro": 1000,
    "xiaomi/mimo-v2.5": 30,
}

FAMILY = {
    "qwen": "#7c3aed",
    "deepseek": "#0ea5e9",
    "google": "#10a37f",
    "google-gemini": "#10a37f",
    "moonshotai": "#f59e0b",
    "z-ai": "#dc2626",
    "mistralai": "#6b7280",
    "meta-llama": "#3b82f6",
    "openai": "#111827",
    "anthropic": "#ea580c",
    "xiaomi": "#a855f7",
}


def fam_of(model):
    return model.split("/")[0]


def weighted(s):
    return sum(W[d] * s.get(d, 0) for d in DIMS) / WS


def load_v1_scored():
    """Per (model, track): list of per-paper weighted-score (mean across critics)."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    # Aggregate per (model, track, paper) → mean weighted across critics
    rows_per_paper = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("""
        SELECT idea_model, track, paper_id, scores_json
        FROM results
        WHERE prompt_version='v1_paper_refs' AND track IN ('B','C') AND idea_index=1
          AND critic_model != '' AND scores_json IS NOT NULL
    """):
        m, t, pid, sj = r
        try:
            s = json.loads(sj)
        except Exception:
            continue
        rows_per_paper[(m, t)][pid].append(weighted(s))
    conn.close()
    # Compress: per-paper mean → list across papers
    out = defaultdict(list)
    for (m, t), per_paper in rows_per_paper.items():
        for pid, vals in per_paper.items():
            out[(m, t)].append((pid, statistics.mean(vals)))
    return out


def load_v1_perdim():
    """Per (model, track, dim): list of scores."""
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


def load_v1_perdomain():
    """Per (model, track, domain): list of per-paper weighted scores."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    conn_p = sqlite3.connect(str(cfg.PAPERS_DB))
    pdom = {pid: dom for pid, dom in conn_p.execute(
        "SELECT paper_id, domain FROM papers WHERE status='filtered'")}
    conn_p.close()
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
        per_paper[(m, t, pdom.get(pid, "?"))][pid].append(weighted(s))
    conn.close()
    out = defaultdict(list)
    for (m, t, dom), pp in per_paper.items():
        for pid, vals in pp.items():
            out[(m, t, dom)].append(statistics.mean(vals))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Finding 1: equalizer table + size-vs-boost plot
# ────────────────────────────────────────────────────────────────────────────
def make_finding1(scored):
    models_with_both = {m for (m, t) in scored if t == "B"} & {m for (m, t) in scored if t == "C"}
    rows = []
    for m in models_with_both:
        b_list = [v for _, v in scored.get((m, "B"), [])]
        c_list = [v for _, v in scored.get((m, "C"), [])]
        if not b_list or not c_list:
            continue
        if len(c_list) < 5:  # require ≥5 papers for Active
            continue
        b_mean = statistics.mean(b_list)
        c_mean = statistics.mean(c_list)
        b_sem = statistics.stdev(b_list) / (len(b_list) ** 0.5) if len(b_list) > 1 else 0
        c_sem = statistics.stdev(c_list) / (len(c_list) ** 0.5) if len(c_list) > 1 else 0
        rows.append({
            "model": m, "params": PARAMS_B.get(m),
            "family": fam_of(m),
            "B": b_mean, "C": c_mean, "boost": c_mean - b_mean,
            "B_sem": b_sem, "C_sem": c_sem,
            "nB": len(b_list), "nC": len(c_list),
        })
    rows.sort(key=lambda r: r["params"] or 0)
    return rows


def write_equalizer_table(rows):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Active-mode boost across 8 open-weight models (Track C $-$ Track B, weighted 5-dim score, 25 papers per model). Smaller models gain more from tool access.}",
        r"\label{tab:equalizer}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Params & Static (B) & Active (C) & Boost & $n_B$ & $n_C$ \\",
        r"\midrule",
    ]
    for r in rows:
        label = r["model"].split("/")[-1]
        lines.append(
            f"{label} & {r['params']:>4}B & {r['B']:.2f}$\\pm${r['B_sem']:.2f} & "
            f"{r['C']:.2f}$\\pm${r['C_sem']:.2f} & \\textbf{{{r['boost']:+.2f}}} & "
            f"{r['nB']} & {r['nC']} \\\\"
        )
    lines += [
        r"\midrule",
        f"\\textbf{{Mean}} & --- & {statistics.mean(r['B'] for r in rows):.2f} & "
        f"{statistics.mean(r['C'] for r in rows):.2f} & "
        f"\\textbf{{{statistics.mean(r['boost'] for r in rows):+.2f}}} & --- & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TAB / "equalizer_table.tex").write_text("\n".join(lines))
    print(f"Wrote {TAB / 'equalizer_table.tex'}")


def plot_size_vs_boost(rows):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in rows:
        fam = r["family"]
        color = FAMILY.get(fam, "#888")
        ax.scatter(r["params"], r["boost"], s=160, color=color,
                   edgecolor="#111", linewidth=1.2, zorder=3, alpha=0.85)
        label = r["model"].split("/")[-1].replace("-instruct", "").replace("-instruct-v0.1", "")
        ax.annotate(label, (r["params"], r["boost"]),
                    xytext=(7, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#888", linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (Billions, log scale)", fontsize=12)
    ax.set_ylabel("Active boost = Track C $-$ Track B (weighted score)", fontsize=12)
    ax.set_title(f"Finding 1: Tool augmentation boost vs. model scale ({len(rows)} models)",
                 fontsize=12, pad=12)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fams = sorted({r["family"] for r in rows})
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=FAMILY.get(f, "#888"), label=f) for f in fams]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "finding1_size_vs_boost.pdf", bbox_inches="tight")
    plt.savefig(FIG / "finding1_size_vs_boost.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG / 'finding1_size_vs_boost.pdf'}")


def plot_active_overlay(rows):
    """Static vs Active side by side, sorted by Static score (shows parameter-gap closure)."""
    rows = sorted(rows, key=lambda r: r["B"])
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(rows)))
    b_vals = [r["B"] for r in rows]
    c_vals = [r["C"] for r in rows]
    labels = [r["model"].split("/")[-1] for r in rows]
    ax.scatter(xs, b_vals, s=140, color="#888", edgecolor="#111", linewidth=1.0,
               label="Static (Track B)", zorder=3)
    ax.scatter(xs, c_vals, s=140, color="#10a37f", edgecolor="#111", linewidth=1.0,
               marker="^", label="Active (Track C)", zorder=4)
    for x, b, c in zip(xs, b_vals, c_vals):
        ax.plot([x, x], [b, c], color="#10a37f", alpha=0.5, linewidth=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Weighted score", fontsize=12)
    ax.set_title("Finding 1: Active-mode lift vs Static baseline", fontsize=12, pad=10)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / "finding1_scatter_with_active.pdf", bbox_inches="tight")
    plt.savefig(FIG / "finding1_scatter_with_active.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG / 'finding1_scatter_with_active.pdf'}")


# ────────────────────────────────────────────────────────────────────────────
# Finding 2: per-dim breakdown
# ────────────────────────────────────────────────────────────────────────────
def make_finding2(per_dim):
    models = sorted({m for (m, t, _) in per_dim if t == "C"})
    rows = []
    for m in models:
        row = {"model": m}
        for d in DIMS:
            b = per_dim.get((m, "B", d), [])
            c = per_dim.get((m, "C", d), [])
            if not b or not c:
                row[d] = None
            else:
                row[d] = statistics.mean(c) - statistics.mean(b)
        rows.append(row)
    rows.sort(key=lambda r: -(r.get("originality") or 0))
    return rows


def write_perdim_table(rows):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-dimension Active-mode boost ($\Delta_d = \text{mean}_C(d) - \text{mean}_B(d)$). Originality is the bottleneck dimension.}",
        r"\label{tab:perdim}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & $\Delta_O$ ($\times 2$) & $\Delta_F$ & $\Delta_C$ ($\times .5$) & $\Delta_I$ ($\times 1.5$) & $\Delta_S$ ($\times .5$) \\",
        r"\midrule",
    ]
    for r in rows:
        label = r["model"].split("/")[-1]
        cells = [f"{r.get(d):+.2f}" if r.get(d) is not None else "n/a" for d in DIMS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "perdim_table.tex").write_text("\n".join(lines))
    print(f"Wrote {TAB / 'perdim_table.tex'}")


def plot_perdim_boost(rows):
    """Bar plot: per-model per-dim boost."""
    import numpy as np
    rows = [r for r in rows if all(r.get(d) is not None for d in DIMS)]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n = len(rows)
    width = 0.16
    xs = np.arange(n)
    colors = ["#dc2626", "#10a37f", "#3b82f6", "#7c3aed", "#f59e0b"]
    for i, d in enumerate(DIMS):
        vals = [r[d] for r in rows]
        ax.bar(xs + (i - 2) * width, vals, width, label=d[:4].upper(), color=colors[i])
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["model"].split("/")[-1] for r in rows], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Active − Static (raw dim score)", fontsize=12)
    ax.set_title("Finding 2: Per-dimension boost decomposition", fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / "finding2_perdim_boost.pdf", bbox_inches="tight")
    plt.savefig(FIG / "finding2_perdim_boost.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG / 'finding2_perdim_boost.pdf'}")


# ────────────────────────────────────────────────────────────────────────────
# Finding 3: per-domain
# ────────────────────────────────────────────────────────────────────────────
def make_finding3(perdomain):
    DOMAINS = ["Biology", "CS", "Chemistry", "Medicine", "Physics"]
    # Aggregate: for each domain, average boost across models with both B+C
    per_dom_model_boost = defaultdict(list)
    models_with_both = set()
    for (m, t, dom), vals in perdomain.items():
        if not vals:
            continue
    models = sorted({m for (m, t, _) in perdomain.keys()
                     if (m, "B", _) in perdomain.keys() and (m, "C", _) in perdomain.keys()})

    rows = []
    for dom in DOMAINS:
        boosts = []
        for m in {mm for (mm, _, _) in perdomain.keys()}:
            b = perdomain.get((m, "B", dom), [])
            c = perdomain.get((m, "C", dom), [])
            if len(b) < 1 or len(c) < 1:
                continue
            boosts.append((m, statistics.mean(c) - statistics.mean(b)))
        rows.append((dom, boosts))
    return rows


def plot_perdomain_boost(rows):
    DOMAINS = [r[0] for r in rows]
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 5))
    means = []
    sems = []
    for dom, boosts in rows:
        vals = [b for _, b in boosts]
        if not vals:
            means.append(0); sems.append(0); continue
        means.append(statistics.mean(vals))
        sems.append(statistics.stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0)
    xs = np.arange(len(DOMAINS))
    bars = ax.bar(xs, means, yerr=sems, capsize=4, color="#10a37f", edgecolor="#111",
                  linewidth=1.0)
    for x, m, _ in zip(xs, means, sems):
        ax.text(x, m + 0.03, f"{m:+.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(DOMAINS, fontsize=10)
    ax.set_ylabel("Mean Active boost across models", fontsize=12)
    ax.set_title("Finding 3: Per-domain Active boost (mean ± SEM across models)",
                 fontsize=12, pad=10)
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / "finding3_perdomain_boost.pdf", bbox_inches="tight")
    plt.savefig(FIG / "finding3_perdomain_boost.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG / 'finding3_perdomain_boost.pdf'}")


def write_domain_table(rows):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-domain Active boost, averaged across all models with both B and C tracks.}",
        r"\label{tab:perdomain}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Domain & Mean Boost & SEM & N models \\",
        r"\midrule",
    ]
    for dom, boosts in rows:
        if not boosts:
            lines.append(f"{dom} & --- & --- & 0 \\\\")
            continue
        vals = [b for _, b in boosts]
        m_val = statistics.mean(vals)
        sem = statistics.stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0
        lines.append(f"{dom} & {m_val:+.3f} & {sem:.3f} & {len(vals)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "domain_table.tex").write_text("\n".join(lines))
    print(f"Wrote {TAB / 'domain_table.tex'}")


def main():
    scored = load_v1_scored()
    per_dim = load_v1_perdim()
    perdomain = load_v1_perdomain()

    f1 = make_finding1(scored)
    print(f"\nFinding 1: {len(f1)} models with both B+C")
    for r in f1:
        print(f"  {r['model']:<45} {r['params']:>5}B  B={r['B']:.2f}  C={r['C']:.2f}  boost={r['boost']:+.2f}  (nC={r['nC']})")
    write_equalizer_table(f1)
    plot_size_vs_boost(f1)
    plot_active_overlay(f1)

    f2 = make_finding2(per_dim)
    print(f"\nFinding 2: {len(f2)} models per-dim")
    for r in f2:
        cells = [f"{r.get(d):+.2f}" if r.get(d) is not None else "  n/a " for d in DIMS]
        print(f"  {r['model']:<45} " + " ".join(cells))
    write_perdim_table(f2)
    plot_perdim_boost(f2)

    f3 = make_finding3(perdomain)
    print(f"\nFinding 3: per-domain boost")
    for dom, boosts in f3:
        vals = [b for _, b in boosts]
        if vals:
            print(f"  {dom:<12} mean={statistics.mean(vals):+.3f}  n_models={len(vals)}")
        else:
            print(f"  {dom:<12} n/a")
    write_domain_table(f3)
    plot_perdomain_boost(f3)


if __name__ == "__main__":
    main()
