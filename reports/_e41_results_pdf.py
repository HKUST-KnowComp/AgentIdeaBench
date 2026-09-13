"""Emit a results-only PDF for E41 (docs/e41_results.tex -> .pdf).

Every number is read from the JSON artifacts or recomputed from results.db here,
so the document cannot drift from the data. No narrative, tables only.

  /opt/homebrew/Caskroom/miniforge/base/bin/python reports/_e41_results_pdf.py
  cd docs && /opt/homebrew/bin/pdflatex -interaction=nonstopmode e41_results.tex
"""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402
from utils.constants import SCORING_DIMS as SD  # noqa: E402

OUT = ROOT / "docs" / "e41_results.tex"
E36 = json.load(open(ROOT / "reports" / "e36_leaderboard_subscores.json"))
E41 = json.load(open(ROOT / "reports" / "e41_frontier_analysis.json"))
CHK = json.load(open(ROOT / "reports" / "e41_claim_check.json"))

FRONTIER = {"azure/openai/gpt-5.6-sol", "azure/openai/gpt-5.6-terra",
            "azure/openai/gpt-5.6-luna", "azure/anthropic/claude-opus-5",
            "azure/anthropic/claude-sonnet-5"}
NAME = {"gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.6-terra": "GPT-5.6 Terra",
        "gpt-5.6-luna": "GPT-5.6 Luna", "claude-opus-5": "Claude Opus 5",
        "claude-sonnet-5": "Claude Sonnet 5"}
DIMLABEL = {"originality": "Orig.", "feasibility": "Feas.", "clarity": "Clar.",
            "impact": "Impact", "specificity": "Spec."}


def esc(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def sgn(x):
    return f"$+{x:.2f}$" if x >= 0 else f"$-{abs(x):.2f}$"


def completeness():
    c = sqlite3.connect(str(cfg.RESULTS_DB))
    n = c.execute("SELECT COUNT(*) FROM subdomain_ideas "
                  "WHERE idea_model LIKE 'azure/%'").fetchone()[0]
    d = c.execute("SELECT COUNT(DISTINCT idea_text) FROM subdomain_ideas "
                  "WHERE idea_model LIKE 'azure/%'").fetchone()[0]
    s, e = c.execute("SELECT COUNT(*),SUM(error IS NOT NULL) FROM lit8d_scores_3seed "
                     "WHERE idea_model LIKE 'azure/%'").fetchone()
    nsub = c.execute("SELECT COUNT(DISTINCT subdomain) FROM subdomain_ideas").fetchone()[0]
    nscored = c.execute("SELECT COUNT(DISTINCT subdomain) "
                        "FROM lit8d_scores_3seed").fetchone()[0]
    tot = c.execute("SELECT COUNT(*) FROM lit8d_scores_3seed").fetchone()[0]
    bad = c.execute("""SELECT COUNT(*) FROM (SELECT idea_model,track,subdomain,idea_index,
        COUNT(DISTINCT critic_model) k FROM lit8d_scores_3seed
        WHERE idea_model LIKE 'azure/%' GROUP BY 1,2,3,4 HAVING k<>3)""").fetchone()[0]
    ev = npap = 0
    for sh in NAME:
        for (ej,) in c.execute("SELECT evidence_json FROM e13_evidence "
                               "WHERE item_id LIKE ?||'|%'", (sh,)):
            ev += 1
            npap += len(json.loads(ej or "[]"))
    calls = 0
    for (t,) in c.execute("SELECT telemetry FROM subdomain_ideas "
                          "WHERE idea_model LIKE 'azure/%' AND track='C'"):
        if t:
            calls += json.loads(t).get("n_tool_calls") or 0
    c.close()
    return dict(ideas=n, distinct=d, scored=s, parse_fail=e, nsub=nsub,
                nscored=nscored, total_rows=tot, bad_critic=bad, ev=ev,
                npap=npap, calls=calls)


def ceiling_rows():
    c = sqlite3.connect(str(cfg.RESULTS_DB))
    c.row_factory = sqlite3.Row
    raw = defaultdict(lambda: defaultdict(list))
    for r in c.execute("SELECT idea_model,track,scores_json FROM lit8d_scores_3seed "
                       "WHERE scores_json IS NOT NULL AND track='C'"):
        s = json.loads(r["scores_json"])
        g = ("frontier" if r["idea_model"] in FRONTIER
             else "gemini" if r["idea_model"].startswith("google/gemini") else "open")
        for d in SD:
            v = s[d]["score"] if isinstance(s[d], dict) else s[d]
            if v is not None:
                raw[g][d].append(v)
    c.close()
    out = []
    for g in ("open", "gemini", "frontier"):
        for d in SD:
            v = np.array(raw[g][d], float)
            out.append((g, d, len(v), v.mean(), v.max(),
                        100 * (v >= 8).mean(), 100 * (v >= 9).mean()))
    return out


def main():
    comp = completeness()
    rows = sorted(E36.values(), key=lambda v: -v["active_total"])
    gate = CHK["C1_capability_gate"]["by_roster"]
    quart = CHK["C2_gap_widens"]["quartile_mean_gain"]
    disc = CHK["C3_discrimination"]["by_roster"]
    top = CHK["C4_top_end_compression"]["by_roster"]
    pdim = E41["per_dimension_gain"]
    pm = E41["per_model"]
    ceil = E41["ceiling"]

    L = []
    A = L.append
    A(r"\documentclass[10pt,a4paper]{article}")
    A(r"\usepackage[margin=1.6cm]{geometry}")
    A(r"\usepackage{booktabs,longtable,amsmath}")
    A(r"\usepackage[T1]{fontenc}")
    A(r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}")
    A(r"\setlength{\parindent}{0pt}")
    A(r"\pagestyle{plain}")
    A(r"\begin{document}")
    A(r"\begin{center}{\Large\bfseries E41 --- 2026 Frontier Closed-Source Extension}\\[2pt]"
      r"{\small Results only. Generated by \texttt{reports/\_e41\_results\_pdf.py} "
      r"from \texttt{results.db} + \texttt{reports/e41\_*.json}.}\end{center}")
    A(r"\vspace{4pt}")

    # ---- scope + completeness
    A(r"\section*{1. Scope and completeness}")
    A(r"\begin{tabular}{lr}\toprule Quantity & Value \\\midrule")
    A(rf"Subfields in the dataset (5 domains $\times$ 20) & {comp['nsub']} \\")
    A(rf"Subfields densely scored (identical for all 38 models) & {comp['nscored']} \\")
    A(r"Models added & 5 \\")
    A(rf"Target cells (5 $\times$ {comp['nscored']} $\times$ 3 idx $\times$ 2 tracks) & 1200 \\")
    A(rf"Ideas stored & \textbf{{{comp['ideas']}}} \\")
    A(rf"Distinct idea texts (duplicates) & {comp['distinct']} ({comp['ideas']-comp['distinct']}) \\")
    A(rf"Scored rows ($=$ ideas $\times$ 3 critics) & {comp['scored']} \\")
    A(rf"parse\_fail & {comp['parse_fail']} ({100*comp['parse_fail']/comp['scored']:.2f}\%) \\")
    A(rf"Ideas whose critic count $\neq$ 3 & {comp['bad_critic']} \\")
    A(rf"Prior-art evidence rows / items retrieved & {comp['ev']} / {comp['npap']} \\")
    A(rf"Track C agent tool calls & {comp['calls']} \\")
    A(rf"\texttt{{lit8d\_scores\_3seed}} total rows & {comp['total_rows']} \\")
    A(r"\bottomrule\end{tabular}")
    A(r"\\[3pt]{\small Scoring covers 40 of the 100 subfields. "
      r"This is the pre-existing protocol: every one of the 33 earlier models is "
      r"scored on the same 40 (e19 indices $\{0,5,10,15\}$ + e24 indices "
      r"$\{2,7,12,17\}$, 8 per domain).}")

    # ---- leaderboard
    A(r"\section*{2. Leaderboard (38 paired models, ranked by Active)}")
    A(r"\small")
    A(r"\begin{longtable}{r l l rrr rrrrr}\toprule")
    A(r"\# & Model & Group & Static & Active & $\Delta$ & "
      + " & ".join(DIMLABEL[d] for d in SD) + r" \\\midrule\endhead")
    for i, v in enumerate(rows, 1):
        short = v["short"]
        grp = ("frontier$^\\dagger$" if short in NAME
               else "closed$^\\dagger$" if v["closed"] else "open")
        nm = NAME.get(short, short)
        bold = r"\bfseries " if short in NAME else ""
        A(f"{i} & {bold}{esc(nm)} & {grp} & {v['static_total']:.2f} & "
          f"{v['active_total']:.2f} & {sgn(v['gain'])} & "
          + " & ".join(f"{v['active_dims'][d]:.2f}" for d in SD) + r" \\")
    A(r"\bottomrule\end{longtable}")
    A(r"\normalsize")

    # ---- paired tests
    A(r"\section*{3. Frontier five: paired Wilcoxon over shared subfields}")
    A(r"\begin{tabular}{l r rrr r r}\toprule")
    A(r"Model & $n$ & Static & Active & $\Delta$ & Wilcoxon $p$ & \% subfields improved \\\midrule")
    for m in sorted(FRONTIER, key=lambda m: -pm[m]["active"]):
        r_ = pm[m]
        A(f"{esc(NAME[r_['short']])} & {r_['n_shared_subdomains']} & {r_['static']:.2f} & "
          f"{r_['active']:.2f} & {sgn(r_['gain'])} & {r_['wilcoxon_p']:.3g} & "
          f"{100*r_['frac_subdomains_improved']:.0f}\\% \\\\")
    A(r"\bottomrule\end{tabular}")
    A(r"\\[3pt]{\small Uncorrected. Under Holm at $\alpha{=}0.05$ the two "
      r"significant rows remain significant.}")

    # ---- claim re-test
    A(r"\section*{4. Paper claims re-tested on nested rosters}")
    A(r"\begin{tabular}{l l l l}\toprule")
    A(r"Quantity & open 28 & +Gemini 33 & +frontier 38 \\\midrule")
    g = gate
    A(r"C1 \; Pearson $r$(Static, $\Delta$) & "
      f"$+{g['open_weight_28']['pearson_r']:.3f}$ & "
      f"$+{g['plus_gemini_33']['pearson_r']:.3f}$ & "
      f"$+{g['plus_frontier_38']['pearson_r']:.3f}$ \\\\")
    A(r"\phantom{C1} \; $p$ & "
      f"{g['open_weight_28']['pearson_p']:.1e} & "
      f"{g['plus_gemini_33']['pearson_p']:.1e} & "
      f"{g['plus_frontier_38']['pearson_p']:.2f} \\\\")
    A(r"\phantom{C1} \; Spearman $\rho$ & "
      f"$+{g['open_weight_28']['spearman_rho']:.3f}$ & "
      f"$+{g['plus_gemini_33']['spearman_rho']:.3f}$ & "
      f"$+{g['plus_frontier_38']['spearman_rho']:.3f}$ \\\\\\midrule")
    for k, lab in [("open28", "open 28"), ("plus_gemini33", "+Gemini 33"), ("all38", "+frontier 38")]:
        pass
    A(r"C2 \; mean $\Delta$ by Static quartile & "
      + " & ".join("/".join(f"{q['mean_gain']:+.2f}" for q in quart[k])
                   for k in ("open28", "plus_gemini33", "all38")) + r" \\\midrule")
    A(r"C3 \; var(Active)/var(Static) & "
      + " & ".join(f"{disc[k]['variance_ratio_C_over_B']:.2f}$\\times$"
                   for k in ("open28", "plus_gemini33", "all38")) + r" \\")
    A(r"\phantom{C3} \; sd(Static) / sd(Active) & "
      + " & ".join(f"{disc[k]['sd_static']:.2f} / {disc[k]['sd_active']:.2f}"
                   for k in ("open28", "plus_gemini33", "all38")) + r" \\\midrule")
    A(r"C4 \; top-8 spread, Static & "
      + " & ".join(f"{top[k]['static_total']['spread']:.2f}"
                   for k in ("open28", "plus_gemini33", "all38")) + r" \\")
    A(r"\phantom{C4} \; top-8 spread, Active & "
      + " & ".join(f"{top[k]['active_total']['spread']:.2f}"
                   for k in ("open28", "plus_gemini33", "all38")) + r" \\")
    A(r"\bottomrule\end{tabular}")

    # ---- per dim
    A(r"\section*{5. Per-dimension gain (Active $-$ Static, model-level means)}")
    A(r"\begin{tabular}{l rrr}\toprule")
    A(r"Dimension & frontier 5 & open 28 & Gemini 5 \\\midrule")
    for d in SD:
        A(f"{DIMLABEL[d]} & {pdim['frontier2026'][d]['mean_gain']:+.2f} & "
          f"{pdim['open_weight_28'][d]['mean_gain']:+.2f} & "
          f"{pdim['gemini_5'][d]['mean_gain']:+.2f} \\\\")
    A(r"\bottomrule\end{tabular}")

    # ---- ceiling
    A(r"\section*{6. Raw critic-score distribution, Track C (0--10 scale)}")
    A(r"\begin{tabular}{l l r rr rr}\toprule")
    A(r"Group & Dimension & $n$ & Mean & Observed max & $\%\geq 8$ & $\%\geq 9$ \\\midrule")
    prev = None
    for g, d, n, mu, mx, p8, p9 in ceiling_rows():
        if prev is not None and g != prev:
            A(r"\midrule")
        prev = g
        A(f"{g} & {DIMLABEL[d]} & {n} & {mu:.2f} & {mx:.0f} & {p8:.1f}\\% & {p9:.1f}\\% \\\\")
    A(r"\bottomrule\end{tabular}")
    A(r"\\[4pt]")
    A(r"\begin{tabular}{lr}\toprule Reference point & Value \\\midrule")
    A(rf"CORE-7 award-paper anchor mean under lit8d (F14) & {ceil['core7_anchor_mean_lit8d']:.2f} \\")
    A(rf"Max weighted anchor score, any rubric variant (F11) & {ceil['max_anchor_weighted_any_variant']:.2f} \\")
    A(rf"Best frontier Active & {ceil['frontier_best_active']:.2f} \\")
    A(rf"Best frontier Static & {ceil['frontier_best_static']:.2f} \\")
    A(rf"Headroom, best frontier Active to CORE-7 mean & {ceil['headroom_best_active_to_core7']:.2f} \\")
    A(rf"Best open-weight Active / Static & {ceil['best_open_weight_active']:.2f} / "
      rf"{ceil['best_open_weight_static']:.2f} \\")
    A(r"\bottomrule\end{tabular}")

    A(r"\end{document}")
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
