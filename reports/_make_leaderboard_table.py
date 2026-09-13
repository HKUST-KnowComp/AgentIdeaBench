"""Emit the paper's leaderboard LaTeX from reports/e36_leaderboard_subscores.json.

One full-width table (tab:leaderboard):
  Model | Static | Active | Delta | Turns | 5 Active per-dimension scores

The separate per-dimension appendix table was folded into the main table on
2026-08-27; `tab:leaderboard-full` no longer exists.

Mean Active turns come from reports/e43_active_turns.json (see _e43_active_turns.py),
computed over exactly the cells that carry lit8d Track-C scores.

E41's five 2026 frontier gateway routes are in the score JSON but are NOT in the
paper's roster (the paper reports 35 evaluated / 33 paired, with the five held-out
closed models being the Gemini family). They are excluded by default so the emitted
table matches the paper; pass --include-e41 to emit all 38.

Design constraints (advisor, 2026-08-03): consistent display names, few columns,
minimal emphasis (bold only the best OPEN-WEIGHT value per numeric column, no
underlines), short caption. Held-out closed-source models (the Gemini family and
the 2026 frontier routes added in E41) are marked with a dagger and are never
bolded (headline statistics use the 28 open-weight models).

Read-only. Prints LaTeX to stdout.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "reports" / "e36_leaderboard_subscores.json"
TURNS = ROOT / "reports" / "e43_active_turns.json"

# E41 gateway routes: scored, but outside the paper's 35-model roster.
E41_SHORTS = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
              "claude-opus-5", "claude-sonnet-5"}

# E42 gateway routes: the 26-model closed OpenAI/Anthropic cutoff ladder. Scored,
# but likewise outside the paper's 35-model roster, so excluded by default.
# Sourced from experiments/e42_cutoff_safe_closed.py ROSTER, keyed by short name.
E42_SHORTS = {"gpt-4o-mini", "gpt-4o", "o1", "o3-mini", "gpt-4.1-nano",
              "gpt-4.1-mini", "gpt-4.1", "o3", "o4-mini", "gpt-5-nano",
              "gpt-5-mini", "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.3-chat",
              "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5",
              "claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5",
              "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-7",
              "claude-opus-4-8"}

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]

# Consistent display names (proper case, family-version then size/variant).
# Keyed by the JSON "short" field; 1:1 with the model identity, presentation only.
NAME = {
    "gemma-2-27b": "Gemma-2 27B",
    "llama-3.1-8b": "Llama-3.1 8B",
    "qwen-2.5-7b": "Qwen2.5 7B",
    "llama-4-maverick": "Llama-4 Maverick",
    "mistral-small-24b": "Mistral Small 24B",
    "qwen-2.5-72b": "Qwen2.5 72B",
    "qwen3-8b": "Qwen3 8B",
    "gemini-2.5-flash-lite": "Gemini-2.5 Flash-Lite",
    "qwen3-32b": "Qwen3 32B",
    "qwen3-coder": "Qwen3 Coder",
    "glm-4.5-air": "GLM-4.5 Air",
    "gemma-3-27b": "Gemma-3 27B",
    "qwen3.5-9b": "Qwen3.5 9B",
    "qwen3-30b-instruct": "Qwen3 30B",
    "gemini-2.5-flash": "Gemini-2.5 Flash",
    "mistral-small-2603": "Mistral Small 2603",
    "deepseek-r1-0528": "DeepSeek-R1",
    "minimax-m2.7": "MiniMax-M2.7",
    "mistral-medium-3.1": "Mistral Medium 3.1",
    "mimo-v2.5": "MiMo-V2.5",
    "gemma-4-31b": "Gemma-4 31B",
    "deepseek-v4-flash": "DeepSeek-V4 Flash",
    "qwen3.5-27b": "Qwen3.5 27B",
    "glm-4.6": "GLM-4.6",
    "mimo-v2.5-pro": "MiMo-V2.5 Pro",
    "deepseek-v4-pro": "DeepSeek-V4 Pro",
    "kimi-k2.5": "Kimi-K2.5",
    "qwen3.5-397b": "Qwen3.5 397B",
    "kimi-k2.6": "Kimi-K2.6",
    "gemini-3.1-pro": "Gemini-3.1 Pro",
    "gemini-3.5-flash": "Gemini-3.5 Flash",
    "gemini-3-flash": "Gemini-3 Flash",
    "glm-5.1": "GLM-5.1",
    # E41 — 2026 frontier closed routes via the internal NVIDIA gateway
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
    # E42 — closed OpenAI / Anthropic cutoff ladder via the internal gateway
    "gpt-4o-mini": "GPT-4o mini",
    "gpt-4o": "GPT-4o",
    "o1": "o1",
    "o3-mini": "o3-mini",
    "gpt-4.1-nano": "GPT-4.1 nano",
    "gpt-4.1-mini": "GPT-4.1 mini",
    "gpt-4.1": "GPT-4.1",
    "o3": "o3",
    "o4-mini": "o4-mini",
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5-mini": "GPT-5 mini",
    "gpt-5": "GPT-5",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.3-chat": "GPT-5.3 Chat",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.5": "GPT-5.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-opus-4-5": "Claude Opus 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-8": "Claude Opus 4.8",
}


def sgn(x):
    return f"$+{x:.2f}$" if x >= 0 else f"$-{abs(x):.2f}$"


def sgn_bold(x):
    return f"$\\mathbf{{+{x:.2f}}}$" if x >= 0 else f"$\\mathbf{{-{abs(x):.2f}}}$"


def disp(v):
    short = v["short"]
    assert short in NAME, f"missing display name for {short}"
    tag = r"$^\dagger$" if v["closed"] else ""
    return NAME[short] + tag


def load(include_e41=False, include_e42=False):
    d = json.load(open(SRC))
    turns = {v["short"]: v["active_turns_mean"] for v in json.load(open(TURNS)).values()}
    rows = []
    for v in d.values():
        if not include_e41 and v["short"] in E41_SHORTS:
            continue
        if not include_e42 and v["short"] in E42_SHORTS:
            continue
        v = dict(v)
        v["active_turns"] = turns.get(v["short"])
        assert v["active_turns"] is not None, f"no turn data for {v['short']}"
        rows.append(v)
    rows.sort(key=lambda v: v["active_total"], reverse=True)
    return rows


def best_open(rows, key):
    """Max value of `key` among open-weight rows (closed models never bolded)."""
    return max(v[key] for v in rows if not v["closed"])


def bold_if(val, is_best, fmt="{:.2f}"):
    s = fmt.format(val)
    return r"\textbf{" + s + "}" if is_best else s


def main_table(rows, appendix=False):
    """Emit tab:leaderboard (paper roster) or, with appendix=True, the complete
    roster table tab:leaderboard-all that lives in the appendix.

    The paper's headline statistics stay on the 28 open-weight models either way;
    the appendix table exists so the closed routes are reported, not promoted.
    """
    bS = best_open(rows, "static_total")
    bA = best_open(rows, "active_total")
    bG = best_open(rows, "gain")
    bd = {d: max(v["active_dims"][d] for v in rows if not v["closed"]) for d in DIMS}
    n = len(rows)
    out = []
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering")
    if appendix:
        out.append(
            r"\caption{\textbf{Complete evaluated roster.} The same measurement as "
            r"Table~\ref{tab:leaderboard}, extended to all " + str(n) + r" paired models "
            r"we have scored, ranked by Active total. Beyond the paper's roster this adds "
            r"the closed OpenAI and Anthropic routes reached through an internal gateway: "
            r"five 2026 frontier models and a 26-model OpenAI/Anthropic release ladder. "
            r"All closed-source models are held out ($\dagger$): they are reported here "
            r"but excluded from every headline statistic, which stays on the 28 "
            r"open-weight models, so best-in-column \textbf{bold} still marks the best "
            r"open-weight value. Columns, weighting, and critic are exactly as in "
            r"Table~\ref{tab:leaderboard}. Two coverage notes. \texttt{gpt-5-chat} is "
            r"absent: it is the non-reasoning serving configuration of \texttt{gpt-5}, "
            r"which is already listed, so it is a second route to one model rather than a "
            r"second model. And $87$ of $18{,}720$ scoring cells on the closed ladder "
            r"($0.46\%$) are missing, $27$ of them because one Anthropic route's safety "
            r"filter rejects three biomedical subfields at input; the affected models' "
            r"means are over $351$--$357$ cells instead of $360$.}")
        out.append(r"\label{tab:leaderboard-all}")
    else:
        out.append(
        r"\caption{\textbf{Model leaderboard.} Static (curated retrieval) and Active "
        r"(agent-controlled retrieval) weighted totals for all " + str(n) + r" paired models, "
        r"ranked by Active total; $\Delta{=}$Active$-$Static. \textbf{Turns} is the mean "
        r"number of tool calls (\texttt{SEARCH}$+$\texttt{FETCH}) an Active rollout issues "
        r"before its final synthesis, out of a budget of 10 "
        r"(Appendix~\ref{app:budget}); it is a behavioral descriptor, not a quality score, "
        r"and is never bolded. The five Active per-dimension scores are each a "
        r"3-hypothesis $\times$ 40-subfield mean after dropping the most generous of three "
        r"critics. Weighted total $O{:}2,I{:}1.5,F{:}1,C{:}0.5,S{:}0.5$ (normalized by "
        r"$5.5$), under the \texttt{lit8d} critic (Appendix~\ref{app:critic}). Best "
        r"open-weight value per column in \textbf{bold}; held-out closed-source models are "
        r"marked $\dagger$ and never bolded, and headline statistics use the 28 open-weight "
        r"models. The dimensions do not move together: originality and impact climb toward "
        r"the frontier, whereas feasibility peaks on a weak model (Llama-4 Maverick "
        r"$6.87$), the dissociation formalized in \S\ref{sec:mechanism}.}"
        )
        out.append(r"\label{tab:leaderboard}")
    out.append(r"\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{0.96}")
    out.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}"
               r" l rrr r rrrrr @{}}")
    out.append(r"\toprule")
    out.append(r"& \multicolumn{3}{c}{Weighted total} & & "
               r"\multicolumn{5}{c}{Active per-dimension} \\")
    out.append(r"\cmidrule(lr){2-4} \cmidrule(lr){6-10}")
    out.append(r"Model & Static & Active & $\Delta$ & Turns & "
               r"Orig. & Feas. & Clar. & Impact & Spec. \\")
    out.append(r"\midrule")
    for v in rows:
        op = not v["closed"]
        s_ = bold_if(v["static_total"], op and abs(v["static_total"] - bS) < 1e-9)
        a_ = bold_if(v["active_total"], op and abs(v["active_total"] - bA) < 1e-9)
        g_ = sgn_bold(v["gain"]) if (op and abs(v["gain"] - bG) < 1e-9) else sgn(v["gain"])
        t_ = f"{v['active_turns']:.1f}"
        dims = [bold_if(v["active_dims"][d],
                        op and abs(v["active_dims"][d] - bd[d]) < 1e-9) for d in DIMS]
        out.append(f"{disp(v)} & {s_} & {a_} & {g_} & {t_} & " + " & ".join(dims) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular*}")
    out.append(r"\end{table*}")
    return "\n".join(out)




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-e41", action="store_true",
                    help="also emit E41's five 2026 frontier gateway routes "
                         "(outside the paper's 35-model roster)")
    ap.add_argument("--include-e42", action="store_true",
                    help="also emit E42's 26 closed OpenAI/Anthropic gateway "
                         "routes (outside the paper's 35-model roster)")
    ap.add_argument("--appendix", action="store_true",
                    help="emit the complete-roster appendix table "
                         "(tab:leaderboard-all) instead of tab:leaderboard")
    args = ap.parse_args()
    print(main_table(load(include_e41=args.include_e41,
                          include_e42=args.include_e42),
                     appendix=args.appendix))
