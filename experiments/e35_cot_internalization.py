#!/usr/bin/env /usr/bin/python3
"""
E35 — CoT internalization probe (SWM appendix case study).

Question (advisor, 2026-07-30): do frontier models already run a
"thought experiment" inside their own chain-of-thought, so that bolting an
external Scientific World Model (SWM) onto them adds little? The SWM null
effect on the deepseek-v4 backbones (E27/F21) is consistent with this, but the
main pipeline stored NO reasoning text: generation ran with thinking OFF
(`reasoning.enabled=False`, see utils/LLM.py:351-358) for a clean cross-year
comparison, so subdomain_ideas.telemetry / swm_ideas.aux_json carry only tool
traces + counts, and reasoning_tokens is 0 for every row. The chain-of-thought
was never persisted and cannot be recovered from the DB.

This probe re-generates a SMALL set of ideation calls with thinking ON
(`SCISYNTH_KEEP_THINKING_ON=1`) and captures the reasoning trace that
`BaseLLM.completion()` returns as (content, reasoning) when a model exposes it.
We then look, in the strong model's trace, for spontaneous thought-experiment
structure (hypothesise a mechanism -> predict an outcome -> name a
falsifying/controlling test) and contrast it with a weak backbone that gained
from SWM (qwen3.5-9b).

Scope / honesty:
  * ONE clear case in the strong model + its absence in the weak one supports an
    *illustrative* appendix case study (a candidate mechanism), NOT a causal
    proof that internalization explains the null SWM effect. Label accordingly.
  * Thinking-ON here differs from the SWM runs' configuration; state this caveat
    in any writeup (the probe shows the model *can* reason this way, not that it
    did so during E27).

Non-invasive: reads only DISTINCT subdomains from swm_ideas; writes only
reports/e35_cot_probe.json. No pipeline table is touched. Both models are
open-weight -> default OPENROUTER_API_KEY (rule 9). Nothing runs on import.

Usage:
  /usr/bin/python3 experiments/e35_cot_internalization.py --smoke      # 2 subs x 2 models
  /usr/bin/python3 experiments/e35_cot_internalization.py --n 6        # 6 subs x 2 models
"""
import os
# Force thinking ON before any LLM import/instantiation reads config/env.
os.environ["SCISYNTH_KEEP_THINKING_ON"] = "1"

import sys
import re
import json
import argparse
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.LLM import create_llm  # noqa: E402

RESULTS_DB = ROOT / "data" / "results.db"
OUT = ROOT / "reports" / "e35_cot_probe.json"

STRONG = "deepseek/deepseek-v4-pro"     # frontier; SWM null (F21)
WEAK = "qwen/qwen3.5-9b"                 # mid/low; largest SWM gain (F21)
JUDGE = "z-ai/glm-5.1"                   # neutral open-weight judge (default key)

# The construct the advisor's claim needs, tested directly (not via keywords):
# does the model, inside its own reasoning, spontaneously perform the checks an
# external SWM would supply? A) simulate a mechanism and predict an outcome;
# B) critique the idea's novelty vs prior work and adjust; C) critique its
# feasibility and adjust. overall = A+B+C (0-3).
JUDGE_PROMPT = (
    "Below is a language model's PRIVATE reasoning (chain-of-thought) while it "
    "invents a scientific hypothesis. Judge whether, INSIDE this reasoning, the "
    "model spontaneously performs the checks an external critic ('scientific "
    "world model') would otherwise supply:\n"
    "A) MECHANISM SIMULATION: mentally runs the proposed mechanism and predicts "
    "a concrete outcome (\"if X then we would observe Y\"), beyond merely naming "
    "a measurement.\n"
    "B) NOVELTY CRITIQUE: weighs whether the idea is genuinely new versus known "
    "/ existing work, and revises accordingly.\n"
    "C) FEASIBILITY CRITIQUE: weighs whether the idea is actually testable / "
    "practical, and revises accordingly.\n"
    "Score each 1 (clearly present) or 0 (absent/superficial). overall = A+B+C.\n"
    'Output ONLY compact JSON on one line: '
    '{"A":0,"B":0,"C":0,"overall":0,"why":"<=15 words"}\n\n'
    "REASONING:\n{trace}"
)

SYSTEM = (
    "You are a research scientist proposing a novel, testable hypothesis. "
    "Think carefully, then commit to a single idea."
)
USER_TMPL = (
    "Subfield: {sub}\n\n"
    "Propose ONE novel, testable scientific hypothesis in this subfield, as a "
    "single 80-150 word paragraph in first-person future tense, naming the "
    "mechanism, method, and the measurement that would confirm or refute it."
)

# Heuristic markers of a thought-experiment step inside the reasoning trace.
TE_MARKERS = [
    "suppose", "imagine", "if we", "what if", "would predict", "we would expect",
    "this predicts", "would imply", "would rule out", "would falsify",
    "counterfactual", "consider the case", "control condition", "null result",
    "if this were true", "if the mechanism", "distinguish between",
    "would observe", "should see", "then we would",
]


def load_subdomains(n):
    """DISTINCT subdomains actually used in the SWM experiment (read-only)."""
    try:
        c = sqlite3.connect(f"file:{RESULTS_DB}?mode=ro", uri=True)
        rows = [r[0] for r in c.execute(
            "SELECT DISTINCT subdomain FROM swm_ideas ORDER BY subdomain")]
        c.close()
        if rows:
            # spread across the alphabetised list for topical variety
            step = max(1, len(rows) // n)
            return rows[::step][:n]
    except Exception as e:
        print(f"[warn] could not read swm_ideas ({e}); using fallback list")
    return [
        "2D material graphene van der Waals heterostructure",
        "CRISPR base editing off-target",
        "large language model in-context learning mechanism",
        "high entropy alloy mechanical property",
        "tumor microenvironment immune evasion",
        "topological superconductor Majorana",
    ][:n]


def marker_hits(text):
    t = (text or "").lower()
    return [m for m in TE_MARKERS if m in t]


def run_one(model, sub):
    llm = create_llm("idea", model)
    out = llm.completion(USER_TMPL.format(sub=sub), system_prompt=SYSTEM)
    if isinstance(out, tuple):
        content, reasoning = out
    else:
        content, reasoning = out, None
    usage = getattr(llm, "last_usage", {}) or {}
    return {
        "model": model,
        "subdomain": sub,
        "content": content,
        "reasoning": reasoning,
        "reasoning_present": bool(reasoning),
        "reasoning_chars": len(reasoning) if reasoning else 0,
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "te_marker_hits": marker_hits(reasoning) if reasoning else [],
    }


def judge_trace(judge_llm, reasoning):
    """LLM-judge the reasoning on the SWM construct (A/B/C). Returns dict or None."""
    if not reasoning:
        return None
    out = judge_llm.completion(JUDGE_PROMPT.format(trace=reasoning[:8000]))
    ans = out[0] if isinstance(out, tuple) else out   # thinking may be on
    m = re.search(r"\{.*\}", ans or "", re.S)
    if not m:
        return {"parse_fail": True, "raw": (ans or "")[:200]}
    try:
        j = json.loads(m.group(0))
        for k in ("A", "B", "C", "overall"):
            j[k] = int(j.get(k, 0))
        return j
    except Exception:
        return {"parse_fail": True, "raw": m.group(0)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 subfields x 2 models, no judge")
    ap.add_argument("--n", type=int, default=8, help="subfields per model")
    ap.add_argument("--models", nargs="*", default=[STRONG, WEAK])
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-judge pass")
    args = ap.parse_args()

    n = 2 if args.smoke else args.n
    use_judge = not (args.smoke or args.no_judge)
    subs = load_subdomains(n)
    print(f"[e35] thinking_on={os.environ.get('SCISYNTH_KEEP_THINKING_ON')} "
          f"| models={args.models} | {len(subs)} subfields | judge={JUDGE if use_judge else 'OFF'}")

    judge_llm = create_llm("idea", JUDGE) if use_judge else None

    records = []
    for model in args.models:
        for sub in subs:
            print(f"  -> {model:30s} | {sub[:50]}")
            try:
                rec = run_one(model, sub)
            except Exception as e:
                rec = {"model": model, "subdomain": sub, "error": str(e)}
                print(f"     [error] {e}")
                records.append(rec)
                continue
            if use_judge and rec.get("reasoning_present"):
                try:
                    rec["judge"] = judge_trace(judge_llm, rec["reasoning"])
                except Exception as e:
                    rec["judge"] = {"error": str(e)}
            records.append(rec)
            if rec.get("reasoning_present"):
                j = rec.get("judge") or {}
                print(f"     reasoning: {rec['reasoning_chars']} chars | "
                      f"TE_kw={rec['te_marker_hits']} | "
                      f"judge A/B/C={j.get('A')}/{j.get('B')}/{j.get('C')} "
                      f"overall={j.get('overall')}")
            elif "error" not in rec:
                print("     reasoning: NONE returned (model did not expose CoT)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": "E35 CoT internalization probe; thinking forced ON. Judge rates "
                "A=mechanism-sim, B=novelty-critique, C=feasibility-critique (0/1), "
                "overall=A+B+C. CAVEAT: thinking-ON here != E27 SWM config; shows "
                "the model CAN reason this way, not that it did during E27.",
        "strong": STRONG, "weak": WEAK, "judge": JUDGE if use_judge else None,
        "subdomains": subs, "records": records,
    }, ensure_ascii=False, indent=2))

    print(f"\n[e35] wrote {OUT}")
    for model in args.models:
        ms = [r for r in records if r.get("model") == model and r.get("reasoning_present")]
        if not ms:
            print(f"  {model}: NO reasoning captured")
            continue
        chars = sum(r["reasoning_chars"] for r in ms) / len(ms)
        line = f"  {model}: {len(ms)} traces | mean {chars:.0f} chars"
        judged = [r["judge"] for r in ms if isinstance(r.get("judge"), dict)
                  and "overall" in r["judge"]]
        if judged:
            mo = sum(j["overall"] for j in judged) / len(judged)
            mA = sum(j["A"] for j in judged) / len(judged)
            mB = sum(j["B"] for j in judged) / len(judged)
            mC = sum(j["C"] for j in judged) / len(judged)
            line += (f" | judge overall {mo:.2f}/3 "
                     f"(A-mech {mA:.2f}, B-novelty {mB:.2f}, C-feas {mC:.2f})")
        print(line)
    print("\nHONEST READ: compare strong vs weak judge overall. If strong is NOT "
          "clearly higher, the 'internalized thought experiment' framing is not "
          "supported; report that and use the data-backed adjacent story instead.")


if __name__ == "__main__":
    main()
