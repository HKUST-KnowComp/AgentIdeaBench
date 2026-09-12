"""Scientific World Model (SWM) — a black-box evaluator-simulator called mid-loop by
the Active agent. Given a candidate hypothesis H (+ domain + literature gathered so
far), it runs a "thought experiment" and returns STRUCTURED, DECOUPLED feedback that
the outer agent turns into a routing decision (SEARCH / REFINE / ACCEPT).

Design brief: docs/research_brief_swm.md. The load-bearing idea is that novelty and
feasibility live on separate channels and a `novel_core` is named + protected during
refinement, so the loop can raise originality WITHOUT regressing feasibility.

Three internal designs (ablation ladder):
  S1 — single LLM inference: one prompt plays Simulator/Skeptic/Referee, emits full JSON.
  S2 — multi-role: Simulator + Novelty-scout + Feasibility-critic (3 calls) + deterministic
       referee (code) that assembles the JSON and picks next_probe.  [headline]
  S3 — S2 + the SWM gets its own small SEARCH budget to ground the mechanism check.
       (implemented via an injected `search_fn`; see run_active_swm_agent.)

Feedback schema (single dict the outer agent parses):
  mechanism_consistency{verdict, explanation, grounded}
  thought_experiment{setup, predicted_outcome, confidence, calibration_note}
  novel_core            (one sentence to PRESERVE)
  dimension_weakness{orig/feas/clar/impact/spec: {severity 0-3, note, channel}}
  feasibility_repair    (edit that raises feasibility without touching novel_core)
  knowledge_gap
  next_probe{action ∈ SEARCH|REFINE|ACCEPT, query, rationale}
  grounding_refs[]

SciPredict (arXiv:2604.10718) caveat is enforced: `confidence` NEVER gates ACCEPT; the
outer policy gates on mechanism_consistency + grounded. /usr/bin/python3.
"""
import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from utils.LLM import IdeaLLM

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
_SHORT = {"originality": "orig", "feasibility": "feas", "clarity": "clar",
          "impact": "impact", "specificity": "spec"}


# --------------------------------------------------------------------------- utils
def _json_extract(text: str) -> Optional[dict]:
    """Pull the first balanced {...} JSON object out of an LLM response."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    # find first { and match braces
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    # tolerant: strip trailing commas
                    frag = re.sub(r",\s*([}\]])", r"\1", t[start:i + 1])
                    try:
                        return json.loads(frag)
                    except Exception:
                        return None
    return None


def _llm(model_name: str) -> IdeaLLM:
    # IdeaLLM.completion handles reasoning models: it allots enough tokens for the
    # thinking phase and returns the final content (raw OpenAI calls returned empty
    # content because qwen3.5 spends the whole budget in the `reasoning` field).
    return IdeaLLM(model_name=model_name)


def _call(llm, system, user, seed=42) -> str:
    resp = llm.completion(user, system_prompt=system)
    if isinstance(resp, tuple):
        content, reasoning = (resp[0] or ""), (resp[1] or "")
        # JSON should be in content; fall back to reasoning if the model dumped it there
        return content if content.strip() else reasoning
    return resp or ""


def _sev(x, default=0):
    try:
        return max(0, min(3, int(round(float(x)))))
    except Exception:
        return default


# --------------------------------------------------------------------------- prompts
_SYS_SINGLE = """You are a Scientific World Model: a rigorous simulator-critic that stress-tests a candidate scientific hypothesis by running a THOUGHT EXPERIMENT against known mechanisms, then returns actionable feedback.

You do NOT rewrite the hypothesis. You (1) check whether its causal chain is consistent with established mechanisms, (2) design the minimal experiment that would test it and predict the outcome, (3) name the ONE sentence carrying its originality (novel_core) that must be preserved, (4) rate each of 5 dimensions for weakness, (5) propose a feasibility repair that does NOT weaken novel_core, (6) name the key missing fact, (7) decide the next action.

CRITICAL: your outcome prediction is a plausibility signal, NOT ground truth — flag it. Never recommend ACCEPT on confidence alone; ACCEPT only if the mechanism is consistent and nothing critical is missing.

Output ONLY a JSON object with EXACTLY these keys:
{"mechanism_consistency":{"verdict":"consistent|partial|inconsistent","explanation":"...","grounded":false},
 "thought_experiment":{"setup":"...","predicted_outcome":"...","confidence":0.0,"calibration_note":"plausibility only, not an oracle"},
 "novel_core":"the one sentence to preserve",
 "dimension_weakness":{"originality":{"severity":0,"note":"...","channel":"novelty"},"feasibility":{"severity":0,"note":"...","channel":"feasibility"},"clarity":{"severity":0,"note":"..."},"impact":{"severity":0,"note":"..."},"specificity":{"severity":0,"note":"..."}},
 "feasibility_repair":"a concrete edit that raises feasibility without touching novel_core",
 "knowledge_gap":"the specific fact needed to judge H, or empty string",
 "next_probe":{"action":"SEARCH|REFINE|ACCEPT","query":"semantic scholar query if SEARCH","rationale":"..."},
 "grounding_refs":[]}
severity is 0-3 (0=fine,3=severe)."""

_SYS_SIM = """You are the SIMULATOR in a Scientific World Model. Given a candidate hypothesis, design the minimal experiment that would test it and predict its outcome by reasoning over known mechanisms; then judge whether the hypothesis's causal chain is consistent with established mechanisms. Your outcome prediction is a plausibility signal, NOT ground truth.
Output ONLY JSON:
{"mechanism_consistency":{"verdict":"consistent|partial|inconsistent","explanation":"...","grounded":false},
 "thought_experiment":{"setup":"...","predicted_outcome":"...","confidence":0.0,"calibration_note":"plausibility only"},
 "knowledge_gap":"the specific fact needed to judge this, or empty string"}"""

_SYS_NOV = """You are the NOVELTY-SCOUT in a Scientific World Model. Identify the ONE sentence in the hypothesis that carries its originality (novel_core) — this must be preserved in any later edit. Rate how original it is against the given literature (severity 0=very original, 3=already known). If it is too close to existing work, propose a Semantic Scholar query that would surface CONTRASTING (not confirmatory) work to push it further.
Output ONLY JSON:
{"novel_core":"the one sentence to preserve",
 "originality_weakness":{"severity":0,"note":"...","channel":"novelty"},
 "contrasting_query":"a semantic scholar query, or empty string"}"""

_SYS_FEAS = """You are the FEASIBILITY-CRITIC in a Scientific World Model. You are given a hypothesis and its novel_core (which you MUST NOT weaken). Rate feasibility, clarity, specificity, impact (severity 0=fine, 3=severe). Then write a concrete feasibility_repair: a minimal edit that makes the hypothesis more testable/doable WITHOUT weakening the novel_core. If you cannot repair feasibility without removing the novelty, set feasibility_repair to "needs evidence" and give a Semantic Scholar query in `evidence_query`.
Output ONLY JSON:
{"feasibility":{"severity":0,"note":"...","channel":"feasibility"},
 "clarity":{"severity":0,"note":"..."},
 "impact":{"severity":0,"note":"..."},
 "specificity":{"severity":0,"note":"..."},
 "feasibility_repair":"...",
 "evidence_query":"a semantic scholar query, or empty string"}"""


def _ctx_block(domain, context, hypothesis):
    ctx = (context or "").strip()
    ctx = (ctx[:4000] + "...") if len(ctx) > 4000 else ctx
    return (f"Research subfield: {domain}\n\n"
            f"Literature gathered so far:\n{ctx or '(none yet)'}\n\n"
            f"Candidate hypothesis H:\n{hypothesis.strip()}")


# --------------------------------------------------------------------------- S1
def _simulate_S1(llm, domain, context, hypothesis, seed):
    raw = _call(llm, _SYS_SINGLE, _ctx_block(domain, context, hypothesis), seed=seed)
    fb = _json_extract(raw) or {}
    return _normalize(fb)


# --------------------------------------------------------------------------- S2 / S3
def _simulate_S2(llm, domain, context, hypothesis, seed, search_fn=None):
    blk = _ctx_block(domain, context, hypothesis)
    sim = _json_extract(_call(llm, _SYS_SIM, blk, seed=seed)) or {}
    nov = _json_extract(_call(llm, _SYS_NOV, blk, seed=seed)) or {}
    novel_core = nov.get("novel_core", "")
    feas_blk = blk + f"\n\nnovel_core (DO NOT WEAKEN): {novel_core}"
    fea = _json_extract(_call(llm, _SYS_FEAS, feas_blk, seed=seed)) or {}

    grounded = False
    grefs = []
    # S3: ground the mechanism check with a couple of real fetches
    if search_fn is not None:
        gq = sim.get("knowledge_gap") or nov.get("contrasting_query") or f"{domain} mechanism"
        try:
            hits = search_fn(gq)[:3]
            if hits:
                grounded = True
                grefs = [h.get("paperId") for h in hits if h.get("paperId")]
                titles = "; ".join((h.get("title") or "")[:80] for h in hits)
                regrounded = _json_extract(_call(
                    llm, _SYS_SIM,
                    blk + f"\n\nRetrieved prior art to ground your mechanism check:\n{titles}",
                    seed=seed)) or {}
                if regrounded:
                    sim = regrounded
        except Exception:
            pass

    fb = {
        "mechanism_consistency": {**sim.get("mechanism_consistency", {}), "grounded": grounded},
        "thought_experiment": sim.get("thought_experiment", {}),
        "novel_core": novel_core,
        "dimension_weakness": {
            "originality": nov.get("originality_weakness", {"severity": 0, "note": "", "channel": "novelty"}),
            "feasibility": fea.get("feasibility", {"severity": 0, "note": "", "channel": "feasibility"}),
            "clarity": fea.get("clarity", {"severity": 0, "note": ""}),
            "impact": fea.get("impact", {"severity": 0, "note": ""}),
            "specificity": fea.get("specificity", {"severity": 0, "note": ""}),
        },
        "feasibility_repair": fea.get("feasibility_repair", ""),
        "knowledge_gap": sim.get("knowledge_gap", ""),
        "grounding_refs": grefs,
        "_contrasting_query": nov.get("contrasting_query", ""),
        "_evidence_query": fea.get("evidence_query", ""),
    }
    fb["next_probe"] = _decide_next(fb)
    return _normalize(fb)


_SYS_PLANNER = """You are the LEAD of a Scientific World Model. Given a candidate hypothesis, decide which panel of expert reviewers to convene to stress-test it. Pick 2-4 experts tailored to THIS hypothesis (e.g. a domain mechanist, a methodologist/experimentalist, a novelty/prior-art scout, a statistician, a specific sub-field specialist). At least one expert must focus on novelty vs prior art and at least one on feasibility/experimental practicality. Mark an expert needs_tool=true only if checking real literature would materially change their judgment.
Output ONLY JSON:
{"panel":[{"role":"short expert title","focus":"what this expert should scrutinize","needs_tool":false,"query":"semantic scholar query if needs_tool else empty"}]}"""

_SYS_SUBAGENT = """You are a {role} on a Scientific World Model review panel. Scrutinize the candidate hypothesis on your focus: {focus}. Run a brief thought experiment from your expertise: what would happen, and where is it weak or wrong? Be concrete and terse.
Output ONLY JSON:
{{"verdict":"support|mixed|refute","key_point":"your single most important finding","dim_concerns":{{"originality":0,"feasibility":0,"clarity":0,"impact":0,"specificity":0}},"suggestion":"one concrete fix or a fact to look up"}}
(dim_concerns severity 0-3; 0 = no concern.)"""

_SYS_AGGREGATOR = """You are the LEAD of a Scientific World Model, synthesizing your expert panel's reviews of a candidate hypothesis into one decision. Combine their findings, resolve disagreements, and produce a final research-outcome prediction plus actionable feedback. Preserve the hypothesis's original idea: name the ONE sentence carrying its novelty (novel_core) that must survive any edit. Keep novelty and feasibility as separate signals. Your outcome prediction is a plausibility signal, NOT ground truth. Never recommend ACCEPT on confidence alone — only if the panel finds the mechanism consistent and nothing critical missing.
Output ONLY JSON with EXACTLY these keys:
{"mechanism_consistency":{"verdict":"consistent|partial|inconsistent","explanation":"...","grounded":false},
 "thought_experiment":{"setup":"the minimal test the panel converged on","predicted_outcome":"the panel's research-outcome prediction","confidence":0.0,"calibration_note":"plausibility only, not an oracle"},
 "novel_core":"the one sentence to preserve",
 "dimension_weakness":{"originality":{"severity":0,"note":"...","channel":"novelty"},"feasibility":{"severity":0,"note":"...","channel":"feasibility"},"clarity":{"severity":0,"note":"..."},"impact":{"severity":0,"note":"..."},"specificity":{"severity":0,"note":"..."}},
 "feasibility_repair":"a concrete edit that raises feasibility without touching novel_core",
 "knowledge_gap":"the specific fact the panel still lacks, or empty string",
 "next_probe":{"action":"SEARCH|REFINE|ACCEPT","query":"semantic scholar query if SEARCH","rationale":"..."},
 "grounding_refs":[]}"""


def _simulate_S4(llm, domain, context, hypothesis, seed, search_fn=None, max_experts=4):
    """Dynamic meta-agent panel: the LEAD picks a bespoke panel of experts for THIS
    hypothesis (role + tool per expert), each reviews, then the LEAD aggregates into
    the final feedback + research-outcome prediction. Same output schema as S2/S3."""
    blk = _ctx_block(domain, context, hypothesis)
    plan = _json_extract(_call(llm, _SYS_PLANNER, blk, seed=seed)) or {}
    panel = plan.get("panel") or []
    if not isinstance(panel, list) or not panel:
        panel = [{"role": "novelty & prior-art scout", "focus": "originality vs existing work",
                  "needs_tool": True, "query": f"{domain}"},
                 {"role": "experimental methodologist", "focus": "feasibility and testability",
                  "needs_tool": False, "query": ""}]
    panel = panel[:max_experts]

    reviews, grefs = [], []
    grounded = False
    for e in panel:
        role = str(e.get("role", "expert"))[:80]
        focus = str(e.get("focus", ""))[:200]
        sub_blk = blk
        if search_fn is not None and e.get("needs_tool") and e.get("query"):
            try:
                hits = search_fn(str(e["query"]))[:3]
                if hits:
                    grounded = True
                    grefs += [h.get("paperId") for h in hits if h.get("paperId")]
                    titles = "; ".join((h.get("title") or "")[:80] for h in hits)
                    sub_blk = blk + f"\n\nPrior art retrieved for your review:\n{titles}"
            except Exception:
                pass
        sys = _SYS_SUBAGENT.format(role=role, focus=focus)
        r = _json_extract(_call(llm, sys, sub_blk, seed=seed)) or {}
        if r:
            r["_role"] = role
            reviews.append(r)

    panel_block = blk + "\n\nEXPERT PANEL REVIEWS:\n" + json.dumps(reviews, ensure_ascii=False)[:6000]
    fb = _json_extract(_call(llm, _SYS_AGGREGATOR, panel_block, seed=seed)) or {}
    if grounded:
        mc = fb.get("mechanism_consistency") or {}
        mc["grounded"] = True
        fb["mechanism_consistency"] = mc
        fb["grounding_refs"] = grefs
    fb["_panel"] = [{"role": r.get("_role"), "verdict": r.get("verdict"),
                     "key_point": r.get("key_point")} for r in reviews]
    return _normalize(fb)


def _simulate_S4b(llm, domain, context, hypothesis, seed, search_fn=None, max_experts=4, gate=False):
    """S4b — dynamic panel (as S4) + S2's FIXED decoupled aggregation.

    Motivation: pure S4's free-form aggregator dilutes the two load-bearing S2 constraints
    (a dedicated Novelty-scout that names+pins `novel_core`, and a Feasibility-critic whose
    repair must NOT weaken it). S4b keeps S4's domain-adaptive panel discovery but feeds the
    panel reviews into S2's fixed Novelty-scout -> Feasibility-critic(novel_core injected) ->
    deterministic referee, instead of a free-text LEAD synthesis. So: dynamic discovery,
    hard-constrained assembly. Same output schema as S2/S4.

    gate=True is design S5: suppress the feasibility_repair when feasibility is already
    strong (severity < 2). Diagnosis (experiments/e27_diagnose.py) showed S4b's *always-on*
    feasibility repair over-edits already-good ideas on saturated strong models (v4-pro
    feasibility -0.39 vs free-form S4). Gating the repair lets a strong idea stand instead of
    forcing a lateral edit, while keeping the full hard-constrained repair when a real
    weakness exists."""
    blk = _ctx_block(domain, context, hypothesis)
    # 1) LEAD convenes a bespoke panel (identical to S4)
    plan = _json_extract(_call(llm, _SYS_PLANNER, blk, seed=seed)) or {}
    panel = plan.get("panel") or []
    if not isinstance(panel, list) or not panel:
        panel = [{"role": "novelty & prior-art scout", "focus": "originality vs existing work",
                  "needs_tool": True, "query": f"{domain}"},
                 {"role": "experimental methodologist", "focus": "feasibility and testability",
                  "needs_tool": False, "query": ""}]
    panel = panel[:max_experts]

    reviews, grefs = [], []
    grounded = False
    for e in panel:
        role = str(e.get("role", "expert"))[:80]
        focus = str(e.get("focus", ""))[:200]
        sub_blk = blk
        if search_fn is not None and e.get("needs_tool") and e.get("query"):
            try:
                hits = search_fn(str(e["query"]))[:3]
                if hits:
                    grounded = True
                    grefs += [h.get("paperId") for h in hits if h.get("paperId")]
                    titles = "; ".join((h.get("title") or "")[:80] for h in hits)
                    sub_blk = blk + f"\n\nPrior art retrieved for your review:\n{titles}"
            except Exception:
                pass
        sys = _SYS_SUBAGENT.format(role=role, focus=focus)
        r = _json_extract(_call(llm, sys, sub_blk, seed=seed)) or {}
        if r:
            r["_role"] = role
            reviews.append(r)

    # 2) FIXED S2-style aggregation: panel reviews become shared context for the three
    #    hard-role calls, so the decoupling + novel_core pinning are guaranteed.
    panel_txt = json.dumps(
        [{"role": r.get("_role"), "verdict": r.get("verdict"),
          "key_point": r.get("key_point"), "suggestion": r.get("suggestion")}
         for r in reviews], ensure_ascii=False)[:5000]
    agg_blk = blk + "\n\nEXPERT PANEL REVIEWS (use as evidence; do not just echo them):\n" + panel_txt

    sim = _json_extract(_call(llm, _SYS_SIM, agg_blk, seed=seed)) or {}
    nov = _json_extract(_call(llm, _SYS_NOV, agg_blk, seed=seed)) or {}
    novel_core = nov.get("novel_core", "")
    feas_blk = agg_blk + f"\n\nnovel_core (DO NOT WEAKEN): {novel_core}"
    fea = _json_extract(_call(llm, _SYS_FEAS, feas_blk, seed=seed)) or {}

    fb = {
        "mechanism_consistency": {**sim.get("mechanism_consistency", {}), "grounded": grounded},
        "thought_experiment": sim.get("thought_experiment", {}),
        "novel_core": novel_core,
        "dimension_weakness": {
            "originality": nov.get("originality_weakness", {"severity": 0, "note": "", "channel": "novelty"}),
            "feasibility": fea.get("feasibility", {"severity": 0, "note": "", "channel": "feasibility"}),
            "clarity": fea.get("clarity", {"severity": 0, "note": ""}),
            "impact": fea.get("impact", {"severity": 0, "note": ""}),
            "specificity": fea.get("specificity", {"severity": 0, "note": ""}),
        },
        "feasibility_repair": fea.get("feasibility_repair", ""),
        "knowledge_gap": sim.get("knowledge_gap", ""),
        "grounding_refs": grefs,
        "_contrasting_query": nov.get("contrasting_query", ""),
        "_evidence_query": fea.get("evidence_query", ""),
        "_panel": [{"role": r.get("_role"), "verdict": r.get("verdict"),
                    "key_point": r.get("key_point")} for r in reviews],
    }
    if gate:
        # S5: only push a feasibility repair when feasibility is genuinely weak (severity>=2);
        # otherwise let the already-strong idea stand rather than force a lateral edit.
        feas_sev = _sev(fb["dimension_weakness"].get("feasibility", {}).get("severity", 0))
        if feas_sev < 2:
            fb["feasibility_repair"] = ""
            fb["_gated"] = True
    fb["next_probe"] = _decide_next(fb)
    return _normalize(fb)


def _decide_next(fb):
    """Deterministic referee: gate on mechanism + knowledge gap, never on confidence."""
    mc = (fb.get("mechanism_consistency") or {}).get("verdict", "partial")
    gap = (fb.get("knowledge_gap") or "").strip()
    dw = fb.get("dimension_weakness") or {}
    max_sev = max((_sev(dw.get(d, {}).get("severity", 0)) for d in DIMS), default=0)
    feas_needs_evidence = str(fb.get("feasibility_repair", "")).strip().lower().startswith("needs evidence")

    if mc != "consistent" or gap or feas_needs_evidence:
        q = (fb.get("_contrasting_query") or fb.get("_evidence_query") or gap or "")
        return {"action": "SEARCH", "query": q,
                "rationale": "mechanism not fully consistent or a key fact is missing → gather evidence"}
    if max_sev >= 2:
        return {"action": "REFINE", "query": "",
                "rationale": "mechanism ok; refine weak dimensions while preserving novel_core"}
    return {"action": "ACCEPT", "query": "", "rationale": "consistent and no severe weakness"}


def _normalize(fb):
    fb = dict(fb or {})
    fb.setdefault("mechanism_consistency", {"verdict": "partial", "explanation": "", "grounded": False})
    fb.setdefault("thought_experiment", {"setup": "", "predicted_outcome": "", "confidence": 0.0,
                                         "calibration_note": "plausibility only, not an oracle"})
    fb.setdefault("novel_core", "")
    dw = fb.get("dimension_weakness") or {}
    for d in DIMS:
        cell = dw.get(d) or {}
        cell["severity"] = _sev(cell.get("severity", 0))
        cell.setdefault("note", "")
        dw[d] = cell
    fb["dimension_weakness"] = dw
    fb.setdefault("feasibility_repair", "")
    fb.setdefault("knowledge_gap", "")
    fb.setdefault("grounding_refs", [])
    if "next_probe" not in fb or not isinstance(fb["next_probe"], dict):
        fb["next_probe"] = _decide_next(fb)
    np_ = fb["next_probe"]
    np_["action"] = str(np_.get("action", "REFINE")).upper()
    if np_["action"] not in ("SEARCH", "REFINE", "ACCEPT"):
        np_["action"] = "REFINE"
    np_.setdefault("query", "")
    np_.setdefault("rationale", "")
    return fb


def simulate(hypothesis, domain, context="", model=None, design="S2",
             llm=None, search_fn: Optional[Callable] = None, seed=42) -> dict:
    """Run the SWM on a candidate hypothesis; return the feedback schema dict.

    design: 'S1' single-inference, 'S2' multi-role, 'S3' multi-role + grounding
    (requires search_fn). model: backbone model id. llm: optional shared IdeaLLM.
    """
    if not (hypothesis or "").strip():
        return _normalize({"next_probe": {"action": "REFINE", "query": "",
                                          "rationale": "empty hypothesis"}})
    model = model or cfg.IDEA_MODELS[0]
    llm = llm or _llm(model)
    if design == "S1":
        return _simulate_S1(llm, domain, context, hypothesis, seed)
    if design == "S3":
        return _simulate_S2(llm, domain, context, hypothesis, seed, search_fn=search_fn)
    if design == "S4":  # dynamic meta-agent panel (LEAD picks bespoke experts + tools)
        return _simulate_S4(llm, domain, context, hypothesis, seed, search_fn=search_fn)
    if design == "S4b":  # dynamic panel + S2's fixed hard-constrained aggregation
        return _simulate_S4b(llm, domain, context, hypothesis, seed, search_fn=search_fn)
    if design == "S5":  # S4b + gated feasibility repair (no over-editing of strong ideas)
        return _simulate_S4b(llm, domain, context, hypothesis, seed, search_fn=search_fn, gate=True)
    return _simulate_S2(llm, domain, context, hypothesis, seed, search_fn=None)


def format_feedback(fb: dict) -> str:
    """Render the SWM JSON into a compact text block for the outer agent's context."""
    mc = fb.get("mechanism_consistency", {})
    te = fb.get("thought_experiment", {})
    dw = fb.get("dimension_weakness", {})
    np_ = fb.get("next_probe", {})
    weak = "; ".join(f"{_SHORT[d]}={dw.get(d,{}).get('severity',0)}" for d in DIMS)
    lines = [
        "SWM FEEDBACK (thought-experiment result — treat outcome as plausibility, not truth):",
        f"- mechanism: {mc.get('verdict','?')} — {mc.get('explanation','')[:280]}"
        + ("  [grounded in retrieved papers]" if mc.get("grounded") else ""),
        f"- thought experiment: {te.get('setup','')[:200]} → predicts: {te.get('predicted_outcome','')[:200]}",
        f"- NOVEL CORE to PRESERVE: {fb.get('novel_core','')[:240]}",
        f"- weakness severity (0-3): {weak}",
        f"- feasibility repair (keep novel_core): {fb.get('feasibility_repair','')[:240]}",
    ]
    if (fb.get("knowledge_gap") or "").strip():
        lines.append(f"- knowledge gap: {fb['knowledge_gap'][:200]}")
    act = np_.get("action", "REFINE")
    if act == "SEARCH":
        lines.append(f"- SUGGESTED NEXT: SEARCH for '{np_.get('query','')}' to close the gap, then re-SIMULATE.")
    elif act == "REFINE":
        lines.append("- SUGGESTED NEXT: REFINE the hypothesis to fix the weak dimensions, "
                     "KEEPING the novel core verbatim, then either re-SIMULATE or FINAL.")
    else:
        lines.append("- SUGGESTED NEXT: the hypothesis is consistent and solid — you may FINAL it.")
    return "\n".join(lines)
