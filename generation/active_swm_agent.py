"""Active + Scientific World Model (SWM) closed loop.

Outer Active agent uses the plain-text tool protocol, extended with a 4th command:

  SEARCH: <query>            -> Semantic Scholar search (as in active_agent)
  FETCH: <paperId>           -> top references (as in active_agent)
  SIMULATE: <draft hypothesis>  -> calls the SWM black box (generation/swm.py) on the
                                draft + literature-gathered-so-far; returns structured
                                thought-experiment feedback; the agent then acts on it
                                (SEARCH more, refine, or FINAL).
  FINAL: <hypothesis>        -> ends the loop

The agent MUST SIMULATE its draft at least once before FINAL. The SWM feedback names a
`novel_core` to preserve, keeps novelty and feasibility on separate channels, and suggests
the next action — this is what lets the loop raise originality without regressing feasibility.

Does NOT modify generation/active_agent.py — reuses its tool functions. Returns the same
shape as run_active_agent plus `swm_trace` and `n_sims`. /usr/bin/python3.
"""
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from openai import OpenAI
from generation.active_agent import (
    _chat_with_retry, _search_papers, _get_paper_references, _format_papers_block)
from generation import swm as swm_mod

SYSTEM_PROMPT_SWM = """\
You are a creative and rigorous scientist. Given a research domain name, you must \
propose a single novel, testable scientific hypothesis in that domain.

You have four plain-text commands. Each turn, output EXACTLY ONE command on the FIRST line:

  SEARCH: <keyword query, 4-10 words>
      -> up to 10 papers with title + abstract + paperId

  FETCH: <paperId from a prior SEARCH result>
      -> the top 15 references of that paper

  SIMULATE: <your current draft hypothesis, one paragraph>
      -> a Scientific World Model runs a THOUGHT EXPERIMENT on your draft and returns
         feedback: a mechanism-consistency verdict, a predicted experimental outcome
         (a plausibility signal, NOT ground truth), the NOVEL CORE you must preserve,
         per-dimension weakness severities, a feasibility repair that keeps the novel
         core, any missing fact, and a suggested next action.

  FINAL: <one-paragraph hypothesis, 80-150 words, first-person future tense, naming \
specific mechanisms / methods / datasets>
      -> ends the loop

Workflow: SEARCH/FETCH to ground yourself, draft a hypothesis, then SIMULATE it. Act on \
the SWM feedback — if it says a fact is missing, SEARCH for it and SIMULATE again; if it \
flags weak dimensions, REFINE your draft (fix feasibility/clarity/specificity) but KEEP \
the NOVEL CORE it identified, then SIMULATE again or FINAL. You MUST SIMULATE at least \
once before FINAL. You have a budget of {max_iters} SEARCH/FETCH calls and {max_sims} \
SIMULATE calls.

Output format is strict: the FIRST line MUST start with `SEARCH:`, `FETCH:`, `SIMULATE:`, \
or `FINAL:`. Do not wrap in code blocks. One command per turn."""

USER_PROMPT_SWM = """\
Domain: {domain}

Start by searching the literature. First command:"""

_CMD_RE = re.compile(
    r"(?im)^\s*(SEARCH|FETCH|SIMULATE|FINAL)\s*[:：]\s*(.+?)"
    r"(?=\n\s*(?:SEARCH|FETCH|SIMULATE|FINAL)\s*[:：]|\Z)", re.DOTALL)


def _parse_command(content: str):
    if not content:
        return (None, None)
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = _CMD_RE.search(text)
    if not m:
        return (None, None)
    cmd = m.group(1).upper()
    arg = m.group(2).strip()
    # SIMULATE/FINAL args are full paragraphs; SEARCH/FETCH are short — keep first para only
    if cmd in ("SEARCH", "FETCH"):
        arg = re.split(r"\n\s*\n", arg, maxsplit=1)[0].strip()
    return (cmd, arg)


def run_active_swm_agent(
    domain: str,
    model_name: str,
    swm_design: str = "S2",
    max_iters: int = 8,
    max_sims: int = 3,
    swm_model: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """Active agent with a mid-loop SWM. Returns {hypothesis, trace, swm_trace, turns,
    n_tool_calls, n_sims, iters_used, error}."""
    api_key = cfg.get_openrouter_key_for_model(model_name)
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    swm_model = swm_model or model_name
    swm_llm = swm_mod._llm(swm_model)

    system = SYSTEM_PROMPT_SWM.format(max_iters=max_iters, max_sims=max_sims)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": USER_PROMPT_SWM.format(domain=domain)}]

    trace, swm_trace, turns = [], [], []
    hypothesis = None
    error = None
    n_tool_calls = 0
    n_sims = 0
    n_malformed = 0
    MAX_MALFORMED = 2
    lit_context = []          # accumulated literature the agent has seen
    last_draft = ""

    # search_fn for S3 grounding (dedupe handled by SS side)
    def _swm_search(q):
        return _search_papers(query=q[:200].strip(), limit=5)

    total_budget = max_iters + max_sims + 4  # room for final + nudges
    for it in range(total_budget):
        try:
            resp = _chat_with_retry(client, model=model_name, messages=messages,
                                    temperature=0.7, seed=seed, max_tokens=6000)
        except Exception as e:
            error = f"api error: {e}"
            break
        if not resp.choices:
            error = "empty response"
            break

        usage = getattr(resp, "usage", None)
        turns.append({"iter": it,
                      "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                      "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                      "total_tokens": getattr(usage, "total_tokens", None) if usage else None})

        content = resp.choices[0].message.content or ""
        cmd, arg = _parse_command(content)

        if cmd is None:
            n_malformed += 1
            if n_malformed > MAX_MALFORMED:
                hypothesis = last_draft or content.strip()
                error = f"malformed_x{n_malformed}"
                break
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": (
                "Your output didn't start with SEARCH:, FETCH:, SIMULATE:, or FINAL:. "
                "Reply again with EXACTLY ONE command on the first line.")})
            continue

        messages.append({"role": "assistant", "content": content})

        if cmd == "FINAL":
            if n_sims == 0:
                # force a SIMULATE first (once)
                messages.append({"role": "user", "content": (
                    "Before FINAL you must SIMULATE your draft hypothesis at least once. "
                    "Output SIMULATE: <your draft hypothesis> now.")})
                continue
            hypothesis = arg
            break

        if cmd == "SIMULATE":
            if n_sims >= max_sims:
                messages.append({"role": "user", "content": (
                    "You've used all SIMULATE calls. Output FINAL: <hypothesis> now "
                    "(80-150 words), keeping the novel core intact.")})
                continue
            last_draft = arg
            ctx = "\n".join(lit_context)[-4000:]
            search_fn = _swm_search if swm_design in ("S3", "S4", "S4b", "S5") else None
            fb = swm_mod.simulate(arg, domain, context=ctx, model=swm_model,
                                  design=swm_design, llm=swm_llm, search_fn=search_fn, seed=seed)
            n_sims += 1
            swm_trace.append({"iter": it, "draft": arg[:1200], "feedback": fb})
            messages.append({"role": "user", "content": swm_mod.format_feedback(fb)})
            continue

        # SEARCH / FETCH — budget check
        if n_tool_calls >= max_iters:
            messages.append({"role": "user", "content": (
                "You've reached the SEARCH/FETCH budget. SIMULATE your draft (if not yet) "
                "then output FINAL: <hypothesis>.")})
            continue

        if cmd == "SEARCH":
            query = arg[:200].strip()
            papers = _search_papers(query=query, limit=10)
            result_str = json.dumps(papers, ensure_ascii=False)
            trace.append({"iter": it, "tool": "search_papers",
                          "args": {"query": query, "limit": 10},
                          "result_preview": result_str[:50_000]})
            n_tool_calls += 1
            block = _format_papers_block(papers, f"SEARCH RESULT for query '{query}':")
            lit_context.append(block[:2500])
            messages.append({"role": "user", "content": block})
        elif cmd == "FETCH":
            pid = re.sub(r"[`\"'\s]+", "", arg.split()[0] if arg.split() else arg)
            pid = pid.strip("()[]{},.;").split("/")[-1][:80]
            if not pid:
                messages.append({"role": "user", "content": (
                    "FETCH requires a paperId from a prior SEARCH result. "
                    "Output another SEARCH: or FETCH: command.")})
                continue
            refs = _get_paper_references(pid)
            result_str = json.dumps(refs, ensure_ascii=False)
            trace.append({"iter": it, "tool": "get_paper_references",
                          "args": {"paper_id": pid}, "result_preview": result_str[:50_000]})
            n_tool_calls += 1
            block = _format_papers_block(refs, f"REFERENCES of paperId={pid}:")
            lit_context.append(block[:2500])
            messages.append({"role": "user", "content": block})

    return {
        "hypothesis": (hypothesis or "").strip(),
        "trace": trace,
        "swm_trace": swm_trace,
        "turns": turns,
        "n_tool_calls": n_tool_calls,
        "n_sims": n_sims,
        "iters_used": len({t["iter"] for t in trace}),
        "error": error,
    }
