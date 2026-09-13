"""
Active Mode — Agentic Idea Generation

Given only a domain name (no refs), an LLM uses Semantic Scholar search tools
to discover literature itself, then proposes a novel hypothesis. Tests the
full research pipeline: retrieval + selection + synthesis.

Tools exposed to the model:
  - search_papers(query, year_range, limit): search SS for papers by keyword
  - get_paper(paper_id): fetch full abstract + refs for a specific paper

Agent loop:
  1. System prompt sets the task + constraints
  2. Model issues tool calls to gather info
  3. Tool results fed back to model
  4. After max_iters (or when model stops calling tools), model emits final
     hypothesis paragraph (80-150 words)

Stored in results.db with track='C' (Active Mode).
"""

import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection.fetch_ss_search import _get, SS_BASE

# Retry schedule for Semantic Scholar 429s. Both call sites previously passed
# [], i.e. a single attempt with no retry, which contradicts _get's docstring
# ("The active agent passes a SHORT schedule"). Under concurrent workers SS
# 429s routinely and a dropped call silently returns an empty result list, so
# the agent explores blind while still spending a tool call. Measured empty
# SEARCH rate was 2.1% (2026-05 run), 20.7% (2026-06), 66.3% (2026-09-01),
# tracking how hard SS was throttling rather than anything about the models.
# A serial probe on 2026-09-02 recovered 5/5 throttled queries within one 2s
# retry, so a short schedule restores the intended behavior. Waits are short
# on purpose: SS recovers in ~1-2s and a 60s wait per 429 would destroy agent
# throughput across the many SEARCH/FETCH calls one rollout makes.
SS_BACKOFFS = [3, 6, 12]
import config as cfg
from openai import OpenAI

# SS recovers from 429 within ~1-2s; the agent makes many SEARCH/FETCH calls
# per idea, so the Phase-1 default [60,120,300]s backoff would make each idea
# take minutes. Short schedule keeps agent throughput sane under SS rate limits.
SS_ACTIVE_BACKOFFS = [3, 6, 12]

# Global SS rate throttle: SS allows ~1 req/s, so with multiple parallel agent
# workers, un-throttled concurrent SS calls cause a 429 storm. This module-level
# lock spaces ALL SS requests >= SS_MIN_INTERVAL apart across every worker, so we
# can raise worker count (LLM calls parallelize) while SS stays under its limit.
import threading as _threading
_SS_LOCK = _threading.Lock()
_SS_LAST = [0.0]
SS_MIN_INTERVAL = 1.5   # SS sustains ~1 req / 1.5s (clean probe: 80% ok at 1.7s)
SS_MAX_ATTEMPTS = 1     # NO hidden retry: the global throttle (~40 SS calls/min)
                        # is the throughput ceiling, and retries multiply SS calls
                        # per idea, starving overall idea completion. On a 429 the
                        # SEARCH returns empty and the agent FINALs from parametric
                        # knowledge (grounding best-effort) — fail-fast maximizes
                        # idea throughput under the SS rate limit.


def _ss_throttle():
    with _SS_LOCK:
        wait = SS_MIN_INTERVAL - (time.time() - _SS_LAST[0])
        if wait > 0:
            time.sleep(wait)
        _SS_LAST[0] = time.time()


def _ss_call(fn):
    """Run a single-attempt SS call (fn returns data or None) under the global
    throttle, retrying up to SS_MAX_ATTEMPTS. EVERY attempt (incl. retries) goes
    through _ss_throttle so all SS requests stay globally spaced — avoids the
    429 storm caused by un-throttled internal backoff retries colliding across
    parallel workers."""
    for attempt in range(SS_MAX_ATTEMPTS):
        _ss_throttle()
        data = fn()
        if data:
            return data
    return None


def is_key_limit_error(e) -> bool:
    """OpenRouter key spend/credit cap exhausted (403). NON-retryable — the whole
    batch must stop and the user must raise the key limit / add credits."""
    s = str(e).lower()
    return ("limit exceeded" in s or "insufficient credits" in s
            or ("403" in s and "limit" in s))


def _chat_with_retry(client, attempts: int = 4, **kwargs):
    """LLM call with exponential backoff on TRANSIENT errors (429 rate-limit,
    5xx, timeouts). The raw agent call had no retry, so a single transient
    OpenRouter 429 turned into an empty hypothesis (blank). Key-limit 403 is
    NON-retryable and propagates immediately so the caller/orchestrator aborts."""
    last = None
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last = e
            if is_key_limit_error(e):
                raise
            s = str(e).lower()
            transient = any(t in s for t in (
                "429", "500", "502", "503", "504", "timeout", "timed out",
                "connection", "rate limit", "temporar", "overloaded"))
            if attempt < attempts - 1 and transient:
                time.sleep(min(2 * (2 ** attempt) + random.uniform(0, 0.5), 20))
                continue
            raise
    raise last


# ---------------------------------------------------------------------------
# Tool schema (OpenAI function calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search Semantic Scholar for scientific papers by keyword. "
                "Returns paper titles, abstracts, year, and citation counts. "
                "Use this to explore the literature in a research area."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (e.g. 'protein structure prediction transformer')",
                    },
                    "year_range": {
                        "type": "string",
                        "description": "Year range like '2022-2025' (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10, max 20)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_references",
            "description": (
                "Given a Semantic Scholar paper ID, return its top-15 references "
                "(title + abstract + citation count) ordered by influential citations. "
                "Use this to dig deeper after finding a promising paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "Semantic Scholar paperId from a previous search result",
                    },
                },
                "required": ["paper_id"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _search_papers(query: str, year_range: Optional[str] = None, limit: int = 10) -> List[dict]:
    """Call SS /paper/search and return summarized results."""
    api_key = cfg.SEMANTIC_SCHOLAR_API_KEY if hasattr(cfg, 'SEMANTIC_SCHOLAR_API_KEY') else None
    limit = max(1, min(limit or 10, 20))
    params = {
        "query": query,
        "fields": "paperId,title,abstract,year,citationCount",
        "limit": limit,
    }
    if year_range:
        params["year"] = year_range

    data = _ss_call(lambda: _get(f"{SS_BASE}/paper/search", params, api_key, 0.2,
                                 backoffs=SS_BACKOFFS))
    if not data:
        return []

    out = []
    for p in (data.get("data") or [])[:limit]:
        if not p:
            continue
        abstract = p.get("abstract") or ""
        out.append({
            "paperId": p.get("paperId"),
            "title": p.get("title"),
            "abstract": abstract[:1500] if abstract else "",
            "year": p.get("year"),
            "citationCount": p.get("citationCount") or 0,
        })
    return out


def _get_paper_references(paper_id: str) -> List[dict]:
    """Fetch refs, return top 15 by (isInfluential, citationCount)."""
    from data_collection.fetch_ss_search import _fetch_refs
    api_key = cfg.SEMANTIC_SCHOLAR_API_KEY if hasattr(cfg, 'SEMANTIC_SCHOLAR_API_KEY') else None
    refs = _ss_call(lambda: _fetch_refs(paper_id, api_key, 0.2, backoffs=SS_BACKOFFS))
    if not refs:
        return []

    # Sort by (isInfluential desc, citationCount desc)
    refs.sort(
        key=lambda r: (1 if r.get("isInfluential") else 0, r.get("citationCount") or 0),
        reverse=True,
    )
    out = []
    for r in refs[:15]:
        abstract = r.get("abstract") or ""
        out.append({
            "paperId": r.get("paperId"),
            "title": r.get("title"),
            "abstract": abstract[:1500] if abstract else "",
            "year": r.get("year"),
            "citationCount": r.get("citationCount") or 0,
            "isInfluential": bool(r.get("isInfluential")),
        })
    return out


def _dispatch_tool(name: str, args: dict) -> str:
    """Execute a tool call and return JSON-serialized result."""
    try:
        if name == "search_papers":
            result = _search_papers(
                query=args.get("query", ""),
                year_range=args.get("year_range"),
                limit=args.get("limit", 10),
            )
        elif name == "get_paper_references":
            result = _get_paper_references(paper_id=args.get("paper_id", ""))
        else:
            return json.dumps({"error": f"unknown tool: {name}"})
        return json.dumps(result)
    except Exception as e:
        logger.warning(f"  tool {name} failed: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_AGENT = """\
You are a creative and rigorous scientist. You will be given a research domain \
name and you must propose a single novel, testable scientific hypothesis in \
that domain.

To do this, you have access to Semantic Scholar search tools. You should:
  1. Search the literature to identify recent work and open problems in the domain
  2. (Optional) Drill into a promising paper's references for deeper context
  3. Synthesize a genuinely novel hypothesis that goes BEYOND what exists

You have a budget of {max_iters} tool calls total. Explore broadly before \
committing to a direction: try multiple distinct angles (different sub-areas, \
different methods, different recent vs foundational works) and use \
`get_paper_references` to drill into the most promising lead. Aim to use most \
of your budget; only stop early if you have already found a genuine, \
non-obvious gap that is well-supported by what you read.

Your FINAL output (after all tool calls) must be exactly one paragraph, \
80-150 words, in first-person future tense, starting with 'Hypothesis: ' or \
'We hypothesize'. Be specific — name mechanisms, methods, datasets. Do NOT \
include any other text, preamble, or explanation in your final output."""

USER_PROMPT_AGENT = """\
Domain: {domain}

Propose a single novel scientific hypothesis in this domain. Use the search \
tools to explore the literature, then output a single 80-150 word hypothesis \
paragraph."""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

# ════════════════════════════════════════════════════════════════════════════
# Text-protocol agent (DEFAULT since 2026-05-18)
# ════════════════════════════════════════════════════════════════════════════
#
# Why: OpenAI function-calling protocol (legacy below) fails for older / smaller
# models that don't emit structured `tool_calls` JSON (e.g. Mistral-7B-v0.1,
# Gemma-3-27B in markdown mode, GPT-4 with degenerate single-call behavior).
# Switching to plain-text commands works for ANY chat model — model outputs
# `SEARCH: <query>` / `FETCH: <paper_id>` / `FINAL: <hypothesis>` and we parse
# regex. Tool data source (Semantic Scholar) is unchanged.
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEXT = """\
You are a creative and rigorous scientist. Given a research domain name, \
you must propose a single novel, testable scientific hypothesis in that \
domain.

To do this, you have access to Semantic Scholar via three plain-text \
commands. Each turn, output EXACTLY ONE command on the FIRST line:

  SEARCH: <keyword query, 4-10 words>
      → returns up to 10 papers with title + abstract + paperId

  FETCH: <paperId from a prior SEARCH result>
      → returns the top 15 references of that paper

  FINAL: <one-paragraph hypothesis, 80-150 words, first-person future \
tense, naming specific mechanisms / methods / datasets>
      → ends the agent loop

You have a budget of {max_iters} tool calls (SEARCH/FETCH counts as one each). \
Explore broadly before committing: try multiple distinct angles (different \
sub-areas, methods, recent vs foundational works), then FETCH the most \
promising lead's references. Aim to use most of your budget; only stop early \
if you have found a genuine non-obvious gap.

Output format is strict: the FIRST line MUST start with `SEARCH:`, `FETCH:`, \
or `FINAL:`. Do not wrap in code blocks. Do not output multiple commands in \
one turn."""

USER_PROMPT_TEXT = """\
Domain: {domain}

Start by searching the literature. First command:"""

# v2_topic_refs (2026-05-18+): give agent the target paper title too.
USER_PROMPT_TEXT_V2 = """\
Domain: {domain}
Paper title: {title}

You are proposing a hypothesis in the same research area as this paper. \
Use the SEARCH tool to ground yourself in the relevant literature, then \
FETCH a promising paper's references for depth. Start with your first command:"""


_CMD_RE = re.compile(
    r"(?im)^\s*(SEARCH|FETCH|FINAL)\s*[:：]\s*(.+?)(?=\n\s*(?:SEARCH|FETCH|FINAL)\s*[:：]|\Z)",
    re.DOTALL,
)


def _parse_command(content: str) -> tuple:
    """Extract (cmd, arg) from model output. None if unparseable."""
    if not content:
        return (None, None)
    # Strip code-block wrappers
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = _CMD_RE.search(text)
    if not m:
        return (None, None)
    cmd = m.group(1).upper()
    arg = m.group(2).strip()
    # Strip trailing markdown / extra noise; first paragraph only
    arg = re.split(r"\n\s*\n", arg, maxsplit=1)[0].strip()
    return (cmd, arg)


def _format_papers_block(papers: list, prefix: str = "") -> str:
    """Format SS results into plain-text block for context."""
    if not papers:
        return "(no results)"
    lines = []
    for i, p in enumerate(papers, 1):
        pid = p.get("paperId") or "?"
        title = p.get("title") or "(untitled)"
        year = p.get("year") or "?"
        abstract = (p.get("abstract") or "").strip()
        line = f"[{i}] paperId={pid} | year={year} | {title}"
        if abstract:
            line += f"\n    {abstract[:600]}"
        lines.append(line)
    return f"{prefix}\n" + "\n".join(lines)


def run_active_agent(
    domain: str,
    model_name: str,
    max_iters: int = 10,
    seed: int = 42,
    system_prompt_override: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """Plain-text active agent (default since 2026-05-18).

    Model emits SEARCH/FETCH/FINAL commands as plain text on the first line
    of each response. Works with any chat model — no function-calling needed.

    Same return shape as legacy version:
        {hypothesis, trace, turns, iters_used, n_tool_calls, n_turns,
         final_prompt_tokens, final_total_tokens, budget_nudge_used, error}

    `system_prompt_override`: replaces SYSTEM_PROMPT_TEXT (with {max_iters}
    substituted if present). Use cautiously — must preserve the SEARCH/FETCH/
    FINAL contract or parsing will fail.
    """
    if cfg.use_gemini_native(model_name):
        # Direct-to-Google native Gemini (own quota; no seed support).
        client = OpenAI(api_key=cfg.GEMINI_NATIVE_KEY,
                        base_url=cfg.GEMINI_NATIVE_BASE_URL, timeout=180.0)
        api_model = cfg.gemini_native_model_id(model_name)
    elif cfg.use_nv_internal(model_name):
        # Internal NVIDIA gateway (frontier closed-source routes).
        client = OpenAI(api_key=cfg.NV_INTERNAL_API_KEY,
                        base_url=cfg.NV_INTERNAL_BASE_URL, timeout=240.0)
        api_model = model_name
    else:
        api_key = cfg.get_openrouter_key_for_model(model_name)
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        api_model = model_name

    if system_prompt_override is not None:
        system = (system_prompt_override.format(max_iters=max_iters)
                  if "{max_iters}" in system_prompt_override
                  else system_prompt_override)
    else:
        system = SYSTEM_PROMPT_TEXT.format(max_iters=max_iters)
    if title:
        user = USER_PROMPT_TEXT_V2.format(domain=domain, title=title)
    else:
        user = USER_PROMPT_TEXT.format(domain=domain)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    trace = []
    turns = []
    hypothesis = None
    error = None
    nudged = False
    n_tool_calls = 0
    n_malformed = 0
    MAX_MALFORMED = 2

    for it in range(max_iters + 3):  # +3 = room for final + nudge + retry on malformed
        try:
            _kw = dict(
                model=api_model,
                messages=messages,
                # Reasoning models (e.g. qwen3.5-397b-a17b) spend 800-1500+ tokens
                # on reasoning per turn; 1500 total truncated their FINAL paragraph
                # to empty -> parsed as malformed. Give room for reasoning + output.
                max_tokens=6000,
            )
            # Anthropic routes on the internal gateway 400 on any `temperature`.
            if not cfg.nv_internal_no_temperature(model_name):
                _kw["temperature"] = 0.7
            # Native Gemini rejects `seed`; everyone else keeps determinism.
            if not cfg.use_gemini_native(model_name):
                _kw["seed"] = seed
            resp = _chat_with_retry(client, **_kw)
        except Exception as e:
            logger.warning(f"  agent call failed (iter {it}): {e}")
            error = f"api error: {e}"
            break

        if not resp.choices:
            error = "empty response"
            break

        # Telemetry
        usage = getattr(resp, "usage", None)
        prompt_tok = getattr(usage, "prompt_tokens", None) if usage else None
        comp_tok = getattr(usage, "completion_tokens", None) if usage else None
        total_tok = getattr(usage, "total_tokens", None) if usage else None
        reason_tok = None
        if usage:
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd:
                reason_tok = getattr(ctd, "reasoning_tokens", None)
        turns.append({
            "iter": it,
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "reasoning_tokens": reason_tok,
            "total_tokens": total_tok,
        })

        content = resp.choices[0].message.content or ""
        cmd, arg = _parse_command(content)

        # Malformed output — nudge once, give up after MAX_MALFORMED
        if cmd is None:
            n_malformed += 1
            if n_malformed > MAX_MALFORMED:
                # Try to extract hypothesis from the rambly output as last resort
                hypothesis = content.strip()
                error = f"malformed_x{n_malformed}"
                break
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": ("Your output didn't start with SEARCH:, FETCH:, "
                            "or FINAL:. Reply again with EXACTLY ONE command "
                            "on the first line."),
            })
            continue

        messages.append({"role": "assistant", "content": content})

        if cmd == "FINAL":
            hypothesis = arg
            break

        # SEARCH or FETCH — check budget
        if n_tool_calls >= max_iters:
            # Force final
            messages.append({
                "role": "user",
                "content": ("You've reached the tool budget. Output FINAL: "
                            "<hypothesis> now (80-150 words, first-person "
                            "future tense)."),
            })
            nudged = True
            continue

        # Dispatch
        if cmd == "SEARCH":
            query = arg[:200].strip()
            logger.info(f"  iter {it}: SEARCH({query[:70]!r})")
            papers = _search_papers(query=query, limit=10)
            result_str = json.dumps(papers, ensure_ascii=False)
            trace.append({
                "iter": it,
                "tool": "search_papers",
                "args": {"query": query, "limit": 10},
                "result_preview": result_str[:50_000],
            })
            n_tool_calls += 1
            messages.append({
                "role": "user",
                "content": _format_papers_block(
                    papers, f"SEARCH RESULT for query '{query}':"),
            })
        elif cmd == "FETCH":
            # Sanitize paper_id: model may include junk like "https://..." or backticks
            pid = re.sub(r"[`\"'\s]+", "", arg.split()[0] if arg.split() else arg)
            pid = pid.strip("()[]{},.;").split("/")[-1][:80]
            if not pid:
                messages.append({
                    "role": "user",
                    "content": ("FETCH requires a paperId from a prior SEARCH "
                                "result. Output another SEARCH: or FETCH: command."),
                })
                continue
            logger.info(f"  iter {it}: FETCH({pid!r})")
            refs = _get_paper_references(pid)
            result_str = json.dumps(refs, ensure_ascii=False)
            trace.append({
                "iter": it,
                "tool": "get_paper_references",
                "args": {"paper_id": pid},
                "result_preview": result_str[:50_000],
            })
            n_tool_calls += 1
            messages.append({
                "role": "user",
                "content": _format_papers_block(
                    refs, f"REFERENCES of paperId={pid}:"),
            })

    iters_used = len({t["iter"] for t in trace})
    final_turn = turns[-1] if turns else {}

    return {
        "hypothesis": (hypothesis or "").strip(),
        "trace": trace,
        "turns": turns,
        "iters_used": iters_used,
        "n_tool_calls": n_tool_calls,
        "n_turns": len(turns),
        "final_prompt_tokens": final_turn.get("prompt_tokens"),
        "final_total_tokens": final_turn.get("total_tokens"),
        "budget_nudge_used": nudged,
        "n_malformed": n_malformed,
        "error": error,
    }


# ════════════════════════════════════════════════════════════════════════════
# Legacy: OpenAI function-calling protocol (kept for reference / fallback)
# ════════════════════════════════════════════════════════════════════════════
#
# This was the default until 2026-05-18. Replaced by plain-text protocol above
# because it failed for models that don't emit structured `tool_calls` JSON
# (Mistral-7B-v0.1, Gemma-3-27B in markdown mode, etc.). Kept here so older
# Track C data can be cross-referenced with the protocol that generated it.
# ════════════════════════════════════════════════════════════════════════════


def _run_active_agent_funccall_legacy(
    domain: str,
    model_name: str,
    max_iters: int = 10,
    seed: int = 42,
    system_prompt_override: Optional[str] = None,
) -> dict:
    """LEGACY function-calling agent. Not used by default. See run_active_agent."""
    api_key = cfg.get_openrouter_key_for_model(model_name)
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    if system_prompt_override is not None:
        system = system_prompt_override.format(max_iters=max_iters) \
            if "{max_iters}" in system_prompt_override else system_prompt_override
    else:
        system = SYSTEM_PROMPT_AGENT.format(max_iters=max_iters)
    user = USER_PROMPT_AGENT.format(domain=domain)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    trace = []
    turns = []
    hypothesis = None
    error = None
    nudged = False

    for it in range(max_iters + 2):
        try:
            resp = _chat_with_retry(
                client,
                model=model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                seed=seed,
            )
        except Exception as e:
            logger.warning(f"  agent call failed (iter {it}): {e}")
            error = f"api error: {e}"
            break

        if not resp.choices:
            error = "empty response"
            break

        usage = getattr(resp, "usage", None)
        prompt_tok = getattr(usage, "prompt_tokens", None) if usage else None
        comp_tok = getattr(usage, "completion_tokens", None) if usage else None
        total_tok = getattr(usage, "total_tokens", None) if usage else None
        reason_tok = None
        if usage:
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd:
                reason_tok = getattr(ctd, "reasoning_tokens", None)
        turns.append({
            "iter": it,
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "reasoning_tokens": reason_tok,
            "total_tokens": total_tok,
        })

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            hypothesis = (msg.content or "").strip()
            break

        if it >= max_iters:
            messages.append({
                "role": "user",
                "content": (
                    "You've reached the tool budget. Based on what you have, "
                    "output your FINAL hypothesis paragraph now (80-150 words, "
                    "no more tool calls, no preamble)."
                )
            })
            nudged = True
            continue

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            logger.info(f"  iter {it}: {tool_name}({tool_args})")
            result = _dispatch_tool(tool_name, tool_args)

            trace.append({
                "iter": it,
                "tool": tool_name,
                "args": tool_args,
                "result_preview": result[:50_000],
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    iters_used = len({t["iter"] for t in trace})
    final_turn = turns[-1] if turns else {}

    return {
        "hypothesis": hypothesis or "",
        "trace": trace,
        "turns": turns,
        "iters_used": iters_used,
        "n_tool_calls": len(trace),
        "n_turns": len(turns),
        "final_prompt_tokens": final_turn.get("prompt_tokens"),
        "final_total_tokens": final_turn.get("total_tokens"),
        "budget_nudge_used": nudged,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Batch runner — stores results into results.db with track='C'
# ---------------------------------------------------------------------------

def run_active_phase(db_papers: str, db_results: str,
                     idea_model: str,
                     smoke: bool = False,
                     n_ideas: int = 1,
                     max_iters: int = 10) -> dict:
    """Run active-mode agent for each (domain-anchor paper × idea).

    Stores hypothesis as track='C' in results.db. For each target paper in
    smoke set, we run the agent with just the paper's domain name (ignoring
    the paper's content). The paper row is just an anchor so Phase 3 can
    reuse the same critic_manager pipeline.
    """
    import sqlite3
    from datetime import datetime, timezone
    from generation.generate_ideas import _clean_idea_text

    conn_p = sqlite3.connect(db_papers, timeout=30)
    conn_p.row_factory = sqlite3.Row
    conn_r = sqlite3.connect(db_results, timeout=30)
    conn_r.execute("PRAGMA busy_timeout = 30000")
    cur = conn_r.cursor()

    query = """
        SELECT * FROM papers WHERE status='filtered'
          AND gt_hypothesis IS NOT NULL AND gt_hypothesis != ''
          AND ranked_refs_json IS NOT NULL
        ORDER BY domain, paper_id
    """
    papers = [dict(r) for r in conn_p.execute(query).fetchall()]
    conn_p.close()

    if smoke:
        from collections import defaultdict
        by_domain = defaultdict(list)
        for p in papers:
            by_domain[p["domain"]].append(p)
        papers = [p for ps in by_domain.values() for p in ps[:5]]
        papers.sort(key=lambda p: (p["domain"], p["paper_id"]))
        logger.info(f"[smoke] {len(papers)} papers ({len(by_domain)} domains) × {idea_model!r} (active mode)")
    else:
        logger.info(f"active mode: {len(papers)} papers × {idea_model!r}")

    stats = {"papers": len(papers), "generated": 0, "errors": 0}
    ts = datetime.now(timezone.utc).isoformat()

    for i, paper in enumerate(papers, 1):
        pid = paper["paper_id"]
        domain = paper["domain"]
        logger.info(f"  [{i}/{len(papers)}] {domain} | {paper['title'][:50]}")

        for idea_idx in range(1, n_ideas + 1):
            # Skip if already generated
            exists = cur.execute(
                "SELECT 1 FROM results WHERE paper_id=? AND idea_model=? "
                "AND track='C' AND idea_index=? AND critic_model=''",
                (pid, idea_model, idea_idx)
            ).fetchone()
            if exists:
                continue

            try:
                result = run_active_agent(domain, idea_model, max_iters=max_iters)
                hypothesis = _clean_idea_text(result["hypothesis"])
                if not hypothesis or len(hypothesis.split()) < 30:
                    logger.warning(f"    empty/short hypothesis; error: {result.get('error')}")
                    stats["errors"] += 1
                    continue
            except Exception as e:
                logger.error(f"    agent error: {e}")
                stats["errors"] += 1
                continue

            # Persist full telemetry (no truncation): trace + per-turn token usage
            telemetry = {
                "trace": result.get("trace", []),
                "turns": result.get("turns", []),
                "iters_used": result.get("iters_used"),
                "n_tool_calls": result.get("n_tool_calls"),
                "n_turns": result.get("n_turns"),
                "final_prompt_tokens": result.get("final_prompt_tokens"),
                "final_total_tokens": result.get("final_total_tokens"),
                "budget_nudge_used": result.get("budget_nudge_used"),
                "error": result.get("error"),
            }
            cur.execute("""
                INSERT OR IGNORE INTO results
                  (paper_id, idea_model, track, idea_index, idea_text,
                   critic_model, raw_response, created_at)
                VALUES (?, ?, 'C', ?, ?, '', ?, ?)
            """, (pid, idea_model, idea_idx, hypothesis,
                  json.dumps(telemetry), ts))
            if cur.rowcount > 0:
                stats["generated"] += 1
                conn_r.commit()
                logger.info(
                    f"    ✓ {len(hypothesis.split())} words, "
                    f"{telemetry['n_tool_calls']} tool calls / "
                    f"{telemetry['iters_used']} iters / "
                    f"final_ctx={telemetry['final_prompt_tokens']} tok / "
                    f"nudge={telemetry['budget_nudge_used']}")

    conn_r.close()
    logger.info(f"active phase done: {stats}")
    return stats


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="CS / Biology / Chemistry / Medicine / Physics")
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--batch", action="store_true", help="Run batch active phase instead of single test")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.batch:
        stats = run_active_phase(
            db_papers=str(cfg.PAPERS_DB),
            db_results=str(cfg.RESULTS_DB),
            idea_model=args.model,
            smoke=args.smoke,
            max_iters=args.max_iters,
        )
        print(stats)
    else:
        assert args.domain, "--domain required for single test"
        print(f"=== Running active agent: {args.model} in {args.domain} ===\n")
        result = run_active_agent(args.domain, args.model, max_iters=args.max_iters)

        print(f"\n--- Tool calls ({len(result['trace'])}) ---")
        for t in result["trace"]:
            print(f"  [{t['iter']}] {t['tool']}: {t['args']}")

        print(f"\n--- Hypothesis ({len(result['hypothesis'].split())} words) ---")
        print(result["hypothesis"])

        if result.get("error"):
            print(f"\n[error] {result['error']}")
