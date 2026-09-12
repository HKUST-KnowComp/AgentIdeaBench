"""E42 — Cutoff-safe closed-source extension (OpenAI + Anthropic via OpenRouter).

Why this exists
---------------
E41 added five 2026 frontier closed models through the internal gateway, but
their knowledge cutoffs are undisclosed and almost certainly postdate the
benchmark's reference window (config.dataset.paper_date_start 2025-04-01 ..
paper_date_end 2026-01-31). They are therefore held out of the headline
statistics and the cutoff regression -- which leaves the main roster with NO
OpenAI or Anthropic models at all.

E42 fills that hole with closed models whose *disclosed* cutoffs land strictly
before 2025-04-01, so they cannot have memorised the target papers and can enter
the F2 capability-gate and F3 cutoff analyses on the same footing as the 28
open-weight models.

Everything downstream is unchanged: same 40 scored subdomains (e19 idx
{0,5,10,15} + e24 idx {2,7,12,17}), idea_index 1..3, Track B (curated refs) and
Track C (agent-controlled retrieval, budget 10), written INSERT OR IGNORE into
`subdomain_ideas`. Generation and status are reused verbatim from E41 -- the
only thing that differs is the roster and the provider path.

Routing: the internal NVIDIA gateway (env API_KEY), same path as E41. This is
forced, not chosen -- OPENROUTER_US_API_KEY, the standing route for OpenAI /
Anthropic / Gemini, is exhausted (250/250 credits used; every one of the 18
OpenRouter candidates returned HTTP 402 on 2026-08-21). The gateway carries the
same OpenAI models under azure/* routes, so the roster below uses gateway ids.
Consequence to keep in mind: idea_model values are gateway routing names, so the
cutoff axis in reports/_make_cross_year_plot.py needs the azure/* keys added
before these models appear on it.

Cutoff provenance is recorded per model in ROSTER below: "disclosed" = stated by
the vendor in model documentation; "family" = inherited from the vendor's stated
family-level cutoff when the individual card does not restate it.

Usage:
  e42_cutoff_safe_closed.py --check-models          # live probe: params + 200 OK
  e42_cutoff_safe_closed.py --gen --smoke           # 1 model x 2 subs
  e42_cutoff_safe_closed.py --gen --track B --workers 6
  e42_cutoff_safe_closed.py --gen --track C --workers 3
  e42_cutoff_safe_closed.py --status
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from experiments.e41_frontier_extend import gen, status, scored_subs, N_IDX  # noqa: F401

# The red line: a model is eligible only if its disclosed knowledge cutoff is
# strictly before the earliest target paper (config paper_date_start).
RED_LINE = "2025-04-01"

# (model_id, cutoff, provenance, group). All routes verified 200 OK against the
# live gateway on 2026-08-21 with the pipeline's exact params; gpt-5-chat is the
# sole route needing `temperature` omitted (config._NV_NO_TEMPERATURE_IDS).
#
# On "exposure" rather than a binary red line
# -------------------------------------------
# RED_LINE below is config.dataset.paper_date_start, but the curated references
# actually shown for the 40 scored subdomains span 2007-2026 (measured: 328 refs,
# 32.0% from 2025 and 6.4% from 2026), so no model in any roster has zero
# exposure and none has full exposure either. What separates the groups is how
# much of that corpus predates the cutoff, and -- more importantly -- whether
# the cutoff is known well enough to compute it at all:
#
#   cutoff year 2023 -> 143/328 = 43.6% of refs already seen
#   cutoff year 2024 -> 202/328 = 61.6%
#   cutoff year 2025 -> 307/328 = 93.6%   (upper bound; see below)
#   cutoff year 2026 -> 328/328 = 100%
#
# subdomain_refs stores `year` only, no month, so a mid-2025 cutoff cannot be
# resolved past the year boundary: a 2025-04-01 cutoff sits somewhere in
# [61.6%, 93.6%] and the exact figure is NOT COMPUTABLE from the stored data.
#
# group "safe"        -- cutoff disclosed AND strictly before RED_LINE. Exposure
#                        computable and bounded (43.6-61.6%). Eligible for the
#                        headline statistics and the cutoff regression.
# group "boundary"    -- cutoff lands exactly ON RED_LINE. Exposure in
#                        [61.6%, 93.6%], point value not computable. Reported as
#                        a flagged group, never pooled into headline numbers.
# group "partial"     -- cutoff sits just PAST RED_LINE, inside the reference
#                        window, but is still a recorded value, so the group can
#                        be placed on the cutoff axis with the caveat attached.
# group "undisclosed" -- no cutoff recorded anywhere in the repo and none
#                        published. Exposure not computable at any granularity.
#                        These belong with the E41 held-out five: run and
#                        reported, never on the cutoff axis, never in headline.
ROSTER = [
    # --- OpenAI, 2023-10 cutoff ---
    ("azure/openai/gpt-4o-mini",         "2023-10-01", "disclosed", "safe"),
    ("azure/openai/gpt-4o",              "2023-10-01", "disclosed", "safe"),
    ("azure/openai/o1",                  "2023-10-01", "disclosed", "safe"),
    ("azure/openai/o3-mini",             "2023-10-01", "disclosed", "safe"),
    # --- OpenAI, 2024-05/06 cutoff ---
    ("us/azure/openai/gpt-4.1-nano",     "2024-06-15", "family",    "safe"),
    ("azure/openai/gpt-4.1-mini",        "2024-06-15", "family",    "safe"),
    ("azure/openai/gpt-4.1",             "2024-06-15", "disclosed", "safe"),
    ("azure/openai/o3",                  "2024-06-01", "disclosed", "safe"),
    ("azure/openai/o4-mini",             "2024-06-01", "disclosed", "safe"),
    ("azure/openai/gpt-5-nano",          "2024-05-30", "disclosed", "safe"),
    ("azure/openai/gpt-5-mini",          "2024-05-30", "disclosed", "safe"),
    # --- OpenAI, 2024-09 cutoff (gpt-5 tier, still pre-red-line) ---
    ("azure/openai/gpt-5",               "2024-09-30", "disclosed", "safe"),
    # --- OpenAI 5.4 tier: reports/_make_cross_year_plot.py records gpt-5.4-mini
    #     and gpt-5.4-nano at 2025-08-31; gpt-5.4 itself inherits the tier value.
    #     Past the red line but recorded -> "partial".
    ("azure/openai/gpt-5.4-nano",        "2025-08-31", "disclosed", "partial"),
    ("azure/openai/gpt-5.4-mini",        "2025-08-31", "disclosed", "partial"),
    ("azure/openai/gpt-5.4",             "2025-08-31", "family",    "partial"),
    ("azure/openai/gpt-5.5",             "2025-12-01", "disclosed", "partial"),
    # --- OpenAI 5.1-5.3: no cutoff recorded in the repo and none published.
    #     gpt-5.1-chat rejects `temperature` (config._NV_NO_TEMPERATURE_IDS).
    ("azure/openai/gpt-5.1",             "",           "none", "undisclosed"),
    ("azure/openai/gpt-5.2",             "",           "none", "undisclosed"),
    ("azure/openai/gpt-5.3-chat",        "",           "none", "undisclosed"),
    # --- Anthropic 4.5 family: config.anti_leakage.known_cutoffs records
    #     haiku-4.5 and sonnet-4.5 at 2025-04-01; opus-4-5 has no record of its
    #     own and inherits the family value, the same provenance rule already
    #     used for gpt-4.1-nano and gpt-5-chat above.
    ("azure/anthropic/claude-haiku-4-5",  "2025-04-01", "config", "boundary"),
    ("azure/anthropic/claude-sonnet-4-5", "2025-04-01", "config", "boundary"),
    ("azure/anthropic/claude-opus-4-5",   "2025-04-01", "family", "boundary"),
    # --- Anthropic 4.6 family: sonnet-4.6 is recorded at 2025-05-15
    #     ("reliable cutoff per Anthropic") in reports/_make_cross_year_plot.py;
    #     opus-4-6 inherits it. Inside the reference window -> "partial".
    ("azure/anthropic/claude-sonnet-4-6", "2025-05-15", "disclosed", "partial"),
    ("azure/anthropic/claude-opus-4-6",   "2025-05-15", "family",    "partial"),
    # --- Anthropic 4.7 / 4.8: no cutoff recorded in the repo and none
    #     published. Run, but they can only ever be reported alongside the E41
    #     held-out five.
    ("azure/anthropic/claude-opus-4-7",   "",           "none", "undisclosed"),
    ("azure/anthropic/claude-opus-4-8",   "",           "none", "undisclosed"),
]

# Deliberately excluded, with the reason, so the boundary is auditable:
#   openai/*, anthropic/* on OpenRouter (gpt-4 2021-09, gpt-4-turbo,
#       claude-3-haiku 2023-08, claude-sonnet-4 / opus-4 / opus-4.1 2025-01)
#       Cutoff-safe with real buffer, and the ONLY way to get Anthropic models
#       into the "safe" group at all -- the gateway carries nothing older than
#       the 4.5 family. Unreachable because OPENROUTER_US_API_KEY is out of
#       credits (250/250 used, HTTP 402 on all 18, checked 2026-08-21).
#       Top the key up and move these four Claude ids into "safe" first.
#   aws/anthropic/* mirrors of routes already listed
#       Same underlying models via Bedrock instead of Azure. A second route to
#       one model is not a second data point.
#   azure/openai/gpt-5.6-*, azure/anthropic/claude-opus-5, claude-sonnet-5
#       The E41 group itself -- already generated and scored, so re-running them
#       here would only duplicate rows. Reported together with this roster's
#       "undisclosed" group.
#   azure/openai/gpt-5-chat, gpt-5.1-chat, gpt-5.2-chat
#       The "-chat" suffix is the non-reasoning serving config of the same base
#       (gpt-5 / 5.1 / 5.2 are all present as plain routes), so each pair is one
#       model served two ways, not two models. Keeping gpt-5-chat would have
#       been the worst case: it sits in "safe", the only group entering the
#       headline statistics and the cutoff regression, where a duplicated base
#       silently violates the independence those fits assume.
#       gpt-5.3-chat is the exception and IS included -- gpt-5.3 has no plain
#       route on the gateway, so -chat is the only way to reach that base.
#   azure/openai/gpt-5.1-codex, gpt-5.1-codex-max, gpt-5.1-codex-mini,
#   gpt-5.2-codex, gpt-5.3-codex
#       Code-specialised variants of bases already covered by gpt-5.1 / 5.2 /
#       gpt-5.3-chat. Not independent capability points for scientific ideation.
#   Non-OpenAI/Anthropic gateway vendors (gcp/google/gemini-3.5+, nvidia/nvidia
#   nemotron-*, nvidia/qwen/qwen3.6-*, nvidia/moonshotai/kimi-k3,
#   nvidia/minimaxai/minimax-m3, azure/zai-org/glm-5.2, perplexity/sonar-*)
#       Out of scope for THIS experiment, which is about closed OpenAI /
#       Anthropic cutoff coverage. Several are genuinely new to the roster and
#       would be worth their own extension; the ones that merely mirror the
#       28-model OpenRouter roster (qwen3.5-9b, qwen3-32b, glm-5.1, kimi-k2.6,
#       minimax-m2.7, deepseek-v4-pro) are second routes to models already
#       measured, not second data points. perplexity/sonar-* additionally ship
#       their own built-in retrieval, which would confound the Static/Active
#       contrast this benchmark is built on.
#   openai/openai/gpt-3.5-turbo
#       Cutoff-safe (2021-09) but too weak for the Active protocol; the 397b
#       precedent was 50% malformed output, which pollutes rather than informs.

MODELS = [m for m, _, _, _ in ROSTER]
CUTOFF = {m: c for m, c, _, _ in ROSTER}
GROUP = {m: g for m, _, _, g in ROSTER}


def check_red_line():
    """Every roster entry must sit strictly before the red line."""
    print(f"red line (paper_date_start): {RED_LINE}")
    FLAG = {"safe": "OK     ", "boundary": "ON-LINE", "partial": "PAST   ",
            "undisclosed": "UNKNOWN"}
    for m, c, prov, grp in ROSTER:
        print(f"  {FLAG[grp]} {m:34s} cutoff={c or 'not recorded':12s} "
              f"({prov}, {grp})")
    strict = [(m, c) for m, c, _, g in ROSTER if g == "safe" and c >= RED_LINE]
    if strict:
        print(f"!!! {len(strict)} 'safe' entries are not strictly before the red line")
    import collections
    tally = collections.Counter(g for _, _, _, g in ROSTER)
    print("  " + "  ".join(f"{g}={tally[g]}" for g in
                           ("safe", "boundary", "partial", "undisclosed")))
    print(f"  total={len(ROSTER)}  anthropic="
          f"{sum(1 for m,_,_,_ in ROSTER if 'anthropic' in m)}")
    return not strict


def check_models(models):
    """Live probe: one minimal generation-role call per model through the real
    LLM path, so parameter quirks (temperature/seed rejection, empty reasoning
    output) surface before a batch is launched rather than during it."""
    from utils.LLM import IdeaLLM
    print(f"probing {len(models)} models (generation role, real LLM path)...")
    ok, bad = [], []
    for m in models:
        which = "gateway" if cfg.use_nv_internal(m) else (
            "US" if cfg.is_us_key_model(m) else "default")
        try:
            out = IdeaLLM(model_name=m).generate_idea(
                "Reply with exactly the word: READY",
                fallback_prompt="Reply with exactly the word: READY",
                system_prompt="You are a terse assistant.")
            txt = (out.get("idea") or "").strip().replace("\n", " ")
            if txt:
                ok.append(m)
                print(f"  OK   {m:34s} route={which:7s} -> {txt[:50]!r}")
            else:
                bad.append((m, "empty response"))
                print(f"  EMPTY{m:34s} route={which:7s} -> ''")
        except Exception as e:
            bad.append((m, str(e)[:160]))
            print(f"  FAIL {m:34s} route={which:7s} -> {str(e)[:160]}")
    print(f"\nprobe: {len(ok)} ok, {len(bad)} failed")
    for m, e in bad:
        print(f"  - {m}: {e}")
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--check-models", action="store_true",
                    help="live one-call probe of every roster model")
    ap.add_argument("--check-red-line", action="store_true",
                    help="verify every roster cutoff precedes paper_date_start")
    ap.add_argument("--models", nargs="*", default=None, help="subset of ROSTER")
    ap.add_argument("--group", choices=["safe", "boundary", "partial",
                                        "undisclosed"], default=None,
                    help="restrict to one cutoff group")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap tasks this batch (0=all)")
    ap.add_argument("--smoke", action="store_true", help="1 model x 2 subs")
    ap.add_argument("--track", choices=["both", "B", "C"], default="both")
    a = ap.parse_args()

    models = a.models if a.models else MODELS
    if a.group:
        models = [m for m in models if GROUP.get(m) == a.group]

    if a.check_red_line:
        check_red_line()
    if a.check_models:
        check_models(models)
    if a.status:
        status(models)
    if a.gen:
        gen(models, a.workers, a.limit, a.smoke, a.track)


if __name__ == "__main__":
    main()
