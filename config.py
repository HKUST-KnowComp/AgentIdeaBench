"""
SciSynthBench Configuration

Reads config.json and exposes:
  - CFG             raw dict
  - IDEA_MODELS     list[str]
  - CRITIC_MODELS   list[str]
  - NUM_CRITICS_PER_EVAL int
  - PARALLEL        dict
  - RETRY           dict
  - DATASET         dict (serper_topics, papers_per_domain, …)
  - DOMAINS         list[str]
  - ANTI_LEAKAGE    dict
  - ROLLING_WINDOW_DAYS int
  - SAFETY_BUFFER_DAYS  int
  - KNOWN_CUTOFFS   dict[str, str]
  - API keys        (resolved from ${ENV_VAR} placeholders)
  - Data paths      PAPERS_DB, RESULTS_DB, SNAPSHOTS_DIR

Also provides two helper functions used by utils/LLM.py:
  - get_provider_for_model(model_name) → str
  - get_api_key(provider)             → str

Run directly for a quick sanity-check:
  python config.py
"""

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-load .env (if present) before anything else
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE, override=False)  # override=False: env vars already set take priority
    except ImportError:
        # dotenv not installed — parse manually (simple KEY=VALUE, skip comments/blanks)
        with open(_ENV_FILE, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#"):
                    continue
                if "=" in _line:
                    _k, _, _v = _line.partition("=")
                    _k, _v = _k.strip(), _v.strip()
                    if _k and _v and _k not in os.environ:
                        os.environ[_k] = _v

# ---------------------------------------------------------------------------
# Load config.json
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent          # scisynthbench/
_CONFIG_PATH = _ROOT / "config.json"

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    CFG: dict = json.load(_f)

# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------

IDEA_MODELS:        list = CFG["models"]["idea_models"]
CRITIC_MODELS:      list = CFG["models"]["critic_models"]
NUM_CRITICS_PER_EVAL: int = CFG["models"].get("num_critics_per_eval", 3)

# ---------------------------------------------------------------------------
# Runtime tuning
# ---------------------------------------------------------------------------

PARALLEL: dict = CFG.get("parallel", {})
RETRY:    dict = CFG.get("retry", {})
GENERATION_PARAMS: dict = CFG.get("generation_params", {"temperature": 0.7, "seed": 42})
SCORING_PARAMS:    dict = CFG.get("scoring_params", {"temperature": 0.0, "seed": 42})

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET: dict = CFG["dataset"]
DOMAINS: list = list(DATASET["serper_topics"].keys())
GENERATION_FORMAT: str = DATASET.get("generation_format", "paragraph")  # "paragraph" | "structured"

# ---------------------------------------------------------------------------
# Anti-leakage
# ---------------------------------------------------------------------------

ANTI_LEAKAGE:        dict = CFG["anti_leakage"]
ROLLING_WINDOW_DAYS: int  = ANTI_LEAKAGE["rolling_window_days"]
SAFETY_BUFFER_DAYS:  int  = ANTI_LEAKAGE["safety_buffer_days"]
KNOWN_CUTOFFS:       dict = ANTI_LEAKAGE["known_cutoffs"]

# ---------------------------------------------------------------------------
# API keys  (resolve ${VAR_NAME} → os.environ)
# ---------------------------------------------------------------------------

def _resolve(val: str) -> str:
    """Expand ${VAR_NAME} placeholders from environment variables."""
    if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
        env_var = val[2:-1]
        return os.environ.get(env_var, "")
    return val or ""


_api = CFG["api"]
OPENROUTER_API_KEY:       str = _resolve(_api.get("openrouter_api_key", ""))
# US-tier OpenRouter key — needed for OpenAI / Anthropic / Gemini providers
# whose access is restricted on the default key.
OPENROUTER_US_API_KEY:    str = os.environ.get("OPENROUTER_US_API_KEY", "")
# Optional native Google Gemini key (AI Studio, OpenAI-compat endpoint). When
# set, `google/gemini*` idea-models route DIRECTLY to Google (own quota),
# bypassing OpenRouter. When empty, behaviour is unchanged (OpenRouter US key).
# Opt-in only → zero impact on existing 30-model runs.
GEMINI_NATIVE_KEY:        str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_NATIVE_BASE_URL:   str = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Internal NVIDIA inference gateway (OpenAI-compatible; see Inference_Tutorial.md).
# Key = env API_KEY, a LiteLLM virtual key ("sk-..."). Opt-in only: when unset,
# nothing routes here and behaviour is unchanged for the existing roster.
NV_INTERNAL_API_KEY:      str = os.environ.get("API_KEY", "")
# The internal gateway route is a collaborator's infrastructure, so its
# endpoint lives in the environment rather than in this file. Leave it unset
# and use_nv_internal() stays off, which is the default for anyone outside
# that network; the public OpenRouter and NIM routes are unaffected.
NV_INTERNAL_BASE_URL:     str = os.environ.get("NV_INTERNAL_BASE_URL", "")
NVIDIA_API_KEY:           str = _resolve(_api.get("nvidia_api_key", ""))
SEMANTIC_SCHOLAR_API_KEY: str = _resolve(_api.get("semantic_scholar_api_key", ""))
DEFAULT_PROVIDER:         str = _api.get("default_provider", "openrouter")

# Model-id prefixes that require the US-tier key
_US_KEY_PREFIXES = ("openai/", "anthropic/", "google/gemini")

# Internal-gateway route prefixes. Gateway ids are ROUTING names and are not
# interchangeable — azure/openai/gpt-5.6-sol and openai/openai/gpt-5.6-sol are
# separate routes. We standardise on the azure/* routes (the tutorial's curated
# catalog); the others are listed so a deliberate route switch still resolves.
_NV_INTERNAL_PREFIXES = ("azure/", "aws/", "gcp/", "switchyard/", "perplexity/",
                         "us/azure/")

# Gateway models that reject `temperature` outright (HTTP 400,
# "`temperature` is deprecated for this model"). Verified 2026-08-12 against
# azure/anthropic/claude-opus-5 and claude-sonnet-5; gpt-5.6-* accept it.
_NV_NO_TEMPERATURE_PREFIXES = ("azure/anthropic/", "aws/anthropic/")

# Individual gateway routes with the same restriction, where the family as a
# whole is fine. gpt-5-chat 400s on any temperature but the default 1
# ("Unsupported value: 'temperature' does not support 0.7 with this model");
# its reasoning siblings gpt-5 / gpt-5-mini / gpt-5-nano accept 0.7. Verified
# 2026-08-21 against the live gateway.
_NV_NO_TEMPERATURE_IDS = frozenset({"azure/openai/gpt-5-chat",
                                     "azure/openai/gpt-5.1-chat"})


def is_us_key_model(model_id: str) -> bool:
    """Whether this model needs OPENROUTER_US_API_KEY (御三家 routing)."""
    if not model_id:
        return False
    return any(model_id.startswith(p) for p in _US_KEY_PREFIXES)


def get_openrouter_key_for_model(model_id: str) -> str:
    """Return the appropriate OpenRouter key for a given model ID.

    OpenAI / Anthropic / Gemini models go through OPENROUTER_US_API_KEY
    (those providers are blocked on the default key). Everything else
    uses OPENROUTER_API_KEY.
    """
    if is_us_key_model(model_id) and OPENROUTER_US_API_KEY:
        return OPENROUTER_US_API_KEY
    return OPENROUTER_API_KEY


def use_gemini_native(model_id: str) -> bool:
    """Whether this idea-model routes directly to Google's native Gemini
    OpenAI-compat endpoint (own quota) instead of OpenRouter. Opt-in only:
    requires GEMINI_NATIVE_KEY to be set AND a google/gemini* model id.
    When the key is unset this is always False → existing runs unaffected."""
    return bool(GEMINI_NATIVE_KEY) and model_id.startswith("google/gemini")


def use_nv_internal(model_id: str) -> bool:
    """Whether this model routes to the internal NVIDIA gateway. Opt-in only:
    requires NV_INTERNAL_API_KEY (env API_KEY) AND a gateway route prefix.
    When the key is unset this is always False → existing runs unaffected."""
    if not (NV_INTERNAL_API_KEY and model_id):
        return False
    return model_id.startswith(_NV_INTERNAL_PREFIXES)


def nv_internal_no_temperature(model_id: str) -> bool:
    """Whether the gateway rejects `temperature` for this model (see
    _NV_NO_TEMPERATURE_PREFIXES). Callers must omit the parameter entirely —
    passing any value, including the config default, returns HTTP 400."""
    if not model_id:
        return False
    return (model_id.startswith(_NV_NO_TEMPERATURE_PREFIXES)
            or model_id in _NV_NO_TEMPERATURE_IDS)


def gemini_native_model_id(model_id: str) -> str:
    """Strip the 'google/' vendor prefix for the native endpoint (native
    expects 'gemini-2.5-flash', not 'google/gemini-2.5-flash')."""
    return model_id[len("google/"):] if model_id.startswith("google/") else model_id

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATA_DIR:      Path = _ROOT / "data"
PAPERS_DB:     Path = DATA_DIR / "papers.db"
RESULTS_DB:    Path = DATA_DIR / "results.db"
SNAPSHOTS_DIR: Path = DATA_DIR / "snapshots"

# ---------------------------------------------------------------------------
# LLM provider helpers  (imported by utils/LLM.py)
# ---------------------------------------------------------------------------

def get_provider_for_model(model_name: str) -> str:
    """Infer provider from model name prefix.

    Rules:
      nvidia/*                       → nvidia            (public NIM endpoint)
      azure/* aws/* gcp/* …          → nv_internal       (internal gateway, opt-in)
      google/gemini* (+ native key)  → gemini_native
      *                              → openrouter  (default)
    """
    if model_name.startswith("nvidia/"):
        return "nvidia"
    if use_nv_internal(model_name):
        return "nv_internal"
    if use_gemini_native(model_name):
        return "gemini_native"
    return "openrouter"


def get_api_key(provider: str) -> str:
    """Return the API key string for a given provider name."""
    if provider == "nvidia":
        return NVIDIA_API_KEY
    if provider == "nv_internal":
        return NV_INTERNAL_API_KEY
    if provider == "semantic_scholar":
        return SEMANTIC_SCHOLAR_API_KEY
    return OPENROUTER_API_KEY   # default


# ---------------------------------------------------------------------------
# Quick sanity-check (python config.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("SciSynthBench — config.json loaded")
    print("=" * 50)

    print(f"\nIDEA_MODELS ({len(IDEA_MODELS)}):")
    for m in IDEA_MODELS:
        print(f"  - {m}")

    print(f"\nCRITIC_MODELS ({len(CRITIC_MODELS)}):")
    for m in CRITIC_MODELS:
        print(f"  - {m}")

    print(f"\nDomains ({len(DOMAINS)}): {DOMAINS}")
    print(f"Papers per domain: {DATASET['papers_per_domain']}")
    print(f"Total target papers: {DATASET['papers_per_domain'] * len(DOMAINS)}")

    print(f"\nAnti-leakage:")
    print(f"  rolling window : {ROLLING_WINDOW_DAYS} days")
    print(f"  safety buffer  : {SAFETY_BUFFER_DAYS} days")

    print(f"\nAPI keys configured:")
    for name, val in [
        ("OPENROUTER_API_KEY",       OPENROUTER_API_KEY),
        ("NVIDIA_API_KEY",           NVIDIA_API_KEY),
        ("SEMANTIC_SCHOLAR_API_KEY", SEMANTIC_SCHOLAR_API_KEY),
    ]:
        status = "✓ set" if val else "✗ NOT SET"
        print(f"  {name}: {status}")

    print(f"\nData paths:")
    print(f"  PAPERS_DB  : {PAPERS_DB}")
    print(f"  RESULTS_DB : {RESULTS_DB}")
