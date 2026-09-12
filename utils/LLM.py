"""
LLM Interaction Module (adapted from LiveIdeaBench for SciSynthBench)

Supports two providers:
  - openrouter  (default for all non-nvidia models)
  - nvidia      (models whose name starts with "nvidia/")

Adds bounded provider/model concurrency and exponential-backoff retry for
transient API failures.
"""

from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import json
import os
import re
import logging
import time
import random
import threading
from typing import Dict, List, Optional, Union, Tuple, Any

from openai import OpenAI
from utils.constants import SCORING_DIMS

# Lazy import of config so this module can be imported before config is fully
# loaded (e.g. during testing).  Functions that actually call the API will
# trigger the import at call time.
_config = None

def _get_config():
    global _config
    if _config is None:
        import config as _c
        _config = _c
    return _config


logger = logging.getLogger(__name__)

REJECTION_PHRASES = [
    "I'm sorry", "I am sorry", "I apologize", "As an AI", "As a language model",
    "As an assistant", "I cannot", "I can't", "I am unable to", "I'm unable to",
    "I am not able to", "I'm not able to"
]

_SEMAPHORE_LOCK = threading.Lock()
_SEMAPHORES: Dict[Tuple[str, str], threading.BoundedSemaphore] = {}
_THREAD_LOCAL = threading.local()


def _int_cfg(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _float_cfg(value: Any, default: float) -> float:
    try:
        parsed = float(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _provider_limit(provider: str) -> int:
    cfg = _get_config()
    provider_limits = cfg.PARALLEL.get("provider_limits", {}) if hasattr(cfg, "PARALLEL") else {}
    default = 1 if provider == "semantic_scholar" else 8
    return _int_cfg(provider_limits.get(provider), default)


def _model_limit(model_name: str) -> Optional[int]:
    cfg = _get_config()
    if model_name in getattr(cfg, "CRITIC_MODELS", []):
        limit = _int_cfg(cfg.PARALLEL.get("model_limit_per_critic", 2), 2)
        return limit
    return None


def _get_or_create_semaphore(kind: str, key: str, limit: int) -> threading.BoundedSemaphore:
    sem_key = (kind, key)
    with _SEMAPHORE_LOCK:
        sem = _SEMAPHORES.get(sem_key)
        if sem is None:
            sem = threading.BoundedSemaphore(limit)
            _SEMAPHORES[sem_key] = sem
        return sem


def _thread_client_cache() -> Dict[Tuple[str, str, str, float], OpenAI]:
    cache = getattr(_THREAD_LOCAL, "openai_clients", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.openai_clients = cache
    return cache


def _get_thread_client(provider: str, base_url: str, api_key: str,
                       timeout: float = 120.0) -> OpenAI:
    cache = _thread_client_cache()
    cache_key = (provider, base_url, api_key, float(timeout))
    client = cache.get(cache_key)
    if client is None:
        kwargs = {
            "base_url": base_url,
            "api_key": api_key,
        }
        if timeout:
            kwargs["timeout"] = timeout
        client = OpenAI(**kwargs)
        cache[cache_key] = client
    return client


@contextmanager
def _acquire_capacity(provider: str, model_name: str):
    provider_sem = _get_or_create_semaphore("provider", provider, _provider_limit(provider))
    model_limit = _model_limit(model_name)
    model_sem = None

    provider_sem.acquire()
    try:
        if model_limit is not None:
            model_sem = _get_or_create_semaphore("model", model_name, model_limit)
            model_sem.acquire()
        yield
    finally:
        if model_sem is not None:
            model_sem.release()
        provider_sem.release()


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None) or {}
    if not headers:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if not retry_after:
        return None

    retry_after = str(retry_after).strip()
    if not retry_after:
        return None

    try:
        return max(0.0, float(retry_after))
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(retry_after)
        if dt.tzinfo is None:
            return None
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


def _exception_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            val = getattr(response, attr, None)
            if isinstance(val, int):
                return val
    return None


def _is_transient_error(exc: Exception) -> bool:
    status_code = _exception_status_code(exc)
    if status_code in {429, 500, 502, 503, 504}:
        return True

    err = str(exc).lower()
    transient_tokens = (
        "429", "500", "502", "503", "504",
        "internal server error", "rate limit",
        "connection", "timeout", "timed out", "network",
        "remotedisconnected", "connectionreset", "temporarily unavailable",
    )
    return any(token in err for token in transient_tokens)


def _retry_wait_seconds(exc: Exception, attempt: int) -> float:
    cfg = _get_config()
    retry_cfg = getattr(cfg, "RETRY", {})
    jitter = _float_cfg(retry_cfg.get("jitter_seconds"), 0.5)

    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return retry_after

    status_code = _exception_status_code(exc)
    if status_code == 429:
        base_wait = _float_cfg(retry_cfg.get("rate_limit_base_wait_seconds"), 2.0)
        max_wait = _float_cfg(retry_cfg.get("rate_limit_max_wait_seconds"), 120.0)
    else:
        base_wait = _float_cfg(retry_cfg.get("base_wait_seconds"), 1.0)
        max_wait = _float_cfg(retry_cfg.get("max_wait_seconds"), 60.0)

    return min(base_wait * (2 ** attempt) + random.uniform(0, jitter), max_wait)


class BaseLLM:
    """Base LLM — handles OpenRouter and NVIDIA NIM (OpenAI-compatible)."""

    def __init__(self, model_name: str, provider: Optional[str] = None,
                 role: str = "default"):
        self.model_name = model_name
        self.role = role  # "generation" | "scoring" | "default"
        cfg = _get_config()
        self.provider = provider or cfg.get_provider_for_model(model_name)
        self._setup_client()
        self._load_sampling_params()
        # Telemetry: last call's usage (prompt/completion/reasoning tokens).
        # Updated after every successful chat.completions.create. Callers
        # can read .last_usage to record context length per call.
        self.last_usage: Optional[Dict[str, Optional[int]]] = None
        # Latency of the last successful chat.completions.create call (ms,
        # wall-clock; includes server-side reasoning + network). Phase-2/3
        # callers persist this to results.telemetry for per-run speed reports.
        self.last_latency_ms: Optional[int] = None

    def _setup_client(self) -> None:
        cfg = _get_config()
        if self.provider == "nvidia":
            api_key = cfg.NVIDIA_API_KEY
            if not api_key:
                raise ValueError("NVIDIA_API_KEY not set")
            self.client = _get_thread_client(
                provider="nvidia",
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
                timeout=120.0,
            )
        elif self.provider == "nv_internal":
            # Internal NVIDIA gateway (OpenAI-compatible). Frontier closed-source
            # routes live here; ids carry their provider prefix verbatim.
            api_key = cfg.NV_INTERNAL_API_KEY
            if not api_key:
                raise ValueError("NV_INTERNAL_API_KEY (env API_KEY) not set")
            self.client = _get_thread_client(
                provider="nv_internal",
                base_url=cfg.NV_INTERNAL_BASE_URL,
                api_key=api_key,
                timeout=240.0,   # frontier reasoning models: observed up to ~25s/call
            )
        elif self.provider == "gemini_native":
            # Direct-to-Google native Gemini (OpenAI-compat endpoint, own quota).
            api_key = cfg.GEMINI_NATIVE_KEY
            if not api_key:
                raise ValueError("GEMINI_NATIVE_KEY (env GEMINI_API_KEY) not set")
            self.client = _get_thread_client(
                provider="gemini_native",
                base_url=cfg.GEMINI_NATIVE_BASE_URL,
                api_key=api_key,
                timeout=180.0,   # pro forces thinking mode → ~20s/call
            )
        else:
            # Default: OpenRouter — pick US-tier key for OpenAI / Anthropic / Gemini,
            # default key for everyone else.
            api_key = cfg.get_openrouter_key_for_model(self.model_name)
            if not api_key:
                raise ValueError(
                    f"No OpenRouter key available for model '{self.model_name}'. "
                    f"OpenAI/Anthropic/Gemini need OPENROUTER_US_API_KEY; "
                    f"others need OPENROUTER_API_KEY."
                )
            self.client = _get_thread_client(
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                timeout=120.0,   # 2-min hard cap; thinking models rarely exceed 90s
            )

    def _load_sampling_params(self) -> None:
        """Load temperature/seed from config based on role."""
        cfg = _get_config()
        if self.role == "generation":
            params = getattr(cfg, "GENERATION_PARAMS", {})
        elif self.role == "scoring":
            params = getattr(cfg, "SCORING_PARAMS", {})
        else:
            params = {}
        self.temperature = params.get("temperature")
        self.seed = params.get("seed")

    def completion(self, prompt: str, system_prompt: Optional[str] = None) -> Union[str, Tuple[str, str]]:
        """Call the LLM; return a string or (content, reasoning) tuple."""
        return self._openai_compatible_completion(prompt, system_prompt)

    def _openai_compatible_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Union[str, Tuple[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        cfg = _get_config()
        retry_cfg = getattr(cfg, "RETRY", {})
        max_retries = _int_cfg(retry_cfg.get("max_retries"), 6)
        retry_count = 0

        while True:
            try:
                with _acquire_capacity(self.provider, self.model_name):
                    # Native Gemini endpoint expects the bare model id
                    # ('gemini-2.5-flash', no 'google/' prefix).
                    api_model = (cfg.gemini_native_model_id(self.model_name)
                                 if self.provider == "gemini_native"
                                 else self.model_name)
                    api_kwargs = {
                        "model": api_model,
                        "messages": messages,
                        "stream": False,
                    }
                    # Some gateway routes (Anthropic on azure/aws) 400 on any
                    # `temperature`; the parameter has to be absent, not default.
                    if (self.temperature is not None
                            and not cfg.nv_internal_no_temperature(self.model_name)):
                        api_kwargs["temperature"] = self.temperature
                    # Native Gemini rejects `seed` ("Unknown name"); gen is
                    # temp-0.7 stochastic anyway (3 idea_indices give diversity).
                    if self.seed is not None and self.provider != "gemini_native":
                        api_kwargs["seed"] = self.seed
                    # Cap max_tokens so OR doesn't pre-reserve credit for the
                    # full 65k-token reasoning ceiling. Without this, US-key
                    # routes (o1, gemini-2.5-pro, etc.) hit 402 even when the
                    # actual response is small. 4000 covers paragraph+Cited
                    # footer; scoring needs less but we set generously.
                    # The gateway has no credit pre-reservation, and LiteLLM
                    # maps max_tokens onto max_output_tokens, which on the
                    # reasoning routes counts REASONING tokens too. At 4000 the
                    # small reasoning models burn the whole budget thinking and
                    # the request 500s ("gpt-5-nano-global unable to complete
                    # request: max_output_tokens") before emitting any content.
                    # Give those routes headroom; the paragraph itself is still
                    # 80-150 words, so this changes what survives, not what the
                    # model is asked to write.
                    if self.role == "generation":
                        api_kwargs["max_tokens"] = (
                            16000 if cfg.use_nv_internal(self.model_name) else 4000)
                    elif self.role == "scoring":
                        api_kwargs["max_tokens"] = 3000
                    # Force thinking OFF on non-US-key OR models for both:
                    #   generation (cleaner cross-year comparison)
                    #   scoring    (judge throughput — observed 4000-token
                    #               reasoning traces stretching 5-min/call)
                    # US-key models (OpenAI/Anthropic/Gemini) retain default
                    # behaviour since US-key has no budget for re-runs.
                    # Exception: a few OR models silently stop responding
                    # when `reasoning.enabled=False` is set. Keep thinking
                    # ON for them.
                    NO_THINKING_OFF = {
                        "minimax/minimax-m2.7",
                        "deepseek/deepseek-r1-0528",  # OR: "Reasoning is mandatory"
                    }
                    # Env-var escape hatch: lets a wrapper request
                    # thinking-ON for *all* models without monkey-patching
                    # cfg.is_us_key_model (which also routes API keys).
                    keep_thinking_on = os.environ.get(
                        "SCISYNTH_KEEP_THINKING_ON", ""
                    ).lower() in ("1", "true", "yes")
                    if (not keep_thinking_on
                            and self.role in ("generation", "scoring")
                            and self.provider == "openrouter"
                            and not cfg.is_us_key_model(self.model_name)
                            and self.model_name not in NO_THINKING_OFF):
                        api_kwargs["extra_body"] = {
                            "reasoning": {"enabled": False}
                        }
                    _t0 = time.time()
                    response = self.client.chat.completions.create(**api_kwargs)
                    self.last_latency_ms = int((time.time() - _t0) * 1000)

                # Check for inline 500 error object
                if (hasattr(response, 'error') and isinstance(response.error, dict)
                        and response.error.get('code') == 500):
                    raise Exception(f"Internal Server Error: {response.error.get('message')}")

                # Capture usage telemetry on success (read by callers via .last_usage)
                usage = getattr(response, "usage", None)
                if usage:
                    pt = getattr(usage, "prompt_tokens", None)
                    ct = getattr(usage, "completion_tokens", None)
                    tt = getattr(usage, "total_tokens", None)
                    rt = None
                    ctd = getattr(usage, "completion_tokens_details", None)
                    if ctd:
                        rt = getattr(ctd, "reasoning_tokens", None)
                    self.last_usage = {
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "reasoning_tokens": rt,
                        "total_tokens": tt,
                    }

                if response and hasattr(response, 'choices') and response.choices:
                    msg = response.choices[0].message
                    # Return (content, reasoning) if the model exposes reasoning
                    if hasattr(msg, 'reasoning') and msg.reasoning:
                        return (msg.content, msg.reasoning)
                    return msg.content
                else:
                    raise ValueError("API response is empty or malformed")

            except Exception as e:
                if not _is_transient_error(e) or retry_count >= max_retries:
                    raise
                wait = _retry_wait_seconds(e, retry_count)
                logger.warning(
                    f"{self.model_name} retry {retry_count+1}/{max_retries} "
                    f"in {wait:.1f}s — {e}"
                )
                time.sleep(wait)
                retry_count += 1


# ---------------------------------------------------------------------------
# Specialised subclasses
# ---------------------------------------------------------------------------

class IdeaLLM(BaseLLM):
    """LLM wrapper for idea generation (Track A / Track B)."""

    def __init__(self, model_name: str, provider: Optional[str] = None):
        super().__init__(model_name, provider=provider, role="generation")

    def generate_idea(
        self,
        prompt: str,
        fallback_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate one idea; optionally retry with fallback on refusal.

        Returns dict with keys: idea, full_response, first_was_rejected,
        first_reject_response, used_fallback.
        """
        response = self.completion(prompt, system_prompt=system_prompt)

        if isinstance(response, tuple):
            idea, full_response = response[0] or "", response[1] or ""
            if len(idea) < 10:
                raise ValueError("Generated idea too short — model may have refused")
        else:
            idea = full_response = response or ""

        first_was_rejected   = _is_rejected(idea)
        first_reject_response = full_response if first_was_rejected else None

        if first_was_rejected and fallback_prompt:
            logger.info(f"{self.model_name} refused; trying fallback prompt")
            fb = self.completion(fallback_prompt, system_prompt=system_prompt)
            if isinstance(fb, tuple):
                idea, full_response = fb[0] or "", fb[1] or ""
            else:
                idea = full_response = fb or ""
            return {
                "idea": idea, "full_response": full_response,
                "first_was_rejected": True,
                "first_reject_response": first_reject_response,
                "used_fallback": True,
                "usage": self.last_usage,
            }

        return {
            "idea": idea, "full_response": full_response,
            "first_was_rejected": first_was_rejected,
            "first_reject_response": first_reject_response,
            "used_fallback": False,
            "usage": self.last_usage,
        }


class CriticLLM(BaseLLM):
    """LLM wrapper for 5-dimension absolute scoring."""

    def __init__(self, model_name: str, provider: Optional[str] = None):
        super().__init__(model_name, provider=provider, role="scoring")

    def score_idea(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Union[str, Tuple[str, str]]:
        """Call the critic; returns raw response (string or reasoning tuple)."""
        return self.completion(prompt, system_prompt=system_prompt)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _is_rejected(text: Optional[str]) -> bool:
    """Whether a generation should be retried with the fallback prompt.

    An empty or absent completion counts as a refusal. Some gateway routes
    answer 200 with content=None and completion_tokens=0 -- observed on
    azure/anthropic/claude-sonnet-4-5 for the lipid-nanoparticle subdomain,
    where the content filter appears to swallow the response. That is exactly
    the case the fallback prompt exists for, but the old signature raised
    AttributeError on None before the fallback could ever be tried.
    """
    if not text:
        return True
    return any(p.lower() in text.lower() for p in REJECTION_PHRASES)


def parse_scores(raw_text: str) -> Dict[str, Optional[int]]:
    """Extract the 5-dimension JSON scores from a critic response.

    Tries three strategies:
      1. ```json ... ``` block
      2. { ... } braces extraction
      3. key: value patterns
    Returns dict with keys: originality, feasibility, clarity, impact, specificity.
    """
    DIMS = SCORING_DIMS

    def _try_parse(s: str) -> Optional[Dict]:
        try:
            d = json.loads(s)
            if any(k in d for k in DIMS):
                return {dim: d.get(dim) for dim in DIMS}
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    # Strategy 1: ```json block
    m = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
    if not m:
        m = re.search(r'```\s*(.*?)\s*```', raw_text, re.DOTALL)
    if m:
        result = _try_parse(m.group(1))
        if result:
            return result

    # Strategy 2: first { ... } block
    m = re.search(r'\{([^{}]*)\}', raw_text, re.DOTALL)
    if m:
        result = _try_parse('{' + m.group(1) + '}')
        if result:
            return result

    # Strategy 3: line-by-line key: value
    scores: Dict[str, Optional[int]] = {dim: None for dim in DIMS}
    for dim in DIMS:
        m = re.search(rf'["\']?{dim}["\']?\s*:\s*(\d+)', raw_text, re.IGNORECASE)
        if m:
            scores[dim] = int(m.group(1))
    return scores


def create_llm(llm_type: str, model_name: str, provider: Optional[str] = None):
    """Factory: return IdeaLLM or CriticLLM."""
    if llm_type.lower() == "idea":
        return IdeaLLM(model_name, provider)
    elif llm_type.lower() == "critic":
        return CriticLLM(model_name, provider)
    raise ValueError(f"Unknown llm_type: {llm_type!r}  (must be 'idea' or 'critic')")
