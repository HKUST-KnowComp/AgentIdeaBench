"""Speed benchmark v2 — measure current critic pool & uncovered idea models
across three thinking modes (default / on / off).

Distinguishes between:
  - default: no `reasoning` parameter (OR decides)
  - on:      reasoning.enabled = True
  - off:     reasoning.enabled = False

For each (model, mode) pair, runs N=3 calls and records latency + json_ok +
completion/reasoning tokens.

Saves: reports/speed_test_v2.json
"""
import sys, time, json, statistics, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from openai import OpenAI

OR_KEY = os.environ.get("OPENROUTER_API_KEY") or cfg.OPENROUTER_API_KEY
US_KEY = os.environ.get("OPENROUTER_US_API_KEY") or getattr(cfg, "OPENROUTER_US_API_KEY", None)

client_or = OpenAI(api_key=OR_KEY, base_url="https://openrouter.ai/api/v1")
client_us = OpenAI(api_key=US_KEY, base_url="https://openrouter.ai/api/v1") if US_KEY else client_or

CRITIC_SYSTEM = """You are a senior reviewer for Nature and Science with 20+ years of experience evaluating scientific ideas.

Score on 5 dimensions (1-10 each): Originality, Feasibility, Clarity, Impact, Specificity.

Output ONLY a JSON object:
{"originality":{"score":<int>,"reasoning":"<short>"},
 "feasibility":{"score":<int>,"reasoning":"<short>"},
 "clarity":{"score":<int>,"reasoning":"<short>"},
 "impact":{"score":<int>,"reasoning":"<short>"},
 "specificity":{"score":<int>,"reasoning":"<short>"}}
"""

CRITIC_USER = """Domain: CS

Background literature (top survey refs):
1. Vaswani et al, "Attention is All You Need" (Transformer)
2. Devlin et al, "BERT: Pre-training of Deep Bidirectional Transformers"
3. Brown et al, "Language Models are Few-Shot Learners" (GPT-3)

Hypothesis to score:
We hypothesize that incorporating contrastive learning objectives into existing graph neural network architectures will improve molecular property prediction by 15-20% on the MoleculeNet benchmark, particularly for tasks with limited labeled data, by enforcing consistency between augmented views of molecular graphs."""

IDEA_SYSTEM = """You are a scientist. Propose a single novel research hypothesis in paragraph form (80-150 words). Do not duplicate ideas in the given references. The hypothesis should be a new direction, not a derivative."""

IDEA_USER = """Domain: Computer Science

Reference papers (background, do not duplicate):
1. Vaswani et al, "Attention is All You Need" (Transformer)
2. Devlin et al, "BERT: Pre-training of Deep Bidirectional Transformers"
3. Brown et al, "Language Models are Few-Shot Learners" (GPT-3)

Propose your novel hypothesis now."""


# Current critic pool (config.json)
CRITICS = [
    "qwen/qwen3.6-plus",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
    "minimax/minimax-m2.7",
    "deepseek/deepseek-v4-flash",
]

# Idea models in config.json that have NEVER been speed-tested
UNCOVERED_IDEAS = [
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4",
]


def call_once(model, system, user, mode, max_tokens, use_us_key=False):
    """mode in {'default','on','off'}"""
    cli = client_us if use_us_key else client_or
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if mode == "on":
        kwargs["extra_body"] = {"reasoning": {"enabled": True}}
    elif mode == "off":
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}

    t0 = time.time()
    try:
        r = cli.chat.completions.create(**kwargs)
        dt = time.time() - t0
        out = (r.choices[0].message.content or "").strip()
        u = r.usage
        comp = getattr(u, "completion_tokens", 0) if u else 0
        reason = 0
        if u:
            ctd = getattr(u, "completion_tokens_details", None)
            if ctd:
                reason = getattr(ctd, "reasoning_tokens", 0) or 0
        return {
            "latency": round(dt, 2),
            "comp_tokens": comp,
            "reason_tokens": reason,
            "out_len": len(out),
            "error": None,
        }
    except Exception as e:
        return {"latency": round(time.time() - t0, 2), "error": str(e)[:120]}


def bench(model, role, mode, n=3, use_us_key=False):
    if role == "critic":
        sys_p, usr_p, max_tok = CRITIC_SYSTEM, CRITIC_USER, 2000
        ok_check = lambda r: r.get("error") is None and r.get("out_len", 0) >= 50
    else:
        sys_p, usr_p, max_tok = IDEA_SYSTEM, IDEA_USER, 4000
        ok_check = lambda r: r.get("error") is None and r.get("out_len", 0) >= 200

    results = [call_once(model, sys_p, usr_p, mode, max_tok, use_us_key) for _ in range(n)]
    succ = [r for r in results if r.get("error") is None]
    lats = [r["latency"] for r in succ]
    ok_rate = sum(1 for r in results if ok_check(r)) / n
    if lats:
        return {
            "p50": round(statistics.median(lats), 2),
            "max": round(max(lats), 2),
            "ok": ok_rate,
            "comp_mean": round(statistics.mean(r["comp_tokens"] for r in succ), 0),
            "reason_mean": round(statistics.mean(r["reason_tokens"] for r in succ), 0),
            "errors": [r["error"] for r in results if r.get("error")],
        }
    return {"p50": None, "max": None, "ok": 0.0, "errors": [r.get("error") for r in results]}


def main():
    out_path = ROOT / "reports" / "speed_test_v2.json"
    results = {"critics": {}, "ideas": {}}

    # ── critics: 3 modes each ───────────────────────────────────────────────
    print("\n=== Phase A: critics × 3 modes ===\n", flush=True)
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for m in CRITICS:
            results["critics"][m] = {}
            for mode in ["default", "on", "off"]:
                tasks.append(pool.submit(bench, m, "critic", mode, 3))
        i = 0
        for m in CRITICS:
            for mode in ["default", "on", "off"]:
                r = tasks[i].result()
                results["critics"][m][mode] = r
                print(f"  critic {m:35s} mode={mode:7s} p50={str(r['p50']):>6} ok={r['ok']:.0%}", flush=True)
                i += 1

    # ── uncovered ideas: default + on (most US-key models don't use OR thinking flag) ───
    print("\n=== Phase B: uncovered idea models × 2 modes (default, on) ===\n", flush=True)
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for m in UNCOVERED_IDEAS:
            results["ideas"][m] = {}
            use_us = cfg.is_us_key_model(m) if hasattr(cfg, "is_us_key_model") else False
            for mode in ["default", "on"]:
                tasks.append((m, mode, pool.submit(bench, m, "idea", mode, 3, use_us)))
        for m, mode, fut in tasks:
            r = fut.result()
            results["ideas"][m][mode] = r
            print(f"  idea   {m:35s} mode={mode:7s} p50={str(r['p50']):>6} ok={r['ok']:.0%}", flush=True)

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ saved: {out_path}\n")

    # ── print sorted summary ────────────────────────────────────────────────
    print("## Critic pool — per-mode summary")
    print(f"{'model':<35} {'def_p50':>8} {'on_p50':>8} {'off_p50':>8} {'def_ok':>7} {'on_ok':>6} {'off_ok':>7}")
    for m, modes in results["critics"].items():
        d = modes.get("default", {})
        o = modes.get("on", {})
        f = modes.get("off", {})
        print(f"{m:<35} {str(d.get('p50')):>8} {str(o.get('p50')):>8} {str(f.get('p50')):>8} "
              f"{d.get('ok',0):>7.0%} {o.get('ok',0):>6.0%} {f.get('ok',0):>7.0%}")

    print("\n## Uncovered idea models — per-mode summary")
    print(f"{'model':<35} {'def_p50':>8} {'on_p50':>8} {'def_ok':>7} {'on_ok':>6}")
    for m, modes in results["ideas"].items():
        d = modes.get("default", {})
        o = modes.get("on", {})
        print(f"{m:<35} {str(d.get('p50')):>8} {str(o.get('p50')):>8} "
              f"{d.get('ok',0):>7.0%} {o.get('ok',0):>6.0%}")


if __name__ == "__main__":
    main()
