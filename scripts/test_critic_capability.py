"""Critic capability + latency benchmark.

For each candidate critic model, run N=3 scoring calls on a fixed hypothesis +
context (with real benchmark prompt). Record:
  - latency p50 / p95 / max (in seconds)
  - JSON parse success rate (3 calls -> 0..3)
  - mean reasoning_tokens (signals thinking effort)
  - mean completion_tokens
  - any 4xx / 5xx errors

Saves: reports/critic_capability_test.json
"""
import sys, time, json, statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from openai import OpenAI

client = OpenAI(api_key=cfg.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

# Realistic critic prompt with all 5 dimensions and Coherence/Factual/Boilerplate caps
SYSTEM = """You are a senior reviewer for Nature and Science with 20+ years of experience evaluating scientific ideas.

Score on 5 dimensions (1-10 each): Originality, Feasibility, Clarity, Impact, Specificity.

Output ONLY a JSON object:
{"originality":{"score":<int>,"reasoning":"<short>"},
 "feasibility":{"score":<int>,"reasoning":"<short>"},
 "clarity":{"score":<int>,"reasoning":"<short>"},
 "impact":{"score":<int>,"reasoning":"<short>"},
 "specificity":{"score":<int>,"reasoning":"<short>"}}
"""

USER = """Domain: CS

Background literature (top survey refs):
1. Vaswani et al, "Attention is All You Need" (Transformer)
2. Devlin et al, "BERT: Pre-training of Deep Bidirectional Transformers"
3. Brown et al, "Language Models are Few-Shot Learners" (GPT-3)

Hypothesis to score:
We hypothesize that incorporating contrastive learning objectives into existing graph neural network architectures will improve molecular property prediction by 15-20% on the MoleculeNet benchmark, particularly for tasks with limited labeled data, by enforcing consistency between augmented views of molecular graphs."""

CANDIDATES = [
    # current pool
    "deepseek/deepseek-v4-flash",
    "moonshotai/kimi-k2-0905",
    "qwen/qwen3-max",
    "x-ai/grok-4",
    "tencent/hunyuan-a13b-instruct",   # to replace
    # Top tier-9 / tier-7 candidates
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3-max-thinking",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "moonshotai/kimi-k2-thinking",
    # Top tier-5 / tier-4
    "qwen/qwen3.6-max-preview",
    "qwen/qwen3.5-397b-a17b",
    "upstage/solar-pro-3",
    "deepseek/deepseek-v3.2-speciale",
    "qwen/qwen3.6-plus",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
    # Other strong recent
    "x-ai/grok-4.3",
    "mistralai/mistral-medium-3-5",
    "qwen/qwen3.5-plus-20260420",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "deepseek/deepseek-r1",
    "minimax/minimax-m2.7",
    "xiaomi/mimo-v2.5-pro",
    "xiaomi/mimo-v2.5",
]


def call_once(model):
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            max_tokens=2000,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
        )
        dt = time.time() - t0
        out = (r.choices[0].message.content or "").strip()
        u = r.usage
        comp = getattr(u, 'completion_tokens', 0) if u else 0
        prompt = getattr(u, 'prompt_tokens', 0) if u else 0
        reason = 0
        if u:
            ctd = getattr(u, 'completion_tokens_details', None)
            if ctd:
                reason = getattr(ctd, 'reasoning_tokens', 0) or 0
        # JSON parse?
        json_ok = False
        try:
            # find first '{' to last matching '}'
            start = out.find('{')
            if start >= 0:
                # naive: try parsing greedy
                json.loads(out[start:])
                json_ok = True
        except Exception:
            pass
        if not json_ok:
            # fallback: regex extract
            import re
            m = re.search(r'\{[^{}]*"score"[^{}]*\}', out, re.DOTALL)
            json_ok = bool(m)
        return {
            "latency": round(dt, 2),
            "prompt_tokens": prompt,
            "completion_tokens": comp,
            "reasoning_tokens": reason,
            "json_ok": json_ok,
            "error": None,
        }
    except Exception as e:
        return {"latency": round(time.time()-t0, 2), "error": str(e)[:100]}


def bench(model, n=3):
    results = []
    for i in range(n):
        results.append(call_once(model))
    lats = [r["latency"] for r in results if r.get("error") is None]
    json_oks = sum(1 for r in results if r.get("json_ok"))
    errors = sum(1 for r in results if r.get("error"))
    if lats:
        p50 = statistics.median(lats)
        p_max = max(lats)
        comp_mean = statistics.mean(r["completion_tokens"] for r in results if r.get("error") is None)
        reason_mean = statistics.mean(r["reasoning_tokens"] for r in results if r.get("error") is None)
    else:
        p50 = p_max = comp_mean = reason_mean = None
    return {
        "model": model,
        "n_calls": n,
        "json_ok_rate": json_oks / n,
        "n_errors": errors,
        "latency_p50": p50,
        "latency_max": p_max,
        "comp_mean": comp_mean,
        "reason_mean": reason_mean,
        "raw": results,
    }


def main():
    out_path = ROOT / "reports" / "critic_capability_test.json"
    results = {}
    for i, m in enumerate(CANDIDATES, 1):
        print(f"  [{i}/{len(CANDIDATES)}] {m}", flush=True)
        results[m] = bench(m, n=3)
    out_path.write_text(json.dumps(results, indent=2))

    # Print sorted by latency_p50 ascending (fast first), filtering by json_ok
    print("\n## Critic capability + speed (sorted by p50 latency)")
    print()
    print(f"{'model':46s}  {'p50':>7s}  {'max':>7s}  {'JSON%':>5s}  {'err':>3s}  {'comp':>5s}  {'reason':>6s}")
    print("-" * 92)
    rows = sorted(results.values(), key=lambda r: (r["latency_p50"] or 999, -r["json_ok_rate"]))
    for r in rows:
        if r["latency_p50"] is None:
            print(f"{r['model']:46s}  {'-':>7s}  {'-':>7s}  {'0%':>5s}  {r['n_errors']:>3d}  {'-':>5s}  {'-':>6s}")
        else:
            print(f"{r['model']:46s}  {r['latency_p50']:>5.1f}s  {r['latency_max']:>5.1f}s  "
                  f"{int(r['json_ok_rate']*100):>4d}%  {r['n_errors']:>3d}  "
                  f"{int(r['comp_mean']):>5d}  {int(r['reason_mean']):>6d}")
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
