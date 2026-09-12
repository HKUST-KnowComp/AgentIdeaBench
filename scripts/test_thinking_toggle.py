"""Wide thinking on/off toggle test across model families.

Tests every candidate in 3 modes: default, reasoning enabled (on), reasoning disabled (off).
Records latency, completion_tokens, reasoning_tokens, and whether JSON is parseable.
Saves full results to reports/thinking_toggle_test.json.
"""
import sys, time, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from openai import OpenAI

client = OpenAI(api_key=cfg.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

PROMPT = """Score this hypothesis on Originality, Feasibility, Clarity, Impact, Specificity (1-10 each).
Return JSON: {"originality":{"score":N,"reasoning":"..."}, ...}.

Hypothesis: We hypothesize that incorporating contrastive learning into GNN architectures will improve molecular property prediction by 15-20% on MoleculeNet."""

CANDIDATES = [
    # Qwen 3.5 (idea_models)
    "qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "qwen/qwen3.5-397b-a17b",
    # Qwen 3 instruct (sizes)
    "qwen/qwen3-8b", "qwen/qwen3-14b", "qwen/qwen3-32b",
    "qwen/qwen3-30b-a3b", "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3-235b-a22b", "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-max",
    # Qwen 3 thinking
    "qwen/qwen3-30b-a3b-thinking-2507",
    "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-max-thinking",
    "qwen/qwen3.6-plus",
    # Qwen 3 VL
    "qwen/qwen3-vl-8b-instruct", "qwen/qwen3-vl-32b-instruct",
    "qwen/qwen3-vl-30b-a3b-instruct", "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3-vl-8b-thinking", "qwen/qwen3-vl-30b-a3b-thinking",
    "qwen/qwen3-vl-235b-a22b-thinking",
    # DeepSeek
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v3.2", "deepseek/deepseek-v3.2-speciale",
    "deepseek/deepseek-v3.2-exp", "deepseek/deepseek-v3.1-terminus",
    "deepseek/deepseek-chat-v3.1", "deepseek/deepseek-r1",
    # MiniMax
    "minimax/minimax-m2.7", "minimax/minimax-m2.5", "minimax/minimax-m2.1",
    "minimax/minimax-m2", "minimax/minimax-m1",
    # Kimi
    "moonshotai/kimi-k2.6", "moonshotai/kimi-k2.5",
    "moonshotai/kimi-k2-thinking", "moonshotai/kimi-k2-0905",
    "moonshotai/kimi-k2",
    # GLM
    "z-ai/glm-5.1", "z-ai/glm-5-turbo", "z-ai/glm-5",
    "z-ai/glm-4.7", "z-ai/glm-4.7-flash",
    "z-ai/glm-4.6", "z-ai/glm-4.5",
    # MiMo
    "xiaomi/mimo-v2.5-pro", "xiaomi/mimo-v2.5",
    "xiaomi/mimo-v2-pro", "xiaomi/mimo-v2-flash",
    # Gemma
    "google/gemma-4-31b-it", "google/gemma-4-26b-a4b-it",
    "google/gemma-3-27b-it", "google/gemma-3-12b-it",
    # Grok
    "x-ai/grok-4", "x-ai/grok-4-fast",
    # Tencent
    "tencent/hunyuan-a13b-instruct",
]


def call(model, mode):
    extra = {}
    if mode == 'on':
        extra = {"reasoning": {"enabled": True, "max_tokens": 2000}}
    elif mode == 'off':
        extra = {"reasoning": {"enabled": False}}
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=1500, temperature=0,
            messages=[{"role": "user", "content": PROMPT}],
            extra_body=extra)
        dt = time.time() - t0
        out = (r.choices[0].message.content or "").strip()
        usage = r.usage
        comp = getattr(usage, 'completion_tokens', 0) if usage else 0
        reason = 0
        if usage:
            ctd = getattr(usage, 'completion_tokens_details', None)
            if ctd:
                reason = getattr(ctd, 'reasoning_tokens', 0) or 0
        json_ok = ('{' in out and '"score"' in out)
        return {"dt": round(dt, 1), "comp": comp, "reason": reason, "json": json_ok, "err": None}
    except Exception as e:
        return {"dt": round(time.time() - t0, 1), "err": str(e)[:80]}


def main():
    results = {}
    for i, m in enumerate(CANDIDATES, 1):
        print(f"  [{i}/{len(CANDIDATES)}] {m}", flush=True)
        results[m] = {mode: call(m, mode) for mode in ['default', 'on', 'off']}

    out = ROOT / "reports" / "thinking_toggle_test.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    # Table
    print()
    print(f"{'model':46s}  {'mode':>7s}  {'time':>6s}  {'comp':>5s}  {'reason':>6s}  json")
    print("-" * 92)
    for m in CANDIDATES:
        for mode in ['default', 'on', 'off']:
            r = results[m][mode]
            if r.get('err'):
                print(f"{m:46s}  {mode:>7s}  {r['dt']:>5.1f}s   FAIL: {r['err'][:50]}")
            else:
                j = "Y" if r['json'] else "N"
                print(f"{m:46s}  {mode:>7s}  {r['dt']:>5.1f}s  {r['comp']:>5}  {r['reason']:>6}  {j}")
        print()


if __name__ == "__main__":
    main()
