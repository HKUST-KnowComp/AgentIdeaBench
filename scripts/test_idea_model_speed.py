"""Idea-model speed benchmark for the 25 candidate models.

For each candidate, run the active-style hypothesis-generation prompt 3 times
in each mode (default / reasoning ON / reasoning OFF — OFF skipped if model
forces thinking). Record:
  - latency p50 / max
  - completion_tokens, reasoning_tokens (mean of 3)
  - JSON OK rate (idea models output paragraph not JSON, so we just check
    non-empty content of >= 50 chars)
  - errors

Output: reports/idea_model_speed_test.json + sorted markdown table to stdout.
"""
import sys, time, json, statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from openai import OpenAI

client = OpenAI(api_key=cfg.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

PROMPT_SYSTEM = """You are a creative and rigorous scientist. Propose a single novel, testable scientific hypothesis in the given research domain. Your output must be exactly one paragraph, 80-150 words, in first-person future tense, starting with 'Hypothesis:' or 'We hypothesize'. Be specific — name mechanisms, methods, datasets."""

PROMPT_USER = "Domain: CS. Propose a novel research hypothesis."

# 25 candidate idea models from selected_models_report
A_CLASS = [   # 17 perfect-toggle
    "qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "qwen/qwen3.5-397b-a17b",
    "xiaomi/mimo-v2.5", "xiaomi/mimo-v2.5-pro",
    "google/gemma-4-26b-a4b-it", "google/gemma-4-31b-it",
    "z-ai/glm-4.7", "z-ai/glm-4.7-flash",
    "z-ai/glm-5", "z-ai/glm-5-turbo",
    "moonshotai/kimi-k2.5", "moonshotai/kimi-k2.6",
    "xiaomi/mimo-v2-flash", "xiaomi/mimo-v2-pro",
    "z-ai/glm-5.1",
    "deepseek/deepseek-v4-flash",
]
B_CLASS = [   # 8 thinking-only
    "qwen/qwen3-30b-a3b-thinking-2507",
    "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "moonshotai/kimi-k2-thinking",
    "minimax/minimax-m2.7",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-v3.2-speciale",
    "x-ai/grok-4",
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
            model=model, max_tokens=1500, temperature=0.7,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": PROMPT_USER},
            ],
            extra_body=extra,
        )
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
            "ok": len(out) >= 50,
            "error": None,
        }
    except Exception as e:
        return {"latency": round(time.time()-t0, 2), "error": str(e)[:100]}


def bench(model, modes, n=3):
    results = {}
    for mode in modes:
        runs = [call(model, mode) for _ in range(n)]
        oks = [r for r in runs if r.get("error") is None]
        if oks:
            lats = [r["latency"] for r in oks]
            comp = statistics.mean(r["comp_tokens"] for r in oks)
            reason = statistics.mean(r["reason_tokens"] for r in oks)
            ok_rate = sum(1 for r in runs if r.get("ok")) / n
            results[mode] = {
                "latency_p50": statistics.median(lats),
                "latency_max": max(lats),
                "comp_tokens_mean": round(comp, 0),
                "reason_tokens_mean": round(reason, 0),
                "ok_rate": ok_rate,
                "n_errors": sum(1 for r in runs if r.get("error")),
            }
        else:
            err = runs[0].get("error", "?")
            results[mode] = {"error": err, "n_errors": n}
    return results


def main():
    out_path = ROOT / "reports" / "idea_model_speed_test.json"
    results = {}
    candidates = [(m, "A") for m in A_CLASS] + [(m, "B") for m in B_CLASS]
    for i, (m, cls) in enumerate(candidates, 1):
        print(f"  [{i}/{len(candidates)}] {cls}: {m}", flush=True)
        modes = ["default", "on", "off"] if cls == "A" else ["default", "on"]
        results[m] = {"class": cls, "modes": bench(m, modes, n=3)}

    out_path.write_text(json.dumps(results, indent=2))

    # Print summary table
    print("\n## Summary (sorted by ON p50 latency, ascending)\n")
    print(f"{'cls':>3s}  {'model':46s}  {'def_p50':>8s}  {'on_p50':>7s}  {'off_p50':>8s}  {'on_max':>7s}  {'on_reason':>9s}  ok%")
    print("-" * 110)
    rows = []
    for m, info in results.items():
        modes = info['modes']
        on = modes.get('on', {})
        rows.append((on.get('latency_p50') or 999, m, info, modes))
    for _, m, info, modes in sorted(rows):
        cls = info['class']
        d = modes.get('default', {})
        o = modes.get('on', {})
        f = modes.get('off', {})
        d_p50 = f"{d.get('latency_p50','-'):.1f}s" if isinstance(d.get('latency_p50'), (int,float)) else str(d.get('latency_p50','-'))
        o_p50 = f"{o.get('latency_p50','-'):.1f}s" if isinstance(o.get('latency_p50'), (int,float)) else str(o.get('latency_p50','-'))
        f_p50 = "FAIL" if f.get('error') else (f"{f.get('latency_p50','-'):.1f}s" if isinstance(f.get('latency_p50'), (int,float)) else "-")
        o_max = f"{o.get('latency_max','-'):.1f}s" if isinstance(o.get('latency_max'), (int,float)) else "-"
        o_reason = str(int(o['reason_tokens_mean'])) if o.get('reason_tokens_mean') is not None else "-"
        o_ok = f"{int(o.get('ok_rate', 0)*100)}%" if o.get('ok_rate') is not None else "-"
        print(f"{cls:>3s}  {m:46s}  {d_p50:>8s}  {o_p50:>7s}  {f_p50:>8s}  {o_max:>7s}  {o_reason:>9s}  {o_ok}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
