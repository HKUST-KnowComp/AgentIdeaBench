"""E29 — add Chinese translations to the human-eval pairs (annotators read 中文).

Reads the pair master (default reports/e29_human_eval_pairs_nlp.json), translates
each unique idea paragraph and each subfield name to Chinese with an open-weight
LLM (default OPENROUTER_API_KEY), and writes `text_zh` (per idea) + `subfield_zh`
(per pair) back into the master in place. Translations are cached by exact text
in reports/e29_zh_cache.json, so re-runs are near-free and idempotent.

Blindness note: translations are added ONLY to left/right text and subfield — no
model names or scores — so the annotation frontend stays blind.

Usage:
  /usr/bin/python3 experiments/e29_translate_ideas.py [--master reports/e29_human_eval_pairs_nlp.json]
                                                      [--model qwen/qwen3.6-plus] [--workers 8]
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: F401  (loads .env / key routing)

CACHE = ROOT / "reports" / "e29_zh_cache.json"
SYS = "你是专业的学术翻译，精通机器学习与自然语言处理领域术语。"
TPL = ("把下面这段英文科研假设完整翻译成流畅、准确的学术中文。"
       "保留专有名词/模型名/指标名（可中英并存），只输出译文，不要加任何解释或前后缀：\n\n{t}")
SUB_TPL = ("把下面这个英文研究子领域名翻译成简洁的中文（10 字以内，可保留关键英文缩写），只输出译文：\n\n{t}")


def _load_cache():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def translate_one(model, text, is_sub=False):
    from utils.LLM import IdeaLLM
    llm = IdeaLLM(model_name=model)
    prompt = (SUB_TPL if is_sub else TPL).format(t=text)
    out = llm.generate_idea(prompt, system_prompt=SYS)
    zh = (out.get("idea") or "").strip()
    # strip accidental wrapping quotes / label prefixes
    for pre in ("译文：", "翻译：", "中文："):
        if zh.startswith(pre):
            zh = zh[len(pre):].strip()
    return zh.strip('"“”')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(ROOT / "reports" / "e29_human_eval_pairs_nlp.json"))
    ap.add_argument("--model", default="qwen/qwen3.6-plus")   # open-weight, strong Chinese
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    master_path = Path(a.master)
    master = json.loads(master_path.read_text())
    pairs = master["pairs"]

    cache = _load_cache()
    # collect unique texts (ideas) + unique subfields
    texts, subs = set(), set()
    for p in pairs:
        texts.add(p["left"]["text"]); texts.add(p["right"]["text"]); subs.add(p["subfield"])
    todo_txt = [t for t in texts if t not in cache]
    todo_sub = [("SUB::" + s) for s in subs if ("SUB::" + s) not in cache]
    print(f"{len(pairs)} pairs; unique ideas {len(texts)} ({len(todo_txt)} to translate), "
          f"subfields {len(subs)} ({len(todo_sub)} to translate)", flush=True)

    def work(item):
        is_sub = item.startswith("SUB::")
        src = item[5:] if is_sub else item
        try:
            return item, translate_one(a.model, src, is_sub=is_sub), None
        except Exception as e:
            return item, None, str(e)[:160]

    jobs = todo_txt + todo_sub
    done = err = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(work, it) for it in jobs]):
            item, zh, e = f.result()
            if zh:
                cache[item] = zh; done += 1
            else:
                err += 1
                if err <= 10:
                    print("  fail:", e, flush=True)
            if (done + err) % 20 == 0 or (done + err) == len(jobs):
                CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
                print(f"  [{done + err}/{len(jobs)}] ok={done} err={err}", flush=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))

    # write translations back into the master
    n_idea = n_sub = 0
    for p in pairs:
        for side in ("left", "right"):
            zh = cache.get(p[side]["text"])
            if zh:
                p[side]["text_zh"] = zh; n_idea += 1
        szh = cache.get("SUB::" + p["subfield"])
        if szh:
            p["subfield_zh"] = szh; n_sub += 1
    master.setdefault("meta", {})["chinese_translation"] = dict(
        model=a.model, cache=str(CACHE.name), n_idea_sides=n_idea, n_subfield=n_sub)
    master_path.write_text(json.dumps(master, ensure_ascii=False, indent=2))
    print(f"wrote {n_idea} idea + {n_sub} subfield translations into {master_path}")


if __name__ == "__main__":
    main()
