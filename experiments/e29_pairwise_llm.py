"""E29 — Pairwise LLM-judge comparison over the 100 CS/AI human-eval pairs.

Runs the paper's pairwise critic (evaluation.pairwise_scorer.resolve_pair, BOTH
orders -> position-debiased) with 3 open-weight judges (the lit8d ensemble) on the
exact 100 pairs humans will judge. Gives, per pair:
  - each judge's position-debiased overall winner (LEFT/RIGHT/split)
  - a majority LLM-judge verdict
so we can later report human-vs-LLM-judge agreement and LLM-vs-LLM agreement, plus a
position-flip (split) rate as a robustness stat.

Idempotent + resumable: new table `e29_pairwise_llm`, PRIMARY KEY (pair_id, judge_model),
INSERT OR IGNORE. Open-weight judges route to the default OPENROUTER_API_KEY (rule 9).
Does NOT touch any existing table. /usr/bin/python3.

Usage:
  /usr/bin/python3 experiments/e29_pairwise_llm.py --run --limit 2      # smoke (2 pairs)
  /usr/bin/python3 experiments/e29_pairwise_llm.py --run --workers 4     # full 100
  /usr/bin/python3 experiments/e29_pairwise_llm.py --status
"""
import argparse, json, sqlite3, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg
from evaluation.pairwise_scorer import resolve_pair

OUT = ROOT / "reports"
MASTER = OUT / "e29_human_eval_pairs.json"
JUDGES = ["qwen/qwen3.6-plus", "moonshotai/kimi-k2.6", "z-ai/glm-5.1"]  # open-weight lit8d ensemble
DB = cfg.RESULTS_DB


def ensure_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS e29_pairwise_llm (
        pair_id TEXT, judge_model TEXT, subfield TEXT,
        overall TEXT, originality TEXT, feasibility TEXT, specificity TEXT,
        critic_pref TEXT, raw_forward TEXT, raw_swap TEXT, telemetry TEXT, created_at TEXT,
        PRIMARY KEY (pair_id, judge_model))""")
    conn.commit()


def load_pairs(path=MASTER):
    return json.loads(Path(path).read_text())["pairs"]


def todo(conn, pairs, judges):
    done = {(r[0], r[1]) for r in conn.execute("SELECT pair_id, judge_model FROM e29_pairwise_llm")}
    tasks = []
    for p in pairs:
        for j in judges:
            if (p["pair_id"], j) not in done:
                tasks.append((p, j))
    return tasks


def run_one(task):
    p, judge = task
    # idea_a = LEFT, idea_b = RIGHT (same presentation the human sees); refs blank (blind, like human)
    res = resolve_pair(p["left"]["text"], p["right"]["text"], judge,
                       domain=p["subfield"], references="")
    # resolve_pair overall: 'A'->LEFT, 'B'->RIGHT, 'split'->position-sensitive
    mp = {"A": "LEFT", "B": "RIGHT", "split": "split", None: None}
    return dict(
        pair_id=p["pair_id"], judge_model=judge, subfield=p["subfield"],
        overall=mp.get(res.get("overall")), originality=mp.get(res.get("originality")),
        feasibility=mp.get(res.get("feasibility")), specificity=mp.get(res.get("specificity")),
        critic_pref=p["critic_pref"],
        raw_forward=res.get("raw_forward", "")[:2000], raw_swap=res.get("raw_swap", "")[:2000],
        telemetry=json.dumps(res.get("telemetry", {}))[:2000])


def save(conn, r):
    conn.execute("""INSERT OR IGNORE INTO e29_pairwise_llm VALUES
        (:pair_id,:judge_model,:subfield,:overall,:originality,:feasibility,:specificity,
         :critic_pref,:raw_forward,:raw_swap,:telemetry,:created_at)""",
                 {**r, "created_at": datetime.now(timezone.utc).isoformat()})
    conn.commit()


def status(conn, pairs, judges):
    n = conn.execute("SELECT COUNT(*) FROM e29_pairwise_llm").fetchone()[0]
    tot = len(pairs) * len(judges)
    print(f"e29_pairwise_llm: {n}/{tot} (pair x judge) done")
    # quick LLM-vs-critic + LLM-vs-LLM if any rows
    rows = conn.execute("SELECT pair_id,judge_model,overall,critic_pref FROM e29_pairwise_llm WHERE overall IS NOT NULL").fetchall()
    if not rows:
        return
    from collections import defaultdict, Counter
    byp = defaultdict(dict); split = 0; agree_crit = 0; nc = 0
    for pid, j, ov, cp in rows:
        byp[pid][j] = ov
        if ov == "split":
            split += 1
        elif ov in ("LEFT", "RIGHT"):
            nc += 1; agree_crit += (ov == cp)
    print(f"  judge verdicts: {len(rows)} | position-split rate: {split/len(rows):.1%}")
    if nc:
        print(f"  LLM-judge vs absolute-critic agreement (non-split): {agree_crit}/{nc} = {agree_crit/nc:.1%}")
    # LLM-LLM majority coverage
    full = [p for p in byp.values() if len(p) == len(judges)]
    if full:
        unan = sum(1 for p in full if len(set(v for v in p.values() if v in ("LEFT", "RIGHT"))) == 1
                   and all(v in ("LEFT", "RIGHT") for v in p.values()))
        print(f"  pairs with all {len(judges)} judges: {len(full)} | unanimous (no split): {unan} ({unan/len(full):.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N pairs")
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--master", default=str(MASTER),
                    help="pair master JSON (use reports/e29_human_eval_pairs_nlp.json for the NLP set)")
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    master_path = Path(args.master)
    conn = sqlite3.connect(str(DB), timeout=60)
    ensure_table(conn)
    pairs = load_pairs(master_path)
    if args.limit:
        pairs = pairs[:args.limit]

    if args.status:
        status(conn, load_pairs(master_path), judges); return
    if not args.run:
        print("use --run (optionally --limit N for smoke) or --status"); return

    tasks = todo(conn, pairs, judges)
    print(f"{len(tasks)} (pair x judge) tasks; judges={judges}; workers={args.workers}")
    if not tasks:
        print("all done."); status(conn, load_pairs(master_path), judges); return

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result(); save(conn, r); done += 1
                print(f"[{done}/{len(tasks)}] {r['pair_id']} {r['judge_model'].split('/')[-1]:<16} "
                      f"overall={r['overall']} (critic={r['critic_pref']})", flush=True)
            except Exception as e:
                print(f"  ERR {t[0]['pair_id']} {t[1]}: {e}", flush=True)
    status(conn, load_pairs(master_path), judges)


if __name__ == "__main__":
    main()
