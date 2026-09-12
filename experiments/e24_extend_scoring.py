"""E24 — extend lit8d 3-seed scoring beyond the pilot 20 subdomains.

Reuses ALL of E21's evidence + scoring machinery (same lit8d critic, same
lit8d_scores_3seed table, same 3-ideas/seed aggregation) but on a COMPLEMENTARY
set of subdomains (indices 2/7/12/17 per domain vs the pilot's 0/5/10/15), adding
20 more subdomains → 40 total. New rows only; pilot rows untouched. Idempotent.

Implementation: monkeypatch e21.pick_subdomains so E21's sample_items/prep/score
pick the extended subdomains, then delegate to E21's staged functions.

  /usr/bin/python3 experiments/e24_extend_scoring.py --prep   # SS evidence (slow)
  /usr/bin/python3 experiments/e24_extend_scoring.py --score
  /usr/bin/python3 experiments/e24_extend_scoring.py --audit
"""
import argparse, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
import experiments.e21_pilot20_3seed as e21

DOMAINS = ["Biology", "CS", "Chemistry", "Medicine", "Physics"]
SUB_IDX_EXT = (2, 7, 12, 17)   # complementary to pilot (0,5,10,15)


def pick_ext(conn):
    picks = []
    for dom in DOMAINS:
        subs = [r[0] for r in conn.execute(
            "SELECT DISTINCT subdomain FROM subdomain_ideas WHERE domain=? ORDER BY subdomain", (dom,))]
        picks += [(dom, subs[i]) for i in SUB_IDX_EXT if i < len(subs)]
    return picks


# monkeypatch: make E21's sample_items use the extended subdomains
e21.pick_subdomains = pick_ext


def main():
    ap = argparse.ArgumentParser()
    for f in ("prep", "score", "audit"): ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--critic-idx", type=int, default=None,
                    help="shard scoring by a single critic (0/1/2) to avoid redundant parallel work")
    a = ap.parse_args()
    if a.prep: e21.prep(min(a.workers, 4))
    if a.score: e21.score(workers=a.workers, only_critic=a.critic_idx)
    if a.audit: e21.audit()


if __name__ == "__main__":
    main()
