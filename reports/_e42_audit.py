"""Read-only integrity audit of the E42 closed-source extension's generated cells.

Checks every (model, track) cell already in subdomain_ideas for the E42 roster and
reports, per model: coverage, protocol failures, refusals, format violations, and
Track-C retrieval health. Prints a summary table plus a per-issue sample so every
number can be traced back to a specific cell.

Nothing is written to the DB. Use --json to dump the full per-cell issue list.
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg                                          # noqa: E402
from experiments.e42_cutoff_safe_closed import ROSTER         # noqa: E402

MIN_WORDS, LO, HI = 30, 80, 150      # <30 = protocol failure; 80-150 = the spec band

# Static (Track B) hypotheses carry a trailing "Cited: [1] <title>; ..." footer that the
# prompt asks for. It is not part of the 80-150-word hypothesis, and its last character
# is whatever the last cited title ends with, so it must be stripped before the length
# and end-punctuation checks. Measuring with it in place reports thousands of false
# "no_end_punct" and inflated word counts.
CITED_FOOTER = re.compile(r"\s*Cited\s*:.*\Z", re.S)   # footer may or may not start a new line


def body(text):
    return CITED_FOOTER.sub("", text or "").strip()
REFUSAL = ("i cannot", "i can't", "i'm sorry", "i am sorry", "as an ai",
           "unable to assist", "cannot assist", "i won't", "i will not")
PROTOCOL_LEAK = ("FINAL:", "SEARCH:", "FETCH:", "SIMULATE:")


def audit():
    e42 = [m for m, _, _, _ in ROSTER]
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    conn.row_factory = sqlite3.Row
    rows = [r for r in conn.execute(
        "SELECT idea_model, domain, subdomain, track, idea_index, idea_text, telemetry "
        "FROM subdomain_ideas WHERE idea_model IN (%s)" % ",".join("?" * len(e42)), e42)]
    conn.close()

    issues = defaultdict(list)                 # (model, track) -> [(kind, sub, idx, note)]
    stats = defaultdict(Counter)               # (model, track) -> Counter
    seen_text = defaultdict(lambda: defaultdict(list))   # (m,tr) -> text -> [cells]

    for r in rows:
        m, tr, sub, idx = r["idea_model"], r["track"], r["subdomain"], r["idea_index"]
        k = (m, tr)
        raw = (r["idea_text"] or "").strip()
        t = body(raw)
        stats[k]["cells"] += 1
        if raw != t:
            stats[k]["has_cited_footer"] += 1
        low = t.lower()
        nw = len(t.split())

        if not t:
            issues[k].append(("empty", sub, idx, "")); stats[k]["empty"] += 1
        elif nw < MIN_WORDS:
            issues[k].append(("too_short", sub, idx, f"{nw}w")); stats[k]["too_short"] += 1
        elif any(p in low[:200] for p in REFUSAL):
            issues[k].append(("refusal", sub, idx, t[:70])); stats[k]["refusal"] += 1
        else:
            if not (LO <= nw <= HI):
                issues[k].append(("len_out_of_band", sub, idx, f"{nw}w"))
                stats[k]["len_out_of_band"] += 1
            if any(p in t for p in PROTOCOL_LEAK):
                issues[k].append(("protocol_leak", sub, idx, t[:70]))
                stats[k]["protocol_leak"] += 1
            if t[-1] not in ".!?\"')" :
                issues[k].append(("no_end_punct", sub, idx, t[-40:]))
                stats[k]["no_end_punct"] += 1
            seen_text[k][t].append((sub, idx))

        if tr == "C":
            tel = {}
            if r["telemetry"]:
                try:
                    tel = json.loads(r["telemetry"])
                except ValueError:
                    issues[k].append(("bad_telemetry_json", sub, idx, ""))
                    stats[k]["bad_telemetry_json"] += 1
            else:
                stats[k]["no_telemetry"] += 1
                issues[k].append(("no_telemetry", sub, idx, ""))
            if tel.get("error"):
                issues[k].append(("telemetry_error", sub, idx, str(tel["error"])[:70]))
                stats[k]["telemetry_error"] += 1
            n = tel.get("n_tool_calls")
            if n == 0:
                issues[k].append(("zero_tool_calls", sub, idx, ""))
                stats[k]["zero_tool_calls"] += 1
            if n is not None:
                stats[k]["tool_calls_sum"] += n
            trace = tel.get("trace") or []
            ne = sum(1 for s in trace
                     if isinstance(s, dict) and len(str(s.get("result_preview") or "")) > 5)
            stats[k]["nonempty_retr_sum"] += ne
            if trace and ne == 0:
                issues[k].append(("all_retrievals_empty", sub, idx, f"{len(trace)} steps"))
                stats[k]["all_retrievals_empty"] += 1

    # exact-duplicate idea text within the same (model, track)
    for k, d in seen_text.items():
        for t, cells in d.items():
            if len(cells) > 1:
                stats[k]["dup_text"] += len(cells)
                issues[k].append(("dup_text", cells[0][0], cells[0][1],
                                  f"{len(cells)} cells share this text"))
    return e42, stats, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--show", type=int, default=6, help="sample issues per kind")
    args = ap.parse_args()

    e42, stats, issues = audit()
    KINDS = ["empty", "too_short", "refusal", "len_out_of_band", "protocol_leak",
             "no_end_punct", "dup_text", "no_telemetry", "bad_telemetry_json",
             "telemetry_error", "zero_tool_calls", "all_retrievals_empty"]

    hdr = f"{'model':<34}{'tr':>3}{'cells':>7}" + "".join(f"{k[:9]:>10}" for k in KINDS[:7])
    print(hdr); print("-" * len(hdr))
    tot = Counter()
    for m in e42:
        for tr in ("B", "C"):
            s = stats.get((m, tr))
            if not s:
                continue
            line = f"{m:<34}{tr:>3}{s['cells']:>7}" + "".join(f"{s[k]:>10}" for k in KINDS[:7])
            print(line)
            for k in KINDS:
                tot[k] += s[k]
            tot["cells"] += s["cells"]
    print("-" * len(hdr))
    print(f"{'TOTAL':<34}{'':>3}{tot['cells']:>7}" + "".join(f"{tot[k]:>10}" for k in KINDS[:7]))

    print("\nTrack-C only:")
    print(f"{'model':<34}{'cells':>7}{'no_tel':>8}{'bad_json':>9}{'tel_err':>8}"
          f"{'0_calls':>8}{'0_retr':>8}{'calls/c':>9}{'retr/c':>8}")
    for m in e42:
        s = stats.get((m, "C"))
        if not s:
            continue
        n = s["cells"]
        print(f"{m:<34}{n:>7}{s['no_telemetry']:>8}{s['bad_telemetry_json']:>9}"
              f"{s['telemetry_error']:>8}{s['zero_tool_calls']:>8}"
              f"{s['all_retrievals_empty']:>8}"
              f"{s['tool_calls_sum']/n:>9.2f}{s['nonempty_retr_sum']/n:>8.2f}")

    # Replicate determinism: the generator sends an identical prompt and a fixed
    # seed for idea_index 1..N, so the only source of variation is provider-side
    # sampling. A model the provider makes deterministic returns the same text for
    # all three replicates, and that cell then contributes one effective sample
    # rather than three to any estimate that resamples over idea_index.
    print("\nReplicate determinism (identical text across idea_index):")
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    conn.row_factory = sqlite3.Row
    grp = defaultdict(lambda: defaultdict(set))
    for r in conn.execute("SELECT idea_model, track, subdomain, idea_text "
                          "FROM subdomain_ideas WHERE idea_index<=3"):
        t = body(r["idea_text"])
        if t:
            grp[(r["idea_model"], r["track"])][r["subdomain"]].add(t)
    conn.close()
    print(f"  {'model':<42}{'tr':>3}{'subs':>6}{'all-same':>10}{'rate':>7}")
    flagged = 0
    for k, subs in sorted(grp.items(),
                          key=lambda kv: -sum(len(v) == 1 for v in kv[1].values())
                          / max(len(kv[1]), 1)):
        same = sum(1 for v in subs.values() if len(v) == 1)
        rate = same / len(subs)
        if rate <= 0.05:
            continue
        flagged += 1
        print(f"  {k[0]:<42}{k[1]:>3}{len(subs):>6}{same:>10}{rate*100:>6.0f}%")
    if not flagged:
        print("  (none above 5%)")

    print("\nIssue samples:")
    by_kind = defaultdict(list)
    for (m, tr), lst in issues.items():
        for kind, sub, idx, note in lst:
            by_kind[kind].append((m, tr, sub, idx, note))
    for k in KINDS:
        v = by_kind.get(k)
        if not v:
            continue
        print(f"\n  [{k}]  total {len(v)}")
        for m, tr, sub, idx, note in v[:args.show]:
            print(f"    {m:<32}{tr} {sub[:34]:<36}idx{idx}  {note}")
        if len(v) > args.show:
            print(f"    ... {len(v)-args.show} more")

    if args.json:
        out = {f"{m}|{tr}": [dict(zip(("kind", "subdomain", "idea_index", "note"), i))
                             for i in lst] for (m, tr), lst in issues.items()}
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
