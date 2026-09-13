"""Compute sim4 = semantic similarity between the STATIC reference set and the
ACTIVE (self-searched) reference set, per subdomain — the missing 4th comparison
for the refs-overlap figure. Reuses the cached embeddings in v3_arrays.npz (no
re-encoding). Read-only. /usr/bin/python3.

sim4 for a (model, subdomain) = cosine( centroid(static refs), centroid(active refs) ),
which equals the mean pairwise cosine between the two sets (vectors are normalized).
"""
import json, re, sqlite3, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

_PID = re.compile(r'"paperId":\s*"([0-9a-f]{40})"')
Z = np.load(ROOT / "reports" / "e11_ref_overlap" / "v3_arrays.npz", allow_pickle=True)
RMAT = Z["rmat"]
IDX = {rid: i for i, rid in enumerate(Z["ref_ids"])}  # pid -> row (light)


def centroid(ids):
    rows = [IDX[r] for r in ids if r in IDX]
    if not rows:
        return None
    c = RMAT[rows].mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else None


def main():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    sref = {}
    for r in conn.execute("SELECT subdomain, refs_json FROM subdomain_refs"):
        sref[r["subdomain"]] = [x["paperId"] for x in json.loads(r["refs_json"])
                                if isinstance(x, dict) and x.get("paperId")]
    per_model = defaultdict(list); allv = []
    for r in conn.execute("SELECT idea_model,subdomain,telemetry FROM subdomain_ideas "
                          "WHERE track='C' AND TRIM(idea_text)!=''"):
        aids = set()
        if r["telemetry"]:
            try:
                tel = json.loads(r["telemetry"])
                for t in tel.get("trace", []):
                    rp = (t.get("result_preview", "") or "") if isinstance(t, dict) else ""
                    try:
                        items = json.loads(rp)
                        if isinstance(items, list):
                            for it in items:
                                if isinstance(it, dict) and it.get("paperId"):
                                    aids.add(it["paperId"])
                            continue
                    except Exception:
                        pass
                    aids.update(_PID.findall(rp))
            except Exception:
                pass
        sids = sref.get(r["subdomain"], [])
        cs, ca = centroid(sids), centroid(list(aids))
        if cs is not None and ca is not None:
            v = float(cs @ ca); per_model[r["idea_model"]].append(v); allv.append(v)
    conn.close()
    out = {"sim4_static_refs_vs_active_refs": round(float(np.mean(allv)), 4), "n_pairs": len(allv),
           "per_model": {m: round(float(np.mean(v)), 4) for m, v in sorted(per_model.items())}}
    (ROOT / "reports" / "e11_ref_overlap" / "v3_refs_sim4.json").write_text(json.dumps(out, indent=2))
    print(f"sim4 (static refs vs active refs) = {out['sim4_static_refs_vs_active_refs']}  (n={out['n_pairs']})")


if __name__ == "__main__":
    main()
