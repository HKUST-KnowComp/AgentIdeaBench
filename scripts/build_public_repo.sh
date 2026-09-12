#!/usr/bin/env bash
# Assemble the public AgentIdeaBench tree from this working repo.
#
# Copies a whitelist into a staging directory. A whitelist, not an exclude
# list: this repo also holds an unsubmitted paper, advisor material, an
# internal NVIDIA gateway tutorial, 1.5 GB of databases and the raw-data
# archive, and none of that may reach a public remote by accident.
#
# Usage:  bash scripts/build_public_repo.sh /path/to/staging
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:?usage: build_public_repo.sh <staging-dir>}"

rm -rf "$DEST"
mkdir -p "$DEST"

# --- top-level files --------------------------------------------------------
for f in README.md DATA_CARD.md LICENSE LICENSE-DATA CITATION.cff \
         .env.example requirements.txt pyproject.toml \
         run.py config.py config.json run_sweep_b_full.sh; do
  cp "$SRC/$f" "$DEST/$f"
done

# --- python packages: every .py -------------------------------------------
for d in analysis baselines data_collection evaluation generation utils; do
  mkdir -p "$DEST/$d"
  cp "$SRC/$d"/*.py "$DEST/$d/"
done

# --- experiment, analysis and operational scripts --------------------------
# reports/ holds the figure and statistics scripts behind the paper. Only the
# code ships; the figures and result JSONs it emits stay local, since the rows
# they aggregate are in release_data/ and the scripts regenerate them.
mkdir -p "$DEST/reports" "$DEST/experiments" "$DEST/scripts"
cp "$SRC/reports"/*.py       "$DEST/reports/"
cp "$SRC/experiments"/*.py   "$DEST/experiments/"
cp "$SRC/experiments"/*.json "$DEST/experiments/"
cp "$SRC/scripts"/*.py       "$DEST/scripts/"
cp "$SRC/scripts/build_public_repo.sh" "$DEST/scripts/"

# --- README figures ---------------------------------------------------------
mkdir -p "$DEST/assets"
cp "$SRC/assets"/*.png "$DEST/assets/"

# --- derived statistics the paper figures read ------------------------------
# reports/*.py plots from these small JSON artifacts rather than from the
# databases, so shipping them is what lets a fresh clone rebuild the paper's
# figures with no API key and no papers.db/results.db. Listed explicitly, not
# globbed: reports/ also holds e29_human_eval_pairs.json and e35_cot_probe.json,
# which carry model-generated idea text and belong in the release_data/ pipeline
# with its redaction pass, not here.
for j in cross_year_cutoff_corrected cutoff_bucket_boost cutoff_quarterly_boost \
         e22_new_axis_stats e23_ablations e26_worldmodel e27_swm \
         e28_idea_diversity e30_review_stats e31_matched_compute \
         e32_recall_only e36_leaderboard_subscores e37_review_r2_stats \
         e38_replay_refs e39_perdim_ci e40_domain_validity e41_claim_check \
         e41_frontier_analysis e43_active_turns f0_partial_correlation \
         release_quarterly_boost primary_roster; do
  cp "$SRC/reports/$j.json" "$DEST/reports/$j.json"
done

# _fig_domain_slopes.py deliberately reads a pre-E42 snapshot instead of the
# live e23_ablations.json, because the live file was recomputed over a roster
# that includes the closed-source models the paper holds out. Locally that
# snapshot sits under archive/, which does not ship, so hand the public tree
# the same bytes under reports/ -- the script falls back to this path.
cp "$SRC/archive/reports/pre_e42_mainstats_20260830/e23_ablations.json" \
   "$DEST/reports/e23_ablations_pre_e42_20260830.json"

# --- the redacted data release ---------------------------------------------
cp -R "$SRC/release_data" "$DEST/release_data"

# --- directories the pipeline expects to exist ------------------------------
mkdir -p "$DEST/data" "$DEST/logs"
touch "$DEST/data/.gitkeep" "$DEST/logs/.gitkeep"

cp "$SRC/scripts/public_gitignore" "$DEST/.gitignore"

# --- refuse to hand back a tree containing anything private -----------------
fail=0
while IFS= read -r pattern; do
  hits=$(cd "$DEST" && find . -name "$pattern" -not -path './.git/*' | head -5)
  if [ -n "$hits" ]; then
    echo "REFUSING: private pattern '$pattern' present in staging:" >&2
    echo "$hits" >&2
    fail=1
  fi
done <<'PATTERNS'
.env
*.zip
*.pptx
*.db
Inference_Tutorial.md
CLAUDE.md
log.md
plan.md
state.md
REPORT.md
CODE_REFERENCE.md
PATTERNS
[ "$fail" -eq 0 ] || exit 1

echo "staged $(find "$DEST" -type f -not -path '*/.git/*' | wc -l | tr -d ' ') files, \
$(du -sh "$DEST" | cut -f1) at $DEST"
