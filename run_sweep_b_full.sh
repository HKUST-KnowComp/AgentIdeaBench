#!/bin/bash
# One-shot full Sweep B run: 10 paper × 3 Qwen size × 4 budget × 3 idea
# + critic scoring (3 critics × all idea) + analyze.
# Total ~90-120 min, ~$15. Uses OR default key (Qwen + critics).
set -e
cd "$(dirname "$0")"

LOG="reports/sweep_b_full_run.log"
echo "=== Sweep B full run started at $(date) ===" | tee "$LOG"

echo "" | tee -a "$LOG"
echo "## Phase 1: idea generation (n_ideas=3, 10 papers, 4 budgets, 3 models)" | tee -a "$LOG"
python experiments/budget_sweep_v2.py --gen --n-ideas 3 --workers 4 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "## Phase 2: critic scoring" | tee -a "$LOG"
python experiments/budget_sweep_v2.py --score --workers 4 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "## Phase 3: analyze + per-dim diagnostic" | tee -a "$LOG"
python experiments/budget_sweep_v2_analyze.py 2>&1 | tee -a "$LOG"
python experiments/budget_sweep_per_dim.py 2>&1 | tee -a "$LOG" || echo "(per_dim optional, continuing)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Sweep B full run finished at $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "## Coverage check after run:" | tee -a "$LOG"
python experiments/budget_sweep_v2.py --check 2>&1 | tee -a "$LOG"
