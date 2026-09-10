#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/experiment_env.sh"
cd "$MICRODUCK_REPO_ROOT"
MICRODUCK_RUNNING_FIXED_PHYSICS=1 bash "$EXPERIMENT_DIR/run_speed2_eval.sh" \
  "$MICRODUCK_ARTIFACTS/speed2/best-so-far/checkpoint.pt" reference13000 123 2.5
MICRODUCK_RUNNING_FIXED_PHYSICS=1 MICRODUCK_RUNNING_SQUARED_PROGRESS=1 \
SPEED2_ACTION_RATE_WEIGHT=0.0 SPEED2_LEARNING_RATE=0.00005 \
bash "$EXPERIMENT_DIR/run_speed2_trial.sh" fixed400 4096 400 \
  "${MICRODUCK_LOAD_RUN:-2026-09-10_04-26-47_zero_action400}" "${MICRODUCK_LOAD_CHECKPOINT:-model_13000.pt}" \
  --env.rewards.forward-progress.weight 10.0
runs=(logs/rsl_rl/running/*_fixed400)
test "${#runs[@]}" -eq 1
for seed in 123 456 789; do
  MICRODUCK_RUNNING_FIXED_PHYSICS=1 bash "$EXPERIMENT_DIR/run_speed2_eval.sh" \
    "${runs[0]}/model_13399.pt" reference_final13399 "$seed" 2.5
  MICRODUCK_RUNNING_FIXED_PHYSICS=0 bash "$EXPERIMENT_DIR/run_speed2_eval.sh" \
    "${runs[0]}/model_13399.pt" randomized_final13399 "$seed" 2.5
done
date --iso-8601=seconds > "$MICRODUCK_ARTIFACTS/speed2/fixed400/evaluations-completed.txt"
