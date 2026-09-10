#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/experiment_env.sh"
cd "$MICRODUCK_REPO_ROOT"
MICRODUCK_RUNNING_SQUARED_PROGRESS=1 SPEED2_ACTION_RATE_WEIGHT=0.0 \
SPEED2_LEARNING_RATE=0.00005 bash "$EXPERIMENT_DIR/run_speed2_trial.sh" \
  entropy02_1000 4096 1000 "${MICRODUCK_LOAD_RUN:-2026-09-10_04-26-47_zero_action400}" "${MICRODUCK_LOAD_CHECKPOINT:-model_13000.pt}" \
  --env.rewards.forward-progress.weight 10.0 --agent.algorithm.entropy-coef 0.02
runs=(logs/rsl_rl/running/*_entropy02_1000)
test "${#runs[@]}" -eq 1
for seed in 123 456 789; do
  bash "$EXPERIMENT_DIR/run_speed2_eval.sh" "${runs[0]}/model_13999.pt" entropy02_final "$seed" 2.5
done
date --iso-8601=seconds > "$MICRODUCK_ARTIFACTS/speed2/entropy02_1000/evaluations-completed.txt"
