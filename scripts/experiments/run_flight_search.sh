#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/experiment_env.sh"
cd "$MICRODUCK_REPO_ROOT"
bash "$EXPERIMENT_DIR/run_flight_trial.sh" flight1000 4096 1000
runs=(logs/rsl_rl/running/*_flight1000)
test "${#runs[@]}" -eq 1
for checkpoint in "${runs[0]}"/model_*.pt; do
  label=$(basename "$checkpoint" .pt)
  MICRODUCK_RUNNING_FIXED_PHYSICS=0 bash "$EXPERIMENT_DIR/run_speed2_eval.sh" \
    "$checkpoint" "flight_run${label#model_}" 123 2.5
done
selected=$(uv run --locked python "$EXPERIMENT_DIR/select_flight_checkpoint.py")
for seed in 123 456 789; do
  MICRODUCK_RUNNING_FIXED_PHYSICS=0 bash "$EXPERIMENT_DIR/run_speed2_eval.sh" \
    "$selected" flight_selected "$seed" 2.5
done
MICRODUCK_RUNNING_FIXED_PHYSICS=0 bash "$EXPERIMENT_DIR/run_speed2_eval.sh" \
  "$selected" flight_selected 456 2.5 30
uv run --locked python "$EXPERIMENT_DIR/select_flight_checkpoint.py" report
unset MICRODUCK_RUNNING_ROBUST_PUSH_MPS MICRODUCK_RUNNING_ROBUST_TRUNK_COM_M
unset MICRODUCK_RUNNING_ROBUST_HEAD_COM_M MICRODUCK_RUNNING_ROBUST_INITIAL_TILT_DEG
MICRODUCK_RUNNING_FIXED_PHYSICS=0 MUJOCO_GL=osmesa uv run --locked python scripts/export.py \
  Mjlab-Running-Flat-MicroDuck --checkpoint-file "$selected" \
  --onnx-file "$MICRODUCK_ARTIFACTS/speed2/flight1000/selected.onnx" --num-envs 1 --device cuda:0 \
  --video False > "$MICRODUCK_ARTIFACTS/speed2/flight1000/export.log" 2>&1
date --iso-8601=seconds > "$MICRODUCK_ARTIFACTS/speed2/flight1000/workflow-completed.txt"
