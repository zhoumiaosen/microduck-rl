#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/experiment_env.sh"
cd "$MICRODUCK_REPO_ROOT"
checkpoint=${1:?checkpoint path}
label=${2:?output label}
seed=${3:-123}
# Nominal evaluation is identical for baseline, control and candidate.
unset MICRODUCK_RUNNING_ROBUST_PUSH_MPS MICRODUCK_RUNNING_ROBUST_TRUNK_COM_M
unset MICRODUCK_RUNNING_ROBUST_HEAD_COM_M MICRODUCK_RUNNING_ROBUST_INITIAL_TILT_DEG
mkdir -p "$MICRODUCK_ARTIFACTS/reward-3070/eval"
uv run --locked python scripts/evaluate_running_checkpoint.py \
  --checkpoint-file "$checkpoint" --speed 2.2 --num-envs 512 \
  --duration-s 10 --warmup-s 1 --seed "$seed" \
  --output-file "$MICRODUCK_ARTIFACTS/reward-3070/eval/${label}_seed${seed}.json" \
  > "$MICRODUCK_ARTIFACTS/reward-3070/eval/${label}_seed${seed}.log" 2>&1
