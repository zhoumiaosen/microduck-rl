#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/experiment_env.sh"
cd "$MICRODUCK_REPO_ROOT"
checkpoint=${1:?checkpoint path}
label=${2:?output label}
seed=${3:-123}
speed=${4:-2.5}
duration=${5:-10}
envs=${6:-512}
unset MICRODUCK_RUNNING_ROBUST_PUSH_MPS MICRODUCK_RUNNING_ROBUST_TRUNK_COM_M
unset MICRODUCK_RUNNING_ROBUST_HEAD_COM_M MICRODUCK_RUNNING_ROBUST_INITIAL_TILT_DEG
mkdir -p "$MICRODUCK_ARTIFACTS/speed2/eval"
uv run --locked python scripts/evaluate_running_checkpoint.py \
  --checkpoint-file "$checkpoint" --speed "$speed" --num-envs "$envs" \
  --duration-s "$duration" --warmup-s 1 --seed "$seed" \
  --output-file "$MICRODUCK_ARTIFACTS/speed2/eval/${label}_v${speed}_t${duration}_s${seed}.json" \
  > "$MICRODUCK_ARTIFACTS/speed2/eval/${label}_v${speed}_t${duration}_s${seed}.log" 2>&1
