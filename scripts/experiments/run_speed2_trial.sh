#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/experiment_env.sh"
cd "$MICRODUCK_REPO_ROOT"
name=${1:?run name}
envs=${2:?environment count}
updates=${3:?update count}
load_run=${4:-${MICRODUCK_LOAD_RUN:-speed-parent-lr2e5}}
checkpoint=${5:-${MICRODUCK_LOAD_CHECKPOINT:-model_11748.pt}}
shift 3
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi
export MICRODUCK_RUNNING_TARGET_MAX_SPEED=2.5
export MICRODUCK_RUNNING_SPEED_CAP=2.6
export MICRODUCK_RUNNING_ACTION_RATE_WEIGHT=${SPEED2_ACTION_RATE_WEIGHT:--0.10}
export MICRODUCK_RUNNING_HIGH_SPEED_STAGE_INTERVAL=${SPEED2_STAGE_INTERVAL:-750}
export MICRODUCK_RUNNING_ROBUST_PUSH_MPS=0
export MICRODUCK_RUNNING_ROBUST_TRUNK_COM_M=0
export MICRODUCK_RUNNING_ROBUST_HEAD_COM_M=0
export MICRODUCK_RUNNING_ROBUST_INITIAL_TILT_DEG=0
out="$MICRODUCK_ARTIFACTS/speed2/$name"
mkdir -p "$out"
env | sort | grep -E '^(MICRODUCK_RUNNING_|SPEED2_|OMP_NUM_THREADS|MKL_NUM_THREADS)' > "$out/environment.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.free --format=csv -l 2 > "$out/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
date --iso-8601=seconds > "$out/started.txt"
uv run --locked train Mjlab-Running-Flat-MicroDuck \
  --env.scene.num-envs "$envs" --agent.resume True \
  --agent.load-run "$load_run" --agent.load-checkpoint "$checkpoint" \
  --agent.max-iterations "$updates" --agent.save-interval 100 \
  --agent.run-name "$name" --agent.seed 42 --agent.logger tensorboard \
  --agent.upload-model False --agent.algorithm.schedule fixed \
  --agent.algorithm.learning-rate "${SPEED2_LEARNING_RATE:-0.00002}" --agent.algorithm.entropy-coef 0.005 \
  "$@" > "$out/train.log" 2>&1
date --iso-8601=seconds > "$out/completed.txt"
