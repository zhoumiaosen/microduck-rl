#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/experiment_env.sh"
cd "$MICRODUCK_REPO_ROOT"
name=${1:?run name}
envs=${2:?environment count}
updates=${3:?update count}
export MICRODUCK_RUNNING_FIXED_PHYSICS=0 MICRODUCK_RUNNING_SQUARED_PROGRESS=1
export MICRODUCK_RUNNING_TARGET_MAX_SPEED=2.5 MICRODUCK_RUNNING_SPEED_CAP=2.6
export MICRODUCK_RUNNING_FORWARD_PROGRESS_WEIGHT=10 MICRODUCK_RUNNING_ACTION_RATE_WEIGHT=0
export MICRODUCK_RUNNING_ROBUST_PUSH_MPS=0 MICRODUCK_RUNNING_ROBUST_TRUNK_COM_M=0
export MICRODUCK_RUNNING_ROBUST_HEAD_COM_M=0 MICRODUCK_RUNNING_ROBUST_INITIAL_TILT_DEG=0
out="$MICRODUCK_ARTIFACTS/speed2/$name"
mkdir -p "$out"
env | sort | grep '^MICRODUCK_RUNNING_' > "$out/environment.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.free --format=csv -l 2 > "$out/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
date --iso-8601=seconds > "$out/started.txt"
uv run --locked train Mjlab-RunningFlight-Flat-MicroDuck \
  --env.scene.num-envs "$envs" --agent.resume True --agent.experiment-name running \
  --agent.load-run "${MICRODUCK_LOAD_RUN:-2026-09-10_04-26-47_zero_action400}" --agent.load-checkpoint "${MICRODUCK_LOAD_CHECKPOINT:-model_13000.pt}" \
  --agent.max-iterations "$updates" --agent.save-interval 100 --agent.run-name "$name" \
  --agent.seed 42 --agent.logger tensorboard --agent.upload-model False \
  --agent.algorithm.schedule fixed --agent.algorithm.learning-rate 0.00005 \
  --agent.algorithm.entropy-coef 0.005 --env.rewards.flight-event.weight 5.0 \
  --env.rewards.flight-event.params.min-forward-speed 1.5 > "$out/train.log" 2>&1
date --iso-8601=seconds > "$out/completed.txt"
