#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/experiment_env.sh"
cd "$MICRODUCK_REPO_ROOT"
name=${1:?run name}
envs=${2:?environment count}
iterations=${3:?iteration count}
weight=${4:?forward reward weight}
out="$MICRODUCK_ARTIFACTS/reward-3070/$name"
mkdir -p "$out"
export MICRODUCK_RUNNING_FORWARD_PROGRESS_WEIGHT="$weight"
env | sort | grep -E '^(MICRODUCK_RUNNING_|OMP_NUM_THREADS|MKL_NUM_THREADS)' > "$out/environment.txt"
git rev-parse HEAD > "$out/source-commit.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.free --format=csv -l 2 > "$out/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
date --iso-8601=seconds > "$out/started.txt"
uv run --locked train Mjlab-Running-Flat-MicroDuck \
  --env.scene.num-envs "$envs" \
  --agent.resume True --agent.load-run "${MICRODUCK_LOAD_RUN:-release-12195}" \
  --agent.load-checkpoint "${MICRODUCK_LOAD_CHECKPOINT:-model_12195.pt}" \
  --agent.max-iterations "$iterations" --agent.save-interval 100 \
  --agent.run-name "$name" --agent.seed 42 --agent.logger tensorboard \
  > "$out/train.log" 2>&1
date --iso-8601=seconds > "$out/completed.txt"
