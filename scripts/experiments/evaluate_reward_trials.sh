#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/experiment_env.sh"
cd "$MICRODUCK_REPO_ROOT"
shopt -s nullglob
for arm in control500 reward6_500; do
  runs=(logs/rsl_rl/running/*_"$arm")
  if [ "${#runs[@]}" -ne 1 ]; then
    printf 'Expected one run for %s, found %s\n' "$arm" "${#runs[@]}" >&2
    exit 1
  fi
  # The first save is only six updates after resume; evaluate later saves.
  for checkpoint in "${runs[0]}"/model_*.pt; do
    iteration=$(basename "$checkpoint" .pt)
    if [ "$iteration" = model_12200 ]; then continue; fi
    if [ -f "$MICRODUCK_ARTIFACTS/reward-3070/eval/${arm}_${iteration}_seed123.json" ]; then continue; fi
    bash "$EXPERIMENT_DIR/run_reward_eval.sh" "$checkpoint" "${arm}_${iteration}" 123
  done
done
