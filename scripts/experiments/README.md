# Local running experiments

These Bash recipes preserve the historical running experiment settings. They
require Linux/WSL, `uv`, NVIDIA tooling, and the checkpoints described below.
They resolve the repository from their own location and run from its root, so
you can invoke them from any working directory. Training still writes mjlab
checkpoints to this checkout's `logs/rsl_rl/running/`.

```bash
bash /path/to/microduck-rl/scripts/experiments/run_speed2_eval.sh \
  /absolute/path/checkpoint.pt baseline 123 2.5
bash /path/to/microduck-rl/scripts/experiments/run_speed2_trial.sh \
  trial_name 64 5 parent_run model_11748.pt
```

Run a 64-environment, 5-update smoke test before a longer trial. The aggregate
search recipes retain their original longer budgets; they do not add a smoke
test automatically.

## Paths and prerequisites

- `UV_PROJECT_ENVIRONMENT` defaults to this checkout's `.venv`. You may set it
  to a dedicated environment elsewhere (for example on WSL's Linux filesystem).
  Commands use `uv run --locked` with this checkout's project and lock file.
- `MICRODUCK_ARTIFACTS` defaults to this checkout's `artifacts`. A relative
  override is relative to the checkout. Evaluation files, logs and receipts
  go below `reward-3070/` or `speed2/` in that directory.
- Evaluation launchers take a checkpoint path as their first argument. Use an
  absolute path for checkpoints outside this checkout; relative paths are
  relative to the checkout, not the caller's directory.
- `MICRODUCK_LOAD_RUN` and `MICRODUCK_LOAD_CHECKPOINT` override trial resume
  defaults. The speed2 trial also accepts these as positional arguments 4 and
  5, followed by extra training CLI flags. A load-run is a run directory name
  under this checkout's `logs/rsl_rl/running/`, not an arbitrary external path.
  To resume an external checkpoint, copy it into a named run directory there
  and provide that directory name and checkpoint filename.

No historical checkpoints or evaluation results are bundled. Default run names
(`release-12195`, `speed-parent-lr2e5`, and
`2026-09-10_04-26-47_zero_action400`) refer only to this checkout's local logs.
Provide the corresponding checkpoints or override the inputs before running.
The entropy/fixed discovery recipes expect their original parent iteration
13000 and final iterations 13999/13399 respectively; changing the parent
iteration also requires adjusting those recipe checkpoint selections.

`run_fixed_speed_discovery.sh` additionally expects
`$MICRODUCK_ARTIFACTS/speed2/best-so-far/checkpoint.pt`.
`run_flight_search.sh` uses the bundled `select_flight_checkpoint.py`, which
preserves the original selector. Before running, evaluate the retained baseline
as label `zero13000` with `run_speed2_eval.sh` at speed 2.5, duration 10, for
seeds 123, 456 and 789. The selector reads these evaluation files from the
configured artifact directory. `evaluate_reward_trials.sh` expects existing
`control500` and `reward6_500` run directories.

## Straight-running search

```bash
uv run --locked --project /path/to/microduck-rl \
  /path/to/microduck-rl/scripts/run_straight_search.py recover \
  --baseline-checkpoint /absolute/path/baseline.pt \
  --flight-run-dir /absolute/path/flight_run \
  --output-dir /absolute/path/straight_results
uv run --locked --project /path/to/microduck-rl \
  /path/to/microduck-rl/scripts/run_straight_search.py train \
  --output-dir /absolute/path/straight_results
```

The recovery phase retains its requirement of exactly nine flight checkpoints
plus the baseline. `train` consumes the selected parent from `recovery.json`.
Path options resolve relative to the caller's directory; omitted defaults
point inside this checkout. The results Markdown is written inside the output
directory alongside its JSON provenance. `--help` does not import GPU packages.
