# Refactor validation

Executed on 2026-09-10 using Ubuntu 22.04 under WSL2, Python 3.12.14,
the committed dependency lock, and an NVIDIA RTX 3070 (8 GB).

| Check | Result |
| --- | --- |
| Unmodified copied-project baseline | 215 passed, 2 skipped |
| Final CPU suite, including reload and script portability regressions | 220 passed, 2 skipped |
| Source-preserving extraction | All 269 original function/class ASTs unchanged |
| Task discovery | 39 tasks; training and play configurations load |
| Robot asset loading | All 11 distinct registered specifications compile |
| Distribution build | Wheel and source distribution built |
| Fresh wheel installation, executed from `/tmp` | 39 tasks and 11 robot specifications pass, without source-path injection |
| Running GPU smoke | 64 environments, 5 completed PPO updates, seed 42, 7,680 samples |
| Smoke numerical checks | Zero logged NaN terminations; actor/critic checkpoint tensors finite |
| Policy interface | Actor observation width 61; ONNX action width 14 |
| Normalized ONNX export | Running smoke and retained swing policy exported; both pass ONNX checker and finite inference |
| Swing checkpoint/ONNX rollout parity | 10 steps, seed 27; max absolute action error `9.53674e-07` (tolerance `2e-5`) |
| Original checkout preservation | All 447 captured source file hashes and existing Git status entries unchanged |

The two skips are the ARM64 GPU-specific Torch check (this is x86_64) and
the Rust runtime adapter test (`rustc` is not installed). These are the same
skips as the baseline. GitHub Actions has been updated but has not run remotely.
At this initial validation, native Windows GPU training, long training runs,
and physical robot deployment were not tested. Five updates verify the training
pipeline, not policy quality. The subsequent running release is recorded below.

## Running release (2026-09-11)

The running policy completed 10,000 additional PPO updates from `model_999.pt`
with 512 environments on the RTX 3070, reaching final checkpoint `model_10998.pt`.
No nonzero NaN terminations were logged. Evaluation and normalized ONNX export
completed, and the [public release](https://huggingface.co/zhoumiaosen/microduck-running)
includes the checkpoint, ONNX policy, replay video, and raw evaluation JSON.

At a 2.0 m/s command, three 10-second evaluations (512 environments each, seeds
123/456/789) averaged 1.499 m/s body-forward speed, 1.234 m/s straight progress,
and 96.74% survival. The 30-second evaluation (seed 456) measured 1.499 m/s
body-forward speed and 91.21% survival. The sustained 2.0 m/s target was not met.
Physical robot deployment remains untested.

During final verification the original checkout gained a new untracked
`STRAIGHT_SPEED_RESULTS.md`. It was not part of the starting snapshot and was
left in the original folder. No previously captured source file changed.

## Reproduce the core checks

From the repository root, after `uv sync --locked`:

```bash
uv run --locked --with pytest pytest -q -rs
uv build
uv run --locked python scripts/verify_install.py
uv run --locked python scripts/check_relative_links.py
uv run --locked train Mjlab-Running-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.save-interval 5 --agent.seed 42 \
  --agent.run-name smoke --agent.logger tensorboard
```

The installed-wheel check is also encoded in `.github/workflows/ci.yml`: install
locked dependencies and the wheel into a separate environment, leave the
checkout, unset `PYTHONPATH`, and run `scripts/verify_install.py` with that
environment's Python.

The local smoke checkpoint is in
`logs/rsl_rl/running/2026-09-10_15-06-43_refactor-smoke/model_4.pt`.
Its exported policy and the retained swing export are in
`outputs/validation/running-smoke.onnx` and `outputs/validation/swing.onnx`.
These generated files remain local and are ignored by Git. The retained swing
checkpoint for reproducing parity is
`experiments/swing/checkpoints/alpha050.pt`:

```bash
mkdir -p outputs/validation
uv run --locked python scripts/export.py Mjlab-SwingPump-MicroDuck \
  --checkpoint-file experiments/swing/checkpoints/alpha050.pt \
  --onnx-file outputs/validation/swing.onnx --num-envs 1 --device cpu
uv run --locked python scripts/verify_swing_onnx_parity.py \
  experiments/swing/checkpoints/alpha050.pt outputs/validation/swing.onnx \
  --device cpu --duration 0.2 --seed 27
```
