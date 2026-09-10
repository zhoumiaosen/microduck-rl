# RTX 3070 running reward experiment — 2026-09-10

Training succeeded on the local RTX 3070 8 GB. Increasing forward-progress reward weight from 5.0 to 6.0 did **not** improve body-forward running speed over the published starting policy in this pilot. Keep the original checkpoint as the reference policy.

## Three-seed checks

Each row uses evaluation seeds 123, 456 and 789, with 512 environments per seed (1,536 trials per policy). Speeds are means of the per-seed results.

| Policy | Additional training updates | Body-forward speed (m/s) | 10-second survival | Speed along original heading (m/s) | Mean absolute heading error (degrees) |
|---|---:|---:|---:|---:|---:|
| Original checkpoint | 0 | 1.6528 | 99.35% | 1.2404 | 31.14 |
| Control, final | 500 | 1.5984 | 99.41% | 1.2420 | 29.28 |
| Reward 6, best screened | 106 | 1.6225 | 97.53% | 1.2926 | 27.43 |

The best screened reward-6 checkpoint is 1.83% slower in body-forward speed than the original and survives less often. Its speed along the original heading is 4.21% higher because it drifts less. That is a trade-off, not an overall faster-and-more-stable policy.

The reward-6 checkpoint was selected at iteration 12300 (106 new updates). The control row above is its final checkpoint, so those two rows are **not** an equal-update causal comparison. Equal-update comparisons appear below. Only seed 123 was used to select the candidate; 456 and 789 are additional evaluation seeds. There was one training seed, so this is a pilot rather than a multi-training-seed result.

## Equal-update comparisons and checkpoint screening

All entries below use evaluation seed 123 and identical evaluation settings.

| Checkpoint | Body-forward speed (m/s) | Survival | Speed along original heading (m/s) |
|---|---:|---:|---:|
| baseline | 1.6507 | 99.61% | 1.2433 |
| control500_model_12300 | 1.5949 | 98.44% | 1.1309 |
| control500_model_12400 | 1.5729 | 96.68% | 1.2050 |
| control500_model_12500 | 1.5667 | 98.83% | 0.9102 |
| control500_model_12600 | 1.5406 | 99.61% | 1.1230 |
| control500_model_12694 | 1.5964 | 99.61% | 1.2455 |
| reward6_500_model_12300 | 1.6225 | 98.24% | 1.3021 |
| reward6_500_model_12400 | 1.6142 | 97.85% | 1.2380 |
| reward6_500_model_12500 | 1.5614 | 99.02% | 1.1721 |
| reward6_500_model_12600 | 1.5790 | 98.63% | 1.2859 |
| reward6_500_model_12694 | 1.5950 | 98.24% | 1.2058 |

At 500 new updates, weight 6 reached 1.5950 m/s versus control 1.5964 m/s, with lower survival (98.24% versus 99.61%). The best reward-6 save occurred early; further training reduced its speed. No screened trained checkpoint exceeded the original's body-forward speed.

## Reproducible setup

- Source: [Vottivott/microduck-playground](https://github.com/Vottivott/microduck-playground) at `828d950134e29a8d04cbb51720a22c8729047fb7`, branch `experiment/reward-speed-3070`. This is the matching source for the running checkpoint used by DuckEMW's running lineage.
- Starting artifact: [HannesVonEssen/microduck-running](https://huggingface.co/HannesVonEssen/microduck-running), iteration 12195.
- Checkpoint SHA-256: `052f6df6683fdae83deb369b7f2c7d13e45f87ed2907adee6e23b1ebf9dfe973`, matched the downloaded release checksum and manifest.
- Ubuntu 22.04 / WSL2; RTX 3070 8 GB; Intel i7-11700F; WSL RAM limit approximately 7.6 GiB.
- Locked dependencies: Python 3.12.14, Torch 2.9.1+cu128, Warp 1.12.0, mjlab 1.3.0, MuJoCo 3.10.0.
- Both arms: seed 42, 512 environments, 24 steps/environment/update, 500 updates, 6,144,000 collected transitions per arm; original actor, critic, optimizer, normalizers and curriculum loaded independently for each arm.
- Same command target 2.2 m/s, reward speed cap 2.4 m/s, final action-rate weight -0.10 and released robustness settings. No physics, observation, network, or reward-function source edits.
- Saved environment configs differ only in forward-progress weight; agent configs differ only in run name. See `artifacts/reward-3070/config-comparison.txt`.
- Existing running config tests: 15 passed. The required 64-environment/five-update smoke test passed and exported valid 61-input/14-output ONNX.
- Benchmarks at 128/256/512 environments: peak observed **total GPU memory** 1329/1487/1709 MiB; final update times 1.28/1.53/1.83 seconds. These are short-run observations, not reserved VRAM guarantees.
- Both full runs logged exactly 500 updates and no nonzero NaN terminations. Intermediate checkpoints saved every 100 iteration indices.

Evaluation uses the existing `scripts/evaluate_running_checkpoint.py`: command 2.2 m/s, 10-second total rollout, first second excluded from speed samples, first tilt above 70 degrees counts as a fall. Body-forward speed averages pre-fall samples; survival is reported separately. Displacement speed includes the full rollout. These are simulation measurements, not physical robot measurements. MuJoCo Warp is not bitwise deterministic.

## Files and rerunning

- `experiment_env.sh`: pinned local environment and shared recipe settings.
- `run_reward_trial.sh NAME ENVIRONMENTS UPDATES WEIGHT`: resumes the untouched release and records logs, configs and GPU memory.
- `run_reward_eval.sh CHECKPOINT LABEL SEED`: identical nominal evaluation.
- `evaluate_reward_trials.sh`: screens the saved checkpoints, skipping already completed records.
- `artifacts/reward-3070/eval/`: all 17 raw evaluation JSON files and logs.
- `artifacts/reward-3070/release/`: original checkpoint, checksum list and provenance.
- `artifacts/reward-3070/baseline.mp4` and `reward6-best.mp4`: matching 720p/50 fps, seed-123, one-robot demonstrations.
- `artifacts/reward-3070/comparison.mp4`: original on the left, best screened reward-6 on the right.
- `artifacts/reward-3070/baseline.onnx`, `reward6-best.onnx`, `reward6-best.pt`: exported policies and selected training checkpoint.
- `logs/rsl_rl/running/*control500/` and `*reward6_500/`: every saved training checkpoint and TensorBoard event log.

From Ubuntu, enter `/mnt/f/Codex_GitHub/Duck/microduck-playground`. For another deliberate trial, use `bash run_reward_trial.sh NEW_NAME 512 500 WEIGHT` with a unique run name. It starts from the original release, not the last trained save.

A sensible next investigation is a more conservative continuation learning rate before another reward sweep, because even the unchanged-reward control lost speed. The present experiment does not isolate whether batch size, optimizer adaptation, exploration or the reward caused that baseline degradation. No further trial has been launched.

