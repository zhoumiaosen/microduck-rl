# Sustained 2.0 m/s search

User authorized inline execution, parameter changes and further training without approval gates. Target: 2.0 m/s average body-forward speed, not instantaneous peak. Evaluate 10-second rollouts, first second excluded from speed, and report survival and original-heading displacement separately. Confirm promising candidates over multiple seeds and a longer rollout; keep physical robot parameters unchanged.

## Evidence and approach
- Previous 512-environment continuations lost speed even at unchanged reward.
- Original optimizer LR was 1.73415e-5; previous early trained LR was 1.13906e-4. Test a fixed 2e-5 LR with both optimizer param groups and algorithm config aligned.
- Use the published speed-focused iteration-11748 parent, verifying its published checksum and measuring its local baseline.
- Benchmark larger 2048/4096 batches after a 64-environment five-update smoke test. Select the largest measured size with adequate GPU and system RAM headroom.
- Try fixed-LR speed continuation, then adjust the command curriculum, tracking-versus-progress balance and smoothness penalty based on measured checkpoint results. Keep actual motor limits, robot mass, gravity, timestep, observation layout and policy normalization unchanged.
- Save and evaluate intermediate checkpoints. Revert to the best evaluated checkpoint when a trial degrades. A failed trial does not establish a physical speed limit.

## Execution record
- [x] Verify/download parent and prepare a separate fixed-LR checkpoint copy without changing learned weights.
- [x] Measure parent and smoke-test low-LR continuation; benchmark batch sizes.
- [x] Run speed-focused parameter trials with saved configurations and memory logs.
- [ ] Evaluate and continue promising checkpoints toward 2.0 m/s; keep failed trials recorded.
- [x] Validate the retained best result over additional evaluation seeds, inspect video and summarize limitations honestly.

## Current measurements

- Parent 11748, seed 123, 512 environments, 10 seconds: command 2.2 gives 1.6843 m/s and 98.44% survival; command 2.5 gives 1.6799 m/s and 99.22% survival.
- Fixed 2e-5 learning-rate smoke passed; saved optimizer learning rate verified.
- 4096-environment benchmark: roughly 3.14 seconds per update, about 5.9 GB total GPU memory. Selected for larger batches.
- Optional squared-progress reward: 18 tests passed, including existing running configuration tests. Matches linear reward at 1.7 m/s, increases the incentive above that reference, caps at 2.6 m/s and pays zero beyond 45 degrees of tilt. Disabled by default. GPU smoke is still required before its long trial.
- `gentle200`: 200 updates from the unchanged speed-parent weights with fixed LR 2e-5, entropy 0.005, original linear progress reward, 4096 environments. Curriculum currently reaches 2.4 m/s; the configured 2.5 m/s stage is reached later. Saves every 100 iterations.
- Gentle trial finished at 1.6913 m/s. Squared reward reached 1.7569 m/s after 200 updates; its extension reached 1.7784 m/s at iteration 12100 and 1.7760 m/s at 12200. Seed 456 confirmed 1.7740 m/s at 12100.
- Extension deliberately stopped after 293 of its planned 400 updates when the latest evaluated save plateaued. Resumed retained iteration 12200 for `half_action200`, with squared progress and action-rate weight -0.05. That setting passed a separate 64-environment/five-update smoke.
- See `SPEED2_RESULTS.md` for measured speeds, survival and the first squared trial's launcher/log error. Checkpoints and TensorBoard records were independently validated.
- Later trials tested action penalty 0, progress weight 10 and fixed LR 5e-5. Retained iteration 13000 averages 1.834 m/s across three seeds, with 95.8% survival. The target remains unmet.
- `run_entropy_continuation.sh` was interrupted after 625 updates when midpoint evaluations lost speed. Its iteration-13600 checkpoint is preserved; its queued final evaluations were cancelled.
- Reference-parameter diagnostics improved the retained policy to 1.878 m/s over 64 environments. The opt-in fixed-physics configuration passed 19 CPU tests and a GPU smoke. `run_fixed_speed_discovery.sh` completed training and all six final evaluations. Its randomized three-seed result was 1.8187 m/s with 92.77% survival, so the earlier retained policy remains preferable.
- `run_flight_search.sh` is now training 1000 updates with the existing controlled-flight event reward under standard randomization. It will screen every saved checkpoint, retain the fastest finite candidate meeting 95% screening survival (including the existing baseline), confirm over three seeds and 30 seconds, then export normalized ONNX. Do not modify the executing flight workflow or its launcher.
