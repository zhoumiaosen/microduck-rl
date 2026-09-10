# Search for sustained 2.0 m/s

Historical body-forward-speed experiments: the 2.0 m/s target was not reached. The flight1000 workflow stopped near update 891 before its evaluation stage. Its nine saved checkpoints have now been recovered and screened by `scripts/run_straight_search.py`; recovery results are in `artifacts/straight2/recovery.json`. The current experiment targets sustained straight running with a matching heading controller and projected-progress reward, following `STRAIGHT_SPEED_PLAN.md`. Current operation is recorded in `artifacts/straight2/status.json`; corrected trial screens are in `artifacts/straight2/training-screen-v2.json`. Reference-parameter training completed all 400 updates and its final comparisons are separate.

The retained checkpoint averages **1.834 m/s across seeds 123, 456 and 789**, versus **1.681 m/s** for the parent: **+9.1%**. Survival is **95.8%**, versus **99.2%** for the parent. Original-heading displacement speed is 1.399 versus 1.349 m/s; heading drift remains. The retained checkpoint is `artifacts/speed2/best-so-far/checkpoint.pt`; detailed aggregate values and checksums are in `artifacts/speed2/validated-summary.json`.

The retained policy's normalized export is `artifacts/speed2/best-so-far/policy.onnx`. ONNX checker passed, input/output dimensions are 61/14, and the graph includes observation-normalizer statistics. Its 10-second illustrative seed-123 video is `artifacts/speed2/best-so-far/videos/play/rl-video-step-0.mp4`; full FFmpeg decoding passed and sampled frames were inspected. Video is one rollout, not the aggregate benchmark.

Longer validation of this exact retained checkpoint (30 seconds, seed 456, command 2.5, 512 environments) gives **1.840 m/s** mean pre-first-fall body-forward speed and **85.5% survival**. Clean survivors average 1.856 m/s. Original-heading displacement speed is only 0.698 m/s. Longer-run reliability and heading control remain unresolved.

The local RTX 3070 8 GB runs 4096 environments at roughly 3.2 seconds per PPO update (24 steps per environment). Total GPU use is approximately 6.2 GB during training and 7.0 GB with a concurrent 512-environment evaluation. Concurrent evaluation temporarily reduces training throughput.

The screening table uses 512 environments, evaluation seed 123, a fixed 2.5 m/s command and 10-second rollouts. Speed excludes the first second and stops counting each robot after its first fall; survival must therefore be read alongside speed. Straight-line speed is displacement along the original heading over the complete rollout. Multiple-seed confirmation is reported separately above.

| Checkpoint | New PPO updates | Mean forward speed | Survival | Original-heading speed |
|---|---:|---:|---:|---:|
| Speed parent 11748 | 0 | 1.6799 m/s | 99.22% | 1.3538 m/s |
| Gentle 11800 | 53 | 1.6573 m/s | 99.61% | See evaluation JSON |
| Gentle 11900 | 153 | 1.6801 m/s | 99.61% | See evaluation JSON |
| Gentle 11947 | 200 | 1.6913 m/s | 99.80% | 1.3290 m/s |
| Squared 11800 | 53 | 1.7119 m/s | 99.02% | 1.3478 m/s |
| Squared 11900 | 153 | 1.7492 m/s | 98.44% | See evaluation JSON |
| Squared 11947 | 200 | 1.7569 m/s | 99.41% | 1.4023 m/s |
| Squared 12000 | 254 | 1.7604 m/s | 98.83% | See evaluation JSON |
| Squared 12100 | 354 | 1.7784 m/s | 98.44% | 1.4139 m/s |
| Squared 12200 | 454 | 1.7760 m/s | 99.02% | See evaluation JSON |
| Half action 12300 | 555 | 1.8008 m/s | 98.05% | 1.4231 m/s |
| Half action 12399 | 654 | 1.8061 m/s | 97.66% | 1.4215 m/s |
| Progress weight 10, 12500 | 756 | 1.8181 m/s | 97.66% | 1.4509 m/s |
| Progress weight 10, 12600 | 856 | 1.8180 m/s | 98.44% | See evaluation JSON |
| LR 5e-5, 12700 | 957 | 1.8277 m/s | 96.29% | See evaluation JSON |
| LR 5e-5, 12800 | 1057 | 1.8351 m/s | 96.68% | 1.4224 m/s |
| LR 5e-5, 12900 | 1157 | 1.8268 m/s | 96.68% | 1.4006 m/s |
| Zero action cost, 12900 | 1158 | 1.8396 m/s | 96.88% | 1.4220 m/s |
| Zero action cost, 13000 | 1258 | 1.8416 m/s | 96.48% | 1.3890 m/s |
| Zero action cost, 13100 | 1358 | 1.8353 m/s | 94.92% | 1.3646 m/s |
| Zero action cost, 13199 | 1457 | 1.8396 m/s | 95.51% | 1.3872 m/s |
| Entropy 0.02, 13200 | 1459 | 1.8396 m/s | 96.29% | See evaluation JSON |
| Entropy 0.02, 13400 | 1659 | 1.8082 m/s | 97.85% | 1.4431 m/s |

The gentle continuation uses a fixed learning rate of 2e-5 in both the PPO algorithm and restored optimizer, entropy coefficient 0.005, 4096 environments, seed 42, linear progress weight 5 and speed cap 2.6. The command curriculum currently reaches 2.4 m/s. Additional robustness perturbations used by the earlier reward-6 experiment are disabled; the underlying robot and actuator model are unchanged. This is a changed training recipe, not a controlled attribution to learning rate alone.

Independent evaluation seed 456 confirms the iteration-12100 improvement: 1.7740 m/s with 98.44% survival, versus approximately 1.68 m/s for the parent. This repeats evaluation, not training; all continuation runs currently use training seed 42.

The planned 400-update extension was deliberately interrupted after the iteration-12200 evaluation showed a plateau near 1.78 m/s. Its `KeyboardInterrupt` is intentional. The retained checkpoint has finite weights and fixed LR 2e-5. The next trial resumes iteration 12200 and changes only the final action-rate penalty from -0.10 to -0.05. This configuration passed a five-update, 64-environment smoke test before the long run.

Command sweep for half-action iteration 12300 (seed 123): command 2.0 gives 1.8063 m/s, 97.66% survival; command 2.5 gives 1.8008 m/s, 98.05%; command 3.0 gives 1.7812 m/s, 98.83%. Raising the command alone does not improve this policy's speed.

Longer check: half-action iteration 12399, seed 456, command 2.0, 30 seconds, 512 environments: 1.8146 m/s mean body-forward speed, 1.8188 m/s clean-survivor speed and 92.97% survival. Original-heading displacement speed is only 0.7002 m/s because of heading drift. This sustains its body-forward pace but does not establish reliable straight-line running.

`progress10_400` resumes half-action iteration 12399, retaining action penalty -0.05 and increasing squared forward-progress weight from 5 to 10 via the command-line reward override. The resolved YAML is authoritative: the shared environment file still lists its default weight 5. The new recipe passed a separate five-update GPU smoke, produced ONNX, and showed zero NaN terminations. The existing curriculum advances from maximum command 2.4 to 2.5 m/s during this continuation.

Motor audit at iteration 12500 (64 environments, command 2.0, seed 123) uses existing BAM instrumentation without changing physics. Knee full-PWM fractions are approximately 51-52%, with current limiting on 69-73% of sampled steps; ankles reach full PWM around 39-40%. These are last-physics-substep snapshots at control frequency, restricted to pre-first-fall samples. This supports actuator saturation as a constraint but does not establish a hard maximum running speed. Full per-actuator values are in `artifacts/speed2/motor-audit-12500.json`.

Matched parent audit: knee full-PWM fractions were 34-41%, with current limiting on 42-53% of samples. The faster gait demands more actuator output. Diagnostic sample mean speed was 1.6741 m/s for the parent; diagnostic runs use only 64 environments and are not replacements for the 512-environment screening evaluations.

The weight-10 trial was deliberately interrupted after its iteration-12600 result remained near 1.818 m/s. `lr5e5_400` resumes a separate copy of that checkpoint with optimizer and algorithm LR both changed to 5e-5. Learned actor/critic weights and environment state are exactly preserved in the prepared copy. A five-update smoke verified the saved 5e-5 optimizer LR, finite weights, and ONNX export. Reward weight 10 and action penalty -0.05 are retained.

The LR trial was deliberately stopped after iteration 12900 regressed slightly relative to 12800. `zero_action400` resumes retained iteration 12800 with the action-change penalty changed from -0.05 to 0.0; fixed LR 5e-5 and speed weight 10 remain. The exact parent/recipe passed a separate 64-environment/five-update smoke. Short-run plateaus and regressions do not establish convergence or a physical ceiling; all selected-best comparisons are subject to checkpoint-selection bias.

`zero_action400` completed all 400 updates. Its best measured checkpoint is iteration 13000. `run_entropy_continuation.sh` resumes that checkpoint for 1000 updates with entropy coefficient 0.02, the repository running recipe's original exploration setting. The 64-environment smoke passed; resolved agent YAML confirms fixed LR 5e-5 and entropy 0.02. The workflow then evaluates final iteration 13999 on seeds 123, 456 and 789. It writes `evaluations-completed.txt` only after all three evaluations finish successfully. The training and evaluation completion markers are separate.

That exploration workflow was subsequently interrupted after retaining iteration 13600: its 401-update evaluation reached only 1.8082 m/s and its 501-update evaluation reached 1.7926 m/s. Its originally queued final evaluations were cancelled by the deliberate interruption.

A paired 64-environment diagnostic on the retained best policy measured 1.8384 m/s and 96.88% survival with standard randomization, versus 1.8778 m/s and 100% survival with reference mass, foot/joint friction and armature. Sensor variability remains enabled. Reference foot friction is 1.0, matching the center of the configured range; motor limits, mass defaults, geometry, gravity and timestep are unchanged. This is a different evaluation condition and is not pooled into the standard benchmark.

An opt-in `MICRODUCK_RUNNING_FIXED_PHYSICS=1` setting now supports that speed-discovery condition. Its test verifies that only four randomization events are removed, while the robot spec, simulation configuration, rewards, actions and observations remain identical. All 19 relevant tests passed, followed by a 64-environment/five-update GPU smoke and ONNX export. The new `fixed400` trial starts from retained iteration 13000 with speed weight 10, action cost 0, LR 5e-5 and entropy 0.005. Standard randomized performance will be measured separately afterward.

The matching 512-environment reference baseline at command 2.5 is 1.8655 m/s with 95.51% survival. The earlier 64-environment diagnostic is not representative of this larger sample's survival. Reference trial gains must be compared with this 1.8655 m/s baseline; standard-condition gains remain compared with the original randomized benchmark.

After 101 fixed-reference training updates, iteration 13100 measured 1.8658 m/s with 97.07% survival under reference conditions, and 1.8353 m/s with 94.53% survival under standard randomization. No speed gain is established at this checkpoint.

A separate 64-environment diagnostic with reference physics and zero static IMU mounting/encoder calibration errors reached 1.9284 m/s with 100% survival at command 2.0. Ordinary observation noise and delays remain. This more idealized condition is not the standard benchmark, has not reached 2.0 m/s, and is not used to claim that the training target is met.

Reference training completed 400 updates. Its final seed-123 reference evaluation is 1.8629 m/s with 96.09% survival, below the matching 1.8655 m/s baseline. It has not established a speed improvement.

The completed three-seed reference-trial evaluation averages 1.8581 m/s with 95.18% survival under reference conditions, and 1.8187 m/s with 92.77% survival under standard randomization. The earlier retained policy remains the better validated standard-condition result. All six reference-trial final evaluations completed successfully; values are recorded in `artifacts/speed2/reference-final-summary.json`.

The next flight-event trial uses the repository's existing `Mjlab-RunningFlight-Flat-MicroDuck` task under standard randomization. Forward reward weight is 10, action-change cost is 0, fixed LR is 5e-5 and entropy is 0.005. Flight-event weight is 5, with minimum forward speed 1.5 m/s and three consecutive airborne control samples required. This rewards controlled takeoff events rather than accumulated airtime. A five-update/64-environment smoke and ONNX export passed; the resolved configuration was checked. The 1000-update run starts from retained iteration 13000.

The flight workflow screens every checkpoint after training, considering only finite speed measurements and at least 95% ten-second survival. It includes the retained baseline so a worse candidate does not automatically replace it. The selected checkpoint receives three-seed confirmation and a 30-second evaluation. The automated target check requires mean speed at least 2.0 m/s on both horizons, with survival at least 95% and 90%, respectively. These are simulation criteria; the script writes the measured result even when the target is not reached.

The squared-progress trial starts from the same parent and changes only the progress reward relative to the gentle trial. It matches the linear reward at 1.7 m/s, increases the speed incentive above that reference, and gates the reward off beyond 45 degrees of tilt. Its 64-environment, five-update GPU smoke and normalized ONNX export passed. All 18 relevant CPU tests passed.

The first squared trial's launcher exited with an error after training: editing the executing shell file shifted its read position and overwrote the text log. `checkpoint-validation.json` independently verifies iteration 11947, finite actor/critic tensors, optimizer LR 2e-5, 200 TensorBoard samples per reward and zero nonzero NaN terminations. The checkpoint and TensorBoard event file are intact. Subsequent runs keep their launcher unchanged while executing.

Artifacts are in `artifacts/speed2/`; learned checkpoints and resolved configurations are in `logs/rsl_rl/running/`. Original source checkpoints are retained. No result here demonstrates real-hardware performance.
