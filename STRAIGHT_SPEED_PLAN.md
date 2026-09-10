# Sustained straight running at 2.0 m/s

Accepted plan: recover and screen nine flight checkpoints plus retained baseline;
confirm the best two and baseline on evaluation seeds 123/456/789 at 10 and 30 s;
then train matched A/B/C recipes from the selected parent. New opt-in behavior
holds the reset heading with yaw-rate = clip(0.8 * wrapped heading error, +/-0.5)
and rewards squared world velocity projected onto that heading (weight 10,
cap 2.6, upright gate 45 degrees). Existing task defaults remain unchanged.

Each A/B/C recipe uses 4096 environments, training seed 42, fixed LR 5e-5,
entropy .005, action-change cost zero, standard randomization and 400 updates.
Flight-event weights are 0/1.5/5 respectively; minimum speed 1.5 and minimum
airborne streak three steps. Smoke each at 64 environments/five updates first.
Save/evaluate every 100 updates sequentially. Continue the eligible best for at
most 1200 updates at LR 2e-5, stopping after four checkpoint screens without an
eligible straight-speed improvement of .02 m/s. Keep the earlier best.

Primary metric: velocity along initial heading, averaged across every environment
and every sample after one-second warmup; zero all samples after first fall.
Nonfinite states count as failures. Keep historical metrics separately.
Screen with >=95% survival. Final tests: 512 environments each, seeds 2027/4093/8191,
30 seconds, commands 2.0 and 2.5. Target passes only if every seed at command 2.5
has straight mean >=2.0 m/s, survival >=95%, and pre-fall heading error <=10 deg.
Record command-2.0 tracking separately. Export normalized 61-to-14 ONNX and video.

No changes to motors, mass, gravity, timestep or sensor randomization. The final
artifact is a policy PLUS a heading controller; equivalent feedback is required
for hardware deployment. The simulation controller reads simulator yaw; a hardware
yaw-estimation error model is not part of this experiment. All work stays local and existing checkpoints are kept.
No automatic unbounded search after this sequence. Report a failure honestly.
