"""MicroDuck roulade terms."""

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from typing import Optional
import math
import numpy as np
import torch
from .common import (
    _DEFAULT_ASSET_CFG,
    _servo_joint_ids,
)
from .recovery import (
    standing_composite_score,
)


# ==============================================================================
# Roulade (forward roll) task — episodic dynamic maneuver
# ==============================================================================
#
# Third attempt at the roulade. What the first two taught us:
#   • origin/roulade (phase-clock + time-windowed reward stages): plateaued
#     face-down at ~90° — time windows are keyframes-in-time, campable local
#     optima (the sit/standup lesson exactly). Also integrated -ω_y as forward
#     progress, which by this codebase's own convention (face-down = +90° pitch
#     = rotation about +y, see set_random_ground_state) is the WRONG SIGN — the
#     progress reward paid for backward rotation.
#   • origin/roulade later commits (keyframe imitation): same waypoint-camping
#     family, dropped per feedback-episodic-pose-landing.
#
# This design uses the proven episodic recipe instead:
#   • ONE dense progress signal: paid INCREMENTS of the max-so-far cumulative
#     forward rotation (potential-based — a camping policy earns zero/step, a
#     full roll earns exactly 2π worth no matter the path or speed).
#   • Landing rewards (composite product, upright, height, rise velocity) are
#     gated on ROLL COMPLETION (max rotation ≥ threshold) — state-based gates,
#     not clock-based. "Do nothing" earns nothing; standing at spawn earns
#     nothing; only rolling opens the standing-attractor annuity.
#   • Reverse curriculum via mid-roll spawns (the face-up partial-roll trick
#     that fixed back-recovery): a slice of episodes starts pitched 50°–185°
#     into the roll, tucked, optionally with forward angular momentum, and the
#     rotation accumulator is initialized to the spawn angle so the progress
#     accounting stays consistent.
#
# RUN-1 LESSON (2026-08): with unsupported rotation counting and uncapped
# paid rate, the optimal policy is a violent ballistic whip ("breakdance") —
# same 2π, finishes sooner, more discounted annuity. Doesn't transfer. Fixes:
#   • SUPPORT GATE: the accumulator only integrates while some robot geom
#     touches the terrain (robot_ground_contact sensor) — a real roulade never
#     leaves the ground; airborne rotation now earns nothing and cannot open
#     the completion gate.
#   • HEAD LATCH: the landing annuity additionally requires head-ground
#     contact to have occurred while accum was in the first-quadrant window —
#     "went over the head" is a requirement, not a 0.5-weight suggestion.
#   • PAID-RATE CAP: progress increments are capped at max_paid_rate; rotation
#     faster than the cap FORFEITS the excess (not deferred), so speed no
#     longer pays. An explicit overspeed penalty backs this up.
#
# Per-env state on the env object (created lazily, reset by
# reset_roulade_state):
#   env._roulade_accum      — supported-only integral of forward pitch rate (rad)
#   env._roulade_max        — max(accum) so far this episode (progress frontier)
#   env._roulade_paid       — frontier already paid out by roulade_progress
#   env._roulade_head_latch — True once the head touched ground mid-first-quadrant

# Forward-roll sign: face-down is +90° pitch = rotation about body +y
# (set_random_ground_state convention), so forward roll = POSITIVE body-frame
# ω_y. Verified empirically (see claude_experiments smoke test): a positive
# qvel about +y pitches the robot nose-down/forward and drives accum upward.
_ROULADE_FWD_SIGN = 1.0


# Sensor names read by the accumulator update (must match the env cfg).
_ROULADE_SUPPORT_SENSOR = "robot_ground_contact"


_ROULADE_HEAD_SENSOR = "head_ground_contact"


# Head-latch window: head-ground contact while accum is inside this window
# marks the episode as a genuine over-the-head roll. In a real roulade the
# head plants at ~60–120° of body rotation; the window is generous around it.
_HEAD_LATCH_LO = math.radians(20.0)


_HEAD_LATCH_HI = math.radians(170.0)


# Head-top axis in jaw_soft's LOCAL frame (measured empirically 2026-08-13:
# world-up expressed in jaw_soft's frame with the robot settled at HOME).
# The latch requires this axis to point DOWN at contact — "the flat top of
# the head on the floor", not the face or the side of the shell (run-5 fix:
# the run-4 policy rolled over the shoulder, which still touched jaw_soft).
_HEAD_TOP_AXIS = (0.882, 0.0, 0.471)


# dot(top_axis_world, -z) threshold. Measured landmarks (trunk pitched 110°):
# passive face-plant (neck at HOME) reads +0.6, full chin-tuck (neck_pitch −1,
# head_pitch +1) reads −0.99 — 0.3 accepts partial tucks while staying far
# from any face/side contact.
_HEAD_TOP_DOWN_MIN = 0.3


# Sagittal flatness gate on the accumulator (run-5): in a clean forward roll
# the body's LATERAL axis stays horizontal the whole way — its world-z
# component is 2(q_y·q_z + q_w·q_x) ≈ 0 for ANY amount of pure pitch, and
# grows toward ±1 as the roll goes over the shoulder instead. Full rotation
# credit while the lateral axis is within ~30° of horizontal, zero beyond
# ~60°: a side roll does not count as rotation, earns no progress, and never
# opens the landing gate.
_FLAT_FULL = 0.5    # |lateral_axis_z| = sin(30°): full credit below


_FLAT_ZERO = 0.866  # sin(60°): zero credit above


def _lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    """World-z component of the body's lateral (y) axis. 0 = flat/sagittal."""
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])


def _head_top_down(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """True where the head-top axis points at the floor (dot with -z > min)."""
    if not hasattr(env, "_roulade_head_body_id"):
        ids, _ = asset.find_bodies("jaw_soft")
        env._roulade_head_body_id = ids[0]
    q = asset.data.body_link_quat_w[:, env._roulade_head_body_id]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a, b, c = _HEAD_TOP_AXIS
    # z-component of R(q) @ axis_local
    axis_world_z = (
        2.0 * (x * z - w * y) * a + 2.0 * (y * z + w * x) * b + (1.0 - 2.0 * (x * x + y * y)) * c
    )
    return axis_world_z < -_HEAD_TOP_DOWN_MIN


def _sensor_any_contact(env: ManagerBasedRlEnv, name: str) -> torch.Tensor | None:
    if name not in env.scene.sensors:
        return None
    found = env.scene.sensors[name].data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)


def _roulade_state(env: ManagerBasedRlEnv) -> tuple:
    if not hasattr(env, "_roulade_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._roulade_accum = z.clone()
        env._roulade_max = z.clone()
        env._roulade_paid = z.clone()
        env._roulade_head_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._roulade_last_update_step = -1
    return env._roulade_accum, env._roulade_max, env._roulade_paid


def _update_roulade_accum(env: ManagerBasedRlEnv, asset: Entity) -> None:
    """Integrate forward pitch rate into the per-env rotation accumulator.

    Step-guarded so that multiple reward terms reading the accumulator in the
    same control step don't double-integrate. The frontier (max) only moves
    forward; backward rocking (wind-up) neither pays nor un-pays.

    SUPPORT GATE (run-1 fix): rotation is integrated only while the robot
    touches the terrain — a roulade is a supported motion; ballistic flips
    accumulate nothing, so they neither get paid nor open the completion gate.

    Also latches env._roulade_head_latch when the head touches the ground
    while accum is inside the first-quadrant window — the landing annuity
    requires this, making "over the head" a hard requirement of the task.
    """
    _roulade_state(env)
    step = int(env.common_step_counter)
    if step != env._roulade_last_update_step:
        omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
        delta = torch.nan_to_num(omega_fwd, nan=0.0) * env.step_dt
        supported = _sensor_any_contact(env, _ROULADE_SUPPORT_SENSOR)
        if supported is not None:
            delta = delta * supported.float()
        # Sagittal flatness gate (run-5): side/shoulder rolls don't count.
        y_z = torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=1.0).abs()
        t = torch.clamp((_FLAT_ZERO - y_z) / (_FLAT_ZERO - _FLAT_FULL), 0.0, 1.0)
        delta = delta * (t * t * (3.0 - 2.0 * t))
        env._roulade_accum = env._roulade_accum + delta
        env._roulade_max = torch.maximum(env._roulade_max, env._roulade_accum)

        head_contact = _sensor_any_contact(env, _ROULADE_HEAD_SENSOR)
        if head_contact is not None:
            in_window = (env._roulade_accum > _HEAD_LATCH_LO) & (
                env._roulade_accum < _HEAD_LATCH_HI
            )
            # Run-5: contact must be with the FLAT TOP of the head (top axis
            # pointing at the floor) — face/side shell contacts don't latch.
            env._roulade_head_latch = env._roulade_head_latch | (
                head_contact & in_window & _head_top_down(env, asset)
            )
        env._roulade_last_update_step = step


def _roulade_completion_gate(
    env: ManagerBasedRlEnv,
    gate_lo: float,
    gate_hi: float,
    require_head: bool = False,
) -> torch.Tensor:
    """Smoothstep on the progress frontier: 0 below gate_lo rad, 1 above gate_hi.

    State-based replacement for the old phase-clock landing window — it can
    only be opened by actually rotating (while SUPPORTED — the accumulator is
    contact-gated), so neither pre-roll standing nor a ballistic flip collects.
    With require_head=True the gate additionally requires the head latch —
    the episode must have rolled over the head to unlock the landing annuity.
    """
    _, max_accum, _ = _roulade_state(env)
    t = torch.clamp((max_accum - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if require_head:
        gate = gate * env._roulade_head_latch.float()
    return gate


def reset_roulade_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    standing_prob: float = 0.5,
    midroll_prob: float = 0.5,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    standing_tilt_max: float = 0.0,
    forward_vel_range: tuple = (0.0, 0.0),
    midroll_pitch_min: float = math.radians(50.0),
    midroll_pitch_max: float = math.radians(185.0),
    midroll_z_min: float = 0.05,
    midroll_z_max: float = 0.10,
    midroll_omega_range: tuple = (0.0, 0.0),
    tuck_overrides: Optional[dict] = None,
    tuck_factor_range: tuple = (0.3, 1.0),
    joint_noise_std: float = 0.0,
):
    """Reset to a standing start or a mid-roll state (reverse curriculum).

    Standing bucket: upright (±standing_tilt_max pitch/roll noise), random yaw,
    HOME joints (left from reset_robot_joints), z in [standing_z_min, _max].
    ``forward_vel_range`` is the élan hook: a per-env forward base velocity
    (body x, mapped to world through the spawn yaw) sampled uniformly — 0 for
    a standstill roll, widen it later to train rolls out of a walk.

    Mid-roll bucket: pitched ``midroll_pitch_min..max`` into the roll (90° =
    on the head, 180° = on the back), random yaw, legs lerped HOME→tuck by a
    per-env factor in ``tuck_factor_range``, z in [midroll_z_min, _max],
    optional forward angular momentum from ``midroll_omega_range``. The
    rotation accumulator is initialized to the spawn pitch so progress
    accounting (and the completion gates) stay consistent: a 170° spawn only
    gets paid for the remaining ~190°.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    accum, max_accum, paid = _roulade_state(env)

    total = standing_prob + midroll_prob
    is_mid = torch.rand(num, device=env.device) < (midroll_prob / max(total, 1e-6))

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    # Pitch per bucket: small noise for standing, mid-roll angle otherwise.
    pitch = (torch.rand(num, device=env.device) * 2 - 1) * standing_tilt_max
    mid_pitch = (
        torch.rand(num, device=env.device) * (midroll_pitch_max - midroll_pitch_min)
        + midroll_pitch_min
    )
    pitch = torch.where(is_mid, mid_pitch, pitch)
    roll = (torch.rand(num, device=env.device) * 2 - 1) * max(standing_tilt_max, math.radians(5.0))

    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    # ZYX intrinsic Euler → quaternion (yaw * pitch * roll), as in
    # set_random_ground_state.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    z_mid = torch.rand(num, device=env.device) * (midroll_z_max - midroll_z_min) + midroll_z_min
    new_z = torch.where(is_mid, z_mid, z_stand)

    env.sim.data.qpos[env_ids, 2] = new_z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    servo_ids = _servo_joint_ids(env, asset)

    # Mid-roll joints: lerp HOME → tuck on the overridden joints, noise on all
    # servo joints (passive_* backlash hinges must stay at 0).
    mid_env_ids = env_ids[is_mid]
    if len(mid_env_ids) > 0 and tuck_overrides:
        u = (
            torch.rand(len(mid_env_ids), device=env.device)
            * (tuck_factor_range[1] - tuck_factor_range[0])
            + tuck_factor_range[0]
        )
        for jnt_idx, angle in tuck_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            home = env.sim.data.qpos[mid_env_ids, col]
            env.sim.data.qpos[mid_env_ids, col] = home + u * (angle - home)
    if len(mid_env_ids) > 0 and joint_noise_std > 0.0:
        cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
        noise = torch.randn(len(mid_env_ids), len(cols), device=env.device) * joint_noise_std
        env.sim.data.qpos[mid_env_ids.unsqueeze(1), cols.unsqueeze(0)] += noise

    # Mid-roll forward angular momentum: rotation about body +y. MuJoCo free
    # joint qvel[3:6] is the angular velocity in the BODY frame, so [0, ω, 0]
    # is the forward-roll axis regardless of spawn yaw (verified in the smoke
    # test — a yawed spawn still rolls straight ahead in its own frame).
    if len(mid_env_ids) > 0 and midroll_omega_range[1] > 0.0:
        omega = (
            torch.rand(len(mid_env_ids), device=env.device)
            * (midroll_omega_range[1] - midroll_omega_range[0])
            + midroll_omega_range[0]
        )
        env.sim.data.qvel[mid_env_ids, 4] = _ROULADE_FWD_SIGN * omega

    # Élan hook: forward base velocity for STANDING spawns, body x → world xy
    # through the spawn yaw. (0, 0) = standstill start, disabled.
    stand_env_ids = env_ids[~is_mid]
    if len(stand_env_ids) > 0 and forward_vel_range[1] > 0.0:
        vx = (
            torch.rand(len(stand_env_ids), device=env.device)
            * (forward_vel_range[1] - forward_vel_range[0])
            + forward_vel_range[0]
        )
        yaw_s = yaw[~is_mid]
        env.sim.data.qvel[stand_env_ids, 0] = vx * torch.cos(yaw_s)
        env.sim.data.qvel[stand_env_ids, 1] = vx * torch.sin(yaw_s)

    # Progress accounting: standing starts at 0, mid-roll at the spawn pitch.
    spawn_angle = torch.where(is_mid, mid_pitch, torch.zeros_like(mid_pitch))
    accum[env_ids] = spawn_angle
    max_accum[env_ids] = spawn_angle
    paid[env_ids] = spawn_angle
    # Head latch: mid-roll spawns are considered already past the head phase
    # (the reverse curriculum teaches roll COMPLETION; requiring a latch they
    # never had the chance to earn would keep their landing gate shut forever).
    # Standing spawns must earn it by actually rolling over the head.
    env._roulade_head_latch[env_ids] = is_mid


def roulade_progress(
    env: ManagerBasedRlEnv,
    target_angle: float = 2 * math.pi,
    max_paid_rate: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay increments of the progress frontier, up to one full roll.

    reward = Δ(min(max_accum, target)) / (step_dt · target), CAPPED at
    max_paid_rate rad/s of paid rotation. Nothing to farm by camping
    face-down (0/step), rocking below the frontier (0/step), or spinning past
    2π (clamped). The accumulator is support-gated, so airborne rotation pays
    nothing either.

    max_paid_rate (run-1 fix): rotation faster than the cap FORFEITS the
    excess — the paid pointer still jumps to the frontier, it just pays the
    capped amount. A violent whip therefore collects LESS total progress
    reward than a controlled ≤cap roll, instead of the same total sooner.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    _, max_accum, paid = _roulade_state(env)
    new_paid = torch.clamp(max_accum, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._roulade_paid = torch.maximum(paid, new_paid)
    return delta / (env.step_dt * target_angle)


def roulade_head_pivot(
    env: ManagerBasedRlEnv,
    sensor_name: str = "head_ground_contact",
    angle_lo: float = math.radians(30.0),
    angle_hi: float = math.radians(240.0),
    rate_norm: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward head-ground contact while rotating forward mid-roll.

    contact × window(accum ∈ [angle_lo, angle_hi]) × clamp(ω_fwd/rate_norm, 0, 1)
    × (0.3 + 0.7·top_down).
    The rate factor is the anti-camping guard: a face-planted robot resting its
    head on the floor has ω_fwd ≈ 0 and earns nothing — the term only pays for
    pivoting OVER the head. The top_down factor (run-5) aligns this dense
    shaping with the latch: any head contact mid-roll pays 30%, contact on the
    FLAT TOP (chin tucked) pays full — the gradient that teaches the tuck.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    accum, _, _ = _roulade_state(env)

    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    contact = (found.view(found.shape[0], -1) > 0).any(dim=-1).float()

    in_window = ((accum > angle_lo) & (accum < angle_hi)).float()
    omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
    rate = torch.clamp(torch.nan_to_num(omega_fwd, nan=0.0) / rate_norm, 0.0, 1.0)
    top = 0.3 + 0.7 * _head_top_down(env, asset).float()
    return contact * in_window * rate * top


def roulade_landing_composite(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """standing_composite_score × completion gate.

    The big annuity: once the roll is (nearly) complete, every step spent
    standing at HOME pose pays — finishing on the feet and staying there
    dominates every partial outcome. Zero before gate_lo of rotation, so the
    standing spawn cannot farm it by doing nothing.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    score = standing_composite_score(
        env,
        target_height=target_height,
        height_std=height_std,
        upright_std=upright_std,
        pose_std=pose_std,
        joint_indices=joint_indices,
        target_overrides=target_overrides,
        asset_cfg=asset_cfg,
    )
    return score * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_upright_after_roll(
    env: ManagerBasedRlEnv,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear cos(tilt) × completion gate — bootstrap pull toward vertical.

    Gradient from ANY orientation (the composite is near-zero far from the
    goal), but only after the roll: before gate_lo it is exactly zero, so it
    cannot oppose the flip the way the old always-on upright term did.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    upright = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    return torch.clamp(upright, min=0.0) * _roulade_completion_gate(
        env, gate_lo, gate_hi, require_head=True
    )


def roulade_height_after_roll(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float = 0.04,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Broad height Gaussian × completion gate — pull up to standing height."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    g = torch.exp(-((z - target_height) / std) ** 2)
    return g * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_landing_sharp(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float = 0.015,
    upright_std: float = 0.3,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Tight-std upright × height Gaussians × completion gate — the last mile.

    Run-4 fix for the 27°-lean / 1-cm-crouch end basin: the broad landing
    composite (upright_std 0.40) scores ~0.5 at that pose, so the policy
    parks there. This is standup's two-layer lesson — the broad layers reach,
    the sharp layers finish. At 27° tilt this term scores ~0.1 (real
    gradient); at vertical it pays ~1.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    upright_g = torch.exp(-tilt_sq / (upright_std * upright_std))
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_g = torch.exp(-((z - target_height) / height_std) ** 2)
    gate = _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)
    return upright_g * height_g * gate


def roulade_stand_tax(
    env: ManagerBasedRlEnv,
    target_height: float,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """SELF-NEGATING height L1 below target, active only after roll completion.

    Returns −max(0, target − z) × completion_gate — use a POSITIVE weight
    (penalty sign convention). The run-3 fix for post-roll crumple-camping:
    the gated landing rewards made standing better than lying in a heap, but
    the heap itself was FREE — with only positive gated terms, "stay crumpled"
    collects ≈0/step, a comfortable basin (the standup static-sit lesson:
    the basin must be net NEGATIVE to force the rise). The gate keeps the
    roll itself untaxed, and requires the head latch so a no-roll episode
    can't be punished into weird avoidance behaviors.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    shortfall = torch.clamp(target_height - z, min=0.0)
    return -shortfall * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_rise_velocity(
    env: ManagerBasedRlEnv,
    max_height: float = 0.125,
    gate_lo: float = math.radians(180.0),
    gate_hi: float = math.radians(260.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """com_upward_velocity × late-roll gate — bootstrap the exit rise.

    The second half of a roulade (supine → sitting-up → standing) is the
    face-up recovery problem, and the standup env proved end-state rewards
    alone have zero gradient at zero motion there: pay for rising vz directly.
    Gated to open from ~180° (on the back) so pre-roll bobbing earns nothing,
    and gated off above max_height so it can't be farmed by hopping.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    reward = torch.clamp(vz, min=0.0) * (z < max_height).float()
    return reward * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_overspeed_penalty(
    env: ManagerBasedRlEnv,
    omega_max: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """max(0, |ω_y| − omega_max)² — quadratic tax on whip-speed rotation.

    Positive quantity; use a negative weight. Complements the paid-rate cap
    in roulade_progress: the cap removes the INCENTIVE to rotate faster than
    ~3 rad/s, this adds an explicit COST above omega_max, so "violent" is
    strictly worse than "controlled" rather than merely not-better. A
    controlled full roll (~2–3 rad/s average) never touches it.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_y = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 1], nan=0.0)
    excess = torch.clamp(omega_y.abs() - omega_max, min=0.0)
    return excess.pow(2)


def roulade_flatness_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """(lateral-axis world-z)² — dense gradient toward a sagittal roll.

    Positive quantity; use a negative weight. Zero when standing, zero
    through an arbitrarily deep CLEAN forward roll (pure pitch keeps the
    lateral axis horizontal), up to 1 when tipped fully onto a shoulder.
    The accumulator's flatness gate makes side rolls unprofitable; this term
    adds the per-step gradient that steers back toward the plane.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=0.0).pow(2)


def roulade_sagittal_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Rotation out of the sagittal plane: body-frame ω_x² + ω_z² (positive;
    use a negative weight). ω_y is the roll axis and stays free."""
    asset: Entity = env.scene[asset_cfg.name]
    omega_b = asset.data.root_link_ang_vel_b
    return torch.nan_to_num(omega_b[:, 0].pow(2) + omega_b[:, 2].pow(2), nan=0.0)


def roulade_lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Body-frame lateral (y) linear velocity² — keeps the roll straight."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b[:, 1].pow(2), nan=0.0)
