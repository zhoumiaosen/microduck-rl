"""MicroDuck pose terms."""

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi
import math
import torch
from .common import (
    _DEFAULT_ASSET_CFG,
    _NECK_JOINT_PATTERNS,
)


def head_pose_tracking(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    std: float = 0.5,
    fine_std: float | None = None,
    fine_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Per-joint Gaussian reward for matching commanded neck/head deltas.

    Mean over the 4 neck/head joints of exp(-(err/std)^2). Result is (N,) in
    [0, 1]. Mean form (vs sum-of-squares) keeps gradient alive when only one
    joint is off — vs SOS where a single big error kills the whole reward.

    `std` is the per-joint tolerance: at err=std the per-joint reward is 1/e
    (~0.37). Pick std on the order of the command range so the gradient
    doesn't die as the curriculum widens.

    `fine_std` (optional) blends in a second, narrow Gaussian:
    (1-fine_weight)·exp(-(err/std)²) + fine_weight·exp(-(err/fine_std)²).
    Rationale: a single wide std (0.5 rad ≈ 29°) makes small errors nearly
    free — a 10° gravity sag on the heavy head costs ~0.03 reward, so the
    policy lets it droop. The narrow component (~0.1 rad) prices those small
    errors while the wide one keeps gradient alive at far commands during
    curriculum widening.

    cmd has shape (N, 4) = deltas from default joint positions in the order
    [neck_pitch, head_pitch, head_yaw, head_roll].

    On backlash models the measured angle is qpos[servo] + qpos[backlash] —
    the OUTPUT link, which is also what the encoder obs
    (joint_pos_rel_backlash) reports. Measuring the servo alone would let the
    head droop the backlash play reward-free AND penalize the policy for
    compensating it (servo biased up = servo-side "error"). On models without
    passive_*_backlash joints the mask is 0 and this reduces to the servo.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

    if not hasattr(env, "_head_pose_neck_ids"):
        ids, names = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
        env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        name_to_id = {n: i for i, n in enumerate(asset.joint_names)}
        bl = [name_to_id.get(f"passive_{n}_backlash") for n in names]
        env._head_pose_bl_ids = torch.tensor(
            [0 if b is None else b for b in bl], device=env.device, dtype=torch.long
        )
        env._head_pose_bl_mask = torch.tensor(
            [0.0 if b is None else 1.0 for b in bl], device=env.device
        )

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = (
        joint_pos[:, neck_ids]
        + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    )
    actual = measured - asset.data.default_joint_pos[:, neck_ids]
    err = actual - cmd
    per_joint = torch.exp(-(err / std) ** 2)
    if fine_std is not None:
        per_joint = (1.0 - fine_weight) * per_joint + fine_weight * torch.exp(
            -(err / fine_std) ** 2
        )
    return per_joint.mean(dim=-1)


def head_pose_bias_penalty(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    tau_s: float = 1.0,
    gate_height_low: float | None = None,
    gate_height_high: float = 0.11,
    gate_tilt_full_deg: float = 20.0,
    gate_tilt_zero_deg: float = 45.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the time-averaged (DC) neck/head tracking error: -mean(|EMA(err)|).

    Companion to ``head_pose_tracking``, which scores the INSTANTANEOUS error.
    Why a separate DC term instead of just tightening that Gaussian's std:
    walking unavoidably shakes a head that is 38% of the robot's mass, so an
    instantaneous tight-tolerance term is a permanent tax on walking that no
    policy can escape — measured at ~0.77/step against an air_time reward of
    ~1.01/step, which is exactly what made velocity run 2026-08-20 abandon
    stepping altogether (wandb 5yay13u4). The steady-state droop IS escapable:
    the policy can bias its neck command up to cancel gravity sag. Averaging
    over ``tau_s`` lets the oscillation cancel and prices only the bias.

    L1 (not Gaussian) on purpose: the gradient stays constant at large bias,
    where a tight Gaussian would be flat and dead.

    On backlash models the measured angle reads through the play, matching
    head_pose_tracking and the encoder obs.

    ``gate_height_low`` (optional): upright gate for recovery envs (standup /
    velstand), same smoothstep shape and semantics as body_ang_vel_at_height —
    zero below gate_height_low or above gate_tilt_zero_deg tilt, full above
    gate_height_high and below gate_tilt_full_deg. The gate multiplies the
    ERROR feeding the EMA (not just the output): while fallen/rising the EMA
    sees zero and decays, so arriving upright starts the bias clock from ~0
    instead of charging the whole ground phase's accumulated error at the
    finish line — that would be a reward wall right before recovery completes,
    the exact failure mode of the retired head_impact_penalty. The output is
    gated too, so a fresh fall stops the charge immediately.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

    if not hasattr(env, "_head_pose_neck_ids"):
        # Share the id cache with head_pose_tracking (either may run first).
        head_pose_tracking(env, command_name=command_name, asset_cfg=asset_cfg)

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = (
        joint_pos[:, neck_ids]
        + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    )
    err = (measured - asset.data.default_joint_pos[:, neck_ids]) - cmd

    if gate_height_low is not None:
        z = torch.nan_to_num(
            asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
            nan=0.0,
        )
        t = torch.clamp(
            (z - gate_height_low) / max(gate_height_high - gate_height_low, 1e-6),
            0.0, 1.0,
        )
        gate = t * t * (3.0 - 2.0 * t)
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        st = torch.clamp(
            (gate_tilt_zero_deg - tilt_deg)
            / max(gate_tilt_zero_deg - gate_tilt_full_deg, 1e-6),
            0.0, 1.0,
        )
        gate = gate * (st * st * (3.0 - 2.0 * st))
        err = err * gate.unsqueeze(-1)
    else:
        gate = None

    if not hasattr(env, "_head_bias_ema"):
        env._head_bias_ema = torch.zeros_like(err)
    # Freshly reset envs: drop the previous episode's accumulated bias.
    fresh = env.episode_length_buf <= 1
    env._head_bias_ema[fresh] = 0.0

    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._head_bias_ema = (1.0 - alpha) * env._head_bias_ema + alpha * err
    out = -env._head_bias_ema.abs().mean(dim=-1)
    if gate is not None:
        out = out * gate
    return out


def body_pose_tracking_6d(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.095,
    xy_std: float = 0.02,
    z_std: float = 0.01,
    angle_std: float = math.radians(8),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean of 6 per-axis Gaussian rewards for tracking commanded body pose.

    cmd has shape (N, 6) = [x, y, z, roll, pitch, yaw] all as deltas from the
    nominal standing pose (xy delta from spawn origin, z delta from
    nominal_height, angles delta from upright = 0).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 6)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    # Position relative to env spawn origin. nan_to_num because MuJoCo can
    # produce NaN on contact instability and we don't want to taint the reward.
    pos_w = asset.data.root_link_pos_w
    origin = env.scene.terrain.env_origins
    rel = torch.nan_to_num(pos_w - origin, nan=0.0)
    x_err = rel[:, 0] - dx
    y_err = rel[:, 1] - dy
    z_err = rel[:, 2] - (nominal_height + dz)

    # ZYX Euler from quat.
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll  = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw   = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    roll_err  = roll  - droll
    pitch_err = pitch - dpitch
    yaw_err   = wrap_to_pi(yaw - dyaw)

    r_x = torch.exp(-(x_err / xy_std) ** 2)
    r_y = torch.exp(-(y_err / xy_std) ** 2)
    r_z = torch.exp(-(z_err / z_std) ** 2)
    r_r = torch.exp(-(roll_err  / angle_std) ** 2)
    r_p = torch.exp(-(pitch_err / angle_std) ** 2)
    r_w = torch.exp(-(yaw_err   / angle_std) ** 2)

    return (r_x + r_y + r_z + r_r + r_p + r_w) / 6.0


def body_pose_tracking_locomotion(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.105,
    xy_std: float = 0.02,
    z_std: float = 0.03,
    angle_std: float = math.radians(30),
    axis_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    vel_gate_command_name: str | None = None,
    vel_gate_std: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
) -> torch.Tensor:
    """Locomotion-aware 6D body pose tracking.

    Same shape as body_pose_tracking_6d (6D cmd, mean of 6 Gaussians), but
    x/y/yaw are measured *relative to the feet support polygon*, not the spawn
    origin. This makes the reward meaningful while the robot walks (or stands):

      x, y  : trunk position − feet-centroid, rotated into trunk body frame.
              dx = +0.02 means "lean trunk 2 cm forward of foot centroid."
      z     : trunk world height (− nominal_height) — locomotion-neutral.
      roll  : trunk world roll                     — locomotion-neutral.
      pitch : trunk world pitch                    — locomotion-neutral.
      yaw   : trunk world yaw − circular-mean(feet site yaws). dyaw = +0.3 rad
              means "twist the trunk 17° relative to where the feet point."

    The body_pose_tracking_6d reward measures x/y/yaw vs spawn origin / world
    yaw, which kills the gradient as soon as the robot translates or turns. This
    version stays meaningful regardless of where in the world the robot is.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 6)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    pos_w = asset.data.root_link_pos_w
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    trunk_yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    roll  = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

    # Feet centroid in world frame.
    foot_pos = asset.data.site_pos_w[:, feet_cfg.site_ids]   # (N, 2, 3)
    foot_quat = asset.data.site_quat_w[:, feet_cfg.site_ids] # (N, 2, 4)
    feet_centroid = foot_pos.mean(dim=1)                     # (N, 3)

    # Trunk xy in body frame relative to feet centroid (rotate world Δxy by −yaw).
    dx_w = pos_w[:, 0] - feet_centroid[:, 0]
    dy_w = pos_w[:, 1] - feet_centroid[:, 1]
    cos_y = torch.cos(trunk_yaw)
    sin_y = torch.sin(trunk_yaw)
    x_body =  cos_y * dx_w + sin_y * dy_w
    y_body = -sin_y * dx_w + cos_y * dy_w

    # Z relative to spawn-origin terrain height (still in world).
    origin = env.scene.terrain.env_origins
    z_world = torch.nan_to_num(pos_w[:, 2] - origin[:, 2], nan=0.0)

    # Feet yaws → circular mean. NOTE: this depends on the site orientation
    # matching the foot pointing direction; if the site frame is rotated, this
    # yaw reference may have an offset (constant per-env, so dyaw=0 still maps
    # to "feet-aligned").
    fqw, fqx, fqy, fqz = foot_quat[..., 0], foot_quat[..., 1], foot_quat[..., 2], foot_quat[..., 3]
    foot_yaws = torch.atan2(2.0 * (fqw * fqz + fqx * fqy), 1.0 - 2.0 * (fqy * fqy + fqz * fqz))  # (N, 2)
    mean_foot_yaw = torch.atan2(torch.sin(foot_yaws).mean(dim=1), torch.cos(foot_yaws).mean(dim=1))

    x_err     = x_body - dx
    y_err     = y_body - dy
    z_err     = z_world - (nominal_height + dz)
    roll_err  = roll  - droll
    pitch_err = pitch - dpitch
    yaw_err   = wrap_to_pi(trunk_yaw - mean_foot_yaw - dyaw)

    r_x = torch.exp(-(x_err / xy_std) ** 2)
    r_y = torch.exp(-(y_err / xy_std) ** 2)
    r_z = torch.exp(-(z_err / z_std) ** 2)
    r_r = torch.exp(-(roll_err  / angle_std) ** 2)
    r_p = torch.exp(-(pitch_err / angle_std) ** 2)
    r_w = torch.exp(-(yaw_err   / angle_std) ** 2)

    # Per-axis weighted mean. Pass axis_weights=(0,0,1,1,1,1) to disable xy
    # tracking — useful when xy lean is mechanically coupled to pitch/roll on
    # the robot, making independent xy commands a noise source rather than a
    # learnable objective.
    wx, wy, wz, wr, wp, wyaw = axis_weights
    total_w = wx + wy + wz + wr + wp + wyaw
    reward = (wx*r_x + wy*r_y + wz*r_z + wr*r_r + wp*r_p + wyaw*r_w) / max(total_w, 1e-6)

    # Optional gate: when vel_gate_command_name is set, scale the reward by a
    # Gaussian on the velocity command's magnitude. With vel_gate_std ≈ 0.1,
    # the gate is ~1 when commanded velocity is 0 and decays to ~exp(-9)≈0
    # by |vel_cmd|≥0.3 — body tracking only meaningfully contributes when the
    # robot is supposed to be standing still. Avoids the tracking vs walking
    # conflict that prevented the previous run from learning either well.
    if vel_gate_command_name is not None:
        # Gate on commanded LINEAR velocity only (xy) — turning in place still
        # leaves body pose meaningful, but walking forward/sideways doesn't.
        vel_cmd = env.command_manager.get_command(vel_gate_command_name)  # (N, 3)
        vel_mag = torch.linalg.vector_norm(vel_cmd[:, :2], dim=-1)
        gate = torch.exp(-(vel_mag / vel_gate_std) ** 2)
        reward = reward * gate

    return reward
