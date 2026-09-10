"""MicroDuck ground pick terms."""

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from typing import Optional
import math
import torch
from .common import (
    _DEFAULT_ASSET_CFG,
    _servo_default_joint_pos,
    _servo_joint_pos,
    _servo_joint_vel,
)


# ==============================================================================
# Ground Pick Rewards
# ==============================================================================

def mouth_ground_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    std: float = 0.03,
    target_height: float = 0.0,
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward for mouth tip approaching the ground, weighted by the approach phase.

    The command for the ground pick task is [cos(2π*phase), sin(2π*phase), 0].
    The approach phase is the first half-cycle (sin > 0, phase ∈ [0, 0.5]),
    smoothly weighted by max(0, sin(2π*phase)).

    Args:
        std: Gaussian std on mouth_tip height (m). 0.03 m gives strong gradient.
        target_height: Target z-height for the mouth tip (m). 0 = ground level.
    """
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]  # (num_envs,)
    proximity = torch.exp(-((mouth_z - target_height) / std) ** 2)

    # Approach weight: max(0, sin(2π*phase)) — peaks at 1 at phase=0.25, zero at 0 and 0.5
    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * proximity


def mouth_perpendicular_to_ground(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward the mouth tip x-axis being vertical (pointing down) during the approach phase.

    A perfectly perpendicular contact gives alignment=1; horizontal gives 0; pointing up gives -1.
    Weighted by max(0, sin(2π*phase)) so it only applies during the descent.
    """
    asset = env.scene[asset_cfg.name]
    # site_quat_w: (num_envs, num_sites, 4) as [w, x, y, z]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]  # (num_envs, 4)
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # z-component of the site x-axis in world frame (first column of rotation matrix)
    x_axis_z = 2.0 * (qx * qz - w * qy)
    # dot with [0, 0, -1]: 1 = perfectly downward, -1 = upward
    alignment = -x_axis_z

    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * alignment


def sit_grounded(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: Optional[str] = None,
    sin_threshold: float = 0.7,
    min_progress_frac: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    upright_cos_threshold: float = 0.5,
) -> torch.Tensor:
    """Positive reward for trunk-ground contact WHILE upright.

    Gated additionally on the trunk's body-frame +Z axis pointing in roughly the
    world-up direction (cosine >= ``upright_cos_threshold``, default 0.5 → up to
    60° tilt accepted). Without this gate, the policy can earn the contact
    bonus by tipping sideways or face-forward — the trunk hits the ground in
    those weird poses, sit_grounded fires, and the policy converges to a
    "fallen" mode that competes with the actual sit pose.

    When ``command_name`` is provided, the reward is gated to the sit window of
    a phase command. Otherwise it's always-on, optionally gated to the late
    part of the episode via ``min_progress_frac``.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    has_contact = (found > 0).float()

    # Upright check: trunk body's +Z (world frame, third column of rotation matrix
    # derived from the trunk quaternion) dot world-up = trunk's body-up · world-up.
    # Equivalently: 1 - 2*(qx² + qy²) for a unit quaternion (w, x, y, z).
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) = (w, x, y, z)
    qx, qy = quat[:, 1], quat[:, 2]
    upright_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
    is_upright = (upright_cos >= upright_cos_threshold).float()

    contact_upright = has_contact * is_upright

    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * contact_upright
        return contact_upright
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * contact_upright


def sit_stability(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: Optional[str] = None,
    ang_vel_std: float = 0.5,
    sin_threshold: float = 0.7,
    min_progress_frac: float = 0.0,
) -> torch.Tensor:
    """Bonus for low body angular velocity.

    Phase-gated when ``command_name`` is set (sit window of a phase command).
    Always-on otherwise, optionally restricted to the late part of the episode
    via ``min_progress_frac``. Encourages a stable rest pose.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_norm = asset.data.root_link_ang_vel_w.norm(dim=-1)
    stillness = torch.exp(-((ang_vel_norm / ang_vel_std) ** 2))
    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * stillness
        return stillness
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * stillness


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 penalty for joint positions deviating from their default (HOME).

    Returns sum of |joint_pos - default| over the selected joints. Unlike the
    Gaussian `pose` reward (which saturates near 1.0 for any small deviation),
    this gives a *linear* gradient at all deviation magnitudes — useful as a
    focused penalty on a subset of joints (e.g. hip_yaw / hip_roll) to prevent
    them drifting to wide-base stances even when other joints are near HOME.
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    err = asset.data.joint_pos[:, jnt_ids] - asset.data.default_joint_pos[:, jnt_ids]
    return torch.sum(torch.abs(err), dim=-1)


def joint_pos_limit_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    margin: float = 0.15,
) -> torch.Tensor:
    """L1 penalty for joint positions entering a ``margin`` (rad) band next to
    their *hard* range limits.

    The base ``joint_pos_limits`` reward only fires past the *soft* limit
    (global ``soft_joint_pos_limit_factor`` = 0.9 → roughly the last 7.5% of
    range) and only by the radians-overshoot magnitude, so it's near-useless
    against a joint parked on its stop. This term instead reads the *hard*
    limits directly and lets each reward set its own wide margin, scoped to
    specific joints.

    Motivating case: with a low-kp position servo and wide ctrlrange the policy
    can command far past a joint's limit "for free" (no command-side cost) and
    park the joint on its hard stop — e.g. hip_yaw slammed to ±limit so the foot
    slides/pivots. The overshoot is *intended* (it's how a low-kp servo reaches
    its target), so the deterrent must live on the qpos side and bite well
    before the stop.

    For each selected joint with hard limits ``[lo, hi]``::

        soft_lo = lo + margin,  soft_hi = hi - margin
        penalty = relu(soft_lo - q) + relu(q - soft_hi)

    summed over joints: zero in the interior, ramping linearly toward each stop.
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, jnt_ids]
    hard = asset.data.joint_pos_limits[:, jnt_ids]  # (num_envs, num_sel_joints, 2)
    soft_lo = hard[..., 0] + margin
    soft_hi = hard[..., 1] - margin
    below = (soft_lo - q).clip(min=0.0)
    above = (q - soft_hi).clip(min=0.0)
    return torch.sum(below + above, dim=-1)


def phase_height_track(
    env: ManagerBasedRlEnv,
    command_name: str,
    stand_z: float,
    sit_z: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward trunk_z tracking a sin-interpolated target between stand and sit heights.

    Used for the sitstand task instead of joint-angle matching for the sit pose —
    rewards the END STATE (low trunk) without prescribing HOW the robot gets there.
    The policy is free to find any motion strategy (deep squat, head-supported
    descent, etc.).

    Command (from GroundPickPhaseCommand): cmd[:, 1] = sin(2π·phase).
    sin = +1 at phase 0.25 (sit peak) → target = sit_z.
    sin = -1 at phase 0.75 (stand peak) → target = stand_z.
    sin = 0 at transitions → target = midpoint.
    """
    cmd = env.command_manager.get_command(command_name)
    sin_phase = cmd[:, 1]
    target_z = (stand_z + sit_z) * 0.5 - (stand_z - sit_z) * 0.5 * sin_phase
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def interpolated_pose_target_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
    source_overrides: Optional[dict] = None,
    target_overrides: Optional[dict] = None,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """Gaussian on joint positions vs a time-interpolated target pose.

    Tracks a target that linearly interpolates from a source pose to a target
    pose over the episode, between progress fractions ``ramp_start_frac`` and
    ``ramp_end_frac``. Before/after the ramp the target is clamped to source /
    final target respectively.

    The point is to enforce smooth descent: snapping to the final target early
    leaves the robot *off-target* relative to where the interpolated target
    currently is, costing pose reward for the duration of the mismatch.

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Optional subset of joints to evaluate.
        source_overrides: ``{joint_index: angle_rad}`` defining the source pose
            (start of the ramp). ``None`` = default/HOME pose.
        target_overrides: same, for the target pose (end of the ramp).
        ramp_start_frac, ramp_end_frac: episode-progress window in [0, 1] over
            which the target moves from source to target.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return torch.exp(-((joint_pos - interp) / std) ** 2).mean(dim=-1)


def interpolated_pose_l1_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
    source_overrides: Optional[dict] = None,
    target_overrides: Optional[dict] = None,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """L1 distance from a time-interpolated target pose (negative — used as penalty).

    Same interpolation schedule as ``interpolated_pose_target_match`` but
    returns ``-mean(|joint_pos - interp|)`` instead of a Gaussian. The L1
    gradient is constant everywhere — useful as a bootstrap signal when the
    Gaussian variant saturates to zero far from target and leaves the policy
    no gradient to discover the target direction.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return -torch.abs(joint_pos - interp).mean(dim=-1)


def interpolated_height_l1_penalty(
    env: ManagerBasedRlEnv,
    start_height: float,
    end_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """L1 distance from a time-interpolated target height (negative — penalty).

    Same role as ``interpolated_pose_l1_penalty`` but on trunk z. Provides a
    constant gradient toward the target height regardless of how far off the
    current z is, complementing the Gaussian ``interpolated_height_target``.
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_z)


def interpolated_height_target(
    env: ManagerBasedRlEnv,
    start_height: float,
    end_height: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """Gaussian on trunk z vs a time-interpolated target height.

    Companion to ``interpolated_pose_target_match`` — same time-interpolation
    logic applied to the trunk height.
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def bilateral_symmetry_penalty(
    env: ManagerBasedRlEnv,
    left_indices: list,
    right_indices: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 penalty on left/right leg asymmetry.

    For a bilaterally-symmetric robot the leg HOME and any symmetric target
    (FOLD, SIT) satisfy ``q_left + q_right == 0`` on each matched joint pair
    (because the left/right joints use mirrored sign conventions). This term
    penalises departures from that constraint.

    Useful when ``mean()`` of pose-target rewards lets the policy get away
    with one-leg-correct solutions (you collect ~half the reward for free
    and the gradient toward fixing the second leg is too weak to escape that
    local minimum). The penalty here has constant L1 gradient regardless of
    magnitude, so any asymmetry pays a cost and the unique zero is the
    fully-symmetric configuration.

    Returns ``-sum_i |q[left_i] + q[right_i]|`` averaged over the N pairs.
    """
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    left = pos[:, left_indices]
    right = pos[:, right_indices]
    return -torch.abs(left + right).mean(dim=-1)


def _multistage_target_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    waypoints,
) -> torch.Tensor:
    """Compute the time-interpolated joint target across N waypoints.

    waypoints: ordered list of dicts {"frac": float in [0,1],
                                       "overrides": dict[int,float] | None}.
    First waypoint should have frac=0.0 (typically HOME, overrides=None).
    Subsequent waypoints define milestones. Between two waypoints the target
    linearly interpolates. Before the first / after the last it clamps.

    Returns a (num_envs, num_joints) tensor of target joint angles.
    """
    asset = env.scene[asset_cfg.name]
    default = _servo_default_joint_pos(env, asset)

    def build_pose(overrides):
        pose = default.clone()
        if overrides:
            for idx, val in overrides.items():
                pose[:, idx] = val
        return pose

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    # Find which segment we're in (broadcast over envs).
    out = build_pose(waypoints[0]["overrides"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0).unsqueeze(-1)
        prev_pose = build_pose(waypoints[i - 1]["overrides"])
        next_pose = build_pose(waypoints[i]["overrides"])
        seg = prev_pose * (1.0 - tau) + next_pose * tau
        # Take this segment's value when progress is in [f0, f1] or past it.
        mask = (progress >= f0).float().unsqueeze(-1)
        out = torch.where(mask > 0, seg, out)
    return out


def _multistage_target_height(
    env: ManagerBasedRlEnv,
    waypoints,
) -> torch.Tensor:
    """Same logic as _multistage_target_pose but for trunk z height.

    waypoints: [{"frac": float, "height": float}, ...].
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    out = torch.full_like(progress, waypoints[0]["height"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0)
        seg = waypoints[i - 1]["height"] * (1.0 - tau) + waypoints[i]["height"] * tau
        mask = (progress >= f0).float()
        out = torch.where(mask > 0, seg, out)
    return out


def multistage_pose_target_match(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """Multi-waypoint variant of interpolated_pose_target_match.

    waypoints: [{"frac": 0.0, "overrides": None},
                {"frac": 0.4, "overrides": FOLD_OVERRIDES},
                {"frac": 0.7, "overrides": SIT_OVERRIDES}]

    Use this to enforce a curriculum-style trajectory through one or more
    intermediate poses (e.g. stand → fold → sit). Same per-joint Gaussian
    semantics as the single-stage version.
    """
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def multistage_pose_l1_penalty(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """L1 companion to multistage_pose_target_match."""
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def multistage_height_target(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.03,
) -> torch.Tensor:
    """Multi-waypoint Gaussian on trunk z."""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def multistage_height_l1_penalty(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to multistage_height_target."""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_z)


def pose_target_match(
    env: ManagerBasedRlEnv,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """Gaussian pose-match against a single fixed target.

    target = ``default_joint_pos`` with the per-index overrides applied. No
    waypoints, no episode-progress interpolation — the same target is rewarded
    from t=0 to the end of the episode.
    """
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def pose_l1_penalty(
    env: ManagerBasedRlEnv,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """L1 companion to ``pose_target_match`` (constant gradient toward target)."""
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def height_target_gaussian(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.02,
) -> torch.Tensor:
    """Gaussian on trunk z against a single fixed target."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_height) / std) ** 2)


def height_l1_penalty(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to ``height_target_gaussian``."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_height)


def trunk_vertical_accel_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty proportional to ``|a_z|`` of the trunk (finite-diff of v_z).

    Captures hard impacts (large deceleration spike on landing) AND incentivises
    a smooth quasi-static descent (constant velocity → a_z ≈ 0). At rest a_z is
    zero so the seated robot pays no cost.

    State is kept on the env in ``_prev_trunk_vz``; at episode reset the
    accel is zeroed to avoid a transient from the previous episode's final
    state leaking into the new one.
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    prev = getattr(env, "_prev_trunk_vz", None)
    if prev is None or prev.shape[0] != vz.shape[0]:
        prev = vz.detach().clone()
    a_z = (vz - prev) / env.step_dt
    # Zero out a_z at reset steps to suppress the cross-episode transient.
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        a_z = torch.where(reset_mask, torch.zeros_like(a_z), a_z)
    env._prev_trunk_vz = vz.detach().clone()
    return -torch.abs(a_z)


def trunk_downward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_down_vel: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty on downward trunk velocity beyond ``max_down_vel``.

    Caps descent SPEED, which ``trunk_vertical_accel_penalty`` alone cannot:
    a fast constant-velocity drop has a_z ≈ 0 the whole way down and pays only
    one impact spike at the bottom — cheap relative to arriving at the target
    pose sooner. This term makes every step of a too-fast descent cost reward,
    so the gentlest descent that stays under the cap is optimal. Zero at rest
    and for any motion slower than the cap (including all upward motion).
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(-vz - max_down_vel, min=0.0)


def seated_stillness(
    env: ManagerBasedRlEnv,
    height_full: float = 0.06,
    height_zero: float = 0.08,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward trunk stillness while seated UPRIGHT: |v| Gaussian, z- and tilt-gated.

    exp(-(|v|/vel_std)²) · smoothstep(z) · smoothstep(tilt). The z gate is full
    below ``height_full`` and zero above ``height_zero`` (inactive during the
    descent). The tilt gate is full below ``tilt_full_deg`` and zero above
    ``tilt_zero_deg`` — WITHOUT it, "lie still on your back" scores as well as
    "sit still upright" (the trunk on its back is inside the seated z band and
    perfectly motionless), which is exactly the exploit run 2 converged to.
    Makes "rest quietly, upright, at the seated height" the only rewarded rest.
    """
    asset = env.scene[asset_cfg.name]
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((height_zero - z) / max(height_zero - height_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)
    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate


def upright_while_tall(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear upright reward weighted by a smoothstep on trunk z.

    Returns ``body_upright_linear * smoothstep((z - low)/(high - low))`` so the
    upright incentive is full while the robot is still standing tall, and
    fades to zero once it has committed to the lower sit configuration (where
    butt-on-ground orientation is fine). Prevents the policy from learning to
    tip backward while still high (which would otherwise farm the descent
    reward via a controlled fall).
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright * smooth


def phase_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """Blend 0..1 le long de la phase [0,1) — 0 = pose STAND, 1 = pose DOWN.

    [0, descent_end)       : 0 -> 1  (se baisser)
    [descent_end, hold_end): 1       (bas)
    [hold_end, rise_end)   : 1 -> 0  (se lever)
    [rise_end, 1.0)        : 0       (haut / repos)
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def kick_pose_target(
    phase: torch.Tensor,
    stand: torch.Tensor,
    back: torch.Tensor,
    forward: torch.Tensor,
    windup_end: float,
    kick_end: float,
    return_end: float,
) -> torch.Tensor:
    """Cible articulaire interpolée d'un geste de shoot à 4 keyframes.

    phase (B,) ∈ [0,1). stand/back/forward (k,) ou (1,k). Retour (B,k).

    [0, windup_end)        STAND   -> BACK     (armement)
    [windup_end, kick_end) BACK    -> FORWARD  (frappe sèche)
    [kick_end, return_end) FORWARD -> STAND    (retour)
    [return_end, 1.0)      STAND             (repos)
    """
    p = phase.unsqueeze(-1)  # (B,1)

    def interp(a, b, s):
        return a + s * (b - a)

    s1 = (p / windup_end).clamp(0.0, 1.0)
    s2 = ((p - windup_end) / (kick_end - windup_end)).clamp(0.0, 1.0)
    s3 = ((p - kick_end) / (return_end - kick_end)).clamp(0.0, 1.0)

    seg1 = interp(stand, back, s1)
    seg2 = interp(back, forward, s2)
    seg3 = interp(forward, stand, s3)  # à s3=1 (phase>=return_end) => STAND

    out = seg1
    out = torch.where(p >= windup_end, seg2, out)
    out = torch.where(p >= kick_end, seg3, out)
    return out


def _kick_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_pose: dict,
    back_pose: dict,
    forward_pose: dict,
    windup_end: float,
    kick_end: float,
    return_end: float,
    joint_names: Optional[list] = None,
):
    """(cur, target) pour le geste de shoot, joints résolus PAR NOM.

    Les 3 poses partagent les mêmes clés (14 joints). L'ordre des noms est
    donné par `stand_pose` (ou par `joint_names` si fourni — un sous-ensemble
    des clés, ex. jambe droite + cou d'un côté, jambe gauche de l'autre, pour
    appliquer des std différents au geste vs à la jambe d'appui).
    """
    if not stand_pose:
        raise ValueError("_kick_pose_error requires a non-empty stand_pose dict")
    asset: Entity = env.scene[asset_cfg.name]
    names = list(joint_names) if joint_names is not None else list(stand_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]

    def vec(d):
        return torch.tensor([d[n] for n in names], device=env.device,
                            dtype=asset.data.joint_pos.dtype)

    stand_v, back_v, fwd_v = vec(stand_pose), vec(back_pose), vec(forward_pose)

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    target = kick_pose_target(phase, stand_v, back_v, fwd_v,
                              windup_end, kick_end, return_end)          # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def kick_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: Optional[dict] = None,
    back_pose: Optional[dict] = None,
    forward_pose: Optional[dict] = None,
    std: float = 0.4,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_names: Optional[list] = None,
) -> torch.Tensor:
    """Gaussienne sur la pose articulaire vs cible interpolée du shoot.

    Reward directif et symétrique : chaque phase impose la config articulaire
    exacte. Résolution PAR NOM. `joint_names` restreint l'évaluation à un
    sous-ensemble (ex. jambe droite + cou tracés serré, jambe gauche d'appui
    tracée lâche pour la laisser équilibrer).
    """
    cur, target = _kick_pose_error(
        env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
        forward_pose or {}, windup_end, kick_end, return_end, joint_names,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def kick_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: Optional[dict] = None,
    back_pose: Optional[dict] = None,
    forward_pose: Optional[dict] = None,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_names: Optional[list] = None,
) -> torch.Tensor:
    """Bootstrap L1 vers la cible interpolée (gradient constant, pénalité<=0)."""
    cur, target = _kick_pose_error(
        env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
        forward_pose or {}, windup_end, kick_end, return_end, joint_names,
    )
    return -(cur - target).abs().mean(dim=-1)


def kick_engagement(
    phase: torch.Tensor,
    windup_end: float,
    return_end: float,
) -> torch.Tensor:
    """Gate d'engagement du geste ∈ [0,1] (pur) — pour pondérer les rewards
    d'équilibre unipède qui ne doivent s'appliquer que hors du repos STAND.

    [0, windup_end)        : 0 -> 1  (montée pendant l'armement)
    [windup_end, return_end): 1       (phase de frappe = appui unipède attendu)
    [return_end, 1.0)      : 0        (repos STAND, appui bipède, CoM centré OK)
    """
    g = torch.zeros_like(phase)
    ramp = phase < windup_end
    g = torch.where(ramp, phase / windup_end, g)
    hold = (phase >= windup_end) & (phase < return_end)
    g = torch.where(hold, torch.ones_like(phase), g)
    return g


def com_over_support_foot(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    std: float = 0.04,
    windup_end: float = 0.35,
    return_end: float = 0.75,
) -> torch.Tensor:
    """Reward gaussien : projection horizontale du CoM proche du pied d'appui,
    gaté sur la phase de frappe (kick_engagement).

    Apprend le transfert latéral du poids sur le pied d'appui (support). Sans
    ça, un geste à un pied issu de poses relevées en appui bipède garde le CoM
    centré entre les deux pieds → bascule et chute dès que l'autre pied se lève.
    Au repos STAND le gate est 0 (appui bipède, CoM centré autorisé).

    `asset_cfg` doit cibler le site du pied d'appui (ex. site_names=["left_foot"]).
    `std` en mètres (rayon de tolérance CoM↔pied, ~taille du pied).
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_xy = asset.data.root_com_pos_w[:, :2]
    foot_id = asset_cfg.site_ids[0]
    foot_xy = asset.data.site_pos_w[:, foot_id, :2]
    dist2 = ((com_xy - foot_xy) ** 2).sum(dim=-1)
    reward = torch.exp(-dist2 / (std ** 2))

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = kick_engagement(phase, windup_end, return_end)
    return gate * reward


def _phase_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    target_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    source_pose: Optional[dict] = None,
):
    """(cur, target) pour la pose interpolée par la phase, résolue PAR NOM.

    Cible = source + blend(phase)·(target_pose - source), source = STAND
    (`source_pose` si fourni, sinon le DEFAULT/HOME du modèle). blend ∈ [0,1]
    (0 = STAND, 1 = target_pose) via `phase_pose_blend`.
    """
    if not target_pose:
        raise ValueError("_phase_pose_error requires a non-empty target_pose dict")

    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = phase_pose_blend(phase, descent_end, hold_end, rise_end)     # (B,)

    names = list(target_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]                       # (B,k)

    source = default.clone()
    if source_pose:
        for j, n in enumerate(names):
            if n in source_pose:
                source[:, j] = source_pose[n]
    target_vec = torch.tensor(
        [target_pose[n] for n in names], device=env.device, dtype=default.dtype
    ).unsqueeze(0)                                                       # (1,k)

    target = source + blend.unsqueeze(-1) * (target_vec - source)        # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def phase_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    std: float = 0.3,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussienne sur la pose articulaire vs cible interpolée STAND<->DOWN.

    Reward directif : indique la config articulaire exacte à chaque phase. Se
    relever (cible → STAND) est récompensé exactement comme se baisser (cible →
    DOWN) — symétrique par construction. Résolution PAR NOM.
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def phase_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bootstrap L1 vers la cible interpolée (pénalité négative).

    Gradient constant partout — donne une direction vers la cible même quand la
    gaussienne ci-dessus a saturé à ~0 loin de la cible.
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return -(cur - target).abs().mean(dim=-1)


def phase_pose_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    target_overrides: Optional[dict] = None,
    phase: str = "approach",
) -> torch.Tensor:
    """Reward matching a target pose, weighted by phase-cycle command.

    Generic helper for phase-conditioned tasks (e.g. sit/stand). The command
    encodes phase as [cos(2π·phase), sin(2π·phase), 0]:
      - "approach" weight = max(0, sin(2π·phase)) — peaks at phase 0.25.
      - "return"   weight = max(0,-sin(2π·phase)) — peaks at phase 0.75.

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Optional subset of joints to evaluate (rest ignored).
        target_overrides: {joint_index: angle_rad}. Joints not listed default
            to asset.data.default_joint_pos (the home/standing pose).
        phase: "approach" or "return".
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    pose_reward = torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)

    cmd = env.command_manager.get_command(command_name)
    if phase == "approach":
        weight = torch.clamp(cmd[:, 1], min=0.0)
    else:
        weight = torch.clamp(-cmd[:, 1], min=0.0)
    return weight * pose_reward


def ground_pick_return_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """Reward for returning to the standing pose after ground pick, weighted by the return phase.

    The return phase is the second half-cycle (sin < 0, phase ∈ [0.5, 1.0]),
    smoothly weighted by max(0, -sin(2π*phase)).

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Subset of joints to evaluate. Use to apply different stds
            to leg joints vs neck/head joints (call this reward twice).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos  = _servo_joint_pos(env, asset)        # (num_envs, n_servo_joints)
    default_pos = _servo_default_joint_pos(env, asset)

    if joint_indices is not None:
        joint_pos   = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]

    pose_reward = torch.exp(-((joint_pos - default_pos) / std) ** 2).mean(dim=-1)

    # Return weight: max(0, -sin(2π*phase)) — peaks at 1 at phase=0.75, zero at 0.5 and 1
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)

    return return_weight * pose_reward


def ground_pick_return_upright(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.4,
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward trunk verticality, weighted by the RETURN phase (stand-up aid).

    Same return weighting as ``ground_pick_return_pose`` (``max(0, -sin(2π·phase))``)
    so it only rewards being upright during the stand-up, never fighting the
    forward lean of the approach. Verticality = ``exp(-tilt²/std²)`` with the same
    tilt proxy as ``body_upright_gaussian`` (``2*(qx²+qy²) ≈ 1-cos(tilt)``). A broad
    std (0.4 rad ≈ 23°) gives gradient even from a fairly tilted crouch.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)  # qx² + qy²
    upright = torch.exp(-tilt_sq / (std * std))
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)
    return return_weight * upright


# --------------------------------------------------------------------------- #
# Ground-pick : gating de phase SEGMENTÉ (durées descente/palier/remontée/repos #
# indépendantes, au lieu de la pondération sinusoïdale max(0,±sin)).            #
#   down-gate  = phase_pose_blend(phase, descent_end, hold_end, rise_end)       #
#               0 (haut) -> 1 (descente) -> 1 (palier bas) -> 0 (remontée/repos) #
#   up-gate    = phase_rise_gate(phase, hold_end, rise_end)                      #
#               0 avant la remontée -> 0..1 (remontée) -> 1 (repos debout)       #
# --------------------------------------------------------------------------- #
def phase_rise_gate(
    phase: torch.Tensor, hold_end: float, rise_end: float
) -> torch.Tensor:
    """Gate montante pour le RETOUR : 0 avant hold_end, 0->1 sur [hold_end,
    rise_end), 1 après (repos debout)."""
    g = torch.zeros_like(phase)
    rising = (phase >= hold_end) & (phase < rise_end)
    g = torch.where(rising, (phase - hold_end) / (rise_end - hold_end), g)
    g = torch.where(phase >= rise_end, torch.ones_like(phase), g)
    return g


def _gp_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def mouth_ground_proximity_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    std: float = 0.10,
    target_height: float = 0.0,
    command_name: str = "twist",
    descent_end: float = 0.25,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """mouth_ground_proximity gaté par la down-gate segmentée (descente+palier)."""
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]
    proximity = torch.exp(-((mouth_z - target_height) / std) ** 2)
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * proximity


def mouth_perpendicular_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    command_name: str = "twist",
    descent_end: float = 0.25,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """mouth_perpendicular_to_ground gaté par la down-gate segmentée."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)
    alignment = -x_axis_z  # 1 = bouche pointe droit vers le bas
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * alignment


def ground_pick_return_pose_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """ground_pick_return_pose gaté par la up-gate segmentée (remontée+repos)."""
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    default_pos = _servo_default_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]
    pose_reward = torch.exp(-((joint_pos - default_pos) / std) ** 2).mean(dim=-1)
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * pose_reward


def ground_pick_return_upright_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.4,
    command_name: str = "twist",
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """ground_pick_return_upright gaté par la up-gate segmentée."""
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright = torch.exp(-tilt_sq / (std * std))
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * upright


def neck_vel_descent_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    hold_end: float = 0.35,
) -> torch.Tensor:
    """Pénalise la vitesse des joints du cou pendant la DESCENTE+palier (freine le
    piqué de la tête).

    Coût = mean(joint_vel²) sur les joints donnés, gaté à 1 pour phase < hold_end
    (descente + palier bas) et 0 ensuite (remontée + repos) -> ne gêne PAS le
    relever du cou. Retourne un coût positif ; à utiliser avec un poids négatif.
    """
    asset = env.scene[asset_cfg.name]
    vel = _servo_joint_vel(env, asset)
    if joint_indices is not None:
        vel = vel[:, joint_indices]
    cost = (vel ** 2).mean(dim=-1)
    phase = _gp_phase(env, command_name)
    gate = (phase < hold_end).to(vel.dtype)  # descente + palier bas uniquement
    return gate * cost


def sample_mouth_payload(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    min_kg: float = 0.01,
    max_kg: float = 0.04,
) -> None:
    """Event de reset : tire une masse d'objet 'tenu dans la bouche' par env (kg),
    stockée sur env._mouth_payload_kg. Utilisée par apply_mouth_payload_force."""
    buf = getattr(env, "_mouth_payload_kg", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device)
        env._mouth_payload_kg = buf
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    buf[env_ids] = torch.rand(len(env_ids), device=env.device) * (max_kg - min_kg) + min_kg


def apply_mouth_payload_force(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["jaw_soft"], site_names=["mouth_tip"]
    ),
    command_name: str = "twist",
    hold_end: float = 0.35,
    ramp: float = 0.05,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Hook par-step (utilisé comme reward de poids 0) : applique le POIDS de
    l'objet tenu dans la bouche comme force externe verticale au mouth_tip, gaté
    sur la remontée (phase >= hold_end, rampe rapide au moment du 'grab').

    Émule une masse ponctuelle au bout de la bouche pendant le relever : la force
    m·g est appliquée au CoM du corps + le couple (p_mouth - p_com) × F, ce qui
    équivaut à l'appliquer au mouth_tip (bon bras de levier pour le cou). Retourne
    0 (ce n'est pas une vraie récompense — juste le hook d'application)."""
    asset: Entity = env.scene[asset_cfg.name]
    payload = getattr(env, "_mouth_payload_kg", None)
    if payload is None:
        return torch.zeros(env.num_envs, device=env.device)
    phase = _gp_phase(env, command_name)
    gate = ((phase - hold_end) / ramp).clamp(0.0, 1.0)  # 0 avant grab -> 1 après
    fz = -(gate * payload) * gravity                     # (N,) force verticale (bas)

    bid = int(asset_cfg.body_ids[0])
    sid = int(asset_cfg.site_ids[0])
    p_mouth = asset.data.site_pos_w[:, sid, :]           # (N,3)
    p_com = asset.data.body_com_pos_w[:, bid, :]         # (N,3)
    F = torch.zeros((env.num_envs, 3), device=env.device, dtype=p_mouth.dtype)
    F[:, 2] = fz
    tau = torch.cross(p_mouth - p_com, F, dim=-1)        # applique F au mouth_tip
    asset.write_external_wrench_to_sim(
        forces=F.unsqueeze(1), torques=tau.unsqueeze(1), body_ids=[bid],
    )
    return torch.zeros(env.num_envs, device=env.device)
