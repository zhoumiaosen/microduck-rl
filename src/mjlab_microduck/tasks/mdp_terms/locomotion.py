"""MicroDuck locomotion terms."""

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from typing import Optional
import torch
from .common import (
    _DEFAULT_ASSET_CFG,
    _servo_joint_vel,
)


def reset_with_forward_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float] = (0.3, 0.8),
    fraction_stages: list[dict] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Warm-start a fraction of reset environments with a random forward velocity.

    The robot spawns already moving in its body-forward direction, so it first
    discovers what coasting at speed feels like. The fraction decreases over
    training, forcing it to progressively earn that speed from rest.

    Args:
        velocity_range: (min, max) forward speed in m/s.
        fraction_stages: list of {"step": int, "fraction": float} dicts, sorted by step.
            The fraction active at the current training step is used.
            Example: [{"step":0,"fraction":0.8}, {"step":2000*24,"fraction":0.0}]
        asset_cfg: robot entity config.
    """
    if fraction_stages is None:
        fraction_stages = [{"step": 0, "fraction": 0.8}]

    # Determine current fraction from training step
    step = env.common_step_counter
    fraction = fraction_stages[0]["fraction"]
    for stage in fraction_stages:
        if step >= stage["step"]:
            fraction = stage["fraction"]

    if len(env_ids) == 0 or fraction <= 0.0:
        return

    n_warmstart = max(1, int(len(env_ids) * fraction))
    perm = torch.randperm(len(env_ids), device=env.device)[:n_warmstart]
    warmstart_ids = env_ids[perm]

    lo, hi = velocity_range
    vx = lo + torch.rand(n_warmstart, device=env.device) * (hi - lo)

    # Build horizontal forward direction from yaw only — ignoring pitch/roll.
    # IMPORTANT: read quaternion from qpos, NOT from root_link_quat_w.
    # root_link_quat_w reads xquat which requires sim.forward() to be current.
    # After reset_base writes a new yaw to qpos, xquat is still stale (old episode).
    # qpos is updated immediately by write_root_pose, so it's always fresh.
    asset: Entity = env.scene[asset_cfg.name]
    qpos_q_adr = asset.data.indexing.free_joint_q_adr[3:7]  # quat indices in qpos
    q = asset.data.data.qpos[warmstart_ids][:, qpos_q_adr]  # (n, 4) [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward_world = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)

    velocities = torch.zeros(n_warmstart, 6, device=env.device)
    velocities[:, :3] = vx.unsqueeze(-1) * forward_world

    asset.write_root_link_velocity_to_sim(velocities, env_ids=warmstart_ids)

    # Spin wheels to match forward velocity — prevents instantaneous no-slip braking.
    # Wheel radius = 0.0175 m (measured).
    # All 4 wheels spin at +ω for forward motion (verified by test_wheel_direction.py).
    _WHEEL_RADIUS = 0.0175
    all_wheel_ids, _ = asset.find_joints(r"^passive_.*")

    if all_wheel_ids:
        joint_pos = asset.data.joint_pos[warmstart_ids].clone()
        joint_vel = asset.data.joint_vel[warmstart_ids].clone()
        omega = vx / _WHEEL_RADIUS  # (n,) rad/s, positive = forward
        joint_vel[:, all_wheel_ids] = omega.unsqueeze(-1).expand(-1, len(all_wheel_ids))
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=warmstart_ids)


def reset_action_history(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """
    Reset cached action history for environments that are being reset.
    This is critical for action rate and acceleration penalty terms.

    This function should be called in the post_reset callback or at episode termination.

    Args:
        env: The environment
        env_ids: Indices of environments being reset
        asset_cfg: Asset configuration
    """
    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]

    # Reset leg action rate cache
    if hasattr(env, '_prev_leg_actions'):
        # Set to current action (or zero if no action yet)
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            env._prev_leg_actions[env_ids] = env.action_manager.action[env_ids][:, leg_joint_indices]
        else:
            env._prev_leg_actions[env_ids] = 0.0

    # Reset neck action rate cache
    if hasattr(env, '_prev_neck_actions'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            env._prev_neck_actions[env_ids] = env.action_manager.action[env_ids][:, neck_joint_indices]
        else:
            env._prev_neck_actions[env_ids] = 0.0

    # Reset leg action acceleration cache
    if hasattr(env, '_prev_leg_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            current_action = env.action_manager.action[env_ids][:, leg_joint_indices]
            env._prev_leg_actions_for_acc[env_ids] = current_action
            env._prev_prev_leg_actions_for_acc[env_ids] = current_action
        else:
            env._prev_leg_actions_for_acc[env_ids] = 0.0
            env._prev_prev_leg_actions_for_acc[env_ids] = 0.0

    # Reset neck action acceleration cache
    if hasattr(env, '_prev_neck_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            current_action = env.action_manager.action[env_ids][:, neck_joint_indices]
            env._prev_neck_actions_for_acc[env_ids] = current_action
            env._prev_prev_neck_actions_for_acc[env_ids] = current_action
        else:
            env._prev_neck_actions_for_acc[env_ids] = 0.0
            env._prev_prev_neck_actions_for_acc[env_ids] = 0.0

    # Reset joint velocity cache for joint accelerations
    if hasattr(asset.data, '_prev_joint_vel'):
        # Get current joint velocities for reset environments
        joint_vel = asset.data.joint_vel[env_ids, :][:, asset_cfg.joint_ids]
        asset.data._prev_joint_vel[env_ids] = joint_vel

    # Reset contact frequency tracking
    if hasattr(env, '_contact_change_count'):
        env._contact_change_count[env_ids] = 0.0
    if hasattr(env, '_contact_change_timer'):
        env._contact_change_timer[env_ids] = 0.0
    if hasattr(env, '_prev_contacts_for_freq'):
        if "feet_ground_contact" in env.scene.sensors:
            contacts = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2]
            env._prev_contacts_for_freq[env_ids] = contacts

    # Reset foot force smoothness tracking
    if hasattr(env, '_prev_foot_forces'):
        if "feet_ground_contact" in env.scene.sensors:
            forces = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2].squeeze(-1)
            env._prev_foot_forces[env_ids] = forces

    # Reset actuator torque rate tracking
    if hasattr(env, '_prev_actuator_forces'):
        env._prev_actuator_forces[env_ids] = asset.data.actuator_force[env_ids].clone()


def joint_accelerations_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize joint accelerations using L2 squared norm.
    Joint accelerations are computed using finite differences of joint velocities.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared joint accelerations
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get current joint velocities
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Get previous joint velocities (stored in asset data)
    # Note: This assumes the environment stores previous joint velocities
    if not hasattr(asset.data, '_prev_joint_vel'):
        # Initialize on first call
        asset.data._prev_joint_vel = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Compute joint accelerations using finite differences
    dt = env.step_dt
    joint_acc = (joint_vel - asset.data._prev_joint_vel) / dt

    # Store current velocities for next step
    asset.data._prev_joint_vel = joint_vel.clone()

    # Return L2 squared norm
    return torch.sum(torch.square(joint_acc), dim=1)


def leg_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize the rate of change of leg actions (action_t - action_{t-1}).
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get leg joint indices
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    # Get current and previous actions for leg joints only
    # Actions are stored in env (assuming the action is available)
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    # Get the joint position action
    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions'):
        env._prev_leg_actions = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = leg_actions - env._prev_leg_actions
    env._prev_leg_actions = leg_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def neck_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize the rate of change of neck actions (action_t - action_{t-1}).
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get neck joint indices
    neck_joint_indices = list(range(5, 9))

    # Get current and previous actions for neck joints only
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions'):
        env._prev_neck_actions = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = neck_actions - env._prev_neck_actions
    env._prev_neck_actions = neck_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def leg_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize leg action accelerations (action_t - 2*action_{t-1} + action_{t-2}).
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get leg joint indices
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions_for_acc'):
        env._prev_leg_actions_for_acc = leg_actions.clone()
        env._prev_prev_leg_actions_for_acc = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = leg_actions - 2 * env._prev_leg_actions_for_acc + env._prev_prev_leg_actions_for_acc

    env._prev_prev_leg_actions_for_acc = env._prev_leg_actions_for_acc.clone()
    env._prev_leg_actions_for_acc = leg_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def neck_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize neck action accelerations (action_t - 2*action_{t-1} + action_{t-2}).
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get neck joint indices
    neck_joint_indices = list(range(5, 9))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions_for_acc'):
        env._prev_neck_actions_for_acc = neck_actions.clone()
        env._prev_prev_neck_actions_for_acc = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = neck_actions - 2 * env._prev_neck_actions_for_acc + env._prev_prev_neck_actions_for_acc

    env._prev_prev_neck_actions_for_acc = env._prev_neck_actions_for_acc.clone()
    env._prev_neck_actions_for_acc = neck_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def reset_rolling_entry(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    speed_range: tuple = (0.25, 0.45),
    wheel_radius: float = 0.0175,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Départ en ROULEMENT sans glissement (élan aux roues).

    Tire une vitesse d'avance v par env ; met la vitesse LINÉAIRE de base (x
    monde) = v ET la vitesse de ROTATION des 4 roues passives = v / r, donc
    ω·r = v => zéro glissement au contact. Évite l'à-coup de l'ancienne poussée
    base-seule (base qui bouge, roues immobiles = patinage brutal au 1er pas).
    À exécuter APRÈS reset_base (qui pose la base ; ne plus lui donner de
    velocity_range).
    """
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = int(env_ids.shape[0])
    lo, hi = speed_range
    v = torch.rand(n, device=env.device) * (hi - lo) + lo  # (n,) vitesse avant

    # Vitesse de base (monde) : uniquement +x.
    root_vel = torch.zeros(n, 6, device=env.device)
    root_vel[:, 0] = v
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    # Rotation des 4 roues passives = v / r (positif = avant, cf. wheel_speed).
    wheel_ids = []
    for name in ("passive_LF_?wheel", "passive_LR_?wheel", "passive_RF_?wheel", "passive_RR_?wheel"):
        ids, _ = asset.find_joints(name)
        wheel_ids.append(ids[0])
    wheel_ids_t = torch.tensor(wheel_ids, device=env.device)
    omega = (v / wheel_radius).unsqueeze(1).repeat(1, len(wheel_ids))  # (n, 4)
    asset.write_joint_velocity_to_sim(omega, joint_ids=wheel_ids_t, env_ids=env_ids)


def wheel_glide_reward(
    env: ManagerBasedRlEnv,
    cap_speed: float = 0.35,
    wheel_radius: float = 0.0175,
) -> torch.Tensor:
    """Récompense le ROULEMENT des roues vers l'avant (glisse), plafonné.

    Contrairement à descent_speed (vitesse de la BASE, qu'on peut atteindre en
    "courant"/poussant), on récompense la rotation des ROUES passives = vraie
    glisse par roulement. Indépendant de toute commande (la tâche pente a une
    commande nulle : la glisse vient de la gravité). Plafonné à ``cap_speed``
    (m/s de vitesse de roulement) -> AUCUNE incitation à accélérer au-delà ; nul
    si les roues reculent (remontée). NaN-safe.
    """
    asset: Entity = env.scene["robot"]
    lf, _ = asset.find_joints("passive_LF_?wheel")
    lr, _ = asset.find_joints("passive_LR_?wheel")
    rf, _ = asset.find_joints("passive_RF_?wheel")
    rr, _ = asset.find_joints("passive_RR_?wheel")
    vel = asset.data.joint_vel
    # Les 4 roues tournent en positif pour l'avant (cf. wheel_speed_reward).
    omega = (vel[:, lf[0]] + vel[:, lr[0]] + vel[:, rf[0]] + vel[:, rr[0]]) / 4.0
    speed = torch.nan_to_num(omega * wheel_radius, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(speed, min=0.0, max=cap_speed)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """
    Reward for staying alive (not terminated)

    Args:
        env: The environment

    Returns:
        Reward tensor of shape (num_envs,) - ones for all envs
    """
    return torch.ones(env.num_envs, device=env.device)


def com_height_target(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_height_min: float = 0.1,
    target_height_max: float = 0.15,
) -> torch.Tensor:
    """
    Reward for keeping the center of mass within a target height range.
    Returns positive reward when in range, negative penalty when outside.

    Args:
        env: The environment
        asset_cfg: Asset configuration
        target_height_min: Minimum target height for CoM (meters)
        target_height_max: Maximum target height for CoM (meters)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Height above terrain spawn origin (world z minus terrain z).
    # env_origins[:, 2] is 0 for flat ground, so this is safe unconditionally.
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0
    # so the penalty is finite (small, since 0 is near the target range).
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )

    # Reward when in range, penalty when outside
    # Use smooth penalty that increases quadratically with distance from range
    below_min = com_height < target_height_min
    above_max = com_height > target_height_max
    in_range = ~(below_min | above_max)

    # Compute penalties for being outside range
    penalty_below = torch.square(com_height - target_height_min) * below_min.float()
    penalty_above = torch.square(com_height - target_height_max) * above_max.float()

    # Reward: +1 when in range, -squared_distance when outside
    reward = in_range.float() - (penalty_below + penalty_above)

    return reward


def crouch_height_target(
    phase: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
) -> torch.Tensor:
    """Cible de hauteur du tronc « en trapèze » le long de la phase [0,1).

    phase ∈ [0, hold_lo)      : descente   height_high -> height_low
    phase ∈ [hold_lo, hold_hi): palier      height_low   (la glisse accroupie)
    phase ∈ [hold_hi, 1.0)    : remontée    height_low  -> height_high

    Args:
        phase: (B,) phase par env, dans [0, 1).
        height_low: hauteur du tronc accroupi (m).
        height_high: hauteur du tronc debout (m).
        hold_lo, hold_hi: bornes du palier bas en fraction de phase.
    Returns:
        (B,) hauteur-cible en mètres.
    """
    descend = phase < hold_lo
    hold = (phase >= hold_lo) & (phase < hold_hi)

    frac_d = phase / hold_lo
    t_descend = height_high + (height_low - height_high) * frac_d

    t_hold = torch.full_like(phase, height_low)

    frac_r = (phase - hold_hi) / (1.0 - hold_hi)
    t_rise = height_low + (height_high - height_low) * frac_r

    return torch.where(descend, t_descend, torch.where(hold, t_hold, t_rise))


def crouch_glide_reward_from_values(
    com_height: torch.Tensor,
    cmd_cos: torch.Tensor,
    cmd_sin: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
) -> torch.Tensor:
    """Récompense gaussienne du suivi de la cible de hauteur (fonction pure).

    Décode la phase depuis [cos, sin] puis compare la hauteur mesurée à la
    cible-trapèze. Retourne exp(-((h - cible)/std)^2) ∈ (0, 1].
    """
    phase = (torch.atan2(cmd_sin, cmd_cos) / (2 * torch.pi)) % 1.0
    target = crouch_height_target(phase, height_low, height_high, hold_lo, hold_hi)
    return torch.exp(-((com_height - target) / std) ** 2)


def crouch_glide_height_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    height_low: float = 0.075,
    height_high: float = 0.11,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward principale : suit la cible de hauteur du tronc le long de la phase.

    La hauteur du CoM est calculée comme dans `com_height_target` (world z moins
    l'origine du terrain, nan->0). La phase provient de la commande GroundPick.
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    cmd = env.command_manager.get_command(command_name)
    return crouch_glide_reward_from_values(
        com_height, cmd[:, 0], cmd[:, 1],
        height_low, height_high, hold_lo, hold_hi, std,
    )


def forward_speed_reward(
    env: ManagerBasedRlEnv,
    vel_ref: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Récompense la vitesse avant du tronc (conserver l'élan / ne pas freiner).

    Indépendante de la commande (la commande porte la phase, pas la vitesse).
    tanh(clamp(vx, 0)/vel_ref) → sature à ~1, ne récompense jamais reculer.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = asset.data.root_link_lin_vel_b[:, 0]
    return torch.tanh(torch.clamp(vx, min=0.0) / vel_ref)


def crouch_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """Blend 0..1 le long de la phase [0,1) — 0 = pose debout, 1 = pose accroupie.

    [0, descent_end)      : 0 -> 1  (se baisser)
    [descent_end, hold_end): 1      (bas / accroupi)
    [hold_end, rise_end)  : 1 -> 0  (se lever)
    [rise_end, 1.0)       : 0       (haut / debout, repos)
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def _crouch_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    crouch_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    stand_pose: Optional[dict] = None,
):
    """(cur, target) joint tensors for the phase-interpolated crouch pose.

    Target interpolates per joint STAND <-> crouch_pose by the 4-segment blend
    b(phase) in [0,1] (0 = standing, 1 = crouch). STAND is `stand_pose` where
    given, else the model DEFAULT (HOME). Joints are resolved BY NAME so the
    passive-wheel interspersing on the roller robot never shifts an index.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = crouch_pose_blend(phase, descent_end, hold_end, rise_end)   # (B,) 0..1

    names = list(crouch_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]                     # (B,k)

    stand = default.clone()                                            # source pose
    if stand_pose:
        for j, n in enumerate(names):
            if n in stand_pose:
                stand[:, j] = stand_pose[n]
    crouch = torch.tensor(
        [crouch_pose[n] for n in names], device=env.device, dtype=default.dtype
    ).unsqueeze(0)                                                     # (1,k)

    target = stand + blend.unsqueeze(-1) * (crouch - stand)           # (B,k)
    cur = asset.data.joint_pos[:, ids]                                # (B,k)
    return cur, target


def crouch_glide_pose_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    crouch_pose: Optional[dict] = None,
    stand_pose: Optional[dict] = None,
    std: float = 0.4,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian match to a phase-interpolated joint pose (stand <-> crouch).

    Directive reward: tells the robot the exact joint configuration to be in at
    each phase. Standing back up (target = stand_pose) is rewarded exactly like
    crouching (target = crouch_pose) — symmetric by construction.
    """
    cur, target = _crouch_pose_error(
        env, asset_cfg, command_name, crouch_pose or {},
        descent_end, hold_end, rise_end, stand_pose,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def crouch_glide_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    crouch_pose: Optional[dict] = None,
    stand_pose: Optional[dict] = None,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 bootstrap toward the phase-interpolated crouch pose (negative penalty).

    Constant gradient everywhere — gives the policy a direction to the target
    pose even when the Gaussian above has saturated to ~0 far from it.
    """
    cur, target = _crouch_pose_error(
        env, asset_cfg, command_name, crouch_pose or {},
        descent_end, hold_end, rise_end, stand_pose,
    )
    return -(cur - target).abs().mean(dim=-1)


def crouch_forward_lean(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pitch: float = 0.08,
    std: float = 0.1,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """Léger penché AVANT du tronc pendant l'accroupi (gaté par le blend crouch).

    Contre la bascule arrière induite par la flexion rapide des hanches. Proxy de
    pitch = projected_gravity_b[:,0] (positif = vers l'avant, vérifié). La porte
    (blend) vaut 1 pendant descente+bas, 0 debout → ne biaise QUE l'accroupi.
    target_pitch petit = "de très peu".
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = crouch_pose_blend(phase, descent_end, hold_end, rise_end)
    lean = asset.data.projected_gravity_b[:, 0]
    return gate * torch.exp(-((lean - target_pitch) ** 2) / std ** 2)


def neck_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize neck joint velocities to keep head stable.
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get neck joint indices (neck_pitch, head_pitch, head_yaw, head_roll).
    # Servo view: passive_* joints (backlash, wheels) don't shift the indices.
    neck_joint_indices = list(range(5, 9))
    joint_vel = _servo_joint_vel(env, asset)
    neck_joint_vel = joint_vel[:, neck_joint_indices]

    # Return L2 squared norm of neck joint velocities
    return torch.sum(torch.square(neck_joint_vel), dim=1)


def leg_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize leg joint velocities to encourage smoother, less dynamic motion.
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get leg joint indices (left hip-ankle: 0-4, right hip-ankle: 9-13).
    # Servo view: passive_* joints (backlash, wheels) don't shift the indices.
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
    joint_vel = _servo_joint_vel(env, asset)
    leg_joint_vel = joint_vel[:, leg_joint_indices]

    # Return L2 squared norm of leg joint velocities
    return torch.sum(torch.square(leg_joint_vel), dim=1)


_NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(neck|head).*",))


_HIP_PITCH_KNEE_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(hip_pitch|knee).*",))


_ROLLER_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def feet_flat_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """Penalize foot sites not being parallel to the ground.

    The foot site frame has Z+ pointing up when flat. We project a unit gravity
    vector (pointing down) into each foot site's local frame. When flat, gravity
    maps to [0,0,-1] in site frame (xy=0, penalty=0). Any tilt rotates Z away
    from world-up, giving nonzero xy components.

    Max value ≈ 2.0 per foot (foot fully sideways), total ≈ 4.0.

    When ``sensor_name`` is given, each foot's penalty is GATED by that foot's own
    ground contact: the airborne (swing) foot is free to tilt, only the stance
    blade is asked to stay flat (so its wheels keep gripping). Without this gate
    the penalty punishes the recovery-foot lift a stride needs — it is minimised
    by keeping BOTH blades flat on the ground, i.e. the swizzle. Assumes the site
    order (left, right) matches the sensor slot order (ankle_l_v1,
    ankle_r_v1) — both left-first in this model.

    Bug note: must normalize gravity PER ENV with dim=-1. Using torch.norm()
    without dim computes a scalar over all envs × 3 dims, making the vector
    ~1/sqrt(num_envs) in magnitude → penalty ~num_envs times too small.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    import torch.nn.functional as F

    asset: Entity = env.scene[asset_cfg.name]
    gravity_w_n = F.normalize(asset.data.gravity_vec_w, dim=-1)  # (B, 3), unit vector per env

    foot_quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N_feet, 4)
    per_foot = torch.zeros(env.num_envs, foot_quats.shape[1], device=env.device)
    for i in range(foot_quats.shape[1]):
        proj = quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)  # (B, 3)
        per_foot[:, i] = torch.sum(torch.square(proj[:, :2]), dim=1)  # xy² only

    if sensor_name is not None:
        from mjlab.sensor import ContactSensor
        sensor: ContactSensor = env.scene[sensor_name]
        contact_time = sensor.data.current_contact_time  # (B, N_feet)
        assert contact_time is not None
        per_foot = per_foot * (contact_time > 0.0).float()

    return per_foot.sum(dim=1)


def feet_tiptoe_alignment(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Reward each foot site's local x-axis pointing downward — tiptoe stance.

    When flat, foot site x points roughly forward (horizontal). Pitching the
    foot forward (heel up, toe down) rotates x toward world -Z. We reward the
    z-component of the foot x-axis being -1 (perfectly downward).

    Per foot: alignment ∈ [-1, 1], summed over both feet ∈ [-2, 2].

    Gated on |vel_cmd_xy| > command_threshold so the policy isn't required to
    stand on tiptoes at rest — only while walking. The companion
    feet_flat_penalty is NOT used in this task; the two would fight.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N, 4) [w, x, y, z]
    w, qx, qy, qz = quats[:, :, 0], quats[:, :, 1], quats[:, :, 2], quats[:, :, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)  # (B, N) — z-component of local x-axis in world
    alignment = (-x_axis_z).sum(dim=-1)  # +1 per foot when pointing straight down

    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.linalg.norm(cmd[:, :2], dim=1)
    active = (cmd_mag > command_threshold).float()
    return alignment * active


def hip_pitch_knee_vel_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _HIP_PITCH_KNEE_CFG,
) -> torch.Tensor:
    """Penalize hip_pitch and knee joint velocities (L2 squared).

    Walking requires rapid oscillation of these sagittal-plane joints.
    Skating uses hip_roll laterally and glides with minimal sagittal movement.
    This penalizes the oscillation without preventing static balance adjustments.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def neck_joint_pos_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _NECK_JOINT_CFG,
    pattern: str = r".*(neck|head).*",
) -> torch.Tensor:
    """Penalize neck/head joint position deviation from default (L2 squared).

    Uses find_joints() every call to avoid stale cached indices when the same
    SceneEntityCfg singleton is reused across robots with different joint layouts
    (e.g. walk robot vs rollers robot where passive wheels shift neck indices).

    ``pattern`` sélectionne les joints comptés (défaut : toute la nuque + la tête).
    La tâche spin passe un motif qui EXCLUT `head_yaw`, pour laisser la tête servir
    de volant d'inertie au lancement de la rotation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Exclude passive_* joints (backlash hinges also contain "neck"/"head").
    if not pattern.startswith(r"^(?!passive_)"):
        pattern = r"^(?!passive_)" + pattern.lstrip("^")
    joint_ids, _ = asset.find_joints(pattern)
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(error), dim=1)


def joint_torques_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize actuator forces (torques) to encourage energy-efficient motion.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared actuator forces
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get actuator forces (scalar actuation in actuation space)
    actuator_forces = asset.data.actuator_force

    # Return L2 squared norm
    return torch.sum(torch.square(actuator_forces), dim=1)


def joint_torque_rate_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize rate of change in actuator torques (proxy for gearbox shock).

    Sudden torque spikes occur when the robot impacts the ground and actuators
    resist the impulse. Penalising this rate encourages soft landings and smooth
    force transitions that protect gearboxes.

    Returns the sum of squared torque differences from the previous step.
    """
    asset: Entity = env.scene[asset_cfg.name]
    current = asset.data.actuator_force  # (num_envs, num_actuators)

    if not hasattr(env, '_prev_actuator_forces'):
        env._prev_actuator_forces = current.clone()
        return torch.zeros(env.num_envs, device=env.device)

    rate = current - env._prev_actuator_forces
    env._prev_actuator_forces = current.clone()
    return torch.sum(torch.square(rate), dim=1)


def feet_grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Positive reward for feet contacting the ground (0, +0.5, or +1.0).

    Uses the contact sensor's `found` field. For the feet_ground_contact sensor
    which has 2 primary foot geoms, `found` has shape (num_envs, 2) with per-foot
    binary contact. We sum and normalize to [0, 1].
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (num_envs, num_feet) or (num_envs, 1)
    if found.dim() > 1:
        found = found.sum(dim=-1)  # collapse foot dimension
    return torch.clamp(found, 0.0, 2.0) / 2.0


def body_impact_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize terrain contact forces above a threshold on protected body parts.

    Used to discourage slamming the trunk shell or head into the ground during
    falls. The sensor should cover the relevant body or subtree with
    reduce='netforce'. Forces below threshold are free; above that the penalty
    grows linearly.

    Args:
        sensor_name: Name of a ContactSensorCfg with fields=("force",),
            reduce="netforce".
        threshold: Contact force (N) below which no penalty is applied.

    Returns:
        Penalty tensor (num_envs,) — N above threshold per step.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force  # (num_envs, N_bodies, 3)
    total_force = forces.sum(dim=1)  # sum over bodies in the subtree
    force_mag = torch.norm(total_force, dim=1)
    return torch.clamp(force_mag - threshold, min=0.0)


def wheel_speed_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    wheel_radius: float = 0.0175,
    vel_scale: float = 0.5,
    bidirectional: bool = False,
) -> torch.Tensor:
    """Reward wheel spin proportional to commanded push.

    All 4 wheels spin positive for forward motion (verified visually).
    tanh saturation at vel_scale m/s equivalent prevents runaway.

    - ``bidirectional=False`` (default): forward only — reward forward spin for
      cmd_x > 0, silent otherwise (cmd_x < 0 handled by the braking reward).
    - ``bidirectional=True``: reward wheel spin in the COMMANDED direction —
      forward for cmd_x > 0, backward for cmd_x < 0 — with magnitude |cmd_x|.
      Lets cmd_x < 0 mean "go backward" instead of "brake".
    """
    cmd_x = env.command_manager.get_command(command_name)[:, 0]  # (B,)

    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    # All 4 wheels spin positive for forward motion (verified by test_wheel_direction.py)
    forward_omega = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]] + vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 4.0

    omega_scale = vel_scale / wheel_radius
    if bidirectional:
        # spin aligned with the command sign (fwd for +, back for -)
        aligned = torch.sign(cmd_x) * forward_omega
        return torch.abs(cmd_x) * torch.tanh(torch.clamp(aligned, min=0.0) / omega_scale)
    return torch.clamp(cmd_x, min=0.0) * torch.tanh(torch.clamp(forward_omega, min=0.0) / omega_scale)


def coasting_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
    stillness_std: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(r".*(hip|knee|ankle).*",)),
) -> torch.Tensor:
    """Reward coasting: low leg-joint velocity while at target speed.

    Returns exp(-vel_error / vel_std²) × exp(-sum(joint_vel²) / stillness_std²).
    Both factors must be high simultaneously — robot is rewarded for being at
    target speed AND keeping its legs still (gliding), not for either alone.

    Typical values when coasting well: ~0.7–1.0.  When actively stomping at
    speed the joint_vel term suppresses the reward toward 0.
    """
    cmd = env.command_manager.get_command(command_name)
    vel_b = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    vel_error = torch.sum(torch.square(cmd[:, :2] - vel_b), dim=1)
    at_speed = torch.exp(-vel_error / vel_std ** 2)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-joint_vel_sq / stillness_std ** 2)

    return at_speed * stillness


def braking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
) -> torch.Tensor:
    """Reward coming to a stop when cmd_x < 0 (brake commanded).

    Returns clamp(-cmd_x, 0) * exp(-fwd_vel² / vel_std²).
    - Silent when cmd_x ≥ 0 (coast or push).
    - At cmd_x = -1 and vel = 0: reward = 1.0 (full stop achieved).
    - At cmd_x = -1 and vel = vel_std: reward ≈ 0.37 (strong gradient).
    vel_std=0.3 m/s gives meaningful gradient down to walking-pace speeds.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    braking_strength = torch.clamp(-cmd_x, min=0.0)
    fwd_vel = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    stopped = torch.exp(-(fwd_vel.clamp(min=0.0) ** 2) / (vel_std ** 2))
    return braking_strength * stopped


def contact_frequency_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_contact_changes_per_sec: float = 4.0,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """
    Penalize high frequency of contact changes to encourage slower stepping.
    Tracks the number of contact state changes per second and penalizes when above threshold.

    Args:
        env: The environment
        sensor_name: Name of the contact sensor
        max_contact_changes_per_sec: Maximum allowed contact changes per second
        command_threshold: Minimum command magnitude to apply penalty

    Returns:
        Penalty tensor of shape (num_envs,) - negative when exceeding threshold
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    # Check if command is above threshold
    if "twist" in env.command_manager._terms:
        cmd = env.command_manager.get_command("twist")
        cmd_vel = cmd[:, :3]
        cmd_norm = torch.linalg.norm(cmd_vel, dim=1)
        active_mask = cmd_norm > command_threshold
    else:
        active_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    sensor = env.scene.sensors[sensor_name]
    contacts = sensor.data.found[:, :2]  # (num_envs, 2)

    # Initialize tracking if needed
    if not hasattr(env, '_contact_change_count'):
        env._contact_change_count = torch.zeros(env.num_envs, device=env.device)
        env._contact_change_timer = torch.zeros(env.num_envs, device=env.device)
        env._prev_contacts_for_freq = contacts.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Detect any contact changes (either foot)
    contact_changed = torch.any(contacts != env._prev_contacts_for_freq, dim=1)

    # Increment change counter
    env._contact_change_count += contact_changed.float()

    # Update timer
    env._contact_change_timer += env.step_dt

    # Calculate current frequency (changes per second)
    # Avoid division by zero
    freq = env._contact_change_count / torch.clamp(env._contact_change_timer, min=0.01)

    # Reset counter and timer every 1 second
    reset_mask = env._contact_change_timer >= 1.0
    env._contact_change_count[reset_mask] = 0.0
    env._contact_change_timer[reset_mask] = 0.0

    # Penalize when frequency exceeds maximum
    # Use quadratic penalty for frequencies above threshold
    excess_freq = torch.clamp(freq - max_contact_changes_per_sec, min=0.0)
    penalty = -torch.square(excess_freq)

    # Update previous contacts
    env._prev_contacts_for_freq = contacts.clone()

    # Apply command threshold mask
    penalty = penalty * active_mask.float()

    return penalty


def standing_phase(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Simple time-based phase for standing task.

    Returns a scalar phase value that cycles from 0 to 1 based on time.
    This allows the policy to have a sense of time progression even when standing.

    Args:
        env: The RL environment
        asset_cfg: Not used, but kept for API consistency

    Returns:
        Phase value [0, 1] as tensor of shape (num_envs, 1)
    """
    # Simple time-based phase that cycles every 2 seconds
    # This gives the policy a time-varying signal
    phase_period = 2.0  # seconds
    time = env.episode_length_buf * env.step_dt
    phase = (time % phase_period) / phase_period

    return phase.unsqueeze(-1)  # Shape: (num_envs, 1)


def air_time_adaptive(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,    # below this: no reward (standing)
    running_threshold: float = 0.5,     # above this: use running air-time window
    walk_threshold_min: float = 0.10,
    walk_threshold_max: float = 0.25,
    run_threshold_min: float = 0.05,
    run_threshold_max: float = 0.25,
) -> torch.Tensor:
    """Air-time reward with separate swing-time windows for walking vs running.

    - command < command_threshold  → 0 (standing, no reward)
    - command_threshold–running_threshold → walk window [walk_min, walk_max]
    - command > running_threshold  → run  window [run_min,  run_max]

    This lets the walking gait keep its deliberate 100–250 ms swing while
    running can use a faster 50–250 ms cadence.
    """
    sensor = env.scene.sensors[sensor_name]
    current_air_time = sensor.data.current_air_time  # (num_envs, num_feet)
    assert current_air_time is not None

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    is_walking = ((total_speed >= command_threshold) & (total_speed < running_threshold)).float()  # (num_envs,)
    is_running = (total_speed >= running_threshold).float()

    # Per-env thresholds broadcast over feet
    tmin = (is_walking * walk_threshold_min + is_running * run_threshold_min).unsqueeze(1)
    tmax = (is_walking * walk_threshold_max + is_running * run_threshold_max).unsqueeze(1)

    in_range = (current_air_time > tmin) & (current_air_time < tmax)
    reward = torch.sum(in_range.float(), dim=1)  # sum over feet

    # Zero reward when standing
    active = (total_speed >= command_threshold).float()
    return reward * active


def stillness_at_zero_command(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    vel_std: float = 0.1,
) -> torch.Tensor:
    """Reward staying still when command is near zero.

    Returns exp(-body_vel² / vel_std²) when command < threshold, else 0.
    This is monotonically decreasing with body speed — moving faster is always
    less rewarding. There is no threshold the robot can cross to 'escape' it,
    unlike gate-based stepping penalties.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    stillness = torch.exp(-body_vel ** 2 / vel_std ** 2)

    return is_standing_cmd * stillness


def joint_vel_l2_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalise leg joint velocities only when command is near zero.

    Targets the standing-shake problem: the policy makes rapid oscillating
    corrections around the home pose when standing. Gated on command so it
    does not affect the walking gait at all.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    leg_indices = list(range(0, 5)) + list(range(9, 14))
    joint_vel = asset.data.joint_vel[:, leg_indices]
    vel_sq = torch.sum(joint_vel ** 2, dim=-1)

    return is_standing_cmd * vel_sq


def foot_step_penalty_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    body_vel_threshold: float = 0.2,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalise stepping when at zero command and the body is not being pushed.

    Symmetric counterpart to the air_time reward:
    - air_time gives  +reward for stepping when command > threshold  (walk)
    - this gives      -reward for stepping when command < threshold  (stand)

    The body-velocity gate prevents penalising recovery steps after a push:
    if the robot is already moving fast (pushed), no penalty is applied so it
    can still take steps to catch itself.

    Returns a value in [0, 1] (use a negative weight in the config).
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors["feet_ground_contact"]

    # Was either foot recently lifted? (last completed air phase > threshold)
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2)
    any_foot_stepped = (air_time > air_time_threshold).any(dim=1).float()

    # Are we in standing mode? (command near zero)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing = (total_speed < command_threshold).float()

    # Is the body still? (not being pushed)
    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    is_still = (body_vel < body_vel_threshold).float()

    return any_foot_stepped * is_standing * is_still


def recovery_stepping_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    velocity_threshold: float = 0.3,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward foot air time only when at zero command AND robot has high velocity (recovering from push).

    This encourages the robot to take steps to recover balance when pushed,
    but does NOT fire during normal walking (command > threshold).

    Args:
        env: The RL environment
        asset_cfg: Asset configuration (unused but kept for API consistency)
        command_name: Name of the velocity command in the command manager
        command_threshold: Speed below which the robot is considered to be in standing mode
        velocity_threshold: Linear velocity threshold to activate stepping reward (m/s)
        air_time_threshold: Minimum air time to count as a step (seconds)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Only fire for standing envs (command near zero)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    # Get base linear velocity magnitude
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # Only reward stepping when velocity is high (being pushed)
    should_step = vel_magnitude > velocity_threshold

    # Get foot air time from contact sensor
    contact_sensor = env.scene.sensors["feet_ground_contact"]
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2) - left and right foot

    # Reward if either foot has been in air recently
    foot_in_air = (air_time > air_time_threshold).any(dim=1)  # (num_envs,)

    # Only give reward when: standing command AND high body velocity AND foot stepped
    reward = is_standing_cmd * should_step.float() * foot_in_air.float()

    return reward


def adaptive_pose_weight(
    env: ManagerBasedRlEnv,
    base_pose_reward: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    velocity_threshold: float = 0.3,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """Reduce pose tracking weight when robot has high velocity (recovering from push).

    This gives the robot freedom to deviate from the standing pose when taking
    recovery steps, while maintaining strict pose tracking when standing still.

    Args:
        env: The RL environment
        base_pose_reward: The original pose reward (before weighting)
        asset_cfg: Asset configuration (unused but kept for API consistency)
        velocity_threshold: Linear velocity threshold to start reducing weight (m/s)
        min_weight: Minimum weight multiplier (0-1) at high velocities

    Returns:
        Weighted reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get base linear velocity magnitude
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # Compute weight: 1.0 when stationary, min_weight at high velocity
    # Use smooth transition via sigmoid-like function
    weight = min_weight + (1.0 - min_weight) * torch.exp(
        -((vel_magnitude - velocity_threshold) / velocity_threshold).clamp(min=0.0) ** 2
    )

    return base_pose_reward * weight


def heading_tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for reducing heading error when cmd[2] encodes heading error.

    Returns exp(-cmd[2]² / std²).
    - At error = 0 (on heading): reward = 1.0.
    - At error = std: reward ≈ 0.37 (strong gradient).
    - At error = 1.0 rad with std=0.5: reward ≈ 0.018 (nearly zero).

    std=0.5 rad (≈28°) gives a meaningful gradient across the expected range.
    """
    cmd = env.command_manager.get_command(command_name)
    heading_error = cmd[:, 2]
    return torch.exp(-(heading_error ** 2) / (std ** 2))


def skating_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.4,
    vel_gate_ref: float = 0.0,
) -> torch.Tensor:
    """Reward feet air time only when pushing (cmd_x > 0).

    Encourages the robot to lift each foot during the recovery phase of the
    skating stroke rather than dragging it on the ground.
    Scaled by cmd_x so the incentive grows with push intensity.

    When ``vel_gate_ref`` > 0 the reward is also multiplied by a forward-speed
    gate so lifting feet without propelling the body (tap-dancing on the spot)
    earns nothing. ``threshold_min`` sets the shortest swing that counts — raise
    it to forbid a frantic high-cadence flutter.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward = reward * torch.clamp(cmd_x, min=0.0)
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        reward = reward * gate
    return reward


def _forward_progress_gate(env: ManagerBasedRlEnv, v_ref: float) -> torch.Tensor | None:
    """0→1 ramp in body forward speed: 0 when standing still, 1 at/above v_ref.

    Used to gate stride-shaping rewards so that stepping which does NOT propel
    the body (e.g. tap-dancing on the spot) earns nothing — the reward for the
    FORM of a stride is only paid when the stride actually does its JOB (moving
    forward). Returns None when disabled (v_ref <= 0)."""
    if v_ref <= 0.0:
        return None
    v_fwd = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (v_fwd.clamp(min=0.0) / v_ref).clamp(max=1.0)


def single_support_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_gate_ref: float = 0.0,
    double_penalty: float = 0.25,
) -> torch.Tensor:
    """Reward single-support (a skating stride), mildly discourage the swizzle.

    Real skating is a STRIDE: push off one blade while the other swings, i.e.
    single support that alternates left/right. A symmetric swizzle keeps BOTH
    blades grounded the whole time and still spins the wheels, so wheel_speed
    alone converges to it.

    Per step, counting blades in contact:
      - exactly 1 blade down (stride)  → + clamp(cmd_x,0) · gate
      - 2 blades down    (double supp) → − double_penalty · clamp(cmd_x,0)
      - 0 blades down    (flight/hop)  →  0

    The POSITIVE single-support reward is gated by forward speed (``vel_gate_ref``)
    so stepping in place (no propulsion) earns nothing — kills the tap-dance hack.
    The double-support penalty is small and UNGATED: brief double support during
    weight transfer / push-off is NORMAL skating, so we only lightly discourage
    PERMANENT double support (the swizzle) rather than forbid it. The real
    anti-swizzle signal is skating_air_time — the swizzle never lifts a foot.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None

    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)  # (num_envs,)
    single = (n_contact == 1).float()
    double = (n_contact >= 2).float()

    cmd_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    single_r = single * cmd_x
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        single_r = single_r * gate
    return single_r - double_penalty * double * cmd_x


def glide_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_ref: float = 0.2,
    stillness_std: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=(r".*(hip|knee|ankle).*",)
    ),
) -> torch.Tensor:
    """Reward the GLIDE phase of a stride: coast on ONE blade with quiet legs.

    Nothing else rewards gliding — skating_air_time pays each swing, so the policy
    maximises swing FREQUENCY (frantic kicking). This term pays staying on one
    foot and coasting, giving the policy a reason to slow down and commit to each
    stroke:

        reward = single_support · forward_gate · stillness · (cmd_x >= 0)

    - single_support: exactly ONE blade in contact. REQUIRED — this is the fix vs
      the earlier broken glide, which omitted it and let a two-blade swizzle-coast
      farm the reward and regress the gait.
    - forward_gate = clamp(v_fwd,0,vel_ref)/vel_ref → 0 when not moving forward.
    - stillness = exp(-Σ leg_joint_vel² / stillness_std²) → high only when legs
      are quiet; a kick (fast joint motion) gets ~0, so only a real glide pays.
    - active on push/coast only (cmd_x >= 0); silent on brake.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    single = (torch.sum((contact_time > 0.0).float(), dim=1) == 1).float()

    forward_gate = _forward_progress_gate(env, vel_ref)
    if forward_gate is None:
        forward_gate = torch.ones(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )
    stillness = torch.exp(-joint_vel_sq / stillness_std ** 2)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    active = (cmd_x >= 0.0).float()
    return single * forward_gate * stillness * active


def leg_symmetry_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_bases: tuple = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"),
) -> torch.Tensor:
    """Reward left/right legs mirroring — the swizzle's defining symmetry.

    The robot uses mirrored L/R sign conventions, so a bilaterally-symmetric config
    satisfies q_left + q_right ≈ 0 per matched joint pair. Returns
    ``-mean_pairs |q_left + q_right|`` (L1, constant gradient); use with a POSITIVE
    weight so asymmetry is penalised and the symmetric swizzle is favoured. L/R index
    pairs are resolved once by name and cached on env.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "_leg_sym_ids"):
        left, right = [], []
        for base in joint_bases:
            li, _ = asset.find_joints([f"left_{base}"])
            ri, _ = asset.find_joints([f"right_{base}"])
            left.append(li[0])
            right.append(ri[0])
        env._leg_sym_ids = (
            torch.tensor(left, device=env.device),
            torch.tensor(right, device=env.device),
        )
    lids, rids = env._leg_sym_ids
    q = asset.data.joint_pos
    return -torch.abs(q[:, lids] + q[:, rids]).mean(dim=-1)


def grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
) -> torch.Tensor:
    """Reward BOTH blades in contact — a classic swizzle stays grounded (no lifting).

    Mirror of single_support_reward but rewarding double support (n_contact >= 2),
    scaled by |cmd_x| so it shapes the push phase in EITHER direction (forward or
    backward — the swizzle env drives cmd_x < 0 as "go backward").
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    return grounded * cmd_x


def gait_symmetry_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize lopsided left/right foot usage (one blade doing most of the work).

    With symmetry augmentation OFF, nothing stops the policy learning an asymmetric
    stride that pushes mostly with one leg — which veers and destabilises (esp. at
    launch). Accumulates per-foot swing time over the episode and penalises the
    normalised imbalance |L - R| / (L + R):
      - balanced alternating stride  -> ~0 (no penalty)
      - one foot swinging much more   -> ~1 (max penalty)
    Only the CUMULATIVE imbalance is penalised — the instantaneous single-support
    asymmetry of a real stride (one foot swinging now) is fine.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    air = sensor.data.current_air_time  # (N, num_feet)
    assert air is not None

    if not hasattr(env, "_swing_accum") or env._swing_accum.shape[0] != env.num_envs:
        env._swing_accum = torch.zeros(env.num_envs, air.shape[1], device=env.device)
    reset = env.episode_length_buf <= 1
    env._swing_accum[reset] = 0.0
    env._swing_accum += (air > 0.0).float() * env.step_dt

    L = env._swing_accum[:, 0]
    R = env._swing_accum[:, 1]
    return torch.abs(L - R) / (L + R + 1e-3)


def heading_hold_reward(
    env: ManagerBasedRlEnv,
    std: float = 0.4,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward holding the SPAWN heading (go straight) — corrective, angle-based.

    Rewards the yaw ANGLE staying near the heading captured at reset:
        reward = exp(-wrap(yaw - yaw_spawn)² / std²)

    This is the RIGHT way to go straight (vs penalising yaw-RATE, which just tells
    the policy 'never turn' → it can't steer back and drifts open-loop). Here a
    drift lowers the reward, and the policy is free to yaw back to recover it.

    The spawn heading is captured per-env on the first step(s) after reset
    (episode_length_buf <= 1), when the robot is still ~at its spawn pose. Reads
    root_link_quat_w, which is fresh at reward time (post physics step). Heading-
    invariant: the reference is each env's own random spawn yaw, so it works with
    the full-circle yaw randomisation at reset.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    if not hasattr(env, "_heading_ref") or env._heading_ref.shape[0] != env.num_envs:
        env._heading_ref = yaw.clone()
    just_reset = env.episode_length_buf <= 1
    env._heading_ref = torch.where(just_reset, yaw, env._heading_ref)

    err = yaw - env._heading_ref
    err = torch.atan2(torch.sin(err), torch.cos(err))  # wrap to [-π, π]
    return torch.exp(-(err ** 2) / std ** 2)


def action_over_limit_penalty(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    overshoot: float = 0.3,
) -> torch.Tensor:
    """Penalise commanding a joint target beyond its hard limit (+ overshoot).

    Policy-side deterrent against over-driving a joint onto its mechanical stop:
    e.g. hip_roll has a ±0.38 rad limit but a ±10 rad ctrlrange, so the low-kp
    servo can be commanded far past the stop to slam it with max torque — a
    fragile sim-only trick that will not transfer.

    Reads the commanded target (raw_action · scale + offset) and penalises only
    the part BEYOND (hard_limit + overshoot):

        penalty = Σ relu(target - (hi + overshoot)) + relu((lo - overshoot) - target)

    Unlike a qpos-limit penalty, this fires on the COMMAND, not the joint
    position — so the joint may still reach its full range (command ≈ limit) and
    no usable amplitude is stolen. Because it constrains the policy's OUTPUT, the
    learned behaviour is baked into the network and transfers to deployment
    WITHOUT any env-side action clip (which would only exist in sim → mismatch).
    ``overshoot`` gives the low-kp servo the headroom to reach near-limit targets
    under load; only the wild over-drive past that is penalised.
    """
    term = env.action_manager.get_term(action_name)
    target = term.raw_action * term.scale + term.offset  # (B, action_dim) abs targets
    jnt_ids = term.target_ids
    hard = env.scene["robot"].data.joint_pos_limits[:, jnt_ids]  # (B, action_dim, 2)
    lo = hard[..., 0] - overshoot
    hi = hard[..., 1] + overshoot
    over = (target - hi).clip(min=0.0) + (lo - target).clip(min=0.0)
    return torch.sum(over, dim=-1)


def forward_lean_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_pitch: float = 0.08,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """Reward leaning slightly forward when pushing, to counteract the backward
    torque from skating strokes.

    Uses projected_gravity_b x-component as a pitch proxy:
      forward_lean = -gravity_b[:, 0]  (positive when leaning forward)

    Only fires when cmd_x > 0. Peaks at target_pitch radians of forward lean.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_lean = asset.data.projected_gravity_b[:, 0]
    push = torch.clamp(cmd_x, min=0.0)
    return push * torch.exp(-((forward_lean - target_pitch) ** 2) / (std ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Gait-shaping penalties ported from mjlab_microban (microban velocity recipe).
# ─────────────────────────────────────────────────────────────────────────────
def no_stepping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalize feet in the air when the commanded speed is below threshold.

    Discourages marching in place when the robot should stand still. Returns the
    count of airborne feet per environment (use with a negative weight).
    Ported from mjlab_microban.
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    below_threshold = cmd_speed < command_threshold

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    in_air = ~found.bool()

    return in_air.float().sum(dim=-1) * below_threshold.float()


def feet_distance_penalty(
    env: ManagerBasedRlEnv,
    min_dist: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the feet getting too close to each other in the horizontal plane.

    Returns ``clamp(min_dist - d, min=0)`` per env (use with a negative weight),
    where ``d`` is the horizontal (xy) distance between the two foot sites.
    Ported from mjlab_microban. Not wired into velocity yet — pinned for later.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
    dist = torch.norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=-1)  # (N,)
    return torch.clamp(min_dist - dist, min=0.0)
