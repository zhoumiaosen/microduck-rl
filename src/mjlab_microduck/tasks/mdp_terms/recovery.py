"""MicroDuck recovery terms."""

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
)


def _fallen_mask(
    env: ManagerBasedRlEnv,
    asset,
    gate_z_below: float,
    gate_tilt_above_deg: float,
) -> torch.Tensor:
    """Per-env float mask: 1.0 where the robot counts as FALLEN — trunk height
    below `gate_z_below` OR tilt beyond `gate_tilt_above_deg`. Used to gate the
    recovery rewards so they only steer while actually fallen and contribute
    exactly zero during clean walking (no walk tax / bounce farming)."""
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    # cos(tilt) = R22 = 1 - 2(qx² + qy²)
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()


def feet_air_time_upright(
    env: ManagerBasedRlEnv,
    gate_tilt_above_deg: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **air_time_kwargs,
) -> torch.Tensor:
    """velocity template feet_air_time, zeroed while FALLEN (tilt > gate).

    velstand: a robot lying on its trunk can still tap its feet rhythmically
    through the air-time window — the observed "lies there shaking a leg"
    exploit. Air time is only meaningful upright.
    """
    from mjlab.tasks.velocity.mdp import feet_air_time as _template_air_time
    reward = _template_air_time(env, **air_time_kwargs)
    asset: Entity = env.scene[asset_cfg.name]
    upright = 1.0 - _fallen_mask(env, asset, 0.0, gate_tilt_above_deg)
    return reward * upright


def upright_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Potential-based upright shaping: Δcos(tilt) per step.

    Pays for PROGRESS toward upright, charges for progress toward fallen, and
    pays exactly ZERO for holding any pose — so no state can farm it (the gated
    state-reward it replaces was farmed from sitting, lying flat, and a
    head-tripod lean across three velstand runs). Potential-based shaping is
    policy-invariant (Ng et al.): it accelerates learning of recovery without
    creating new optima. A full prone→stand recovery collects Δ≈+1 total
    (× weight); a fall costs the same on the way down.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    cos_tilt = torch.nan_to_num(
        1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=1.0
    )
    if not hasattr(env, "_upright_potential_prev"):
        env._upright_potential_prev = cos_tilt.clone()
    # Freshly reset envs: no spurious delta from the previous episode's pose.
    fresh = env.episode_length_buf <= 1
    env._upright_potential_prev[fresh] = cos_tilt[fresh]
    delta = cos_tilt - env._upright_potential_prev
    env._upright_potential_prev = cos_tilt.clone()
    return delta


def height_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ceiling: float = 0.115,
) -> torch.Tensor:
    """Potential-based height shaping: Δ min(trunk z, ceiling) per step.

    The z-axis companion to ``upright_progress`` (velstand crouch-endpoint
    lesson): the last mile of a recovery — extending the knees out of a deep
    crouch — is mostly a HEIGHT change at modest tilt, exactly where the
    Gaussian upright/pose rewards are flat and Δcos(tilt) is tiny. Rising pays,
    falling charges, holding pays zero, so gait bobbing nets zero and nothing
    can farm it. Capped at ``ceiling`` (just below full-stand trunk z ≈ 0.117)
    so hopping above stance height pays nothing extra.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    pot = torch.clamp(z, max=ceiling)
    if not hasattr(env, "_height_potential_prev"):
        env._height_potential_prev = pot.clone()
    fresh = env.episode_length_buf <= 1
    env._height_potential_prev[fresh] = pot[fresh]
    delta = pot - env._height_potential_prev
    env._height_potential_prev = pot.clone()
    return delta


def fallen_state_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_tilt_above_deg: float = 40.0,
    release_tilt_below_deg: float | None = None,
    release_z_above: float | None = None,
) -> torch.Tensor:
    """1.0 while FALLEN (weight it negative): a flat per-step tax on staying
    down. Without it, lying still is ~0/step while attempting recovery costs
    action-rate/torque penalties — waiting for the fallen_too_long recycle was
    the rational policy. (Penalties on bad states are safe; it's POSITIVE
    rewards gated on bad states that get farmed.)

    With ``release_*`` set, the tax has HYSTERESIS (velstand crouch-endpoint
    lesson): a fall arms it and it keeps paying until the robot is genuinely
    up (tilt < release_tilt AND z > release_z), not merely under the arming
    gate. Without it, a crouch just below the 40° gate is a zero-cost rest
    state — recoveries learned to park there instead of finishing the stand.
    Arms only on a genuine fall, so gait-cycle tilt wobble is never taxed."""
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, 0.0, gate_tilt_above_deg).bool()
    if release_tilt_below_deg is None:
        return fallen.float()
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    up = cos_tilt > math.cos(math.radians(release_tilt_below_deg))
    if release_z_above is not None:
        up &= z > release_z_above
    if not hasattr(env, "_fallen_tax_armed"):
        env._fallen_tax_armed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    env._fallen_tax_armed[fresh] = False
    env._fallen_tax_armed |= fallen
    env._fallen_tax_armed &= ~up
    return env._fallen_tax_armed.float()


def recovery_success(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    fallen_tilt_deg: float = 40.0,
    min_fallen_s: float = 0.5,
    up_tilt_deg: float = 25.0,
    up_z: float = 0.105,
) -> torch.Tensor:
    """One-shot bounty on a COMPLETED recovery: fires on the frame where an env
    that has been fallen (tilt > fallen_tilt for ≥ min_fallen_s) becomes
    genuinely upright (tilt < up_tilt AND trunk z > up_z). Hysteresis: re-arms
    only by being fallen again, so oscillating around the gate pays nothing.
    Gives the sparse-but-strong endpoint gradient the dense gated terms lack.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = cos_tilt < math.cos(math.radians(fallen_tilt_deg))
    up = (cos_tilt > math.cos(math.radians(up_tilt_deg))) & (z > up_z)
    if not hasattr(env, "_recovery_fallen_s"):
        env._recovery_fallen_s = torch.zeros(env.num_envs, device=env.device)
        env._recovery_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fresh = env.episode_length_buf <= 1
    env._recovery_fallen_s[fresh] = 0.0
    env._recovery_armed[fresh] = False
    env._recovery_fallen_s = torch.where(
        fallen, env._recovery_fallen_s + env.step_dt, torch.zeros_like(env._recovery_fallen_s)
    )
    env._recovery_armed |= env._recovery_fallen_s >= min_fallen_s
    fired = env._recovery_armed & up
    env._recovery_armed &= ~fired
    return fired.float()


def body_upright_linear(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
) -> torch.Tensor:
    """Linear reward for body uprightness — provides gradient at every tilt angle.

    Returns +1 when fully upright, 0 when horizontal (prone/supine), -1 when inverted.
    Unlike flat_orientation (Gaussian), this has non-zero gradient everywhere, so the
    robot always has a signal to rotate toward upright even when starting from prone.

    Computed as the z-component of the body's local Z-axis expressed in world frame,
    which equals R[2,2] = 1 - 2*(qx² + qy²) for quaternion [w, x, y, z].
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4): [w, x, y, z]
    qx = quat[:, 1]
    qy = quat[:, 2]
    reward = 1.0 - 2.0 * (qx * qx + qy * qy)
    if gate_z_below is not None:
        # Recovery-gated variant (velstand): active only while fallen, exactly
        # zero during clean walking so it can't dilute the tracking rewards.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def body_upright_gaussian(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.1,
) -> torch.Tensor:
    """Gaussian reward on tilt magnitude — sharp pull toward fully vertical.

    Complements ``body_upright_linear`` (which is ``cos(tilt)`` and whose
    gradient ``sin(tilt)`` *vanishes* at the target). This Gaussian's
    gradient is non-zero near vertical and tapers as you move away, so it
    creates a strong differential pull in the regime where the linear
    version is weakest.

    Uses ``2*(qx² + qy²) = 1 - cos(tilt) ≈ tilt²/2`` as a tilt-squared
    proxy and applies ``exp(-tilt²/std²)``. Default std=0.1 rad ≈ 5.7°.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)  # ≈ 1 − cos(tilt); small-angle: tilt²/2
    return torch.exp(-tilt_sq / (std * std))


def upright_gaussian_at_height(
    env: ManagerBasedRlEnv,
    std: float,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``body_upright_gaussian`` weighted by smoothstep on trunk z.

    Full Gaussian-upright reward when ``z >= height_high``, zero when
    ``z <= height_low``, smoothstep in between. Use this when the upright
    incentive should only apply at the target standing height — otherwise
    the policy can find a "crouch low and vertical" local optimum that
    collects upright reward without ever rising.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_g = torch.exp(-tilt_sq / (std * std))
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright_g * smooth


def body_ang_vel_at_height(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    tilt_full_deg: float | None = None,
    tilt_zero_deg: float = 45.0,
) -> torch.Tensor:
    """Trunk ``sum(ω_xy²)`` penalty gated by trunk z (and optionally tilt).

    Height-gated arrival damper: zero below ``height_low`` (ground recovery —
    flips/rolls need large trunk rotation and must stay free), full above
    ``height_high``. Same formula as mjlab's body_angular_velocity_penalty
    (world-frame ω_xy, z-rotation free) but returns the gated POSITIVE cost;
    use a negative weight.

    ``tilt_full_deg`` (optional but STRONGLY recommended): additionally gate
    by tilt — full cost only when tilt ≤ tilt_full_deg, zero when
    ≥ tilt_zero_deg, smoothstep between. LESSON (2026-07 run that broke
    front-recovery): with a height gate alone, the final straighten of a
    bent-over rise (tilt 60°→0 happening INSIDE the z gate) is itself a
    large trunk rotation — taxing it builds a reward wall right before the
    finish, and the policy parks bent-over below the gate instead. With the
    tilt gate, the approach TO vertical is free; only residual wobble
    AROUND vertical (the overshoot→tip→retry oscillation) is damped.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    cost = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if tilt_full_deg is not None:
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        s = torch.clamp(
            (tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6),
            0.0,
            1.0,
        )
        gate = gate * (s * s * (3.0 - 2.0 * s))
    return cost * gate


def standing_composite_score(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Smooth multiplicative goal-state score (product of three Gaussians).

    Returns ``height_score * upright_score * pose_score``, each ∈ [0, 1].
    Because the factors *multiply*, a deficiency in any one term collapses
    the whole reward — the policy can't claim 80% of this by being perfect
    on 2-of-3. Gradient is non-zero everywhere, so the score works during
    the rise (not just at the goal like a binary bonus would).

    Use to break Nash-equilibrium compromises (e.g., a "lean trunk at the
    right height" basin that satisfies the additive rewards' partial sums).
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_score = torch.exp(-((z - target_height) / height_std) ** 2)

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err_sq = ((joint_pos - target) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    return height_score * upright_score * pose_score


def standing_success_bonus(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_tol: float,
    upright_threshold: float,
    pose_tol: float,
    joint_indices: list,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Binary bonus: 1.0 iff height, uprightness AND pose are all within tol.

    Creates a discrete goal-state attractor that gradient-based pose/upright/
    height rewards can't fully match by themselves. Surrounding compromises
    (lean trunk to balance head-forward CoM, park 1cm short of target z,
    etc.) collect partial gradient credit but ZERO bonus — the bonus is
    available only at the true goal state, so it changes the policy's
    relative preference once the rest of the rewards have brought it close.
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_ok = (z - target_height).abs() <= height_tol

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright_ok = upright >= upright_threshold

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err = (joint_pos - target).abs().max(dim=-1).values  # tightest joint
    pose_ok = pose_err <= pose_tol

    return (height_ok & upright_ok & pose_ok).float()


def com_upward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_height: float = 0.08,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
    max_vz: float | None = None,
) -> torch.Tensor:
    """Reward upward CoM velocity to incentivize dynamic standup motion.

    Gated by height: only active while the CoM is below `max_height` (the
    standing target). Once standing, the reward is zero so the robot has no
    incentive to keep squatting to farm upward-velocity reward.

    ``max_vz`` (optional): cap the rewarded velocity. Uncapped, the reward is
    proportional to vz, which pays MORE per step for an explosive launch —
    a violent-rise incentive. With a cap, any rise ≥ max_vz earns the same,
    so the gentlest rise that reaches the cap is optimal (the |a_z| penalty
    then picks the smooth one). The bootstrap property is preserved: any
    upward motion still pays immediately.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0
    com_z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    below_target = (com_z < max_height).float()
    reward = torch.clamp(vz, min=0.0, max=max_vz) * below_target
    if gate_z_below is not None:
        # Recovery-gated (velstand): without the gate this pays for dip-and-rise
        # during gait whenever the trunk crosses max_height → bounce incentive.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def fallen_too_long(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float = 0.10,
    gate_tilt_above_deg: float = 40.0,
    max_duration_s: float = 5.0,
) -> torch.Tensor:
    """Terminate envs that have been continuously FALLEN for `max_duration_s`.

    For envs that mix walking with fall recovery (velstand): the fell_over
    termination gets disabled by curriculum so the policy can attempt recovery,
    but without a backstop a failed recovery farms recovery-reward for the whole
    20 s episode, starving the walk of data (audit: ~25% walking share). This
    gives every fall a fair recovery window, then recycles the env.
    """
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg).bool()
    if not hasattr(env, "_fallen_timer_s"):
        env._fallen_timer_s = torch.zeros(env.num_envs, device=env.device)
    # Freshly reset envs start with a clean timer.
    env._fallen_timer_s[env.episode_length_buf <= 1] = 0.0
    env._fallen_timer_s = torch.where(
        fallen, env._fallen_timer_s + env.step_dt, torch.zeros_like(env._fallen_timer_s)
    )
    return env._fallen_timer_s >= max_duration_s


def root_height_below(
    env: ManagerBasedRlEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the trunk drops below ``min_height`` in world z.

    Utilisé par roller_slope comme « tombé dans le vide » : le terrain a un
    plat de sortie au bas de la rampe, donc une descente normale ne passe
    jamais sous le niveau du plat de sortie le plus bas. Choisir min_height
    en dessous de ce niveau => la terminaison ne se déclenche que si le robot
    quitte le solide et chute dans le vide. Indépendant de la géométrie exacte
    de la rampe (longueur/pente).
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height


def descent_speed_reward(
    env: ManagerBasedRlEnv,
    cap: float = 0.8,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Récompense la vitesse d'avance vers le BAS de la pente (monde +x).

    La rampe descend en +x, donc la vitesse linéaire monde en x mesure la
    progression de descente. Plafonnée à ``cap`` m/s : encourage à se laisser
    glisser sans pousser à dévaler de plus en plus vite. Nulle si le robot
    recule/remonte (vx < 0). Sans cette récompense, l'optimum est de rester
    immobile et droit (le robot « freine » au lieu de glisser). NaN-safe.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = torch.nan_to_num(
        asset.data.root_link_lin_vel_w[:, 0], nan=0.0, posinf=0.0, neginf=0.0
    )
    return torch.clamp(vx, min=0.0, max=cap)
