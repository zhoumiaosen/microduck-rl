"""MicroDuck swing terms."""

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat
import math
import torch
from .common import (
    _DEFAULT_ASSET_CFG,
)


# =============================================================================
# Swing pumping — ideal-string pendulum state and rewards
# =============================================================================

_SWING_SITE_NAMES = (
    "swing_anchor_left",
    "swing_anchor_right",
    "swing_attach_left",
    "swing_attach_right",
)


# Positive swing reward is paid only while the mechanism remains inside a
# physically credible envelope.  The ordinary penalties remain active and
# provide gradients everywhere; this multiplicative gate prevents a policy
# from trading a little more arc for a visibly diagonal/crossed-string swing.
_SWING_GATE_LATERAL_SAFE_M = 0.012


_SWING_GATE_LATERAL_SCALE_M = 0.004


_SWING_GATE_ALIGNMENT_SAFE = 0.04


_SWING_GATE_ALIGNMENT_SCALE = 0.005


_SWING_GATE_DEEP_SLACK_SAFE_M = 0.0095


_SWING_GATE_DEEP_SLACK_SCALE_M = 0.003


_SWING_GATE_EXTENSION_SAFE_M = 0.012


_SWING_GATE_EXTENSION_SCALE_M = 0.001


def _swing_kinematics(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return angle, angular rate, lateral offset, slack, and length imbalance.

    Angle is measured in the sagittal x-z plane from the hanging-down
    direction using the midpoint of the two strings.  Site velocity gives an
    exact phase rate without differentiating a wrapped angle.  Slack and
    imbalance are measured from the actual spatial-tendon lengths, so reward
    shaping cannot silently treat the strings as rods.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = env.__dict__.setdefault("_swing_kinematics_cache", {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        site_ids, _ = asset.find_sites(_SWING_SITE_NAMES, preserve_order=True)
        tendon_ids, _ = asset.find_tendons(
            ("swing_string_left", "swing_string_right"), preserve_order=True
        )
        ids = (site_ids, tendon_ids)
        cache[key] = ids
    site_ids, tendon_ids = ids

    sites = asset.data.site_pos_w[:, site_ids]
    site_vel = asset.data.site_lin_vel_w[:, site_ids]
    anchor = 0.5 * (sites[:, 0] + sites[:, 1])
    attach = 0.5 * (sites[:, 2] + sites[:, 3])
    attach_vel = 0.5 * (site_vel[:, 2] + site_vel[:, 3])
    rope = attach - anchor

    x, y, z = rope[:, 0], rope[:, 1], rope[:, 2]
    vx, vz = attach_vel[:, 0], attach_vel[:, 2]
    radius_sq = torch.clamp(x * x + z * z, min=1e-8)
    angle = torch.atan2(x, -z)
    angle_rate = (-z * vx + x * vz) / radius_sq

    nominal_length = 0.38
    lateral = y / nominal_length
    lengths = asset.data.tendon_len[:, tendon_ids]
    # A little constraint tolerance is normal. Only pay attention to genuine
    # loss of tension (the tendon being shorter than its maximum length).
    # One loose cord is already a failure of two-string support.  Taking the
    # maximum (rather than averaging both cords) prevents a taut cord from
    # hiding a deeply slack partner during a crossed/twisted swing.
    slack = torch.clamp(nominal_length - lengths - 5e-4, min=0.0).amax(dim=1)
    imbalance = torch.abs(lengths[:, 0] - lengths[:, 1])
    return angle, angle_rate, lateral, slack, imbalance


def _swing_string_lengths(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return both actual spatial-tendon lengths with cached tendon IDs."""
    asset: Entity = env.scene[asset_cfg.name]
    cache = env.__dict__.setdefault("_swing_string_tendon_ids_cache", {})
    key = id(asset)
    tendon_ids = cache.get(key)
    if tendon_ids is None:
        tendon_ids, _ = asset.find_tendons(
            ("swing_string_left", "swing_string_right"), preserve_order=True
        )
        cache[key] = tendon_ids
    return asset.data.tendon_len[:, tendon_ids]


def swing_state_observation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Privileged critic state: normalized angle/rate/lateral/string slack."""
    angle, rate, lateral, slack, _ = _swing_kinematics(env, asset_cfg)
    return torch.stack(
        (angle / math.pi, rate / 8.0, lateral, slack / 0.02), dim=-1
    )


def swing_physical_validity_factor(
    lateral_m: torch.Tensor,
    alignment_penalty: torch.Tensor,
    deepest_string_slack_m: torch.Tensor,
    longest_string_extension_m: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smoothly suppress positive reward outside the strict swing envelope.

    The factor is exactly one in the safe interior and decays rapidly before
    the 20 mm lateral and 0.05 alignment validation boundaries.  Deep string
    slack is included only beyond the 370 mm length threshold; ordinary
    unilateral tension redistribution remains governed by the separate slack
    penalty rather than being treated as impossible geometry.
    """
    lateral_excess = torch.clamp(
        torch.abs(lateral_m) - _SWING_GATE_LATERAL_SAFE_M, min=0.0
    )
    alignment_excess = torch.clamp(
        alignment_penalty - _SWING_GATE_ALIGNMENT_SAFE, min=0.0
    )
    slack_excess = torch.clamp(
        deepest_string_slack_m - _SWING_GATE_DEEP_SLACK_SAFE_M, min=0.0
    )
    if longest_string_extension_m is None:
        longest_string_extension_m = torch.zeros_like(lateral_m)
    extension_excess = torch.clamp(
        longest_string_extension_m - _SWING_GATE_EXTENSION_SAFE_M, min=0.0
    )
    normalized_cost = (
        (lateral_excess / _SWING_GATE_LATERAL_SCALE_M).square()
        + (alignment_excess / _SWING_GATE_ALIGNMENT_SCALE).square()
        + (slack_excess / _SWING_GATE_DEEP_SLACK_SCALE_M).square()
        + (extension_excess / _SWING_GATE_EXTENSION_SCALE_M).square()
    )
    return torch.exp(-torch.clamp(normalized_cost, max=20.0))


def swing_physical_validity_gate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Current validity factor, irreversibly zero after sustained violation."""
    _, _, lateral, slack, _ = _swing_kinematics(env, asset_cfg)
    alignment = swing_attachment_alignment_penalty(env, asset_cfg)
    lengths = _swing_string_lengths(env, asset_cfg)
    longest_extension = lengths.amax(dim=1) - 0.38
    cord_invalid = (lengths.amax(dim=1) > 0.394) | (
        lengths.amax(dim=1) < 0.370
    )
    current_factor = swing_physical_validity_factor(
        lateral * 0.38, alignment, slack, longest_extension
    )
    invalid_episode = _swing_update_episode_invalid_latch(
        env, lateral * 0.38, alignment, cord_invalid=cord_invalid
    )
    return current_factor * (~invalid_episode).float()


def _swing_update_episode_invalid_latch(
    env: ManagerBasedRlEnv,
    lateral_m: torch.Tensor,
    alignment_penalty: torch.Tensor,
    lateral_limit_m: float = 0.020,
    alignment_limit: float = 0.050,
    grace_steps: int = 2,
    cord_invalid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Latch a sustained strict-gate violation for the rest of the episode.

    Reward terms call the validity gate several times per control step.  The
    global step cache makes this state update exactly once, while keeping the
    environment alive for the complete 24-second hardware-relevant rollout.
    """
    if lateral_limit_m <= 0.0 or alignment_limit <= 0.0:
        raise ValueError("swing geometry limits must be positive")
    if grace_steps < 1:
        raise ValueError("grace_steps must be at least one")
    step = int(env.common_step_counter)
    cached_step = getattr(env, "_swing_invalid_latch_cache_step", None)
    if cached_step == step:
        return env._swing_episode_invalid

    count = getattr(env, "_swing_invalid_geometry_steps", None)
    latched = getattr(env, "_swing_episode_invalid", None)
    if count is None or count.shape[0] != env.num_envs:
        count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    if latched is None or latched.shape[0] != env.num_envs:
        latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    fresh = env.episode_length_buf <= 1
    count = torch.where(fresh, torch.zeros_like(count), count)
    latched = torch.where(fresh, torch.zeros_like(latched), latched)
    invalid = (torch.abs(lateral_m) > lateral_limit_m) | (
        alignment_penalty > alignment_limit
    )
    if cord_invalid is not None:
        invalid = invalid | cord_invalid
    count = torch.where(invalid, count + 1, torch.zeros_like(count))
    latched = latched | (count >= grace_steps)
    env._swing_invalid_geometry_steps = count
    env._swing_episode_invalid = latched
    env._swing_invalid_latch_cache_step = step
    return latched


def swing_invalid_episode_debt(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Persistent, action-sensitive cost after a sustained invalid mode.

    The unit baseline is irreversible for the rest of the episode, preserving
    the full-horizon debt semantics.  Its bounded severity multiplier remains
    sensitive to the current geometry, however, so PPO can still distinguish
    actions that recover toward the plane from actions that let the violation
    grow.  A purely binary debt provided no such post-latch credit assignment.
    """
    _, _, lateral, _, _ = _swing_kinematics(env, asset_cfg)
    alignment = swing_attachment_alignment_penalty(env, asset_cfg)
    lateral_m = lateral * 0.38
    lengths = _swing_string_lengths(env, asset_cfg)
    cord_invalid = (lengths.amax(dim=1) > 0.394) | (
        lengths.amax(dim=1) < 0.370
    )
    invalid_episode = _swing_update_episode_invalid_latch(
        env, lateral * 0.38, alignment, cord_invalid=cord_invalid
    )
    normalized_severity = torch.maximum(
        (torch.abs(lateral_m) / 0.020).square(),
        alignment / 0.050,
    )
    extension_severity = (
        torch.clamp(lengths.amax(dim=1) - 0.392, min=0.0) / 0.002
    ).square()
    normalized_severity = torch.maximum(
        normalized_severity, extension_severity
    ).clamp(max=8.0)
    return invalid_episode.float() * (1.0 + normalized_severity)


def swing_validity_observation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Six-dimensional privileged critic context for physical validity.

    The deployable actor still receives only IMU/encoder/history signals.
    These exact mechanism quantities make the critic Markov for the gated
    reward without changing either actor or critic tensor dimensionality.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _, _, lateral, slack, imbalance = _swing_kinematics(env, asset_cfg)
    alignment = swing_attachment_alignment_penalty(env, asset_cfg)
    lengths = _swing_string_lengths(env, asset_cfg)
    lateral_velocity = torch.nan_to_num(
        asset.data.root_link_lin_vel_w[:, 1], nan=0.0, posinf=0.0, neginf=0.0
    )
    angular_velocity = torch.nan_to_num(
        asset.data.root_link_ang_vel_w, nan=0.0, posinf=0.0, neginf=0.0
    )
    invalid_episode = getattr(
        env,
        "_swing_episode_invalid",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    invalid_episode = torch.where(
        env.episode_length_buf <= 1,
        torch.zeros_like(invalid_episode),
        invalid_episode,
    )
    extension_fault = torch.clamp(
        lengths.amax(dim=1) - 0.392, min=0.0
    ) / 0.002
    cord_fault = torch.maximum(
        torch.maximum(slack / 0.01, imbalance / 0.01), extension_fault
    )
    return torch.stack(
        (
            lateral * 0.38 / 0.02,
            lateral_velocity / 0.2,
            angular_velocity[:, 0] / 2.0,
            angular_velocity[:, 2] / 2.0,
            alignment / 0.05,
            torch.maximum(cord_fault, 2.0 * invalid_episode.float()),
        ),
        dim=-1,
    )


def swing_plane_heading_observation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Deployable IMU cue for alignment with the initial swing plane.

    A pure fore-aft swing rotates about world y, so the trunk's body-y axis
    remains in world y and its world x/z components stay zero. Those two
    components reveal yaw/roll drift that projected gravity alone cannot
    observe, particularly the yaw accumulated from left/right actuator
    mismatch. The leading zero preserves the existing three-slot command
    layout and makes the mirror transform ``[even, odd, odd]`` unchanged.

    Hardware needs only the orientation estimate already produced from the
    IMU, expressed relative to the known still-start frame; no string, anchor,
    camera, or motion-capture state is exposed to the actor.
    """
    asset: Entity = env.scene[asset_cfg.name]
    rotation_w_b = matrix_from_quat(asset.data.root_link_quat_w)
    body_y_axis_w = rotation_w_b[:, :, 1]
    zeros = torch.zeros_like(body_y_axis_w[:, 0])
    return torch.stack(
        (
            zeros,
            torch.nan_to_num(body_y_axis_w[:, 0], nan=0.0),
            torch.nan_to_num(body_y_axis_w[:, 2], nan=0.0),
        ),
        dim=-1,
    )


def swing_frontier_observation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Privileged critic context for the history-dependent frontier reward.

    The deployable actor remains entirely proprioceptive. The critic, however,
    must distinguish two identical instantaneous poses reached before versus
    after a large peak; otherwise their future frontier rewards differ despite
    identical critic observations. This reuses the critic's three zero command
    slots, preserving checkpoint tensor shapes during continuation.
    """
    zeros = torch.zeros(env.num_envs, device=env.device)
    positive = getattr(env, "_swing_peak_positive", zeros)
    negative = getattr(env, "_swing_peak_negative", zeros)
    fresh = env.episode_length_buf <= 1
    positive = torch.where(fresh, zeros, positive)
    negative = torch.where(fresh, zeros, negative)
    progress = torch.clamp(
        env.episode_length_buf.float() / float(env.max_episode_length),
        0.0,
        1.0,
    )
    return torch.stack(
        (
            positive / math.pi,
            negative / math.pi,
            progress,
        ),
        dim=-1,
    )


def swing_height_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Normalized rise of the seat midpoint above the hanging bottom.

    Potential rise is already quadratic in small swing angle. Squaring it
    again made the useful 10--20 degree regime scale as angle^4, leaving almost
    no dense gradient after the policy discovered a small sustaining pump.
    """
    angle, _, _, _, _ = _swing_kinematics(env, asset_cfg)
    height = torch.clamp(1.0 - torch.cos(angle), 0.0, 2.0)
    return height * swing_physical_validity_gate(env, asset_cfg)


def swing_late_height_reward(
    env: ManagerBasedRlEnv,
    time_power: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Favor large arcs late in the rollout, not just a transient reset spike.

    The policy receives no episode clock, so it cannot farm this term through
    a scripted time schedule. It must infer amplitude and phase from deployable
    proprioception and maintain or build the physical swing. The critic gets
    episode progress through its privileged frontier slots to keep value
    estimation Markov.
    """
    angle, _, _, _, _ = _swing_kinematics(env, asset_cfg)
    height = torch.clamp(1.0 - torch.cos(angle), 0.0, 2.0)
    progress = torch.clamp(
        env.episode_length_buf.float() / float(env.max_episode_length),
        0.0,
        1.0,
    )
    return (
        height
        * torch.pow(progress, time_power)
        * swing_physical_validity_gate(env, asset_cfg)
    )


def swing_energy_reward(
    env: ManagerBasedRlEnv,
    max_equivalent_height: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pendulum energy expressed as equivalent rise in string lengths.

    Kinetic energy at the bottom is useful because it predicts the height the
    swing will reach a moment later.  Clipping prevents a numerical or
    high-frequency velocity spike from dominating the objective.
    """
    angle, rate, _, _, _ = _swing_kinematics(env, asset_cfg)
    equivalent_height = 1.0 - torch.cos(angle) + 0.5 * (0.38 / 9.81) * rate.square()
    return torch.clamp(
        equivalent_height, 0.0, max_equivalent_height
    ) * swing_physical_validity_gate(env, asset_cfg)


def swing_bidirectional_peak_progress(
    env: ManagerBasedRlEnv,
    target_angle: float = math.pi,
    max_paid_rate: float = 8.0,
    frontier_power: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay only new frontiers on both sides of the swing.

    This is deliberately non-farmable: holding an angle or repeating an
    already-reached arc pays zero.  Separate positive and negative frontiers
    prevent a one-sided fling from satisfying the pumping objective. A power
    greater than one makes later, genuinely large breakthroughs much more
    valuable than endlessly polishing the first few degrees.
    """
    angle, _, _, _, _ = _swing_kinematics(env, asset_cfg)
    fresh = env.episode_length_buf <= 1
    if not hasattr(env, "_swing_peak_positive"):
        env._swing_peak_positive = torch.zeros(env.num_envs, device=env.device)
        env._swing_peak_negative = torch.zeros(env.num_envs, device=env.device)
    pos = torch.clamp(angle, min=0.0, max=target_angle)
    neg = torch.clamp(-angle, min=0.0, max=target_angle)
    env._swing_peak_positive[fresh] = 0.0
    env._swing_peak_negative[fresh] = 0.0
    old_pos = env._swing_peak_positive
    old_neg = env._swing_peak_negative
    dpos = torch.clamp(pos - old_pos, min=0.0)
    dneg = torch.clamp(neg - old_neg, min=0.0)
    cap = max_paid_rate * env.step_dt
    paid_pos = old_pos + torch.clamp(dpos, max=cap)
    paid_neg = old_neg + torch.clamp(dneg, max=cap)
    paid = (
        torch.pow(paid_pos / target_angle, frontier_power)
        - torch.pow(old_pos / target_angle, frontier_power)
        + torch.pow(paid_neg / target_angle, frontier_power)
        - torch.pow(old_neg / target_angle, frontier_power)
    )
    env._swing_peak_positive = torch.maximum(env._swing_peak_positive, pos)
    env._swing_peak_negative = torch.maximum(env._swing_peak_negative, neg)
    return paid * swing_physical_validity_gate(env, asset_cfg) / (
        2.0 * env.step_dt
    )


def swing_lateral_penalty(
    env: ManagerBasedRlEnv,
    tolerance_m: float = 0.03,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Squared out-of-plane displacement, scaled to a useful tolerance.

    The kinematic helper normalizes lateral displacement by string length for
    observations.  Reward scaling it by the full 38 cm string made even a
    visibly severe 10--15 cm excursion cheap.  Three centimetres is a much
    more meaningful tolerance for this small seat.
    """
    _, _, lateral, _, _ = _swing_kinematics(env, asset_cfg)
    lateral_m = lateral * 0.38
    return (lateral_m / tolerance_m).square()


def swing_lateral_barrier_penalty(
    env: ManagerBasedRlEnv,
    safe_m: float = 0.018,
    scale_m: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Hinge barrier for out-of-plane excursions beyond the accepted envelope.

    The ordinary quadratic term softly encourages a planar swing everywhere.
    This term is deliberately almost free inside the envelope and rises only
    near the 20 mm validation gate, so a high-energy planar policy is not
    globally taxed to prevent rare large sideways excursions.
    """
    if safe_m < 0.0 or scale_m <= 0.0:
        raise ValueError("safe_m must be non-negative and scale_m positive")
    _, _, lateral, _, _ = _swing_kinematics(env, asset_cfg)
    excess_m = torch.clamp(torch.abs(lateral * 0.38) - safe_m, min=0.0)
    return (excess_m / scale_m).square()


def swing_lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    tolerance_m_s: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Suppress out-of-plane kinetic energy while preserving fore-aft speed.

    A displacement-only barrier is weakest at the centre crossing, exactly
    where a diagonal swing carries maximum lateral kinetic energy. Penalizing
    the trunk's world-y velocity prevents repeatedly rebuilding that mode
    without taxing the desired world-x pendulum velocity.
    """
    if tolerance_m_s <= 0.0:
        raise ValueError("tolerance_m_s must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    lateral_velocity = torch.nan_to_num(
        asset.data.root_link_lin_vel_w[:, 1], nan=0.0, posinf=0.0, neginf=0.0
    )
    return (lateral_velocity / tolerance_m_s).square()


def swing_out_of_plane_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    tolerance_rad_s: float = 1.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Suppress trunk roll/yaw kinetics while leaving swing pitch unpenalized.

    The desired planar pendulum rotates around world y. Squared world-x/z
    angular speed captures the kinetic precursor to attachment-span roll/yaw
    without taxing the fore-aft pumping degree of freedom.
    """
    if tolerance_rad_s <= 0.0:
        raise ValueError("tolerance_rad_s must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    angular_velocity = torch.nan_to_num(
        asset.data.root_link_ang_vel_w, nan=0.0, posinf=0.0, neginf=0.0
    )
    out_of_plane_squared = (
        angular_velocity[:, 0].square() + angular_velocity[:, 2].square()
    )
    return out_of_plane_squared / (tolerance_rad_s * tolerance_rad_s)


def swing_attachment_alignment_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize yaw/roll that crosses the seat attachment points.

    Midpoint-only pendulum rewards cannot distinguish a normal planar swing
    from rotating the seat sideways while the two strings cross.  Comparing
    the directed left-to-right attachment span with the corresponding anchor
    span gives zero for an aligned seat, one at 90 degrees, and two when the
    attachment order has fully flipped.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = env.__dict__.setdefault("_swing_alignment_site_ids_cache", {})
    key = id(asset)
    site_ids = cache.get(key)
    if site_ids is None:
        site_ids, _ = asset.find_sites(_SWING_SITE_NAMES, preserve_order=True)
        cache[key] = site_ids
    sites = asset.data.site_pos_w[:, site_ids]
    anchor_span = sites[:, 1] - sites[:, 0]
    attach_span = sites[:, 3] - sites[:, 2]
    denom = torch.clamp(
        torch.linalg.vector_norm(anchor_span, dim=1)
        * torch.linalg.vector_norm(attach_span, dim=1),
        min=1e-8,
    )
    cosine = torch.clamp(
        torch.sum(anchor_span * attach_span, dim=1) / denom,
        min=-1.0,
        max=1.0,
    )
    return 1.0 - cosine


def swing_attachment_alignment_barrier_penalty(
    env: ManagerBasedRlEnv,
    safe_penalty: float = 0.04,
    scale: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Hinge barrier for rare seat/string yaw excursions.

    ``swing_attachment_alignment_penalty`` is approximately half the squared
    misalignment angle near zero. Keeping a soft term preserves centering;
    this barrier makes only excursions near the 0.05 validation gate sharply
    uneconomical, avoiding the oscillation between clean-lateral and
    clean-alignment policies seen with larger global weights alone.
    """
    if safe_penalty < 0.0 or scale <= 0.0:
        raise ValueError("safe_penalty must be non-negative and scale positive")
    penalty = swing_attachment_alignment_penalty(env, asset_cfg)
    excess = torch.clamp(penalty - safe_penalty, min=0.0)
    return (excess / scale).square()


def swing_predictive_geometry_barrier_from_values(
    predicted_lateral_m: torch.Tensor,
    predicted_alignment_penalty: torch.Tensor,
    lateral_safe_m: float = 0.012,
    lateral_scale_m: float = 0.008,
    alignment_safe_penalty: float = 0.04,
    alignment_scale: float = 0.01,
    max_penalty: float = 16.0,
) -> torch.Tensor:
    """Continuous barrier on short-horizon predicted gate geometry."""
    if lateral_safe_m < 0.0 or lateral_scale_m <= 0.0:
        raise ValueError("invalid predictive lateral barrier parameters")
    if alignment_safe_penalty < 0.0 or alignment_scale <= 0.0:
        raise ValueError("invalid predictive alignment barrier parameters")
    if max_penalty <= 0.0:
        raise ValueError("max_penalty must be positive")
    lateral_excess = torch.clamp(
        torch.abs(predicted_lateral_m) - lateral_safe_m, min=0.0
    )
    alignment_excess = torch.clamp(
        predicted_alignment_penalty - alignment_safe_penalty, min=0.0
    )
    raw_penalty = (lateral_excess / lateral_scale_m).square() + (
        alignment_excess / alignment_scale
    ).square()
    # Constant-velocity projection is deliberately local. At high angular
    # speed its first-order attachment span can briefly approach zero, making
    # the normalized direction ill-conditioned and generating enormous
    # extrapolation-only costs. The cap retains useful pre-gate gradients but
    # delegates severe states to current-geometry penalties and violation debt.
    return torch.clamp(raw_penalty, max=max_penalty)


def swing_predictive_geometry_barrier_penalty(
    env: ManagerBasedRlEnv,
    horizon_s: float = 0.08,
    lateral_safe_m: float = 0.012,
    lateral_scale_m: float = 0.008,
    alignment_safe_penalty: float = 0.04,
    alignment_scale: float = 0.01,
    max_penalty: float = 16.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize a gate crossing predicted from current mechanism velocity.

    Current-position barriers react only after a sideways mode has developed.
    An 80 ms constant-velocity projection supplies a dense precursor signal:
    lateral translation uses trunk world-y velocity, while attachment
    alignment uses the exact relative velocity of the two seat sites.  The
    reward is privileged during training; the actor contract remains the same
    deployable IMU/encoder observation vector.
    """
    if horizon_s < 0.0:
        raise ValueError("horizon_s must be non-negative")
    asset: Entity = env.scene[asset_cfg.name]
    _, _, lateral, _, _ = _swing_kinematics(env, asset_cfg)
    lateral_velocity = torch.nan_to_num(
        asset.data.root_link_lin_vel_w[:, 1], nan=0.0, posinf=0.0, neginf=0.0
    )
    predicted_lateral_m = lateral * 0.38 + horizon_s * lateral_velocity

    cache = env.__dict__.setdefault("_swing_alignment_site_ids_cache", {})
    key = id(asset)
    site_ids = cache.get(key)
    if site_ids is None:
        site_ids, _ = asset.find_sites(_SWING_SITE_NAMES, preserve_order=True)
        cache[key] = site_ids
    sites = asset.data.site_pos_w[:, site_ids]
    site_velocities = asset.data.site_lin_vel_w[:, site_ids]
    anchor_span = sites[:, 1] - sites[:, 0]
    attachment_span = sites[:, 3] - sites[:, 2]
    attachment_span_velocity = site_velocities[:, 3] - site_velocities[:, 2]
    predicted_attachment_span = (
        attachment_span + horizon_s * attachment_span_velocity
    )
    denom = torch.clamp(
        torch.linalg.vector_norm(anchor_span, dim=1)
        * torch.linalg.vector_norm(predicted_attachment_span, dim=1),
        min=1e-8,
    )
    predicted_cosine = torch.clamp(
        torch.sum(anchor_span * predicted_attachment_span, dim=1) / denom,
        min=-1.0,
        max=1.0,
    )
    predicted_alignment_penalty = 1.0 - predicted_cosine
    return swing_predictive_geometry_barrier_from_values(
        predicted_lateral_m,
        predicted_alignment_penalty,
        lateral_safe_m=lateral_safe_m,
        lateral_scale_m=lateral_scale_m,
        alignment_safe_penalty=alignment_safe_penalty,
        alignment_scale=alignment_scale,
        max_penalty=max_penalty,
    )


def swing_string_slack_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize slack or unequal string loading, normalized to centimetres."""
    _, _, _, slack, imbalance = _swing_kinematics(env, asset_cfg)
    return (slack / 0.01).square() + (imbalance / 0.01).square()


def swing_string_extension_barrier_from_lengths(
    lengths_m: torch.Tensor,
    safe_length_m: float = 0.392,
    scale_m: float = 0.002,
    max_penalty: float = 16.0,
) -> torch.Tensor:
    """Squared pre-limit barrier on the longer of the two swing strings."""
    if safe_length_m <= 0.0 or scale_m <= 0.0 or max_penalty <= 0.0:
        raise ValueError("invalid string-extension barrier parameters")
    excess = torch.clamp(lengths_m.amax(dim=-1) - safe_length_m, min=0.0)
    return torch.clamp((excess / scale_m).square(), max=max_penalty)


def swing_string_extension_barrier_penalty(
    env: ManagerBasedRlEnv,
    safe_length_m: float = 0.392,
    scale_m: float = 0.002,
    max_penalty: float = 16.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize approaching the 0.395 m physical tendon limit."""
    return swing_string_extension_barrier_from_lengths(
        _swing_string_lengths(env, asset_cfg),
        safe_length_m=safe_length_m,
        scale_m=scale_m,
        max_penalty=max_penalty,
    )
