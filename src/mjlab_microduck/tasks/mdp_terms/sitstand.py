"""MicroDuck sitstand terms."""

from dataclasses import dataclass as _dataclass
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
import math
import torch
from .common import (
    _DEFAULT_ASSET_CFG,
    _NECK_JOINT_PATTERNS,
    _servo_default_joint_pos,
    _servo_joint_pos,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sit↔Stand posture command + posture-conditioned rewards (sitstand env).
#
# One policy, both directions: the command is a single sit/stand flag carried
# in the twist slot (cmd = [sit_flag, 0, 0], so "stand" is the all-zero
# command — same deployment idle as every other policy). All task rewards
# below select their target (SIT keyframe + SIT_Z vs HOME + STAND_Z) from the
# live command, per env, so the same reward stack drives the descent, the
# seated rest, the rise and the standing rest. Uses the _servo_* helpers →
# backlash-model compatible.
# ─────────────────────────────────────────────────────────────────────────────


class SitStandCommand(UniformVelocityCommand):
    """Posture command: cmd = [sit_flag, 0, 0] with dwell-time resampling and a
    SLEWED internal target blend.

    sit_flag ∈ {0.0, 1.0}. Resampled by the command manager on the cfg's
    resampling_time_range (the dwell time in each posture) and on episode
    reset. cfg.sit_prob is the probability a resample commands SIT; with the
    reset-state mix this trains all four (start-state × command) combinations,
    including "hold what you're already doing".

    ``alpha`` (0 = STAND target, 1 = SIT target) slews toward the flag at a
    constant rate (full transition in cfg.ramp_s seconds) and is what the
    posture_* rewards track. THE anti-crash mechanism: with a binary target,
    arriving early pays the full goal-state jackpot for every step saved,
    while the linear speed-cap penalties integrate to a bounded excess-
    distance cost — an instant drop beat a 1 s descent by ~7×. With the
    slewed target, being AHEAD of the ramp scores ~0 on the height/composite
    stack (z far from the commanded height), so tracking the slow setpoint IS
    the argmax; the caps remain as backstops for overshoot/bounce. The OBS
    stays the raw binary flag (deployment: runtime writes 0/1; the trained
    response to a flip is the ~ramp_s glide).

    On episode reset, alpha is initialised from the robot's ACTUAL trunk
    height, not the flag — a seated spawn must not be dragged upward by a
    stand-initialised ramp (and vice versa).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sit_prob = float(getattr(cfg, "sit_prob", 0.5))
        self._ramp_s = float(getattr(cfg, "ramp_s", 2.0))
        self._sit_z = float(getattr(cfg, "sit_z", 0.060))
        self._stand_z = float(getattr(cfg, "stand_z", 0.115))
        self._env_ref = env
        self._alpha = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def alpha(self) -> torch.Tensor:
        """Slewed target blend: 0 = STAND target, 1 = SIT target."""
        return self._alpha

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        sit = (torch.rand(n, device=self.device) < self._sit_prob).float()
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = sit

    def _alpha_from_height(self) -> torch.Tensor:
        z = torch.nan_to_num(
            self.robot.data.root_link_pos_w[:, 2]
            - self._env_ref.scene.terrain.env_origins[:, 2],
            nan=self._stand_z,
        )
        return torch.clamp(
            (self._stand_z - z) / max(self._stand_z - self._sit_z, 1e-6), 0.0, 1.0
        )

    def compute(self, dt: float) -> None:
        super().compute(dt)
        # Episode-start re-init of the blend from the ACTUAL trunk height.
        # Done here (not in reset()) because the command manager resets BEFORE
        # the set_ground_state event teleports the robot, so reset() would read
        # the pre-teleport height. On the first compute of an episode the spawn
        # state is in place.
        fresh = self._env_ref.episode_length_buf <= 1
        if fresh.any():
            self._alpha = torch.where(fresh, self._alpha_from_height(), self._alpha)
        # Constant-rate slew of the target blend toward the commanded flag.
        step = dt / max(self._ramp_s, 1e-6)
        delta = self.vel_command_b[:, 0] - self._alpha
        self._alpha += torch.clamp(delta, -step, step)

    def _update_command(self) -> None:
        pass  # No heading controller / standing-env machinery.

    def _update_metrics(self) -> None:
        pass  # No velocity-tracking metrics for a posture flag.


@_dataclass(kw_only=True)
class SitStandCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SitStandCommand
    # Probability that a resample commands SIT (vs STAND).
    sit_prob: float = 0.5
    # Seconds for the internal target blend to traverse STAND↔SIT in full.
    ramp_s: float = 2.0
    # Rest heights, used to initialise the blend from the spawn state.
    sit_z: float = 0.060
    stand_z: float = 0.115

    def build(self, env: ManagerBasedRlEnv) -> "SitStandCommand":
        return SitStandCommand(self, env)


def _posture_blend(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target blend ∈ [0, 1] (0 = STAND, 1 = SIT) for the posture rewards.

    Uses the SitStandCommand's slewed ``alpha`` (the moving setpoint) when the
    term exposes it; falls back to the raw binary flag otherwise.
    """
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is not None:
        return alpha
    return env.command_manager.get_command(command_name)[:, 0]


def _posture_targets(
    env: ManagerBasedRlEnv,
    asset: Entity,
    command_name: str,
    sit_overrides: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(target blend, per-env joint target) for the commanded posture.

    STAND target = default_joint_pos (HOME); SIT target = HOME with the
    keyframe overrides applied; the SLEWED blend interpolates between them,
    so mid-ramp the rewarded pose folds in sync with the descending height.
    """
    blend = _posture_blend(env, command_name)
    stand_target = _servo_default_joint_pos(env, asset)
    sit_target = stand_target.clone()
    for idx, val in sit_overrides.items():
        sit_target[:, idx] = val
    target = stand_target + blend.unsqueeze(-1) * (sit_target - stand_target)
    return blend, target


def _posture_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(slewed target trunk z, actual trunk z) per env."""
    blend = _posture_blend(env, command_name)
    target_z = stand_z + blend * (sit_z - stand_z)
    asset = env.scene["robot"]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return target_z, z


def posture_pose_match(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian pose-match against the commanded posture's target pose."""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def posture_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to ``posture_pose_match`` (constant gradient to target)."""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def posture_height_gaussian(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian on trunk z against the commanded posture's target height."""
    del asset_cfg  # trunk z read via _posture_height
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return torch.exp(-((z - target_z) / std) ** 2)


def posture_height_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to ``posture_height_gaussian`` — the transition driver.

    While the robot rests in the *wrong* posture this charges a constant
    per-step cost (~|Δz| = 55 mm), which is what makes "ignore the command"
    a net-negative strategy in both directions.
    """
    del asset_cfg
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return -torch.abs(z - target_z)


def posture_composite(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    sit_z: float,
    stand_z: float,
    height_std: float = 0.03,
    upright_std: float = 0.40,
    pose_std: float = 0.40,
    head_std: float | None = None,
    head_command_name: str = "head_pose",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Multiplicative goal score vs the commanded posture (height·upright·pose
    [·head]).

    The posture-conditioned version of ``standing_composite_score``: a
    deficiency in any factor collapses the whole term, so partial-sum
    compromises (plank, flop, lean) never pay. Both rest states demand an
    upright trunk, so the upright factor is posture-independent.

    ``head_std`` (optional): adds a fourth factor on the neck/head joints vs
    the ``head_pose`` command (same error convention as head_pose_tracking).
    Without it the goal state is head-blind: the trained policy rested with
    the head dangling to the floor — trunk upright, legs in pose, z on target
    all held while the head hung, costing only the light tracking term. With
    the factor, "arrived" REQUIRES the head at its commanded pose, so head
    assist stays free mid-transition (composite is ≈0 there anyway) but must
    be retracted to collect the goal reward.
    """
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)

    height_score = torch.exp(-((z - target_z) / height_std) ** 2)

    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    pose_err_sq = ((joint_pos - target[:, joint_indices]) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    score = height_score * upright_score * pose_score

    if head_std is not None:
        if not hasattr(env, "_head_pose_neck_ids"):
            ids, _ = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
            env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        neck_ids = env._head_pose_neck_ids
        head_cmd = env.command_manager.get_command(head_command_name)
        actual = asset.data.joint_pos[:, neck_ids] - asset.data.default_joint_pos[:, neck_ids]
        head_err_sq = ((actual - head_cmd) ** 2).mean(dim=-1)
        score = score * torch.exp(-head_err_sq / (head_std * head_std))

    return score


def posture_stillness(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    band_full: float = 0.012,
    band_zero: float = 0.03,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward trunk stillness while AT the commanded posture, upright.

    Generalizes ``seated_stillness`` to both rest states: exp(-(|v|/std)²)
    gated by a smoothstep on |z − commanded z| (full inside ``band_full``,
    zero beyond ``band_zero`` → inactive during transitions) and by trunk
    tilt (a tilted rest — back/face/side — earns nothing). Additionally gated
    on the target ramp being COMPLETE (|flag − alpha| small), so stillness
    never pays mid-transition. Makes "rest quietly, upright, at the commanded
    height" the peak of the stack.
    """
    asset = env.scene[asset_cfg.name]
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)

    flag = env.command_manager.get_command(command_name)[:, 0]
    blend = _posture_blend(env, command_name)
    ramp_done = ((flag - blend).abs() < 0.02).float()

    err = torch.abs(z - target_z)
    t = torch.clamp((band_zero - err) / max(band_zero - band_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)

    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)

    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate * ramp_done


def posture_rise_bootstrap(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_height: float,
    max_vz: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Upward-vz reward, active only when STAND is commanded and z < max_height.

    The standup-env lesson: destination-only rewards have zero gradient at
    zero motion, so "stay seated and eat the L1" is a local optimum — paying
    for the rise *motion* itself makes any attempt immediately positive.
    Gated off above ``max_height`` (set just ABOVE the stand target so the
    final cm still pays; gating at exactly STAND_Z parks the policy short).
    Zero whenever SIT is commanded, so it can never fight the descent.
    ``max_vz`` caps the rewarded speed (any rise ≥ the cap earns the same, so
    an explosive launch can't out-earn a gentle one).
    """
    asset = env.scene[asset_cfg.name]
    sit = env.command_manager.get_command(command_name)[:, 0]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return torch.clamp(vz, min=0.0, max=max_vz) * (z < max_height).float() * (1.0 - sit)


def trunk_upward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_up_vel: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty on upward trunk velocity beyond ``max_up_vel``.

    Mirror of ``trunk_downward_velocity_penalty`` for the rise: charges every
    step of a too-fast (violent) stand-up, so the explosive rise can't be
    amortised against arriving-standing reward. Zero at rest, for any rise
    slower than the cap, and for all downward motion. Introduce via
    curriculum AFTER the rise is discovered (attempt-tax lesson).
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(vz - max_up_vel, min=0.0)
