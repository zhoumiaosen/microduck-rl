"""MicroDuck events terms."""

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from typing import Optional
import math
import numpy as np
import torch
from .common import (
    _DEFAULT_ASSET_CFG,
    _servo_joint_ids,
)


# ==============================================================================
# Domain Randomization Events
# ==============================================================================


def randomize_delayed_actuator_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    operation: str = "scale",
):
    """Randomize firmware PD gains per episode (NON-accumulating).

    Under the canonical BAM actuator (``bam.mjlab.BamActuator``) gains are scaled
    per-env via ``set_gains``/``reset_gains`` (the actuator owns ``kp_scale``/
    ``kd_scale``), so we never touch the MuJoCo model — no accumulation risk. The
    sampled per-joint factors are averaged into a single scalar per env (the
    actuator applies one scale across its joints), matching the previous behavior.
    Non-BAM actuators are skipped (e.g. the roller XmlActuator, which doesn't
    expose set_gains).

    Args:
        env: The environment
        env_ids: Environment IDs to randomize (None = all envs)
        kp_range: (min, max) for kp randomization
        kd_range: (min, max) for kd randomization
        asset_cfg: Asset configuration
        operation: unused (kept for cfg compatibility; scaling is always applied)
    """
    del operation
    from bam.mjlab import BamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    for actuator in asset.actuators:
        if not isinstance(actuator, BamActuator):
            continue
        n_joints = len(actuator.ctrl_ids)
        kp_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kp_range[1] - kp_range[0]) + kp_range[0]
        kd_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kd_range[1] - kd_range[0]) + kd_range[0]
        # Restore nominal first (prevents accumulation), then apply fresh scale.
        actuator.reset_gains(env_ids)
        actuator.set_gains(
            env_ids,
            kp_scale=kp_samples.mean(dim=1, keepdim=True),
            kd_scale=kd_samples.mean(dim=1, keepdim=True),
        )


@requires_model_fields("dof_frictionloss", "dof_damping")
def expand_bam_friction_fields(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
):
    """No-op startup event whose only purpose is the decorator above.

    bam's BamActuator (mjlab_frictionloss branch) writes a per-env friction
    budget into MuJoCo's dof_frictionloss/dof_damping every step, which
    requires those model fields to be expanded per world. mjlab expands
    exactly the fields declared by event functions via requires_model_fields,
    so every env using the BAM actuator must register this event.
    """


def randomize_bam_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Per-episode joint-friction randomization for the BAM actuator (NON-accumulating).

    Under BAM, MuJoCo's dof_frictionloss is zeroed (BAM computes friction in
    compute()), so stock dr.dof_frictionloss is a no-op. Instead this samples a
    per-env scalar in ``scale_range`` and applies it to the FrictionDRBamActuator's
    ``friction_scale``, which multiplies BAM's velocity-independent friction budget
    (Coulomb + Stribeck + load). Restores nominal (1.0) first to avoid accumulation.
    No-op on actuators without a friction_scale hook.
    """
    from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    lo, hi = scale_range
    for actuator in asset.actuators:
        if isinstance(actuator, FrictionDRBamActuator):
            actuator.reset_friction_scale(env_ids)
            samples = torch.rand(len(env_ids), 1, device=env.device) * (hi - lo) + lo
            actuator.set_friction_scale(env_ids, samples)


def randomize_mass_and_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize body mass and inertia together with the same scaling factor.

    This maintains physical consistency - mass and inertia must scale together
    to avoid creating invalid inertia tensors that cause simulation instability.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        scale_range: (min, max) scaling factor applied to both mass and inertia
        asset_cfg: Asset configuration specifying which bodies to randomize
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # Get body indices
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    # Sample ONE random scale per environment (applied to both mass and inertia)
    num_envs = len(env_ids)
    num_bodies = len(body_indices)
    scales = torch.rand(num_envs, num_bodies, device=env.device) * (scale_range[1] - scale_range[0]) + scale_range[0]

    # Store original values on first call
    if not hasattr(env, '_original_mass_inertia'):
        env._original_mass_inertia = {
            'mass': env.sim.model.body_mass[0, body_indices].clone(),
            'inertia': env.sim.model.body_inertia[0, body_indices].clone(),
        }

    # Reset to original first (to prevent accumulation)
    original = env._original_mass_inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] = original['mass'].unsqueeze(0).expand(num_envs, -1)
    env.sim.model.body_inertia[env_ids[:, None], body_indices] = original['inertia'].unsqueeze(0).expand(num_envs, -1, -1)

    # Apply same scale to both mass and inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] *= scales
    env.sim.model.body_inertia[env_ids[:, None], body_indices] *= scales.unsqueeze(-1)  # Scale all 3 inertia components


def randomize_imu_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_angle_deg: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize IMU sensor mounting orientation by small angles.
    
    Simulates slight mounting errors or calibration offsets in the real robot.
    The IMU orientation is randomized by rotating around random axes by up to max_angle_deg.
    
    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        max_angle_deg: Maximum rotation angle in degrees (default 2.0°)
        asset_cfg: Asset configuration
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    
    asset: Entity = env.scene[asset_cfg.name]

    # IMU site is the first site (index 0) in robot.xml
    # Sites: imu (0), left_foot (1), right_foot (2)
    site_id = 0
    
    # Store original orientation on first call
    if not hasattr(env, '_original_imu_quat'):
        env._original_imu_quat = env.sim.model.site_quat[0, site_id].clone()
    
    # Generate random rotations for each environment
    num_envs = len(env_ids)
    max_angle_rad = max_angle_deg * torch.pi / 180.0
    
    # Random rotation angles [-max_angle, +max_angle] for each axis
    angles = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * max_angle_rad
    
    # Convert Euler angles to quaternions (small angle approximation for efficiency)
    # For small angles: quat ≈ [1, θx/2, θy/2, θz/2]
    half_angles = angles / 2.0
    quats_delta = torch.zeros(num_envs, 4, device=env.device)
    quats_delta[:, 0] = 1.0  # w component
    quats_delta[:, 1:] = half_angles  # x, y, z components
    
    # Normalize the quaternion
    quats_delta = quats_delta / torch.norm(quats_delta, dim=1, keepdim=True)
    
    # Get original quaternion and apply delta rotation
    original_quat = env._original_imu_quat.unsqueeze(0).expand(num_envs, -1)
    
    # Quaternion multiplication: q_new = q_delta * q_original
    # q1 * q2 = [w1*w2 - dot(v1,v2), w1*v2 + w2*v1 + cross(v1,v2)]
    w1, x1, y1, z1 = quats_delta[:, 0], quats_delta[:, 1], quats_delta[:, 2], quats_delta[:, 3]
    w2, x2, y2, z2 = original_quat[:, 0], original_quat[:, 1], original_quat[:, 2], original_quat[:, 3]
    
    new_quat = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
    ], dim=1)
    
    # Apply to the selected environments
    env.sim.model.site_quat[env_ids, site_id] = new_quat


def randomize_base_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_pitch_deg: float = 10.0,
    max_roll_deg: float = 5.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize base orientation at episode start to force reactive behavior.

    Adds random pitch and roll to the robot's base orientation at the start of
    each episode. This prevents the policy from memorizing a single initial state
    and forces it to use feedback to adapt to different orientations.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        max_pitch_deg: Maximum pitch angle in degrees (forward/backward tilt)
        max_roll_deg: Maximum roll angle in degrees (side-to-side tilt)
        asset_cfg: Asset configuration
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # Generate random pitch and roll angles
    max_pitch_rad = max_pitch_deg * torch.pi / 180.0
    max_roll_rad = max_roll_deg * torch.pi / 180.0

    pitch = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_pitch_rad
    roll = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_roll_rad
    yaw = torch.zeros(num_envs, device=env.device)  # Keep yaw at 0

    # Convert Euler angles (roll, pitch, yaw) to quaternion
    # Using the standard aerospace sequence (ZYX)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    quat_w = cr * cp * cy + sr * sp * sy
    quat_x = sr * cp * cy - cr * sp * sy
    quat_y = cr * sp * cy + sr * cp * sy
    quat_z = cr * cp * sy - sr * sp * cy

    new_quat = torch.stack([quat_w, quat_x, quat_y, quat_z], dim=1)

    # Normalize quaternion
    new_quat = new_quat / torch.norm(new_quat, dim=1, keepdim=True)

    # Get root position index (freejoint starts at qpos index 0)
    # Freejoint: [x, y, z, qw, qx, qy, qz]
    root_quat_idx = 3  # Quaternion starts at index 3

    # Apply the randomized orientation to selected environments
    env.sim.data.qpos[env_ids, root_quat_idx:root_quat_idx+4] = new_quat


def set_face_down_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Set the robot to a prone (belly-down) orientation for stand-up training.

    Rotates the robot 90° forward around the pitch axis (Y) so the front/belly
    faces the ground and legs point upward. Combined with a random yaw.

    Quaternion derivation:
        quat_pitch90 = [s, 0, s, 0]   where s = sqrt(2)/2  (90° around Y)
        quat_yaw     = [cy, 0, 0, sy]
        combined     = quat_yaw * quat_pitch90 = [s*cy, -s*sy, s*cy, s*sy]
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    new_quat = torch.stack(
        [
            s * cy,   # w
            -s * sy,  # x
            s * cy,   # y
            s * sy,   # z
        ],
        dim=1,
    )

    # Freejoint qpos: [x, y, z, qw, qx, qy, qz, ...]
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.5,
):
    """Randomly initialize each env as face-down (belly) or face-up (back), with random yaw.

    Face-down:  +90° pitch → quat = [s*cy, -s*sy,  s*cy,  s*sy]
    Face-up:    -90° pitch → quat = [s*cy,  s*sy, -s*cy,  s*sy]

    Args:
        face_down_prob: probability of sampling face-down (vs face-up). A curriculum
            can ramp this from a high initial value (easier task) toward 0.5.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    face_down = torch.stack([ s * cy, -s * sy,  s * cy,  s * sy], dim=1)
    face_up   = torch.stack([ s * cy,  s * sy, -s * cy,  s * sy], dim=1)

    mask = torch.rand(num, device=env.device) < face_down_prob  # True → face-down
    new_quat = torch.where(mask.unsqueeze(1), face_down, face_up)

    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_ground_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.4,
    face_up_prob: float = 0.4,
    sitting_prob: float = 0.2,
    standing_prob: float = 0.0,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    sitting_z_min: float = 0.07,
    sitting_z_max: float = 0.09,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    sitting_joint_overrides: Optional[dict] = None,
    sitting_joint_noise_std: float = 0.0,
    sitting_tilt_max: float = 0.0,
    face_up_roll_max: float = 0.0,
):
    """Reset to a random ground state: face-down, face-up, sitting, or standing.

    Broader than ``set_random_prone_orientation`` — used by the stand-up env so
    the policy learns to recover from any plausible pose, including the sitting
    keyframe (rest state of the sit policy) and an already-standing pose (so it
    also learns to *hold* a stand, not only to rise).

    Modes (probabilities are normalized; they need not sum to 1.0):
      - face-down (belly to floor): +90° pitch, random yaw, z in [prone_z_min, prone_z_max].
      - face-up   (back to floor):  -90° pitch, random yaw, z in [prone_z_min, prone_z_max].
      - sitting:                    upright (±sitting_tilt_max), random yaw, z low,
                                    joints set to ``sitting_joint_overrides``.
      - standing:                   upright (±sitting_tilt_max), random yaw, z in
                                    [standing_z_min, standing_z_max], joints left at
                                    HOME (whatever ``reset_robot_joints`` set).

    Args:
        sitting_joint_overrides: ``{qpos_joint_index: angle_rad}`` to write into
            ``qpos[7+idx]`` for envs sampled into the sitting bucket. ``None``
            keeps joints at whatever ``reset_robot_joints`` already set.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    total = face_down_prob + face_up_prob + sitting_prob + standing_prob
    p_fd  = face_down_prob / total
    p_fu  = (face_down_prob + face_up_prob) / total
    p_sit = (face_down_prob + face_up_prob + sitting_prob) / total

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    face_down = torch.stack([ s * cy, -s * sy,  s * cy,  s * sy], dim=1)
    face_up   = torch.stack([ s * cy,  s * sy, -s * cy,  s * sy], dim=1)
    # Upright sitting: yaw-only by default, with optional ±sitting_tilt_max
    # pitch/roll noise so the policy doesn't overfit to perfectly-upright starts.
    if sitting_tilt_max > 0.0:
        pitch = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        roll  = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll  * 0.5); sr = torch.sin(roll  * 0.5)
        # ZYX intrinsic Euler → quaternion (yaw * pitch * roll).
        sit_w = cr * cp * cy + sr * sp * sy
        sit_x = sr * cp * cy - cr * sp * sy
        sit_y = cr * sp * cy + sr * cp * sy
        sit_z = cr * cp * sy - sr * sp * cy
        sitting = torch.stack([sit_w, sit_x, sit_y, sit_z], dim=1)
    else:
        sitting = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)

    u = torch.rand(num, device=env.device)
    is_fd    = u < p_fd
    is_fu    = (u >= p_fd) & (u < p_fu)
    is_sit   = (u >= p_fu) & (u < p_sit)
    is_stand = u >= p_sit

    # Face-up partial-roll noise: rotate the supine pose about the body's long
    # axis by uniform ±face_up_roll_max. WHY (2026-07, back-recovery was
    # seed-lucky): the reward landscape between supine and prone is FLAT —
    # upright_linear (cos tilt) is ≈0 through the whole roll, height doesn't
    # change — so rolling off the back only pays via the front-rise path that
    # follows, a long-horizon dependency that noisy exploration rarely finds
    # from a perfectly flat supine start. With roll noise, a fraction of
    # face-up spawns start near-on-side (partway along the roll): the policy
    # learns roll-completion from easy starts and generalizes back to flat
    # supine — a built-in reverse curriculum. Uniform sampling keeps every
    # difficulty represented (flat back |roll|<15° ≈ 17% at ±90°), so no
    # annealing schedule is needed, and varied post-fall poses are realistic
    # DR for deployment anyway.
    if face_up_roll_max > 0.0:
        theta = (torch.rand(num, device=env.device) * 2 - 1) * face_up_roll_max
        ct = torch.cos(theta * 0.5)
        st = torch.sin(theta * 0.5)
        # Log-roll = rotation about the body's LONG axis, which is body z (the
        # spine: trunk z is up when standing → horizontal when lying). NOT body
        # x — supine leaves body x pointing skyward, so an x-roll would only
        # spin the robot in place like the yaw noise already does.
        # Body-frame rotation → right-multiply: q_fu ⊗ [ct, 0, 0, st].
        w, x, y, z = face_up[:, 0], face_up[:, 1], face_up[:, 2], face_up[:, 3]
        face_up = torch.stack(
            [
                w * ct - z * st,
                x * ct + y * st,
                y * ct - x * st,
                w * st + z * ct,
            ],
            dim=1,
        )

    # Sitting and standing share the same upright orientation (identity + optional
    # ±sitting_tilt_max); they differ only in trunk height and joint pose.
    new_quat = face_down.clone()
    new_quat[is_fu]    = face_up[is_fu]
    new_quat[is_sit]   = sitting[is_sit]
    new_quat[is_stand] = sitting[is_stand]

    # Random z per env: prone heights for face-down/up, low for sit, ~standing for stand.
    z_prone = torch.rand(num, device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
    z_sit   = torch.rand(num, device=env.device) * (sitting_z_max - sitting_z_min) + sitting_z_min
    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    new_z = z_prone.clone()
    new_z = torch.where(is_sit, z_sit, new_z)
    new_z = torch.where(is_stand, z_stand, new_z)

    env.sim.data.qpos[env_ids, 2]   = new_z
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6]  = 0.0

    # Sitting-bucket joint overrides (e.g. knee/ankle bent to keyframe).
    # Override keys are SERVO indices (14-joint layout); translate to entity
    # joint indices so models with interleaved passive_* joints (backlash)
    # write the intended joints. qpos column = 7 + entity joint index
    # (robot free joint first, all hinges 1-dof).
    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    if sitting_joint_overrides:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            for jnt_idx, angle in sitting_joint_overrides.items():
                env.sim.data.qpos[sit_env_ids, 7 + servo_ids[jnt_idx]] = angle

    # Joint noise for sitting envs: Gaussian noise on every actuated joint
    # so the policy sees a distribution of plausible "sit" starts rather than
    # a single canonical pose. Captures real-world transfer where the robot's
    # joint angles won't match the SIT keyframe exactly when the standup
    # policy takes over from the sit policy.
    if sitting_joint_noise_std > 0.0:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            # Servo joints only: passive_* joints (backlash hinges) have tiny
            # ranges and must stay at 0 on reset.
            n_sit = len(sit_env_ids)
            cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
            noise = torch.randn(n_sit, len(cols), device=env.device) * sitting_joint_noise_std
            env.sim.data.qpos[sit_env_ids.unsqueeze(1).long(), cols.unsqueeze(0)] += noise


# Deep-crouch anchor pose (velstand run-5): the "stuck" mid-recovery basin —
# knees folded under the body, trunk pitched forward, feet flat. Values chosen
# by extending the HOME zig-zag (hip fwd / knee back / ankle fwd, sign
# conventions per the SIT keyframe fold directions) to deep flexion, inside
# the ±1.57 joint limits. hip_yaw/hip_roll/neck stay at HOME.
_CROUCH_ANCHOR_BY_NAME = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


def set_random_crouch_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    depth_min: float = 0.35,
    depth_max: float = 1.0,
    pitch_max_deg: float = 55.0,
    joint_noise: float = 0.12,
    z_stand: float = 0.115,
    z_deep: float = 0.06,
):
    """Reset selected envs into a random mid-recovery crouch.

    Reverse curriculum for the recovery last mile (velstand run-5 lesson):
    prone-init episodes spend most of their fallen budget getting TO the deep
    crouch and are recycled shortly after reaching it, so the crouch→stand
    mile gets almost no on-policy data — the policy converged to parking
    there. Seeding resets ACROSS that mile (depth λ ∈ [depth_min, depth_max]
    between standing and the deep-crouch anchor, trunk pitch and z scaled
    with λ) makes the frontier dense from step 0 of the episode.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    lam = torch.rand(num, device=env.device) * (depth_max - depth_min) + depth_min

    # Joints: lerp HOME → anchor on the leg pitch chain, uniform noise on the
    # servo joints only (passive_* backlash hinges have ±1° ranges — noise
    # there would spawn them pinned outside their limits).
    joints = asset.data.default_joint_pos[env_ids].clone()
    for name, anchor in _CROUCH_ANCHOR_BY_NAME.items():
        ids, _ = asset.find_joints(f"^{name}$")
        j = ids[0]
        joints[:, j] = joints[:, j] + lam * (anchor - joints[:, j])
    noise_mask = torch.zeros(joints.shape[1], device=joints.device)
    noise_mask[_servo_joint_ids(env, asset)] = 1.0
    joints += (torch.rand_like(joints) * 2 - 1) * joint_noise * noise_mask

    # Base orientation: forward pitch scaled with depth (the stuck basin is a
    # forward crouch from both fall directions), random yaw, small roll noise.
    pitch = lam * math.radians(pitch_max_deg) \
        + (torch.rand(num, device=env.device) * 2 - 1) * math.radians(10.0)
    pitch = torch.clamp(pitch, min=math.radians(5.0))
    roll = (torch.rand(num, device=env.device) * 2 - 1) * math.radians(8.0)
    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5); sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    # ZYX intrinsic Euler → quaternion (yaw * pitch * roll), as in
    # set_random_ground_state's sitting branch.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    # Trunk height scaled with depth, small upward margin to settle cleanly.
    z = z_stand + lam * (z_deep - z_stand) \
        + torch.rand(num, device=env.device) * 0.01

    env.sim.data.qpos[env_ids, 2] = z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qpos[env_ids, 7:] = joints
    env.sim.data.qvel[env_ids, :] = 0.0


def maybe_set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    prone_prob: float = 0.0,
    face_down_prob: float = 0.5,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    crouch_prob: float = 0.0,
):
    """Reset event that overrides orientation to prone with probability `prone_prob`.

    With prob `prone_prob`, replaces the upright orientation (already set by
    reset_base) with a prone orientation; otherwise leaves it upright. Among the
    overridden envs, `face_down_prob` picks face-down (belly) vs face-up (back).

    Also lifts z to [prone_z_min, prone_z_max] for the overridden envs so the
    head/neck clearance is sufficient — the vel-env reset z (~0.125) would
    clip the head through the ground at 90° pitch.

    At prone_prob=2/3 and face_down_prob=0.5 you get a balanced 33/33/33 split
    of upright/face-down/face-up resets, which is the standard mixture for
    learning fall recovery alongside normal upright start.

    With ``crouch_prob`` > 0, an additional exclusive slice of envs is reset
    into a random mid-recovery crouch via ``set_random_crouch_state`` (reverse
    curriculum for the recovery last mile — see its docstring).
    """
    if prone_prob <= 0.0 and crouch_prob <= 0.0:
        return
    # env_ids=None means "all envs" (the initial global reset passes None —
    # the old early-return silently skipped prone init there).
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    env_ids_t = env_ids.to(env.device, dtype=torch.long) if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=env.device, dtype=torch.long)
    # One draw partitions envs into exclusive prone / crouch / untouched slices.
    u = torch.rand(len(env_ids_t), device=env.device)
    selected = env_ids_t[u < prone_prob]
    crouch_selected = env_ids_t[(u >= prone_prob) & (u < prone_prob + crouch_prob)]
    if len(selected) > 0:
        set_random_prone_orientation(
            env, selected, asset_cfg=asset_cfg, face_down_prob=face_down_prob
        )
        # Override z so the prone body has head/neck clearance when settling.
        z = torch.rand(len(selected), device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
        env.sim.data.qpos[selected, 2] = z
    if len(crouch_selected) > 0:
        set_random_crouch_state(env, crouch_selected, asset_cfg=asset_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Non-accumulating domain randomization (restore-nominal-then-apply).
#
# The stock mdp.randomize_field with operation="add"/"scale" + mode="reset"
# reads the CURRENT model value and applies the op to it, with no restore to
# nominal — so on every episode reset the perturbation STACKS on the previous
# one and the parameter random-walks away from nominal over training. For
# body_ipos (CoM) this was the long-standing microduck instability: the CoM
# drifted centimeters off-center over hundreds of resets → progressively
# unbalanced robot → falls more → reward/episode-length collapse after the early
# peak. These functions mirror randomize_mass_and_inertia: cache the nominal
# once, restore it before each draw, then apply a freshly-sampled perturbation —
# so it is re-sampled per episode but never accumulates.
# ─────────────────────────────────────────────────────────────────────────────
def randomize_com(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ranges: tuple[float, float],
    field: str = "body_ipos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Randomize body CoM (body_ipos) per episode WITHOUT accumulating.

    Drop-in replacement for the buggy mdp.randomize_field(add, body_ipos, reset).
    ``ranges`` is (lo, hi) applied to all 3 CoM axes; the com_range curriculum
    updates this same ``ranges`` param. ``field`` is declared so the event can run
    with ``domain_randomization=True`` (mjlab reads params["field"] to expand that
    model field per-env).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    mf = getattr(env.sim.model, field)
    # Key the cache by (field, body set): multiple randomize_com events can share
    # the same field (e.g. trunk + head both randomize body_ipos) and must NOT
    # collide on a single _original_body_ipos attr — their body counts differ.
    _bidx = body_indices.tolist() if hasattr(body_indices, "tolist") else list(body_indices)
    cache_attr = f"_original_{field}_" + "_".join(str(int(i)) for i in _bidx)
    # Cache nominal on first call (model[0] is still nominal at that point).
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, body_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_bodies = len(body_indices)

    # Restore nominal first (prevents accumulation), then add a fresh offset.
    mf[env_ids[:, None], body_indices] = nominal.unsqueeze(0).expand(num_envs, -1, -1)
    lo, hi = ranges
    offsets = torch.rand(num_envs, num_bodies, 3, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], body_indices] += offsets
    return torch.tensor(float(hi))


def randomize_dof_field_scaled(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    field: str,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Scale a per-dof model field (e.g. dof_frictionloss/dof_damping) per episode
    WITHOUT accumulating: restore nominal, then apply a fresh scale.

    ``field`` doubles as the domain_randomization field name. NOTE: under the BAM
    actuator, dof_frictionloss and dof_damping are zeroed in edit_spec (BAM models
    friction itself), so scaling them is a no-op — these only matter with the XML
    position actuator. Kept correct to avoid the accumulation footgun if re-enabled.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(len(asset.indexing.joint_ids)))[joint_ids]
    dof_indices = asset.indexing.joint_v_adr[joint_ids]

    mf = getattr(env.sim.model, field)
    cache_attr = f"_original_{field}"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, dof_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_dofs = len(dof_indices)

    mf[env_ids[:, None], dof_indices] = nominal.unsqueeze(0).expand(num_envs, -1)
    lo, hi = scale_range
    scales = torch.rand(num_envs, num_dofs, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], dof_indices] *= scales
    return torch.tensor(float(hi))
