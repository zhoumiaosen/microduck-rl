"""MicroDuck common terms."""

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
import torch


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# Name patterns matching the 4 neck/head actuated joints. Used by head_pose
# tracking reward and by UniformPoseCommand asset hookups.
_NECK_JOINT_PATTERNS = [r".*neck_pitch.*", r".*head_pitch.*", r".*head_yaw.*", r".*head_roll.*"]


def _servo_joint_ids(env: "ManagerBasedRlEnv", asset: Entity) -> list:
    """Entity-local indices of the servo (non-``passive_``) joints, cached.

    All joint-index-based reward/event params in this module (``joint_indices``,
    ``target_overrides``, qpos-column math) are written against the canonical
    14-servo layout. On models with extra unactuated joints — backlash hinges,
    roller wheels, the jaw linkage, all named ``passive_*`` — the entity joint
    array is wider and interleaved, so raw indices would select the wrong
    joints. Index through this list to recover the servo-only view; on plain
    models it is the identity.
    """
    cache = env.__dict__.setdefault("_servo_joint_ids_cache", {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        ids, _ = asset.find_joints(r"^(?!passive_).*")
        cache[key] = ids
    return ids


def _servo_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_pos[:, _servo_joint_ids(env, asset)]


def _servo_joint_vel(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_vel[:, _servo_joint_ids(env, asset)]


def _servo_default_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.default_joint_pos[:, _servo_joint_ids(env, asset)]
