"""MicroDuck running terms."""

from dataclasses import dataclass as _dataclass
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
import math
import torch
from .commands import (
    VelocityCommandCommandOnly,
    VelocityCommandCommandOnlyCfg,
)
from .common import (
    _DEFAULT_ASSET_CFG,
)


def running_forward_progress_from_velocity(
    velocity_x: torch.Tensor,
    speed_cap: float = 1.2,
) -> torch.Tensor:
    """Linear forward-speed objective used by the running task.

    Unlike :func:`forward_speed_reward`, this deliberately does not saturate at
    ordinary walking speed.  Backward motion receives no reward and very large
    velocities are capped so a single physics outlier cannot become a jackpot.
    """
    if speed_cap <= 0.0:
        raise ValueError("speed_cap must be positive")
    velocity_x = torch.nan_to_num(velocity_x, nan=0.0, posinf=speed_cap, neginf=0.0)
    return torch.clamp(velocity_x, min=0.0, max=speed_cap) / speed_cap


def running_forward_progress(
    env: ManagerBasedRlEnv,
    speed_cap: float = 1.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward forward trunk speed with useful gradient above walking speeds."""
    asset: Entity = env.scene[asset_cfg.name]
    return running_forward_progress_from_velocity(
        asset.data.root_link_lin_vel_b[:, 0], speed_cap=speed_cap
    )


def running_projected_progress_from_values(velocity_xy, target_heading, upright_cos, speed_cap=2.6):
    direction = torch.stack((target_heading.cos(), target_heading.sin()), dim=-1)
    speed = (velocity_xy * direction).sum(dim=-1)
    return running_squared_progress_from_values(speed, upright_cos, speed_cap)


def running_projected_progress(env, speed_cap=2.6, asset_cfg=_DEFAULT_ASSET_CFG):
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_term("twist")
    return running_projected_progress_from_values(
        asset.data.root_link_lin_vel_w[:, :2], command.target_heading,
        -asset.data.projected_gravity_b[:, 2], speed_cap,
    )


def running_squared_progress_from_values(
    velocity_x: torch.Tensor, upright_cos: torch.Tensor, speed_cap: float = 2.6
) -> torch.Tensor:
    """Match linear reward at 1.7 m/s, favor faster motion, and reject falls."""
    if speed_cap <= 0.0:
        raise ValueError("running speed cap must be positive")
    speed = torch.nan_to_num(velocity_x, nan=0.0, posinf=0.0, neginf=0.0)
    score = speed.clamp(0.0, speed_cap).square() / (speed_cap * 1.7)
    upright = torch.isfinite(upright_cos) & (upright_cos >= 2.0**-0.5)
    return torch.where(upright, score, 0.0)


def running_squared_progress(
    env: ManagerBasedRlEnv,
    speed_cap: float = 2.6,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return running_squared_progress_from_values(
        asset.data.root_link_lin_vel_b[:, 0],
        -asset.data.projected_gravity_b[:, 2],
        speed_cap,
    )


def running_flight_event(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    min_forward_speed: float = 0.3,
    max_tilt_deg: float = 50.0,
    min_airborne_steps: int = 3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay once when a stable, forward-moving flight phase begins.

    This is intentionally an *event*, not an airtime reward: extending an
    uncontrolled ballistic phase never increases the return.  Requiring three
    consecutive 50 Hz samples rejects one-frame contact-sensor flicker.  The
    state cache is reset on a fresh episode so spawning in the air cannot
    collect a reward.
    """
    if min_airborne_steps < 1:
        raise ValueError("min_airborne_steps must be at least one")
    sensor = env.scene[sensor_name]
    contacts = sensor.data.found.reshape(env.num_envs, -1).any(dim=-1)
    airborne = ~contacts

    air_steps = getattr(env, "_running_airborne_steps", None)
    if air_steps is None or air_steps.shape != airborne.shape:
        air_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    fresh_episode = env.episode_length_buf == 0
    air_steps = torch.where(airborne, air_steps + 1, torch.zeros_like(air_steps))
    air_steps = torch.where(fresh_episode, torch.zeros_like(air_steps), air_steps)
    onset = air_steps == min_airborne_steps
    env._running_airborne_steps = air_steps

    asset: Entity = env.scene[asset_cfg.name]
    forward = torch.nan_to_num(asset.data.root_link_lin_vel_b[:, 0], nan=0.0)
    gravity_z = torch.nan_to_num(asset.data.projected_gravity_b[:, 2], nan=0.0)
    max_tilt_cos = math.cos(math.radians(max_tilt_deg))
    stable = (-gravity_z) >= max_tilt_cos
    return (onset & stable & (forward >= min_forward_speed)).float()


def running_planar_drift_cost_from_values(
    lateral_velocity: torch.Tensor,
    yaw_rate: torch.Tensor,
    lateral_command: torch.Tensor,
    yaw_command: torch.Tensor,
    lateral_weight: float = 4.0,
) -> torch.Tensor:
    """Positive straight-line error cost; use with a negative reward weight."""
    lateral_error = torch.nan_to_num(lateral_velocity - lateral_command, nan=0.0)
    yaw_error = torch.nan_to_num(yaw_rate - yaw_command, nan=0.0)
    return yaw_error.square() + lateral_weight * lateral_error.square()


def running_planar_drift_cost(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    lateral_weight: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize body-frame lateral drift and yaw-rate command error."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return running_planar_drift_cost_from_values(
        asset.data.root_link_lin_vel_b[:, 1],
        asset.data.root_link_ang_vel_b[:, 2],
        command[:, 1],
        command[:, 2],
        lateral_weight=lateral_weight,
    )


def running_command_ranges_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    speed_stages: list[dict],
) -> torch.Tensor:
    """Advance a forward-only running speed band over training.

    A band avoids spending most samples near zero while an explicit standing
    bucket in the command cfg still trains the deployment idle state.
    """
    del env_ids

    from typing import cast

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    current_min = float(speed_stages[0]["min_speed"])
    current_max = float(speed_stages[0]["max_speed"])
    for stage in speed_stages:
        if env.common_step_counter >= stage["step"]:
            current_min = float(stage["min_speed"])
            current_max = float(stage["max_speed"])
    if not (0.0 <= current_min <= current_max):
        raise ValueError(f"invalid running speed band: {(current_min, current_max)}")

    cfg.ranges.lin_vel_x = (current_min, current_max)
    return torch.tensor([current_max], device=env.device)


def running_heading_yaw_rate(target_heading, current_heading):
    delta = target_heading - current_heading
    error = torch.atan2(delta.sin(), delta.cos())
    return torch.nan_to_num(0.8 * error, nan=0.0).clamp(-0.5, 0.5)


class RunningStraightCommand(VelocityCommandCommandOnly):
    """Hold the reset heading using an ordinary yaw-rate command in rad/s."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.target_heading = self.robot.data.heading_w.clone()
        self._pending_heading = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids):
        extras = super().reset(env_ids)
        # Reset hooks precede sim.forward(): derived heading is still stale here.
        self._pending_heading[env_ids] = True
        return extras

    def _update_command(self):
        # Command updates run after sim.forward(), including immediately after reset.
        self.target_heading[self._pending_heading] = self.robot.data.heading_w[self._pending_heading]
        self._pending_heading[:] = False
        self.vel_command_b[:, 2] = running_heading_yaw_rate(
            self.target_heading, self.robot.data.heading_w
        )
        self.vel_command_b[self.is_standing_env] = 0.0


@_dataclass(kw_only=True)
class RunningStraightCommandCfg(VelocityCommandCommandOnlyCfg):
    def build(self, env):
        return RunningStraightCommand(self, env)
