"""Compatibility patches required by the pinned mjlab/rsl_rl versions.

Install once, even when the public task facade is re-imported. Keep reward,
advantage sanitization and passive-joint export behavior unchanged.
"""
import torch

from mjlab.managers.reward_manager import RewardManager as _RewardManager
from rsl_rl.algorithms.ppo import PPO as _PPO
from mjlab.rl import exporter_utils as _exporter_utils
from mjlab.envs.mdp.actions import JointPositionAction as _JointAction

# importlib.reload retains the module dictionary. Preserve the true originals
# and installed flag so an interactive reload cannot wrap our own wrappers.
if "_installed" not in globals():
    _orig_reward_compute = _RewardManager.compute
    _orig_compute_returns = _PPO.compute_returns
    _installed = False

def _nan_safe_reward_compute(self, dt: float) -> torch.Tensor:
    result = _orig_reward_compute(self, dt)
    # _episode_sums is updated inside compute() before nan_to_num can act.
    # Sanitize in-place so per-term metrics don't show NaN.
    for key in self._episode_sums:
        torch.nan_to_num_(self._episode_sums[key], nan=0.0)
    return torch.nan_to_num(result, nan=0.0)

def _safe_compute_returns(self, obs) -> None:
    _orig_compute_returns(self, obs)
    st = self.storage
    torch.nan_to_num_(st.advantages, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nan_to_num_(st.returns,    nan=0.0, posinf=0.0, neginf=0.0)

def _get_base_metadata_no_passive(env, run_path):
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, _JointAction)
    full_names = list(robot.joint_names)
    keep_idx = [i for i, n in enumerate(full_names) if not n.startswith("passive_")]
    joint_names = [full_names[i] for i in keep_idx]
    joint_name_to_ctrl_id = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    ctrl_ids = [joint_name_to_ctrl_id[n] for n in joint_names]
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
    damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]
    default_jp = robot.data.default_joint_pos[0].cpu().tolist()
    return {
        "run_path": run_path,
        "joint_names": joint_names,
        "joint_stiffness": stiffness.tolist(),
        "joint_damping": damping.tolist(),
        "default_joint_pos": [default_jp[i] for i in keep_idx],
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": joint_action._scale[0].cpu().tolist()
        if isinstance(joint_action._scale, torch.Tensor)
        else joint_action._scale,
    }

def install_runtime_patches() -> None:
    """Install the existing training/export fixes exactly once per process."""
    global _installed
    if _installed:
        return
    _RewardManager.compute = _nan_safe_reward_compute
    _PPO.compute_returns = _safe_compute_returns
    _exporter_utils.get_base_metadata = _get_base_metadata_no_passive
    # The velocity exporter may already hold a reference to the old function.
    try:
        from mjlab.tasks.velocity.rl import exporter
    except ImportError:
        pass  # This legacy module is absent in the pinned mjlab release.
    else:
        if hasattr(exporter, "get_base_metadata"):
            exporter.get_base_metadata = _get_base_metadata_no_passive
    _installed = True
