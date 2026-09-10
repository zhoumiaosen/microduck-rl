"""The module split must preserve checkpoint imports and patch behavior."""
import importlib
import pickle

from mjlab.managers.reward_manager import RewardManager
from rsl_rl.algorithms.ppo import PPO

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks import runtime_patches
from mjlab_microduck.tasks.runtime_patches import install_runtime_patches


def test_reimport_does_not_stack_runtime_patches():
    reward_compute = RewardManager.compute
    compute_returns = PPO.compute_returns
    install_runtime_patches()
    importlib.reload(mdp)
    assert RewardManager.compute is reward_compute
    assert PPO.compute_returns is compute_returns


def test_legacy_checkpoint_globals_resolve():
    for name in (
        'VelocityCommandCommandOnlyCfg', 'RunningStraightCommandCfg',
        'GroundPickPhaseCommandCfg', 'UniformPoseCommandCfg', 'SitStandCommandCfg',
        'running_forward_progress', '_servo_joint_ids', '_DEFAULT_ASSET_CFG',
    ):
        # Protocol 0 GLOBAL models a previously saved reference to tasks.mdp.
        restored = pickle.loads(f'cmjlab_microduck.tasks.mdp\n{name}\n.'.encode())
        assert restored is getattr(mdp, name)


def test_patch_module_reload_keeps_original_methods():
    original_reward = runtime_patches._orig_reward_compute
    original_returns = runtime_patches._orig_compute_returns
    installed_reward = RewardManager.compute
    installed_returns = PPO.compute_returns
    importlib.reload(runtime_patches)
    runtime_patches.install_runtime_patches()
    assert runtime_patches._orig_reward_compute is original_reward
    assert runtime_patches._orig_compute_returns is original_returns
    assert RewardManager.compute is installed_reward
    assert PPO.compute_returns is installed_returns
