import math

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks import microduck_running_env_cfg as running


def test_squared_progress_preserves_reference_pay_and_increases_speed_incentive():
    speed = torch.tensor([0.0, 1.7, 2.0, 2.6, 4.0])
    score = mdp.running_squared_progress_from_values(speed, torch.ones_like(speed), 2.6)
    torch.testing.assert_close(score, speed.clamp(0, 2.6).square() / (2.6 * 1.7))
    torch.testing.assert_close(score[1], torch.tensor(1.7 / 2.6))
    assert score[2] - score[1] > (2.0 - 1.7) / 2.6
    assert score[3] == score[4]


def test_squared_progress_does_not_pay_backward_fallen_or_nonfinite_states():
    speed = torch.tensor([-1.0, 2.0, 2.0, float('nan'), float('inf'), 2.0, 2.0])
    upright = torch.tensor([1.0, math.cos(math.radians(60)), 1.0, 1.0, 1.0, float('nan'), float('inf')])
    score = mdp.running_squared_progress_from_values(speed, upright, 2.6)
    torch.testing.assert_close(score, torch.tensor([0.0, 0.0, 4.0 / (2.6 * 1.7), 0.0, 0.0, 0.0, 0.0]))


def test_squared_progress_is_opt_in_and_keeps_the_rest_of_the_recipe(monkeypatch):
    monkeypatch.setattr(running, 'RUNNING_SQUARED_PROGRESS', False, raising=False)
    baseline = running.make_microduck_running_env_cfg()
    monkeypatch.setattr(running, 'RUNNING_SQUARED_PROGRESS', True)
    candidate = running.make_microduck_running_env_cfg()
    assert baseline.rewards['forward_progress'].func is mdp.running_forward_progress
    assert candidate.rewards['forward_progress'].func is mdp.running_squared_progress
    assert candidate.rewards['forward_progress'].weight == baseline.rewards['forward_progress'].weight
    assert candidate.rewards['forward_progress'].params == baseline.rewards['forward_progress'].params
    assert candidate.rewards.keys() == baseline.rewards.keys()
    assert candidate.observations.keys() == baseline.observations.keys()
    assert candidate.sim == baseline.sim


def test_fixed_physics_only_disables_selected_randomization_events(monkeypatch):
    names = {"randomize_mass_inertia", "randomize_joint_friction", "randomize_armature", "foot_friction"}
    monkeypatch.setattr(running, "RUNNING_FIXED_PHYSICS", False, raising=False)
    baseline = running.make_microduck_running_env_cfg()
    assert names <= baseline.events.keys()
    monkeypatch.setattr(running, "RUNNING_FIXED_PHYSICS", True)
    candidate = running.make_microduck_running_env_cfg()
    assert candidate.events.keys() == baseline.events.keys() - names
    assert candidate.scene.entities["robot"].spec_fn is baseline.scene.entities["robot"].spec_fn
    assert candidate.sim == baseline.sim
    assert candidate.actions == baseline.actions
    assert candidate.observations == baseline.observations
    assert candidate.rewards == baseline.rewards
