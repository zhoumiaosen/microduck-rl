import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks import microduck_running_env_cfg as running


def test_heading_rate_sign_wrap_and_cap():
    target = torch.tensor([0., 0., -math.pi + .1, math.pi - .1])
    current = torch.tensor([1., -1., math.pi - .1, -math.pi + .1])
    torch.testing.assert_close(mdp.running_heading_yaw_rate(target, current),
                               torch.tensor([-.5, .5, .16, -.16]))


def test_projected_progress_rejects_sideways_backward_and_fallen_motion():
    velocity = torch.tensor([[2., 0.], [0., 2.], [-2., 0.], [2., 0.], [float('inf'), 0.]])
    upright = torch.tensor([1., 1., 1., 0., 1.])
    reward = mdp.running_projected_progress_from_values(velocity, torch.zeros(5), upright, 2.6)
    torch.testing.assert_close(reward, torch.tensor([4 / (2.6 * 1.7), 0., 0., 0., 0.]))
    rotated = mdp.running_projected_progress_from_values(torch.tensor([[0., 2.]]),
                  torch.tensor([math.pi / 2]), torch.ones(1), 2.6)
    torch.testing.assert_close(rotated, reward[:1])


def test_controller_only_recaptures_reset_environments(monkeypatch):
    command = object.__new__(mdp.RunningStraightCommand)
    command._env = SimpleNamespace(episode_length_buf=torch.tensor([10, 0]))
    command.robot = SimpleNamespace(data=SimpleNamespace(heading_w=torch.tensor([1., -2.])))
    command.target_heading = torch.tensor([0., 0.])
    command._pending_heading = torch.zeros(2, dtype=torch.bool)
    command.vel_command_b = torch.zeros(2, 3)
    command.is_standing_env = torch.tensor([False, False])
    monkeypatch.setattr(mdp.VelocityCommandCommandOnly, 'reset', lambda self, ids: {})
    command.reset(torch.tensor([1]))
    # Reset hooks run before sim.forward(); the fresh pose arrives afterward.
    command.robot.data.heading_w[1] = 2.
    command._update_command()
    torch.testing.assert_close(command.target_heading, torch.tensor([0., 2.]))
    torch.testing.assert_close(command.vel_command_b[:, 2], torch.tensor([-.5, 0.]))
    command.robot.data.heading_w += .1
    command._update_command()
    torch.testing.assert_close(command.target_heading, torch.tensor([0., 2.]))


def test_straight_mode_opt_in_preserves_sim_and_action_contract(monkeypatch):
    baseline = running.make_microduck_running_env_cfg()
    monkeypatch.setattr(running, 'RUNNING_STRAIGHT', True, raising=False)
    candidate = running.make_microduck_running_env_cfg()
    assert isinstance(candidate.commands['twist'], mdp.RunningStraightCommandCfg)
    assert candidate.rewards['forward_progress'].func is mdp.running_projected_progress
    assert candidate.sim == baseline.sim
    assert candidate.actions == baseline.actions
    assert candidate.observations.keys() == baseline.observations.keys()


def test_fixed_window_counts_falls_and_nonfinite_states_as_zero():
    helpers = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/evaluate_running_checkpoint.py'))
    step = helpers['_straight_step']
    alive = torch.ones(3, dtype=torch.bool)
    speed = torch.tensor([2., 2., 2.])
    first, alive = step(speed, alive, torch.tensor([False, True, False]))
    second, alive = step(torch.tensor([2., 2., float('nan')]), alive, torch.zeros(3, dtype=torch.bool))
    torch.testing.assert_close((first + second) / 2, torch.tensor([2., 0., 1.]))
    assert alive.tolist() == [True, False, False]


def test_target_requires_all_three_held_out_seeds_and_stability():
    helpers = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/run_straight_search.py'))
    check = helpers['meets_target']
    rows = [{'seed': seed, 'command_speed_mps': 2.5,
             'straight_progress_speed_mps': {'mean': 2.01}, 'survival_fraction': .96,
             'mean_absolute_heading_error_deg': 9.} for seed in (2027, 4093, 8191)]
    assert check(rows)
    assert not check(rows[:2])
    rows[-1]['survival_fraction'] = .94
    assert not check(rows)
    rows[-1]['survival_fraction'] = .96
    rows[-1]['mean_absolute_heading_error_deg'] = 11.
    assert not check(rows)
