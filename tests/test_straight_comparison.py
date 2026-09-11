import json
import math
import subprocess
import sys

import pytest

from scripts.experiments.run_straight_comparison import (
    ROOT, eligible_score, receipt_outputs, schedule, select_candidate,
    trial_environment,
)


def row(speed=1.8, survival=.96):
    return {'straight_progress_speed_mps': {'mean': speed},
            'survival_fraction': survival}


@pytest.mark.parametrize('rows', [[], [{}], [row(2.1, .89)],
                                [row(float('nan'))], [row(None)],
                                [row(survival=float('inf'))]])
def test_rejects_invalid_candidate(rows):
    assert eligible_score(rows) == -math.inf


def test_scores_valid_candidate():
    assert eligible_score([row()]) == 1.8
    assert eligible_score([row(1.7, .89), row(1.9, .99)]) == pytest.approx(1.8)


def test_environment_isolation():
    base = {'PATH': 'keep', 'MICRODUCK_RUNNING_FLIGHT': '9',
            'MICRODUCK_RUNNING_STRAIGHT': '0'}
    env = trial_environment(base, -.05)
    assert env['PATH'] == 'keep'
    assert 'MICRODUCK_RUNNING_FLIGHT' not in env
    assert env['MICRODUCK_RUNNING_STRAIGHT'] == '1'
    assert env['MICRODUCK_RUNNING_ACTION_RATE_WEIGHT'] == '-0.05'
    assert base['MICRODUCK_RUNNING_STRAIGHT'] == '0'
    assert env['PYTHONPATH'] == f'{ROOT / "src"}:{ROOT}'


def test_schedule_has_independent_parents_and_exact_budget():
    stages = schedule('parent.pt')
    assert len(stages) == 6
    assert sum(s['updates'] for s in stages) == 1215
    assert {s['parent'] for s in stages} == {'parent.pt'}
    assert [s['envs'] for s in stages] == [64, 512] * 3
    assert [s['updates'] for s in stages] == [5, 400] * 3


def test_dry_run_never_loads_checkpoint(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/experiments/run_straight_comparison.py'),
                             '--parent', '/missing/parent.pt', '--output-dir', str(tmp_path),
                             '--dry-run'], check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan['max_training_updates'] == 1215
    assert not list(tmp_path.iterdir())


def test_receipts_reject_stale_config_and_changed_output(tmp_path):
    from scripts.experiments.run_straight_comparison import sha256
    output = tmp_path / 'checkpoint.pt'
    output.write_bytes(b'checkpoint')
    receipt = tmp_path / 'receipt.json'
    config = {'parent_sha256': 'abc', 'lr': .00002}
    receipt.write_text(json.dumps({'config': config, 'outputs': {str(output): sha256(output)}}))
    assert receipt_outputs(receipt, config) == [output]
    with pytest.raises(ValueError, match='configuration'):
        receipt_outputs(receipt, {**config, 'lr': .00005})
    output.write_bytes(b'changed')
    with pytest.raises(ValueError, match='output'):
        receipt_outputs(receipt, config)


def test_validation_never_keeps_worse_candidate():
    parent = {'label': 'parent', 'short': [row(1.8)], 'long': [row(1.8)]}
    worse = {'label': 'A', 'short': [row(1.9)], 'long': [row(1.7)]}
    unstable = {'label': 'B', 'short': [row(2.1, .94)], 'long': [row(2.1)]}
    better = {'label': 'C', 'short': [row(1.9)], 'long': [row(1.9)]}
    assert select_candidate(parent, [worse, unstable]) is parent
    assert select_candidate(parent, [worse, better]) is better


def test_failed_stage_cannot_spend_budget_again(tmp_path):
    from scripts.experiments.run_straight_comparison import Comparison
    comparison = Comparison.__new__(Comparison)
    comparison.output = tmp_path
    comparison.identity = {'parent_sha256': 'abc'}
    attempts = []
    def fail(folder):
        attempts.append(folder)
        raise RuntimeError('simulated training failure')
    with pytest.raises(RuntimeError, match='simulated'):
        comparison.stage('A_train', {'updates': 400}, fail)
    with pytest.raises(ValueError, match='Incomplete stage'):
        comparison.stage('A_train', {'updates': 400}, fail)
    assert len(attempts) == 1


def test_preparation_changes_only_copied_optimizer_lr(tmp_path):
    torch = pytest.importorskip('torch')
    from scripts.experiments.run_straight_comparison import prepare_parent, sha256
    data = {'iter': 13594, 'infos': {'env_state': {'common_step_counter': 326688}},
            'optimizer_state_dict': {'param_groups': [{'lr': .001}, {'lr': .002}]}}
    for kind, width, outputs in (('actor', 61, 14), ('critic', 76, 1)):
        shapes = {'mlp.0.weight': (512, width), 'mlp.2.weight': (256, 512),
                  'mlp.4.weight': (128, 256), 'mlp.6.weight': (outputs, 128),
                  'obs_normalizer._mean': (1, width), 'obs_normalizer._var': (1, width),
                  'obs_normalizer._std': (1, width)}
        data[kind + '_state_dict'] = {key: torch.ones(shape) for key, shape in shapes.items()}
    parent = tmp_path / 'parent.pt'
    torch.save(data, parent)
    digest = sha256(parent)
    copied = tmp_path / 'owned/checkpoint.pt'
    info = prepare_parent(parent, copied, .00002)
    assert info['learning_rates'] == [.00002, .00002]
    assert sha256(parent) == digest
    saved = torch.load(copied, weights_only=True)
    for kind in ('actor_state_dict', 'critic_state_dict'):
        assert all(torch.equal(tensor, saved[kind][name]) for name, tensor in data[kind].items())


def test_curriculum_check_allows_stage_transition_within_trial():
    from scripts.experiments.run_straight_comparison import verify_curriculum_metrics
    stages = [{'step': '0', 'max_speed': '2.4'}, {'step': '270000', 'max_speed': '2.5'}]
    metrics = {'Curriculum/running_speed_range': [2.4, 2.4, 2.5]}
    verify_curriculum_metrics(metrics, 'running_speed_range', stages, 'max_speed', 264000, 400)
    with pytest.raises(ValueError, match='curriculum'):
        verify_curriculum_metrics({'Curriculum/running_speed_range': [2.4, 2.4]},
                                  'running_speed_range', stages, 'max_speed', 264000, 400)


def test_controller_cli_clears_inherited_running_flags(monkeypatch, tmp_path):
    import os
    from types import SimpleNamespace
    from scripts.experiments import run_straight_comparison as launcher
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setenv('MICRODUCK_RUNNING_HIGH_SPEED_STAGE_INTERVAL', '1')
    monkeypatch.setitem(sys.modules, 'mjlab_microduck', SimpleNamespace(__file__=str(ROOT / 'src/mjlab_microduck/__init__.py')))
    observed = {}
    monkeypatch.setattr(launcher, 'check_controller', lambda *_: observed.update(os.environ))
    monkeypatch.setattr(sys, 'argv', ['launcher', '--parent', str(tmp_path / 'parent.pt'),
                                    '--output-dir', str(tmp_path), '--check-controller'])
    launcher.main()
    assert 'MICRODUCK_RUNNING_HIGH_SPEED_STAGE_INTERVAL' not in observed


def test_live_yaw_check_follows_standing_mask_after_resampling():
    import torch
    from types import SimpleNamespace
    from mjlab_microduck.tasks import mdp
    from scripts.experiments.run_straight_comparison import check_yaw_correction
    term = object.__new__(mdp.RunningStraightCommand)
    term.robot = SimpleNamespace(data=SimpleNamespace(heading_w=torch.tensor([0., 0.])))
    term.target_heading = torch.tensor([.3, .3])
    term._pending_heading = torch.zeros(2, dtype=torch.bool)
    term.vel_command_b = torch.zeros(2, 3)
    term.is_standing_env = torch.tensor([True, False])
    term._update_command()
    check_yaw_correction(term)
    term.is_standing_env = torch.tensor([False, True])
    term._update_command()
    check_yaw_correction(term)
    term.vel_command_b[1, 2] = .24
    with pytest.raises(AssertionError):
        check_yaw_correction(term)
