"""Run the bounded, matched straight-running comparison (maximum 1,215 updates).

Use the existing runtime Python with PYTHONPATH=<worktree>/src:<worktree>.
Dry run needs only Python's standard library. Incomplete stages fail closed;
investigate their logs before any manual recovery or additional training.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TASK = 'Mjlab-Running-Flat-MicroDuck'
TRIALS = (('A', 2e-5, 0.0), ('B', 5e-5, 0.0), ('C', 2e-5, -.05))
SEEDS = (123, 456, 789)
HELDOUT = (2027, 4093, 8191)
SETTINGS = {
    'MICRODUCK_RUNNING_STRAIGHT': '1',
    'MICRODUCK_RUNNING_ENABLE_HEADING_FEEDBACK': '0',
    'MICRODUCK_RUNNING_ENABLE_SYMMETRY': '0',
    'MICRODUCK_RUNNING_FIXED_PHYSICS': '0',
    'MICRODUCK_RUNNING_SQUARED_PROGRESS': '1',
    'MICRODUCK_RUNNING_TARGET_MAX_SPEED': '2.5',
    'MICRODUCK_RUNNING_SPEED_CAP': '2.6',
    'MICRODUCK_RUNNING_FORWARD_PROGRESS_WEIGHT': '10',
    'MICRODUCK_RUNNING_CURRICULUM_START_ITERATION': '0',
    'MUJOCO_GL': 'egl', 'WANDB_MODE': 'offline',
}


def trial_environment(base: dict[str, str], action_rate: float) -> dict[str, str]:
    env = {k: v for k, v in base.items() if not k.startswith('MICRODUCK_RUNNING_')}
    env.update(SETTINGS)
    env['MICRODUCK_RUNNING_ACTION_RATE_WEIGHT'] = str(action_rate)
    env['PYTHONPATH'] = f'{ROOT / "src"}:{ROOT}'
    return env


def eligible_score(rows: list[dict]) -> float:
    try:
        speeds = [r['straight_progress_speed_mps']['mean'] for r in rows]
        survival = [r['survival_fraction'] for r in rows]
        if not rows or any(v is None or not math.isfinite(v) for v in speeds + survival):
            return -math.inf
        if any(not 0 <= v <= 1 for v in survival) or statistics.mean(survival) < .90:
            return -math.inf
        return statistics.mean(speeds)
    except (KeyError, TypeError, ValueError):
        return -math.inf


def select_candidate(parent, candidates):
    best = parent
    for candidate in candidates:
        if (eligible_score(candidate['short']) > -math.inf
                and statistics.mean(r['survival_fraction'] for r in candidate['short']) >= .95
                and eligible_score(candidate['long']) > eligible_score(best['long'])):
            best = candidate
    return best


def schedule(parent):
    return [{'label': f'{label}_{phase}', 'trial': label, 'parent': str(parent),
             'lr': lr, 'action_rate': rate, 'updates': updates, 'envs': envs,
             'seed': 42, 'rollout_steps': 24, 'save_interval': 100,
             'entropy': .005, 'schedule': 'fixed'}
            for label, lr, rate in TRIALS
            for phase, updates, envs in (('smoke', 5, 64), ('train', 400, 512))]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def receipt_outputs(path, config):
    receipt = json.loads(path.read_text())
    if receipt['config'] != config:
        raise ValueError(f'Stale receipt configuration: {path}')
    outputs = receipt['outputs']
    if not outputs or any(not Path(p).is_file() or sha256(p) != digest
                          for p, digest in outputs.items()):
        raise ValueError(f'Stale receipt output: {path}')
    return [Path(p) for p in outputs]


def checkpoint_info(path, lr=None):
    import torch
    data = torch.load(path, map_location='cpu', weights_only=True)
    for kind, width, outputs in (('actor', 61, 14), ('critic', 76, 1)):
        state = data[f'{kind}_state_dict']
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError(f'Nonfinite {kind} tensors: {path}')
        for key, shape in {'mlp.0.weight': (512, width), 'mlp.2.weight': (256, 512),
                           'mlp.4.weight': (128, 256), 'mlp.6.weight': (outputs, 128),
                           'obs_normalizer._mean': (1, width),
                           'obs_normalizer._var': (1, width),
                           'obs_normalizer._std': (1, width)}.items():
            if tuple(state[key].shape) != shape:
                raise ValueError(f'Wrong {kind} shape {key}: {path}')
    groups = data['optimizer_state_dict']['param_groups']
    rates = [group['lr'] for group in groups]
    if not rates or any(not math.isfinite(rate) for rate in rates):
        raise ValueError(f'Invalid optimizer rates: {path}')
    if lr is not None and any(not math.isclose(rate, lr) for rate in rates):
        raise ValueError(f'Wrong optimizer rates: {rates}, expected {lr}')
    counter = (data.get('infos') or {}).get('env_state', {}).get('common_step_counter')
    return {'checkpoint': str(path), 'sha256': sha256(path), 'iteration': int(data['iter']),
            'common_step_counter': counter,
            'learning_rates': rates, 'actor_width': 61, 'critic_width': 76, 'action_width': 14}


def prepare_parent(parent, copied, lr):
    import torch
    source = torch.load(parent, map_location='cpu', weights_only=True)
    for group in source['optimizer_state_dict']['param_groups']:
        group['lr'] = lr
    copied.parent.mkdir(parents=True, exist_ok=False)
    torch.save(source, copied)
    original = torch.load(parent, map_location='cpu', weights_only=True)
    restored = torch.load(copied, map_location='cpu', weights_only=True)
    for key in ('actor_state_dict', 'critic_state_dict'):
        if not all(torch.equal(value, restored[key][name]) for name, value in original[key].items()):
            raise ValueError(f'Preparation altered {key}')
    return checkpoint_info(copied, lr)


def nan_metrics(run_dir, required=False):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(run_dir), size_guidance={'scalars': 0}).Reload()
    tags = [tag for tag in events.Tags()['scalars'] if 'nan_state' in tag]
    if required and not tags:
        raise ValueError(f'Missing NaN termination metrics: {run_dir}')
    for tag in tags:
        if any(not math.isfinite(e.value) or e.value != 0 for e in events.Scalars(tag)):
            raise ValueError(f'Nonzero NaN termination: {run_dir}, {tag}')
    return {tag: [e.value for e in events.Scalars(tag)] for tag in tags}


def verify_curriculum_metrics(metrics, name, stages, field, start_counter, updates):
    observed = [values for tag, values in metrics.items() if name in tag and values]
    for index, counter in ((0, start_counter + 24), (-1, start_counter + updates * 24)):
        expected = float([s for s in stages if int(s['step']) <= counter][-1][field])
        if not observed or any(not math.isclose(values[index], expected, abs_tol=1e-6) for values in observed):
            raise ValueError(f'Restored curriculum {name} mismatch at counter {counter}: {observed}, expected {expected}')


def verify_training(run_dir, stage, source):
    import yaml
    # BaseLoader reads the trainer's Python-tagged YAML as data, not Python objects.
    agent = yaml.load((run_dir / 'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    env = yaml.load((run_dir / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    expected = {'seed': 42, 'num_steps_per_env': 24, 'max_iterations': stage['updates'],
                'save_interval': 100}
    if any(int(agent[key]) != value for key, value in expected.items()):
        raise ValueError(f'Training agent configuration mismatch: {run_dir}')
    algorithm = agent['algorithm']
    if (algorithm['schedule'] != 'fixed' or float(algorithm['learning_rate']) != stage['lr']
            or float(algorithm['entropy_coef']) != .005
            or int(env['scene']['num_envs']) != stage['envs']):
        raise ValueError(f'Training recipe mismatch: {run_dir}')
    curriculum = env['curriculum']
    speed_stages = curriculum['running_speed_range']['params']['speed_stages']
    rate_stages = curriculum['action_rate_weight']['params']['weight_stages']
    if (float(speed_stages[-1]['max_speed']) != 2.5
            or float(rate_stages[-1]['weight']) != stage['action_rate']):
        raise ValueError(f'Curriculum mismatch: {run_dir}')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(run_dir), size_guidance={'scalars': 0}).Reload()
    metrics = {tag: [e.value for e in events.Scalars(tag)]
               for tag in events.Tags()['scalars']
               if 'Curriculum' in tag or 'learning_rate' in tag.lower()}
    for name, stages, field in (('running_speed_range', speed_stages, 'max_speed'),
                                ('action_rate_weight', rate_stages, 'weight')):
        verify_curriculum_metrics(metrics, name, stages, field, source['common_step_counter'], stage['updates'])
    return {'agent': agent, 'curriculum': curriculum, 'first_metrics': metrics,
            'nan_metrics': nan_metrics(run_dir, required=True)}


class Comparison:
    def __init__(self, parent, output):
        self.parent, self.output = parent, output
        self.started = time.time()
        self.source = checkpoint_info(parent)
        self.identity = {'parent': self.source, 'schedule': schedule(parent),
                         'settings': SETTINGS, 'worktree': str(ROOT),
                         'source_commit': subprocess.check_output(
                             ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
        manifest = output / 'comparison-manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())['identity'] != self.identity:
                raise ValueError('Comparison manifest changed; refusing continuation')
            self.started = json.loads(manifest.read_text())['started_unix']
        else:
            write(manifest, {'identity': self.identity, 'started_unix': self.started,
                             'python': sys.version, 'max_training_updates': 1215})

    def stage(self, label, config, action):
        folder = self.output / 'stages' / label
        config = {'identity': self.identity, **config}
        receipt = folder / 'completed.json'
        if receipt.exists():
            return receipt_outputs(receipt, config)
        if folder.exists():
            raise ValueError(f'Incomplete stage; inspect before retrying: {folder}')
        folder.mkdir(parents=True)
        write(folder / 'started.json', {'config': config, 'started_unix': time.time()})
        try:
            outputs = action(folder)
            write(receipt, {'config': config, 'outputs': {str(p): sha256(p) for p in outputs},
                            'completed_unix': time.time()})
            return outputs
        except Exception as error:
            write(folder / 'failed.json', {'error': repr(error), 'failed_unix': time.time()})
            raise

    def run(self, args, folder, rate=0.0, watch=None):
        env = trial_environment(os.environ, rate)
        write(folder / 'command.json', {'args': list(map(str, args)), 'cwd': str(ROOT),
                                      'settings': {k: v for k, v in env.items()
                                                   if k.startswith('MICRODUCK_RUNNING_')}})
        print(folder.name, flush=True)
        with (folder / 'process.log').open('x') as log:
            if watch is None:
                subprocess.run(args, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            else:
                with subprocess.Popen(args, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT) as process:
                    try:
                        while True:
                            try:
                                code = process.wait(timeout=20)
                                subprocess.CompletedProcess(args, code).check_returncode()
                                break
                            except subprocess.TimeoutExpired:
                                watch()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise

    def train(self, spec):
        label = 'straight_comparison_' + hashlib.sha256(str(self.output).encode()).hexdigest()[:10] + '_' + spec['label']
        runs = ROOT / 'logs/rsl_rl/running'
        prepared = runs / ('parent_' + label)
        def perform(folder):
            if list(runs.glob('*_' + label)):
                raise ValueError(f'Existing unreceipted training run: {label}')
            copied = prepared / 'checkpoint.pt'
            preparation = prepare_parent(self.parent, copied, spec['lr'])
            if self.source['common_step_counter'] is None or self.source['common_step_counter'] < 5500 * 24:
                raise ValueError('Parent lacks a restored counter activating final action-rate curriculum')
            args = [str(Path(sys.executable).parent / 'train'), TASK,
                    '--env.scene.num-envs', str(spec['envs']), '--agent.resume', 'True',
                    '--agent.experiment-name', 'running', '--agent.load-run', prepared.name,
                    '--agent.load-checkpoint', copied.name, '--agent.max-iterations', str(spec['updates']),
                    '--agent.save-interval', '100', '--agent.run-name', label,
                    '--agent.seed', '42', '--agent.num-steps-per-env', '24',
                    '--agent.logger', 'tensorboard', '--agent.upload-model', 'False',
                    '--agent.algorithm.schedule', 'fixed', '--agent.algorithm.learning-rate', str(spec['lr']),
                    '--agent.algorithm.entropy-coef', '0.005']
            def directories():
                return [p for p in runs.glob('*_' + label) if p != prepared]
            def watch():
                for run_dir in directories():
                    nan_metrics(run_dir)
                    for checkpoint in run_dir.glob('model_*.pt'):
                        # A file being written is checked after the trainer closes it.
                        if time.time() - checkpoint.stat().st_mtime > 10:
                            checkpoint_info(checkpoint, spec['lr'])
            self.run(args, folder, spec['action_rate'], watch)
            found = directories()
            if len(found) != 1:
                raise ValueError(f'Ambiguous training output: {found}')
            checkpoints = sorted(found[0].glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
            if not checkpoints:
                raise ValueError('Training produced no checkpoints')
            validation = [checkpoint_info(p, spec['lr']) for p in checkpoints]
            if validation[-1]['common_step_counter'] != self.source['common_step_counter'] + spec['updates'] * 24:
                raise ValueError('Completed rollout count differs from the bounded update budget')
            write(folder / 'validation.json', {'prepared': preparation, 'checkpoints': validation,
                                               'resolved': verify_training(found[0], spec, self.source)})
            if sha256(self.parent) != self.source['sha256']:
                raise ValueError('Parent checkpoint changed')
            return [*checkpoints, folder / 'validation.json']
        return [p for p in self.stage(spec['label'], spec, perform) if p.suffix == '.pt']

    def evaluate(self, checkpoint, label, seeds, duration):
        rows = []
        for seed in seeds:
            config = {'checkpoint': str(checkpoint), 'sha256': sha256(checkpoint),
                      'seed': seed, 'duration': duration, 'command': 2.0, 'envs': 512, 'warmup': 1}
            def perform(folder):
                output = folder / 'evaluation.json'
                self.run([sys.executable, 'scripts/evaluate_running_checkpoint.py',
                          '--checkpoint-file', str(checkpoint), '--speed', '2.0', '--num-envs', '512',
                          '--seed', str(seed), '--duration-s', str(duration), '--warmup-s', '1',
                          '--output-file', str(output)], folder)
                return [output]
            outputs = self.stage(f'eval_{label}_s{seed}_t{duration}', config, perform)
            row = json.loads(outputs[0].read_text())
            expected = {'checkpoint': str(checkpoint), 'seed': seed, 'duration_s': duration,
                        'command_speed_mps': 2.0, 'num_envs': 512,
                        'heading_controller': 'RunningStraightCommand', 'actor_observation_dim': 61}
            if any(row.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Evaluation provenance mismatch: {outputs[0]}')
            rows.append({**row, 'checkpoint_sha256': config['sha256']})
        return rows

    def export(self, checkpoint, label, video=False):
        def perform(folder):
            output = folder / 'policy.onnx'
            args = [sys.executable, 'scripts/export.py', TASK, '--checkpoint-file', str(checkpoint),
                    '--onnx-file', str(output), '--num-envs', '1', '--device', 'cuda:0', '--video', str(video)]
            if video:
                args += ['--video-length', '500', '--running-speed', '2.0', '--seed', '2027',
                         '--episode-length-s', '11', '--video-width', '640', '--video-height', '360',
                         '--disable-shadows', 'True']
            self.run(args, folder)
            import numpy as np
            import onnx
            import onnxruntime as ort
            onnx.checker.check_model(onnx.load(output))
            session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
            result = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 61), dtype=np.float32)})[0]
            if result.shape != (1, 14) or not np.isfinite(result).all():
                raise ValueError('ONNX dimensions or finite inference failed')
            outputs = [output]
            if video:
                source = checkpoint.parent / 'videos/play/rl-video-step-0.mp4'
                copied = folder / 'selected-10s.mp4'
                shutil.copy2(source, copied)
                outputs.append(copied)
            return outputs
        return self.stage(label + '_export', {'checkpoint': str(checkpoint),
                          'sha256': sha256(checkpoint), 'video': video}, perform)

    def execute(self):
        finalists, screening = [], []
        for spec in schedule(self.parent):
            checkpoints = self.train(spec)
            if spec['updates'] == 5:
                self.export(checkpoints[-1], spec['label'])
                continue
            candidates = []
            for checkpoint in checkpoints:
                rows = self.evaluate(checkpoint, spec['label'] + '_' + checkpoint.stem, (123,), 30)
                candidate = {'checkpoint': str(checkpoint), 'label': spec['trial'], 'screen': rows}
                candidates.append(candidate)
                screening.append(candidate)
                write(self.output / 'screening.json', screening)
            eligible = [c for c in candidates if eligible_score(c['screen']) > -math.inf]
            if eligible:
                finalists.append(max(eligible, key=lambda c: eligible_score(c['screen'])))
        parent = {'checkpoint': str(self.parent), 'label': 'parent'}
        for candidate in [parent, *finalists]:
            checkpoint = Path(candidate['checkpoint'])
            for duration, field in ((10, 'short'), (30, 'long')):
                candidate[field] = self.evaluate(checkpoint, 'validation_' + candidate['label'], SEEDS, duration)
        selected = select_candidate(parent, finalists)
        frozen = {'selected': selected, 'parent': parent, 'finalists': finalists}
        selection_path = self.output / 'frozen-selection.json'
        if selection_path.exists() and json.loads(selection_path.read_text()) != frozen:
            raise ValueError('Frozen selection mismatch; held-out results cannot reselect')
        write(selection_path, frozen)
        heldout = {}
        for name, candidate in (('parent', parent), ('selected', selected)):
            heldout[name] = {str(duration): self.evaluate(Path(candidate['checkpoint']),
                            'heldout_' + name, HELDOUT, duration) for duration in (10, 30)}
        aggregates = {name: {duration: aggregate(rows) for duration, rows in durations.items()}
                      for name, durations in heldout.items()}
        chosen, baseline = aggregates['selected'], aggregates['parent']
        target_met = all(chosen[d]['straight_progress_speed_mps'] >= 2.0
                         and chosen[d]['survival_fraction'] >= threshold
                         for d, threshold in (('10', .95), ('30', .90)))
        useful = (chosen['30']['straight_progress_speed_mps'] >= baseline['30']['straight_progress_speed_mps'] + .02
                  and chosen['30']['survival_fraction'] >= max(.90, baseline['30']['survival_fraction'] - .01))
        destination = self.output / 'selected/checkpoint.pt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(selected['checkpoint']):
            raise ValueError('Selected destination checkpoint differs')
        if not destination.exists():
            shutil.copy2(selected['checkpoint'], destination)
        exports = self.export(destination, 'selected', video=True)
        def parity_stage(folder):
            parity_output = folder / 'parity.json'
            self.run([sys.executable, str(Path(__file__)), '--parent', str(destination),
                      '--output-dir', str(folder), '--parity-onnx', str(exports[0])], folder)
            return [parity_output]
        parity = self.stage('selected_parity', {'checkpoint_sha256': sha256(destination),
                           'onnx_sha256': sha256(exports[0])}, parity_stage)
        result = {'target_met': target_met, 'useful_improvement': useful,
                  'selected_label': selected['label'], 'selected_checkpoint': checkpoint_info(destination),
                  'retained_parent': selected['label'] == 'parent', 'heldout': heldout,
                  'aggregates': aggregates, 'validation': frozen, 'screening': screening,
                  'training_updates': 1215, 'wall_time_s': time.time() - self.started,
                  'recipe': self.identity, 'exports': list(map(str, exports)), 'parity': str(parity[0]),
                  'limitations': 'Simulation only. Policy plus RunningStraightCommand controller; ONNX excludes heading controller. A plateau is not a physical speed limit.'}
        write(self.output / 'result.json', result)
        report(self.output / 'REPORT.md', result)
        return result


def aggregate(rows):
    return {'straight_progress_speed_mps': statistics.mean(r['straight_progress_speed_mps']['mean'] for r in rows),
            'body_forward_speed_mps': statistics.mean(r['forward_speed_mps']['mean'] for r in rows),
            'survival_fraction': statistics.mean(r['survival_fraction'] for r in rows),
            'final_absolute_heading_error_deg': statistics.mean(r['final_absolute_heading_error_deg']['mean'] for r in rows),
            'mean_absolute_heading_error_deg': statistics.mean(r['mean_absolute_heading_error_deg'] for r in rows)}


def report(path, result):
    lines = ['# Straight-running comparison', '',
             f"Target met: {result['target_met']}. Useful improvement: {result['useful_improvement']}.",
             f"Selected: {result['selected_label']}; parent retained: {result['retained_parent']}.",
             f"Training updates: {result['training_updates']}/1215; wall time: {result['wall_time_s']:.0f} seconds.", '',
             '| Policy | Seconds | Seed | Straight m/s | Body m/s | Survival | Final heading deg | Mean heading deg |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, durations in result['heldout'].items():
        for duration, rows in durations.items():
            for row in rows:
                lines.append(f"| {name} | {duration} | {row['seed']} | {row['straight_progress_speed_mps']['mean']:.4f} | "
                             f"{row['forward_speed_mps']['mean']:.4f} | {row['survival_fraction']:.2%} | "
                             f"{row['final_absolute_heading_error_deg']['mean']:.2f} | {row['mean_absolute_heading_error_deg']:.2f} |")
    lines += ['', result['limitations'], '', 'Recipe, checkpoint hashes, all rejected screening and validation candidates,',
              'aggregate results, export paths and parity evidence are preserved in `result.json`.']
    path.write_text('\n'.join(lines) + '\n')


def verify_parity(checkpoint, onnx_path, output):
    """Compare checkpoint and ONNX actions before stepping the SAME real observation."""
    from dataclasses import asdict
    import numpy as np
    import onnxruntime as ort
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner
    import mjlab_microduck.tasks  # noqa: F401
    env_cfg, agent_cfg = load_env_cfg(TASK, play=True), load_rl_cfg(TASK)
    env_cfg.scene.num_envs, env_cfg.seed, env_cfg.episode_length_s = 1, 2027, 11.0
    command = env_cfg.commands['twist']
    command.ranges.lin_vel_x, command.ranges.lin_vel_y = (2.0, 2.0), (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = command.rel_turn_in_place_envs = 0.0
    command.resampling_time_range = (11.0, 11.0)
    raw = ManagerBasedRlEnv(cfg=env_cfg, device='cuda:0')
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    try:
        if type(raw.command_manager.get_term('twist')).__name__ != 'RunningStraightCommand':
            raise ValueError('Parity requires matching heading controller')
        runner = (load_runner_cls(TASK) or OnPolicyRunner)(env, asdict(agent_cfg), device='cuda:0')
        runner.load(str(checkpoint), map_location='cuda:0')
        policy = runner.get_inference_policy(device='cuda:0')
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        obs, errors, observations = env.get_observations(), [], []
        for _ in range(10):
            with torch.inference_mode():
                actions = policy(obs)
                if agent_cfg.clip_actions is not None:
                    actions = actions.clamp(-agent_cfg.clip_actions, agent_cfg.clip_actions)
                inputs = obs['actor'].cpu().numpy()
                actual = session.run(None, {session.get_inputs()[0].name: inputs})[0]
                expected = actions.cpu().numpy()
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
                errors.append(float(np.max(np.abs(actual - expected))))
                observations.append(inputs.tolist())
                obs, _, _, _ = env.step(actions)
        write(output / 'parity.json', {'checkpoint_sha256': sha256(checkpoint),
              'onnx_sha256': sha256(onnx_path), 'steps': 10, 'atol': 2e-5,
              'max_absolute_errors': errors, 'actor_observations': observations,
              'controller': 'RunningStraightCommand', 'command_speed_mps': 2.0})
    finally:
        env.close()


def check_controller(parent, output):
    """Exercise eight real environments through full and partial reset ordering."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks.mdp_terms.running import running_heading_yaw_rate
    import mjlab_microduck.tasks  # noqa: F401
    source = checkpoint_info(parent)
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs, cfg.seed = 8, 42
    raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        raw.common_step_counter = source['common_step_counter']
        raw.reset(seed=42)
        term = raw.command_manager.get_term('twist')
        if type(term).__name__ != 'RunningStraightCommand':
            raise ValueError('Wrong controller in live integration check')
        torch.testing.assert_close(term.target_heading, term.robot.data.heading_w)
        # Distinct old targets make accidental recapture of all environments observable.
        term.target_heading += .3
        old_targets = term.target_heading.clone()
        ids = torch.tensor([1, 3, 6], device=raw.device)
        remaining = torch.tensor([0, 2, 4, 5, 7], device=raw.device)
        raw.reset(env_ids=ids)
        torch.testing.assert_close(term.target_heading[ids], term.robot.data.heading_w[ids])
        torch.testing.assert_close(term.target_heading[remaining], old_targets[remaining])
        expected = running_heading_yaw_rate(term.target_heading, term.robot.data.heading_w)
        torch.testing.assert_close(term.command[:, 2], expected)
        targets = term.target_heading.clone()
        term.time_left[:] = 0.0
        raw.command_manager.compute(dt=0.0)
        torch.testing.assert_close(term.target_heading, targets)
        torch.testing.assert_close(term.command[:, 2], expected)
        if term._pending_heading.any():
            raise ValueError('Pending heading capture remained after sim.forward and command update')
        write(output / 'controller-check.json', {'num_envs': 8, 'partial_reset_ids': [1, 3, 6],
              'full_reset_capture': True, 'partial_reset_capture': True, 'other_targets_preserved': True,
              'resampling_preserved': True, 'yaw_correction_verified': True,
              'restored_common_step_counter': raw.common_step_counter,
              'actual_command_range': list(term.cfg.ranges.lin_vel_x),
              'actual_action_rate_weight': raw.reward_manager.get_term_cfg('action_rate_l2').weight,
              'checkpoint_sha256': source['sha256']})
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check-controller', action='store_true',
                        help='Only run the eight-environment live reset/controller check; no training')
    parser.add_argument('--parity-onnx', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    parent, output = args.parent.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if args.dry_run:
        print(json.dumps({'max_training_updates': 1215, 'stages': schedule(parent),
                          'common_settings': SETTINGS, 'screening': {'seeds': [123], 'duration': 30},
                          'validation': {'seeds': SEEDS, 'durations': [10, 30]},
                          'heldout': {'seeds': HELDOUT, 'durations': [10, 30]}}, indent=2))
        return
    environment = trial_environment(os.environ, 0.0)
    os.environ.clear()
    os.environ.update(environment)
    import mjlab_microduck
    if not Path(mjlab_microduck.__file__).resolve().is_relative_to(ROOT / 'src'):
        raise ValueError('Runtime package does not resolve to this worktree')
    if args.check_controller:
        check_controller(parent, output)
    elif args.parity_onnx:
        verify_parity(parent, args.parity_onnx, output)
    else:
        Comparison(parent, output).execute()


if __name__ == '__main__':
    main()
