"""Bounded local experiment; each subprocess finishes before the next uses CUDA.

Run with uv run scripts/run_straight_search.py --help for path options. `recover` evaluates saved policies without training;
`train` consumes the completed recovery. Completed evaluation files are reusable.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/straight2'
BASELINE = ROOT / 'artifacts/speed2/best-so-far/checkpoint.pt'
FLIGHT = ROOT / 'logs/rsl_rl/running/2026-09-10_05-58-21_flight1000'
SEEDS = (123, 456, 789)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def environment(straight):
    env = os.environ.copy()
    env.update(MICRODUCK_RUNNING_STRAIGHT=str(int(straight)),
               MICRODUCK_RUNNING_ENABLE_HEADING_FEEDBACK='0',
               MICRODUCK_RUNNING_ENABLE_SYMMETRY='0',
               MICRODUCK_RUNNING_FIXED_PHYSICS='0', MICRODUCK_RUNNING_SQUARED_PROGRESS='1',
               MICRODUCK_RUNNING_TARGET_MAX_SPEED='2.5', MICRODUCK_RUNNING_SPEED_CAP='2.6',
               MICRODUCK_RUNNING_FORWARD_PROGRESS_WEIGHT='10', MICRODUCK_RUNNING_ACTION_RATE_WEIGHT='0',
               MICRODUCK_RUNNING_ROBUST_PUSH_MPS='0', MICRODUCK_RUNNING_ROBUST_TRUNK_COM_M='0',
               MICRODUCK_RUNNING_ROBUST_HEAD_COM_M='0', MICRODUCK_RUNNING_ROBUST_INITIAL_TILT_DEG='0',
               MICRODUCK_RUNNING_CURRICULUM_START_ITERATION='0', MUJOCO_GL='osmesa')
    return env


def run(args, label, straight):
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'status.json', {'state': 'running', 'operation': label, 'started_unix': time.time(), 'args': args})
    print(label, flush=True)
    with (OUT / f'{label}.log').open('w') as log:
        subprocess.run(args, cwd=ROOT, env=environment(straight), stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def validate(checkpoint, expected_lr=None):
    import torch
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for key in ('actor_state_dict', 'critic_state_dict'):
        if not all(torch.isfinite(t).all() for t in data[key].values()):
            raise ValueError(f'Nonfinite {key}: {checkpoint}')
    rates = [group['lr'] for group in data['optimizer_state_dict']['param_groups']]
    if expected_lr is not None and any(not math.isclose(rate, expected_lr) for rate in rates):
        raise ValueError(f'Wrong restored learning rate: {rates}, expected {expected_lr}')
    return {'checkpoint': str(checkpoint), 'iteration': data['iter'], 'learning_rates': rates,
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def evaluate(checkpoint, label, straight=False, seed=123, duration=10, speed=2.5):
    name = f'{label}_s{seed}_t{duration}_v{speed}'
    path = OUT / 'eval' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        run([sys.executable, 'scripts/evaluate_running_checkpoint.py', '--checkpoint-file', str(checkpoint),
             '--speed', str(speed), '--num-envs', '512', '--seed', str(seed),
             '--duration-s', str(duration), '--warmup-s', '1', '--output-file', str(path)], name, straight)
    row = json.loads(path.read_text())
    expected = 'RunningStraightCommand' if straight else 'VelocityCommandCommandOnly'
    if row['checkpoint'] != str(checkpoint) or row['heading_controller'] != expected:
        raise ValueError(f'Evaluation provenance mismatch: {path}')
    return row


def score(row):
    speed = row['straight_progress_speed_mps']['mean']
    survival = row['survival_fraction']
    return speed if (math.isfinite(speed) and math.isfinite(survival) and survival >= .95) else -math.inf


def meets_target(rows):
    target_rows = [r for r in rows if r['command_speed_mps'] == 2.5]
    return (len(target_rows) == 3 and {r['seed'] for r in target_rows} == {2027, 4093, 8191}
            and all(score(r) >= 2.0 and math.isfinite(r['mean_absolute_heading_error_deg'])
                    and r['mean_absolute_heading_error_deg'] <= 10 for r in target_rows))


def recover():
    checkpoints = [BASELINE, *sorted(FLIGHT.glob('model_*.pt'))]
    if len(checkpoints) != 10:
        raise ValueError(f'Expected baseline and nine flight checkpoints, got {len(checkpoints)}')
    rows = []
    for checkpoint in checkpoints:
        label = 'baseline' if checkpoint == BASELINE else 'recovery_' + checkpoint.stem
        validation = validate(checkpoint)
        row = evaluate(checkpoint, label)
        rows.append({'label': label, 'validation': validation, 'screen': row})
        write(OUT / 'recovery-screen.json', rows)
    eligible = sorted((r for r in rows if score(r['screen']) > -math.inf),
                      key=lambda r: score(r['screen']), reverse=True)
    finalists = eligible[:2]
    if rows[0] not in finalists:
        finalists.append(rows[0])
    for candidate in finalists:
        checkpoint = Path(candidate['validation']['checkpoint'])
        candidate['confirmation'] = [evaluate(checkpoint, candidate['label'], seed=seed, duration=duration)
                                     for seed in SEEDS for duration in (10, 30)]
    eligible = [c for c in finalists if all(score(r) > -math.inf for r in c['confirmation'] if r['duration_s'] == 10)]
    chosen = max(eligible, key=lambda c: statistics.mean(r['straight_progress_speed_mps']['mean']
                 for r in c['confirmation'] if r['duration_s'] == 30)) if eligible else rows[0]
    write(OUT / 'recovery.json', {'parent': chosen['validation']['checkpoint'], 'finalists': finalists})


def train_chunk(parent, label, weight, updates, lr, envs=4096):
    import torch
    source = validate(parent)
    receipt = OUT / f'{label}-validation.json'
    if receipt.exists():
        completed = json.loads(receipt.read_text())
        if completed['source']['sha256'] != source['sha256']:
            raise ValueError(f'Resume source mismatch: {label}')
        result = Path(completed['result']['checkpoint'])
        if validate(result, lr)['sha256'] != completed['result']['sha256']:
            raise ValueError(f'Resume checkpoint changed: {label}')
        return result
    prepared = ROOT / 'logs/rsl_rl/running' / f'straight2_parent_{label}'
    prepared.mkdir(parents=True, exist_ok=False)
    data = torch.load(parent, map_location='cpu', weights_only=False)
    for group in data['optimizer_state_dict']['param_groups']:
        group['lr'] = lr
    copied = prepared / parent.name
    torch.save(data, copied)
    validate(copied, lr)
    task = 'Mjlab-RunningFlight-Flat-MicroDuck' if weight else 'Mjlab-Running-Flat-MicroDuck'
    args = ['uv', 'run', '--locked', 'train', task, '--env.scene.num-envs', str(envs),
            '--agent.resume', 'True', '--agent.experiment-name', 'running',
            '--agent.load-run', prepared.name, '--agent.load-checkpoint', copied.name,
            '--agent.max-iterations', str(updates), '--agent.save-interval', '100',
            '--agent.run-name', label, '--agent.seed', '42', '--agent.logger', 'tensorboard',
            '--agent.upload-model', 'False', '--agent.algorithm.schedule', 'fixed',
            '--agent.algorithm.learning-rate', str(lr), '--agent.algorithm.entropy-coef', '0.005']
    if weight:
        args += ['--env.rewards.flight-event.weight', str(weight),
                 '--env.rewards.flight-event.params.min-forward-speed', '1.5']
    run(args, label, True)
    directories = list((ROOT / 'logs/rsl_rl/running').glob(f'*_{label}'))
    directories = [d for d in directories if d != prepared]
    if len(directories) != 1:
        raise ValueError(f'Ambiguous training run: {directories}')
    checkpoints = sorted(directories[0].glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
    result = checkpoints[-1]
    validation = validate(result, lr)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(directories[0]), size_guidance={'scalars': 0}).Reload()
    for tag in events.Tags()['scalars']:
        if 'nan_state' in tag and any(e.value != 0 for e in events.Scalars(tag)):
            raise ValueError(f'NaN terminations in {label}')
    write(OUT / f'{label}-validation.json', {'source': source, 'result': validation,
                                           'environment': {k: v for k, v in environment(True).items() if k.startswith('MICRODUCK_')}})
    return result


def export(checkpoint, label, video=False):
    path = OUT / f'{label}.onnx'
    args = [sys.executable, 'scripts/export.py', 'Mjlab-Running-Flat-MicroDuck',
            '--checkpoint-file', str(checkpoint), '--onnx-file', str(path), '--num-envs', '1',
            '--device', 'cuda:0', '--video', str(video)]
    if video:
        args += ['--video-length', '1500', '--running-speed', '2.5', '--seed', '2027',
                 '--episode-length-s', '31', '--video-width', '640', '--video-height', '360',
                 '--disable-shadows', 'True']
    run(args, label + '_export', True)
    import onnx
    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value == 61
    assert model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value == 14
    assert any('obs_normalizer' in item.name for item in model.graph.initializer)
    write(OUT / f'{label}-export.json', {'checkpoint': str(checkpoint), 'onnx': str(path),
          'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'input_dim': 61, 'output_dim': 14,
          'video': str(checkpoint.parent / 'videos/play/rl-video-step-0.mp4') if video else None,
          'controller': 'RunningStraightCommand; gain 0.8; yaw-rate cap +/-0.5 rad/s'})


def train():
    parent = Path(json.loads((OUT / 'recovery.json').read_text())['parent'])
    candidates = []
    control = evaluate(parent, 'parent_with_controller_v2', True)
    candidates.append({'checkpoint': str(parent), 'weight': 0, 'screen': control})
    for name, weight in (('A', 0), ('B', 1.5), ('C', 5)):
        smoke = train_chunk(parent, f'straight2v2_{name}_smoke', weight, 5, 5e-5, 64)
        export(smoke, f'{name}_v2_smoke')
        current = parent
        for chunk in range(1, 5):
            label = f'straight2v2_{name}_{chunk * 100}'
            current = train_chunk(current, label, weight, 100, 5e-5)
            row = evaluate(current, label, True)
            candidates.append({'checkpoint': str(current), 'weight': weight, 'screen': row})
            write(OUT / 'training-screen-v2.json', candidates)
    eligible = [c for c in candidates if score(c['screen']) > -math.inf]
    if not eligible:
        write(OUT / 'result.json', {'target_reached': False, 'reason': 'No candidate reached 95% screening survival', 'candidates': candidates})
        return
    best = max(eligible, key=lambda c: score(c['screen']))
    current = Path(best['checkpoint'])
    plateau_reference = score(best['screen'])
    stale = 0
    # Smoke the lower-LR recipe independently; production resumes the unmodified winner.
    smoke = train_chunk(current, 'straight2v2_continue_smoke', best['weight'], 5, 2e-5, 64)
    export(smoke, 'continue_v2_smoke')
    for chunk in range(1, 13):
        label = f'straight2v2_continue_{chunk * 100}'
        current = train_chunk(current, label, best['weight'], 100, 2e-5)
        candidate = {'checkpoint': str(current), 'weight': best['weight'], 'screen': evaluate(current, label, True)}
        candidates.append(candidate)
        value = score(candidate['screen'])
        if value > score(best['screen']):
            best = candidate
        if value >= plateau_reference + .02:
            stale = 0
            plateau_reference = value
        else:
            stale += 1
        write(OUT / 'training-screen-v2.json', candidates)
        if stale >= 4:
            break
    checkpoint = Path(best['checkpoint'])
    final = [evaluate(checkpoint, 'final', True, seed, 30, speed)
             for seed in (2027, 4093, 8191) for speed in (2.0, 2.5)]
    passed = meets_target(final)
    write(OUT / 'result.json', {'target_reached': passed, 'selected': best,
          'validation': validate(checkpoint), 'final_evaluations': final,
          'deployment': 'Policy plus reset-heading controller: yaw rate clip(0.8 * wrapped error, +/-0.5 rad/s); simulator yaw feedback, hardware estimator not validated'})
    lines = ['# Straight-running experiment results', '',
             f'2.0 m/s target reached: **{passed}**.', '',
             'Speed is progress along the initial heading over a fixed window, with zero contribution after falling.',
             'All results use the policy plus a reset-heading controller (gain 0.8, yaw-rate cap 0.5 rad/s).', '',
             '| Seed | Command m/s | Straight m/s | Survival | Heading error deg |',
             '|---|---:|---:|---:|---:|']
    for r in final:
        lines.append(f"| {r['seed']} | {r['command_speed_mps']} | {r['straight_progress_speed_mps']['mean']:.3f} | "
                     f"{r['survival_fraction']:.1%} | {r['mean_absolute_heading_error_deg']:.1f} |")
    lines += ['', f'Source checkpoint: `{checkpoint}`.',
              f'Full provenance and acceptance results: `{OUT / "result.json"}`.',
              'The controller uses simulator yaw. Equivalent heading feedback is required for hardware use; a hardware estimator has not been validated. These are simulation results.']
    (OUT / 'STRAIGHT_SPEED_RESULTS.md').write_text('\n'.join(lines) + '\n')
    retained = OUT / 'selected/checkpoint.pt'
    retained.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, retained)
    export(retained, 'selected', video=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('recover', 'train'))
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--baseline-checkpoint', type=Path, default=BASELINE)
    parser.add_argument('--flight-run-dir', type=Path, default=FLIGHT)
    args = parser.parse_args()
    OUT = args.output_dir.expanduser().resolve()
    BASELINE = args.baseline_checkpoint.expanduser().resolve()
    FLIGHT = args.flight_run_dir.expanduser().resolve()
    os.chdir(ROOT)
    try:
        {'recover': recover, 'train': train}[args.phase]()
        write(OUT / f'{args.phase}-completed.json', {'completed_unix': time.time()})
        write(OUT / 'status.json', {'state': 'complete', 'phase': args.phase})
    except Exception as error:
        write(OUT / 'failure.json', {'error': repr(error), 'time': time.time()})
        status = json.loads((OUT / 'status.json').read_text()) if (OUT / 'status.json').exists() else {}
        write(OUT / 'status.json', {**status, 'state': 'failed', 'error': repr(error)})
        raise
