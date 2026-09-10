# MicroDuck RL

Local reinforcement-learning environments, policy experiments, and printable
hardware add-ons for [Pollen Robotics' MicroDuck](https://github.com/pollen-robotics/microduck).

This independent snapshot refactors
[`Vottivott/microduck-playground`](https://github.com/Vottivott/microduck-playground),
itself derived from [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl).
It starts a fresh local Git history on `main`; it does not preserve the source
Git history and has no configured publishing destination. It is not an official
Pollen Robotics release. All registered RL task families are retained.
See [provenance](docs/PROVENANCE.md), [NOTICE](NOTICE), and the
[current validation record](docs/VALIDATION.md).

The experiment results and media below are retained source artifacts, not new
measurements of this refactor. External policy links remain attributed to their
original publishers.

## Experiments

Animated previews play directly in the table. Click one—or use its explicit
full-video link—to open the complete silent MP4.

<table>
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Preview</th>
      <th>Result and artifacts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Self-pumped swing</strong></td>
      <td>
        <a href="experiments/swing/media/alpha050_seed27.mp4">
          <img src="experiments/swing/media/preview.gif" width="280" alt="Animated preview of MicroDuck pumping itself on a swing">
        </a>
      </td>
      <td>
        Starts still and reaches a 173.20° strict full span.<br>
        <a href="experiments/swing/media/alpha050_seed27.mp4">Full video</a> ·
        <a href="experiments/swing/README.md">Experiment</a> ·
        <a href="integrations/pollen-microduck/README.md">Runtime adapter</a> ·
        <a href="https://huggingface.co/HannesVonEssen/microduck-swing">ONNX on Hugging Face</a>
      </td>
    </tr>
    <tr>
      <td><strong>Fast running</strong></td>
      <td>
        <a href="experiments/running/media/preview.mp4">
          <img src="experiments/running/media/preview.gif" width="280" alt="Animated preview of MicroDuck running on flat ground">
        </a>
      </td>
      <td>
        Robustified iteration-12,195 simulation candidate: 1.651 m/s nominal,
        and 1.612 m/s under backlash plus disturbance stress.<br>
        <a href="experiments/running/media/preview.mp4">Full video</a> ·
        <a href="experiments/running/README.md">Experiment</a> ·
        <a href="https://huggingface.co/HannesVonEssen/microduck-running">ONNX on Hugging Face</a>
      </td>
    </tr>
    <tr>
      <td><strong>Stilt walking</strong></td>
      <td>
        <a href="experiments/stilts/media/preview.mp4">
          <img src="experiments/stilts/media/preview.gif" width="280" alt="Animated preview of MicroDuck walking on green 10 cm stilts">
        </a>
      </td>
      <td>
        Blend-0.50 policies for 10, 15, 20, 25, and 50 cm, plus
        1.0, 1.4, and 2.0 m simulation stilts (10 cm shown).<br>
        <a href="experiments/stilts/media/preview.mp4">Full video</a> ·
        <a href="experiments/stilts/README.md">Experiment</a> ·
        <a href="hardware/stilts/README.md">Hardware</a> ·
        <a href="https://huggingface.co/HannesVonEssen/microduck-stilts">Policies and videos</a>
      </td>
    </tr>
  </tbody>
</table>

Each preview is a direct simulation demonstration of the policy linked in its
row. Compact machine-readable evaluation records live beside each experiment.

## Hardware galleries

The retained swing seat keeps the battery centered without occupying the
head-and-leg pumping corridors. It includes compliant locating pads, a padded
strap, and a removable buckle. The source generators, printable millimetre
meshes, MuJoCo collision hulls, and clearance reports are under
[`hardware/swing-seat`](hardware/swing-seat/README.md).

<table>
  <tr>
    <td align="center"><img src="hardware/swing-seat/renders/seat_front.png" width="300" alt="Retained swing seat, front view"><br><sub>Front</sub></td>
    <td align="center"><img src="hardware/swing-seat/renders/seat_three_quarter.png" width="300" alt="Retained swing seat, three-quarter view"><br><sub>Three-quarter</sub></td>
    <td align="center"><img src="hardware/swing-seat/renders/seat_side.png" width="300" alt="Retained swing seat, side view"><br><sub>Side</sub></td>
  </tr>
</table>

The stilt system replaces the removable soles and preserves explicit tip
contact geometry. The gallery uses the demonstrated green 10 cm blend-0.50
configuration. Parametric generators and printable meshes are under
[`hardware/stilts`](hardware/stilts/README.md).

<table>
  <tr>
    <td align="center"><img src="hardware/stilts/renders/stilts_front.png" width="230" alt="MicroDuck green stilts, front view"><br><sub>Front</sub></td>
    <td align="center"><img src="hardware/stilts/renders/stilts_three_quarter.png" width="230" alt="MicroDuck green stilts, three-quarter view"><br><sub>Three-quarter</sub></td>
    <td align="center"><img src="hardware/stilts/renders/stilts_side.png" width="230" alt="MicroDuck green stilts, side view"><br><sub>Side</sub></td>
    <td align="center"><img src="hardware/stilts/renders/printed_stilt.jpg" width="230" alt="Green 3D-printed MicroDuck replacement sole and stilt prototype"><br><sub>3D-printed prototype</sub></td>
  </tr>
</table>

## Setup and task discovery

Use Linux or WSL2 with Python 3.12 and [`uv`](https://docs.astral.sh/uv/).
GPU training requires a CUDA-capable NVIDIA GPU accessible from that Linux
session. Native Windows training is not the supported setup. Run these commands
from this checkout's root (in WSL, for example,
`cd /mnt/f/Codex_GitHub/Duck/microduck-rl`):

```bash
uv python install 3.12
uv sync --locked --python 3.12
uv run --locked list-envs
```

`uv.lock` fixes dependency revisions, including BAM. Keep the lockfile and
`pyproject.toml` together; do not upgrade dependencies to troubleshoot setup
without checking compatibility. See [validation](docs/VALIDATION.md) for checks
that have actually run and platform limitations.

## Train and resume

Start with a five-iteration smoke run, then run the full configuration after
checking its output. TensorBoard logging keeps these examples local.

```bash
uv run --locked train Mjlab-SwingPump-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.logger tensorboard

uv run --locked train Mjlab-SwingPump-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
```

Logs and checkpoints are written under `logs/rsl_rl/<experiment_name>/<run>/`.
To resume, set `RUN` to the existing run directory name and `MODEL` to its
checkpoint filename. Use the same task and environment settings as the saved
run; `--agent.load-run` is relative to that task's experiment log directory.

```bash
RUN=your_existing_run_directory
MODEL=model_4.pt
uv run --locked train Mjlab-SwingPump-MicroDuck \
  --agent.resume True \
  --agent.load-run "$RUN" \
  --agent.load-checkpoint "$MODEL" \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.logger tensorboard
```

The retained [running continuation recipe](experiments/running/README.md) has
specific curriculum settings for its published checkpoint. Additional running
trial scripts live in [`scripts/experiments/`](scripts/experiments/); inspect
their settings before starting a run.

## Evaluate and export

Set `CHECKPOINT` to the full path of a swing checkpoint from the run above.
The swing evaluator measures a deterministic physical rollout and writes JSON;
it does not certify hardware readiness.

```bash
CHECKPOINT=/absolute/path/to/model_4.pt
mkdir -p outputs
uv run --locked python scripts/evaluate_swing_checkpoint.py "$CHECKPOINT" \
  --output outputs/swing-evaluation.json --device cpu --duration 24 --seed 72

uv run --locked python scripts/export.py Mjlab-SwingPump-MicroDuck \
  --checkpoint-file "$CHECKPOINT" \
  --onnx-file outputs/swing.onnx --num-envs 1 --device cpu
```

Always export through `scripts/export.py`: it embeds observation normalization,
training action clipping where configured, and policy metadata. For swing
runtime requirements, use the [policy-specific adapter](integrations/pollen-microduck/README.md).
For a running checkpoint, use its task and evaluator instead:

```bash
RUNNING_CHECKPOINT=/absolute/path/to/running_checkpoint.pt
uv run --locked python scripts/evaluate_running_checkpoint.py \
  --checkpoint-file "$RUNNING_CHECKPOINT" \
  --task-id Mjlab-Running-Flat-MicroDuck --speed 2.2 \
  --num-envs 64 --duration-s 10 --warmup-s 1 \
  --output-file outputs/running-evaluation.json
```

## Repository layout

```text
experiments/
  running/               clean policy preview and result summary
  stilts/                policy index, executed curriculum, continuation guide
  swing/                 selected checkpoints, evaluation, media, methodology
hardware/
  stilts/                parametric stilt generator and printable meshes
  swing-seat/            retained-seat generator, meshes, clearance reports
src/mjlab_microduck/     tasks, robot models, actuator model, rewards
scripts/                evaluation, export, rendering, and selection tools
  experiments/          local running trial and continuation scripts
integrations/            policy-specific deployment adapters
tests/                   CPU configuration and invariant tests
docs/                    supporting research and training notes
```

## Scope and safety

These are simulation experiments, not hardware safety certifications. The
swing model simulates two elastic tension-only cords and randomized actuator
and sensor dynamics, but real cord knots, frame flex, textile contact, servo
temperature, and assembly tolerances remain. Extreme-height stilts require an
engineered load path and fall protection. Use a safety tether, current limits,
an emergency stop, a clear exclusion zone, and conservative incremental tests.

## License

Software is licensed under Apache-2.0; see [`LICENSE`](LICENSE). As in the
upstream project, 3D hardware design files are licensed under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International; see
[`LICENSE-HARDWARE`](LICENSE-HARDWARE). Third-party MicroDuck assets retain
their original attribution and terms. See [`NOTICE`](NOTICE) and the
hardware-specific READMEs.
