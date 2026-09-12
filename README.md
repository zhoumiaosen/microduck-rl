# MicroDuck RL

Train, evaluate, and export reinforcement-learning policies for [Pollen Robotics' MicroDuck](https://github.com/pollen-robotics/microduck), a small biped with 14 actuators. Built with **mjlab, MuJoCo Warp, and PPO**, with 39 registered tasks covering locomotion and additional behaviors.

[**Download the trained model**](https://huggingface.co/zhoumiaosen/microduck-straight-running) · [**Watch the running video**](https://huggingface.co/zhoumiaosen/microduck-straight-running/resolve/main/run.mp4) · [Setup](#setup) · [Train](#train-and-resume) · [Validation](docs/VALIDATION.md)

## Published running policy

The current release reaches **1.844 m/s straight progress over 10 seconds** and **1.806 m/s over 30 seconds** in simulation, using a heading-hold controller. The sustained **2.0 m/s target was not met**.

| Evaluation | Parent straight progress | Selected straight progress | Selected survival |
| --- | ---: | ---: | ---: |
| 10 seconds, three held-out seeds | 1.83940 m/s | 1.84439 m/s | 98.05% |
| 30 seconds, three held-out seeds | 1.79879 m/s | 1.80560 m/s | 93.16% |

The observed 30-second gain is **+0.00681 m/s (+0.38%)**, below the predefined 0.02 m/s useful-improvement threshold. Repeat evaluations varied, so this small gain is not established as repeatable. Each evaluation uses 512 environments, command 2.0 m/s, a one-second warmup, and held-out seeds 2027/4093/8191. Selection was frozen before these tests. Straight progress measures travel along the initial heading and counts failed trajectories as zero thereafter; it is distinct from body-forward speed.

The [Hugging Face release](https://huggingface.co/zhoumiaosen/microduck-straight-running) contains selected checkpoint `model_13600.pt`, normalized ONNX, video, model card, file hashes, recipe, and [raw evaluations](https://huggingface.co/zhoumiaosen/microduck-straight-running/tree/main/evaluation). Three independent continuations plus smoke tests consumed 1,215 updates on an RTX 3070. Trial A's early checkpoint won validation; later checkpoints generally lost stability.

The comparison parent is the stronger intermediate straight-running checkpoint, not the [older public model](https://huggingface.co/zhoumiaosen/microduck-running). That older release used a different control configuration, so this is not a controlled improvement claim over it. Physical robot deployment remains untested.

## Setup

Use **Linux or WSL2**, Python **3.12**, and [uv](https://docs.astral.sh/uv/). GPU training requires a CUDA-capable NVIDIA GPU available inside Linux/WSL. Native Windows GPU training is not supported by this setup.

From the root of your cloned checkout:

```bash
uv python install 3.12
uv sync --locked --python 3.12
uv run --locked list-envs
```

Keep `uv.lock` and `pyproject.toml` together. The lockfile records the dependency versions used by this project. See [validation](docs/VALIDATION.md) for tested configurations and remaining limitations.

## Try the published model

Download the public release without needing a Hugging Face login:

```bash
uv run --locked hf download zhoumiaosen/microduck-straight-running \
  --revision 9241cc4e0a5e99e84f2025558c1cda74ecde60fc \
  --local-dir artifacts/pretrained/microduck-straight-running
source artifacts/pretrained/microduck-straight-running/environment.sh
```

Measure it at a 2.0 m/s command:

```bash
mkdir -p outputs
uv run --locked python scripts/evaluate_running_checkpoint.py \
  --checkpoint-file artifacts/pretrained/microduck-straight-running/model_13600.pt \
  --speed 2.0 --num-envs 512 --duration-s 10 --warmup-s 1 --seed 2027 \
  --output-file outputs/running-evaluation.json
```

Use seeds 4093 and 8191 for the other evaluations, and repeat all three with `--duration-s 30` for the long evaluations. Start from a clean shell without other `MICRODUCK_RUNNING_*` overrides. Reducing `--num-envs` can help on smaller GPUs, but changes the evaluation population.

Record a ten-second replay and export the policy:

```bash
MUJOCO_GL=egl uv run --locked python scripts/export.py Mjlab-Running-Flat-MicroDuck \
  --checkpoint-file artifacts/pretrained/microduck-straight-running/model_13600.pt \
  --onnx-file outputs/running.onnx --num-envs 1 --seed 2027 \
  --running-speed 2.0 --episode-length-s 11 \
  --video True --video-length 500 --video-width 1280 --video-height 720
```

The video is saved under `videos/play/` beside the checkpoint. Headless rendering requires working EGL support. The published video can also be watched directly from the release link above.

## Train and resume

Start with a small smoke run and inspect its output before launching longer training:

```bash
uv run --locked train Mjlab-Running-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.run-name smoke --agent.logger tensorboard --agent.upload-model False
```

Then start a fresh running policy, for example:

```bash
MICRODUCK_RUNNING_TARGET_MAX_SPEED=2.0 MICRODUCK_RUNNING_SPEED_CAP=2.2 \
uv run --locked train Mjlab-Running-Flat-MicroDuck \
  --env.scene.num-envs 512 --agent.max-iterations 10000 \
  --agent.seed 42 --agent.run-name running --agent.logger tensorboard \
  --agent.upload-model False
```

This fresh-run example is not an exact reproduction of the published continuation. The curriculum gradually increases command speed; setting a target does not mean the robot immediately trains at or achieves that speed. Training can take several hours.

Logs and checkpoints live in `logs/rsl_rl/<experiment_name>/<run>/`. To continue an existing running run, replace the example directory and checkpoint below:

```bash
RUN=your_existing_run_directory
MODEL=model_999.pt
MICRODUCK_RUNNING_TARGET_MAX_SPEED=2.0 MICRODUCK_RUNNING_SPEED_CAP=2.2 \
uv run --locked train Mjlab-Running-Flat-MicroDuck \
  --agent.resume True --agent.load-run "$RUN" --agent.load-checkpoint "$MODEL" \
  --env.scene.num-envs 512 --agent.max-iterations 10000 \
  --agent.logger tensorboard --agent.upload-model False
```

On resume, `max-iterations` specifies additional updates. Match the task and training settings to the saved run. Further experiment launchers are documented in [scripts/experiments](scripts/experiments/README.md).

## Policy interface

The policy runs at **50 Hz**, taking **61 actor observations** and producing **14 actions**. The observation layout includes 48 proprioceptive values plus a 13-value command block. Preserve the matching runtime's observation order, joint order, action scaling, and actuator model.

The current release requires `RunningStraightCommand`, enabled by the downloaded environment file. It holds the reset heading through a yaw-rate command. **The ONNX file does not include this heading controller**; keep the matching command construction and reset behavior when integrating it.

Always export using [scripts/export.py](scripts/export.py): it embeds observation normalization and action clipping where configured. Do not apply normalization twice or treat the action vector as direct motor commands. Runtime integration is separate from training; see the [upstream robot runtime](https://github.com/pollen-robotics/microduck) and the [retained swing adapter](integrations/pollen-microduck/README.md) for their respective contracts.

## Earlier running experiment

This running demonstration was inherited from the source project. Its results are separate from this repository's newly trained policy above; the linked upstream model retains its original attribution.

| Experiment | Preview | Documentation |
| --- | --- | --- |
| Earlier running policy | [![Upstream running demonstration](experiments/running/media/preview.gif)](experiments/running/media/preview.mp4) | [Results and model](experiments/running/README.md) |

## Development

```bash
uv run --locked --with pytest pytest -q -rs
uv build
uv run --locked python scripts/verify_install.py
uv run --locked python scripts/check_relative_links.py
```

The existing [GitHub Actions workflow](.github/workflows/ci.yml) runs CPU tests, packaging checks, documentation-link checks, and selected export and hardware-generation checks. Local straight-running release validation recorded **239 passed, 2 skipped**, successful package builds, and ONNX action parity within `2e-5`; see the [dated validation record](docs/VALIDATION.md). Local results do not assert the status of remote CI.

| Path | Contents |
| --- | --- |
| `src/mjlab_microduck/` | Tasks, robot models, actuator model, and shared policy terms |
| `scripts/` | Training helpers, evaluation, export, and verification |
| `experiments/` | Retained experiment documentation and curated artifacts |
| `hardware/` | Add-on generators, printable meshes, and galleries |
| `integrations/` | Policy-specific runtime adapters |
| `tests/` | Configuration and behavior regression tests |
| `docs/` | Provenance, validation, and supporting notes |

Generated training logs, new checkpoints, exports, virtual environments, and release staging files are ignored by Git. Download the published model from Hugging Face; curated historical experiment artifacts remain tracked for reproducibility.

## Attribution and license

This independent refactor is based on [Vottivott/microduck-playground](https://github.com/Vottivott/microduck-playground), itself derived from [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl). It is not an official Pollen Robotics release. Fresh Git history does not imply original authorship of the inherited work. See [provenance](docs/PROVENANCE.md) and [NOTICE](NOTICE).

Software: [Apache-2.0](LICENSE). 3D hardware designs: [CC BY-NC-SA 4.0](LICENSE-HARDWARE). Simulation results do not establish hardware readiness.
