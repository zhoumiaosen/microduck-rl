# MicroDuck RL

Train, evaluate, and export reinforcement-learning policies for [Pollen Robotics' MicroDuck](https://github.com/pollen-robotics/microduck), a small biped with 14 actuators. Built with **mjlab, MuJoCo Warp, and PPO**, with 39 registered tasks covering locomotion and additional behaviors.

[**Download the trained model**](https://huggingface.co/zhoumiaosen/microduck-running) · [**Watch the running video**](https://huggingface.co/zhoumiaosen/microduck-running/resolve/main/run.mp4) · [Setup](#setup) · [Train](#train-and-resume) · [Validation](docs/VALIDATION.md)

## Published running policy

The current release reaches **1.50 m/s average body-forward speed in simulation**. It was trained toward a 2.0 m/s command; that sustained-speed target was **not met**.

| Evaluation | Mean body-forward speed | Mean straight progress | Survival |
| --- | ---: | ---: | ---: |
| 10 seconds, averaged over three seeds | 1.499 m/s | 1.234 m/s | 96.74% |
| 30 seconds, seed 456 | 1.499 m/s | See raw evaluation | 91.21% |

Each evaluation uses 512 environments at a 2.0 m/s command. Heading drift reduces straight-line progress, so body-forward speed should not be confused with travel along the original heading. These are simulation results; this release has not been validated on a physical robot.

The [Hugging Face release](https://huggingface.co/zhoumiaosen/microduck-running) contains the final PPO checkpoint, normalized ONNX policy, video, full model card, file hashes, and [raw evaluations](https://huggingface.co/zhoumiaosen/microduck-running/tree/main/evaluation). Training used an RTX 3070, 512 environments, and 10,000 additional updates resumed from iteration 999. The released checkpoint is `model_10998.pt`.

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
uv run --locked hf download zhoumiaosen/microduck-running \
  --revision 49f70558bd8f42f0f05fcdea64f4bec596c9a3a6 \
  --local-dir artifacts/pretrained/microduck-running
```

Measure it at a 2.0 m/s command:

```bash
mkdir -p outputs
uv run --locked python scripts/evaluate_running_checkpoint.py \
  --checkpoint-file artifacts/pretrained/microduck-running/model_10998.pt \
  --speed 2.0 --num-envs 512 --duration-s 10 --warmup-s 1 --seed 123 \
  --output-file outputs/running-evaluation.json
```

Use seeds 456 and 789 for the other short evaluations; use seed 456 and `--duration-s 30` for the long evaluation. Reducing `--num-envs` can help on smaller GPUs, but changes the evaluation population.

Record a ten-second replay and export the policy:

```bash
MUJOCO_GL=egl uv run --locked python scripts/export.py Mjlab-Running-Flat-MicroDuck \
  --checkpoint-file artifacts/pretrained/microduck-running/model_10998.pt \
  --onnx-file outputs/running.onnx --num-envs 1 --seed 123 \
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

The existing [GitHub Actions workflow](.github/workflows/ci.yml) runs CPU tests, packaging checks, documentation-link checks, and selected export and hardware-generation checks. Remote CI has not yet run for this repository. Earlier local validation recorded **220 passed, 2 skipped**; see the [dated validation record](docs/VALIDATION.md).

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
