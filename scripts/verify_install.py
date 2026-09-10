"""Verify task discovery and robot assets using the installed Python package.

Run with the wheel environment's Python from outside the source checkout.
"""
from pathlib import Path

import mjlab_microduck
import mjlab_microduck.tasks  # noqa: F401 -- explicitly fail if plugin loading failed
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg


def main() -> None:
    tasks = [task for task in list_tasks() if 'MicroDuck' in task or 'Microduck' in task]
    assert tasks, 'No MicroDuck tasks registered'
    models = set()
    for task in tasks:
        for play in (False, True):
            cfg = load_env_cfg(task, play=play)
            assert cfg.observations and cfg.actions, task
            for entity in cfg.scene.entities.values():
                spec_fn = entity.spec_fn
                if spec_fn not in models:
                    model = spec_fn().compile()
                    assert model.nq > 0, task
                    models.add(spec_fn)
        assert load_rl_cfg(task).num_steps_per_env > 0, task
    print(f'Package: {Path(mjlab_microduck.__file__).resolve()}')
    print(f'PASS: {len(tasks)} tasks, train/play configurations, {len(models)} robot specifications')


if __name__ == '__main__':
    main()
