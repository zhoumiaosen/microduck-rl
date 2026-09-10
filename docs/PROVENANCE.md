# Source and refactor provenance

This repository is an independent snapshot of
[Vottivott/microduck-playground](https://github.com/Vottivott/microduck-playground)
at commit `828d950134e29a8d04cbb51720a22c8729047fb7`, including local working-tree
changes present on 2026-09-10. That project derives from
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl).
Fresh Git history does not imply original authorship of the inherited work.
See [NOTICE](../NOTICE), [LICENSE](../LICENSE), and [LICENSE-HARDWARE](../LICENSE-HARDWARE).

## Snapshot

The initial copy contained 447 files (93,284,788 bytes), individually verified
against their source SHA-256 hashes before refactoring. The source checkout's
three modified tracked files were the running evaluator, shared task terms,
and running task configuration. Its untracked running experiment scripts,
regression tests, and result/plan documents were included as well.

The original filenames and hashes are recorded in
[source-files.json](provenance/source-files.json); names precede the relocation
of experiment scripts. The separately recovered flight checkpoint selector is
recorded in [additional-source-files.json](provenance/additional-source-files.json).
Source Git history, virtual environments, caches, and bulk training outputs
were not copied. Curated checkpoints and media supporting the existing
reproducibility tests were retained.

## Changes in this repository

- Extracted the 269 task functions/classes into focused `mdp_terms` modules and
  a single-install runtime patch module. Function/class bodies are unchanged;
  `tasks.mdp` remains the compatibility import path for saved configurations.
- Moved local trial launchers into `scripts/experiments`, made their paths
  portable, and retained the existing experiment settings.
- Added compatibility and installed-package checks, retained the dependency
  lock, and packaged both software/hardware licenses and attribution notices.
- Rewrote setup and usage documentation for a standalone local project.

Existing experiment scores, videos, and result documents are historical source
artifacts. They are not new measurements produced by this refactor. See
[VALIDATION.md](VALIDATION.md) for the checks performed on this repository.
