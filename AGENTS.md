# Repository Guidelines

## Project Overview

CAIDBench is a Python benchmark framework for continual AI-generated image
detection and continual deepfake/face-forgery detection. Keep the framework
boundaries clear:

- `caidbench/engine/`: generic continual training, evaluation, checkpointing,
  and output writing.
- `caidbench/methods/`: method-specific continual learning behavior.
- `caidbench/data/`: AID Arrow loading, protocols, dataset sources, transforms,
  and packers.
- `caidbench/models/`, `caidbench/losses/`, `caidbench/memory/`: reusable model,
  objective, and replay components.
- `configs/` and `protocols/`: YAML entry points for experiments.
- `tests/`: pytest smoke and unit coverage.

## Setup and Commands

Install locally from the repository root:

```bash
pip install -e .
```

Useful optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[clip]"
pip install -e ".[sprompts]"
pip install -e ".[hub]"
```

Run tests with:

```bash
pytest
```

Run a focused test during development with:

```bash
pytest tests/test_smoke.py
pytest tests/test_method_compat.py
```

Train from a config with:

```bash
caid-train --config configs/dfil.yaml
```

For smoke tests or local runs without experiment tracking, set
`logging.backend=none`.

## Coding Conventions

- Use modern Python with type annotations where the surrounding code already
  does.
- Preserve lazy method registration through `caidbench.registry`; new method
  classes should be registered from `caidbench/methods`.
- Keep generic loop behavior in `Trainer`; put paper/method-specific training
  logic in the corresponding method class.
- Prefer structured config dictionaries and YAML-compatible values. Avoid
  hard-coding dataset paths, devices, or experiment names in reusable modules.
- Keep tensor/device movement consistent with helpers such as `move_to_device`
  and `batch_to_device`.
- Return scalar train metrics from methods through `train_metrics` or scalar
  entries in `observe`; avoid logging non-scalar tensors as metrics.

## Logging Rules

Read `docs/LOGGING_GUIDE.md` before changing logging behavior.

- Preserve the separation between human-readable event logs and scalar metric
  logs.
- Keep console output stable, short, and readable.
- Keep progress logging on a fixed cadence.
- Avoid per-batch noise unless debugging is explicitly enabled.
- Keep experiment payloads structured and scalar-only.
- When adding new logs, match existing prefixes and field style.

## Data and Outputs

- AID Arrow metadata is reconstructed from `mapping.json`, split JSON files, and
  optional `caid_meta.jsonl`; continual `task_id` values come from the YAML
  protocol at load time.
- Avoid committing generated experiment outputs, checkpoints, cache files, or
  large local datasets unless explicitly requested.
- Treat `outputs/`, `data/`, and local sidecars as generated or environment
  specific unless a task explicitly targets them.

## Testing Guidance

- Add or update focused pytest coverage when changing behavior in shared
  trainers, data loading, metrics, method registration, or method-specific
  algorithms.
- For method changes, prefer small deterministic CPU tests with the
  `small_conv` backbone pattern used in existing tests.
- For logging changes, assert the user-visible line shape and structured scalar
  payload behavior separately when practical.

## Git Hygiene

- Do not revert or overwrite unrelated uncommitted work.
- Keep edits scoped to the requested behavior and the relevant module boundary.
- If the working tree is already dirty, inspect affected files before editing
  them and preserve user changes.
