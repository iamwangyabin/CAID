# CAIDBench Configs

These configs are complete experiment entry points for the generated CAIDBench
continual protocols. Use them with `caid-train --config ...`.

## CAIDBench Default Protocol

`configs/caidbench/finetune.yaml` wires together the pieces required for a
run:

- dataset root: `scenario.data.path`
- task protocol: `scenario.protocol`
- selected metadata index: `index_path` inside the protocol YAML

Example:

```bash
caid-train --config configs/caidbench/finetune.yaml
```

For local smoke runs without SwanLab:

```bash
caid-train \
  --config configs/caidbench/finetune.yaml \
  --override logging.backend=none
```

By default, checkpoints contain only tensor state dicts and scalar/list/dict
metadata, so they can be loaded with `torch.load(..., weights_only=True)`.
Runs save `task_{index}.pt` after every continual task/stage, save `base.pt`
after the first task, and keep updating `last.pt`. Set
`checkpoint.save_each_task=false` only when you explicitly want to skip
per-task checkpoint files.

Resume from the last completed task with:

```bash
caid-train \
  --config configs/caidbench/finetune.yaml \
  --override resume_from=/path/to/run/last.pt logging.backend=none
```

For a quick debug pass where each epoch only consumes a few train batches:

```bash
caid-train \
  --config configs/caidbench/finetune.yaml \
  --override logging.backend=none train.epochs=1 train.debug_max_steps_per_epoch=2 eval.max_batches_per_task=2
```

The CAIDBench configs default to the lab-server Arrow package at
`/home/home/yabin/CAIDBench`. For a quick server smoke run, use:

```bash
caid-train \
  --config configs/caidbench/finetune.yaml \
  --override \
    logging.backend=none \
    train.epochs=1 \
    train.debug_max_steps_per_epoch=2 \
    eval.max_batches_per_task=2
```

`train.debug_max_steps_per_epoch` limits only train-loader consumption. Without
an eval limit, validation still runs over every selected test sample. Use
`eval.max_batches_per_task` for fast end-to-end smoke runs.

Eval/test dataloaders default to single-process loading
(`train.eval_num_workers=0`) to avoid accumulating worker file descriptors
during long many-task runs. Train loaders still use `train.num_workers`.
Override `train.eval_num_workers` explicitly if you want multiprocessing eval.

The same value can be placed in YAML:

```yaml
train:
  epochs: 1
  debug_max_steps_per_epoch: 2
  eval_num_workers: 0
eval:
  # seen: standard lower-triangle CL eval; all: full matrix including future tasks;
  # current: fastest smoke mode with only the diagonal task.
  scope: seen
  max_batches_per_task: 2
```

## Evaluation Scope and Matrix Semantics

CAIDBench stores metric matrices such as `acc_matrix.csv`, `auc_matrix.csv`,
`ap_matrix.csv`, and `f1_matrix.csv`.

- Rows are model states after training task `i`, written as `after_task=i`.
- Columns are evaluation tasks `j`.
- In the default `eval.scope=seen` mode, only `j <= i` is evaluated. This is the
  standard continual-learning lower triangle used for retention and forgetting.
- In `eval.scope=all`, every row evaluates all tasks. Columns with `j > i` are
  future/unseen tasks and measure cross-generator or future-domain
  generalization.
- In `eval.scope=current`, only the current task is evaluated, producing a fast
  diagonal smoke run.

The standard summary fields (`average_accuracy`, `average_forgetting`,
`average_auc`, `average_ap`, and `average_f1`) are always computed from seen
tasks only. Future-task results from `eval.scope=all` are reported separately as
`future_average_*` fields in `summary.json`, `eval/future_*` scalar logs, and
`future_weighted_curves.csv`.

For future-task generalization, run a full matrix:

```bash
caid-train \
  --config configs/caidbench/finetune.yaml \
  --override eval.scope=all
```

For a full matrix smoke run with bounded evaluation cost:

```bash
caid-train \
  --config configs/caidbench/finetune.yaml \
  --override \
    logging.backend=none \
    train.epochs=1 \
    train.debug_max_steps_per_epoch=2 \
    eval.scope=all \
    eval.max_batches_per_task=2
```

Evaluation cost depends on scope. With 90 tasks and 2000 test samples per task:

| scope | total eval samples across the full run |
|---|---:|
| `seen` | `2000 * (1 + 2 + ... + 90) = 8,190,000` |
| `all` | `2000 * 90 * 90 = 16,200,000` |
| `current` | `2000 * 90 = 180,000` |

Use `eval.max_batches_per_task` during debugging to avoid spending most runtime
in evaluation.

If the CAIDBench Arrow dataset is stored somewhere else, override only the data
root:

```bash
caid-train \
  --config configs/caidbench/finetune.yaml \
  --override scenario.data.path=/path/to/caidbench_arrow_root
```

To run the model-appearance order instead of the default order:

```bash
caid-train \
  --config configs/caidbench/finetune.yaml \
  --override scenario.protocol=protocols/caidbench/model_appearance_order_protocol.yaml
```
