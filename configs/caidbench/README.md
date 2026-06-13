# CAIDBench Configs

These configs are complete experiment entry points for the generated CAIDBench
continual protocols. Use them with `caid-train --config ...`.

## CAIDBench Default Protocol

`configs/caidbench/finetune_default.yaml` wires together the pieces required for a
run:

- dataset root: `scenario.data.path`
- task protocol: `scenario.protocol`
- selected metadata index: `index_path` inside the protocol YAML

Example:

```bash
caid-train --config configs/caidbench/finetune_default.yaml
```

For local smoke runs without SwanLab:

```bash
caid-train \
  --config configs/caidbench/finetune_default.yaml \
  --override logging.backend=none
```

By default, checkpoints contain only tensor state dicts and scalar/list/dict
metadata, so they can be loaded with `torch.load(..., weights_only=True)`.
Runs save `base.pt` after the first task and keep updating `last.pt`; per-task
`task_*.pt` files are disabled unless you set `checkpoint.save_each_task=true`.

Resume from the last completed task with:

```bash
caid-train \
  --config configs/caidbench/finetune_default.yaml \
  --override resume_from=/path/to/run/last.pt logging.backend=none
```

For a quick debug pass where each epoch only consumes a few train batches:

```bash
caid-train \
  --config configs/caidbench/finetune_default.yaml \
  --override logging.backend=none train.epochs=1 train.debug_max_steps_per_epoch=2
```

Eval/test dataloaders default to single-process loading
(`train.eval_num_workers=0`) to avoid accumulating worker file descriptors
during long many-task runs. Override `train.eval_num_workers` explicitly if you
want multiprocessing eval.

The same value can be placed in YAML:

```yaml
train:
  epochs: 1
  debug_max_steps_per_epoch: 2
  eval_num_workers: 0
```

If the CAIDBench Arrow dataset is stored somewhere else, override only the data
root:

```bash
caid-train \
  --config configs/caidbench/finetune_default.yaml \
  --override scenario.data.path=/path/to/caidbench_arrow_root
```

To run the model-appearance order instead of the default order:

```bash
caid-train \
  --config configs/caidbench/finetune_default.yaml \
  --override scenario.protocol=protocols/caidbench/model_appearance_order_protocol.yaml
```
