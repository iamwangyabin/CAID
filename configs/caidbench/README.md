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
