# CAIDBench Configs

These configs are complete experiment entry points for the generated CAIDBench
continual protocols. Use them with `caid-train --config ...`.

## CAIDBench 90-Task Protocol

`configs/caidbench/finetune_90.yaml` wires together the pieces required for a
run:

- dataset root: `scenario.data.path`
- task protocol: `scenario.protocol`
- selected metadata index: `index_path` inside the protocol YAML

Example:

```bash
caid-train --config configs/caidbench/finetune_90.yaml
```

For local smoke runs without SwanLab:

```bash
caid-train \
  --config configs/caidbench/finetune_90.yaml \
  --override logging.backend=none
```

If the CAIDBench Arrow dataset is stored somewhere else, override only the data
root:

```bash
caid-train \
  --config configs/caidbench/finetune_90.yaml \
  --override scenario.data.path=/path/to/caidbench_arrow_root
```
