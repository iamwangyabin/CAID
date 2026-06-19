# Protocols

Protocols define continual task order, per-task metadata filters, and any
paired protocol index file. They do not define where the dataset root lives,
which transform to use, or which method to train.

A runnable experiment needs all three layers:

- `scenario.data`: dataset backend, root path, and column names
- `scenario.protocol`: task order, filters, and protocol-local index path
- `method` / `train`: algorithm and optimization settings

For generated CAIDBench protocols, prefer the ready-to-run configs under
`configs/caidbench/` instead of passing a protocol path by itself.

Example:

```bash
python train.py --config configs/caidbench/finetune.yaml
```

The protocol file used by that config is:

```yaml
scenario:
  protocol: protocols/caidbench/default_protocol.yaml
```

The matching dataset root is in the same config:

```yaml
scenario:
  data:
    backend: caidbench
    path: /home/home/yabin/CAIDBench
```

The selected index is declared by the protocol file itself:

```yaml
index_path: continual_index.parquet
tasks:
  ...
```
