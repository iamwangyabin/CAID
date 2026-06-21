# CAIDBench Protocols

This directory contains generated CAIDBench continual task protocols.

Use the matching runnable config instead of passing this protocol alone:

```bash
python train.py --config configs/caidbench/finetune.yaml
```

`default_protocol.yaml` defines the default task order and filters. It also
declares the paired index file in the same directory:

```yaml
index_path: continual_index.parquet
```

`model_appearance_order_protocol.yaml` uses the same index but orders tasks by
model appearance chronology. Select it by overriding `scenario.protocol`.
`model_appearance_order_protocol_50.yaml` is the 50-task subset of that order.

The dataset root lives in the runnable config:

```yaml
scenario:
  data:
    backend: caidbench
    path: /home/home/yabin/CAIDBench
  protocol: protocols/caidbench/default_protocol.yaml
```
