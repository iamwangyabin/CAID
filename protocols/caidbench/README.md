# CAIDBench Protocols

This directory contains generated CAIDBench continual task protocols.

Use the matching runnable config instead of passing this protocol alone:

```bash
caid-train --config configs/caidbench/finetune_90.yaml
```

`continual_90_protocol.yaml` defines the 90-task order and filters. It also
declares the paired index file in the same directory:

```yaml
index_path: continual_90_index.parquet
```

The dataset root lives in the runnable config:

```yaml
scenario:
  data:
    backend: caidbench
    path: /home/home/yabin/caidbench_datasets_merged
  protocol: protocols/caidbench/continual_90_protocol.yaml
```
