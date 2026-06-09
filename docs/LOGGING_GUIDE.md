# Logging Guide

This repository uses two logging layers:

1. Event logs for human-readable progress.
2. Metric logs for scalar training and evaluation data.

## Rules

- Keep event logs short and stable.
- Use a fixed bracketed component prefix for event logs.
- Log lifecycle boundaries, state transitions, and expensive post-task work.
- Keep metric logs structured and scalar-only.
- Include task, epoch, step, progress, ETA, learning rate, loss, and accuracy when available.
- Do not mix narrative text and raw metric payloads in the same line.
- Do not print every batch unless debugging is explicitly enabled.

## Recommended cadence

- Run start: once per run.
- Task start and task end: once per task.
- Epoch start and epoch end: once per epoch.
- In-epoch progress: at a fixed interval, plus the first and last batch.
- Long post-task work: first batch, every 50 batches, and the last batch.
- Evaluation summary: once after each task, across all already seen tasks.

## Output shape

Event log example:

```text
[Component] phase=... status=... key=value ...
```

Metric log example:

```text
train task=... epoch=... step=... progress=... eta=... lr=... loss=... acc=...
```

## Backend behavior

- Console output is for readable progress.
- Experiment tracking receives structured scalar payloads.
- Non-scalar tensors, arrays, and blobs should be excluded from metric logs unless they are stored as artifacts.
