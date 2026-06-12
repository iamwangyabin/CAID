# Incremental CAIDBench Dataset Plan

This document turns the model timeline draft into a CAIDBench dataset assembly
contract. The companion protocol is:

```text
protocols/examples/model_timeline_incremental.yaml
```

## Goal

Build one AID-style Arrow dataset that supports chronological continual
detection tasks across GAN image generation, diffusion image generation,
face-forgery, inpainting, talking-head, and video-generation-frame sources.

The dataset package should stay separate from the task order. Samples are stored
once, while the YAML protocol selects each incremental task by metadata.

## Canonical Metadata Contract

Every caidbench sample should normalize into the existing CAIDBench sidecar
schema:

```text
path,label,split,dataset,domain,generator,manipulation,video_id,frame_idx,scene,task_hint,preprocess_profile
```

Use these conventions for the timeline dataset:

| column | required convention |
|---|---|
| `dataset` | Source package name from the draft, for example `CNNDetection`, `GenImage`, `DF40`, `OpenFake`, `DFLIP-3K`, `ForgeryNet`, `Dolos`, or `AIGIBench`. |
| `task_hint` | Canonical task id from `model_timeline_incremental.yaml`, for example `progan`, `sd15`, or `flux2_dev`. This is the primary selector. |
| `generator` | Human-facing model/checkpoint name, for example `ProGAN`, `SD1.5`, or `FLUX.2-dev`. |
| `manipulation` | Operation family, for example `image_gan`, `image_diffusion`, `face_swap`, `face_reenactment`, `inpainting`, `talking_head`, or `video_generation_frame`. |
| `domain` | Coarse domain/modality, for example `natural_image`, `face`, `inpainting`, `talking_head`, `video_frame`, or `anime`. |
| `split` | Normalize to `train`, `val`, and `test`. If an upstream source only has validation data used for evaluation, copy or map that split to `test` in the caidbench package. |
| `label` | `0` for real, `1` for fake/generated/manipulated. |

The protocol filters on `dataset` and `task_hint`. This avoids relying on
upstream generator naming, which often differs across source packages.

## Source Coverage

Existing CAIDBench packers can start from these source families:

| source family | current support |
|---|---|
| `CNNDetection` / ForenSynths-style folders | `caid-pack-dataset --kind cnn_detection` |
| `GenImage` | `caid-pack-dataset --kind genimage` |
| DeepFakeBench-style face-forgery frames | `caid-pack-dataset --kind deepfakebench` |
| `CDDB` | `caid-pack-dataset --kind cddb` |

The timeline draft also needs source-specific adapters or verified manifests
for:

```text
DF40
ForgeryNet
Dolos
OpenFake
DFLIP-3K
AIGIBench
```

For those sources, the first implementation step should be a scanner that emits
normalized records with the metadata contract above. Once records are normalized,
the existing `write_aid_dataset` path can package them as Arrow plus sidecars.

## Dataset Assembly Workflow

1. Inventory local roots for each source package.
2. For each source, scan image/frame files and produce normalized metadata rows.
3. Set `task_hint` using the canonical ids in the protocol.
4. Normalize source-specific evaluation splits into `test`.
5. Deduplicate exact file duplicates by content hash where a source overlap is
   known or suspected.
6. Write one combined AID-style Arrow directory.
7. Run protocol-count validation before training.

Recommended output layout:

```text
data/datasets/model_timeline_incremental/
  data-*.arrow
  dataset_info.json
  state.json
  mapping.json
  train.json
  val.json
  test.json
  caid_meta.jsonl
  caid_info.json
```

## Quality Gates

Before using the caidbench dataset for experiments, check:

| check | expected result |
|---|---|
| missing labels | zero rows with `label=-1` |
| missing task ids | zero rows with empty `task_hint` |
| task train counts | every protocol task has non-zero train rows unless intentionally held out |
| task test counts | every protocol task has non-zero test rows |
| split leakage | no identical content hash appears in both train and test for the same task unless explicitly allowed |
| source leakage | test-only sources are not also selected by a task's train filter |
| video leakage | frames from the same `video_id` should not cross train/test when evaluating video/frame tasks |

## Example Commands

Pack an existing supported source:

```bash
caid-pack-dataset \
  --kind genimage \
  --root /data/raw/GenImage \
  --out /data/arrow/GenImage \
  --preprocess-profile model_timeline_v1
```

Inspect available subsets:

```bash
caid-inspect-aid --root /data/arrow/GenImage
```

Train with the caidbench timeline protocol after the combined Arrow dataset is
available:

```yaml
scenario:
  data:
    backend: aid_arrow
    path: data/datasets/model_timeline_incremental
    image_column: image
  protocol: protocols/examples/model_timeline_incremental.yaml
```

## Items Requiring Verification

The pasted draft is treated as the desired experimental timeline, not as a
verified availability report. Before final release, verify:

| item | reason |
|---|---|
| `RDDM` | Draft says the exact scene needs checking. |
| `Tiny-Random-Sana` | Looks like a test/synthetic variant; decide whether it belongs in the benchmark. |
| `Anima` | Draft says the concrete model series needs checking. |
| closed/commercial models | Confirm redistribution rights and whether only generated outputs, not model assets, can be packaged. |
| future/fast-moving checkpoints | Confirm dates, naming, and source availability before freezing a benchmark version. |
