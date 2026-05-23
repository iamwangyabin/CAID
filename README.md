# CAIDBench

CAIDBench is a benchmark framework for continual AI-generated image detection
and continual deepfake/face-forgery detection. It keeps dataset protocols,
training loops, metrics, replay buffers, backbones, and method-specific logic in
separate modules so different continual detectors can run under one evaluator.

## Install

```bash
cd CAID
pip install -e .
```

Optional extras:

```bash
pip install -e ".[clip]"       # CLIP / open_clip backbones
pip install -e ".[arrow,hub]"  # AID Arrow datasets and remote mirrors
pip install -e ".[dev]"        # tests
```

## Quick Start

Create a manifest from `root/domain/split/{real,fake}/*.jpg`:

```bash
caid-make-manifest --root /path/to/dataset --out data/manifest.csv
```

Train a method:

```bash
caid-train --config configs/dfil.yaml
```

Training logs to SwanLab by default. Log in once before cloud runs:

```bash
swanlab login -k <api-key>
```

You can override the SwanLab project or mode in any config:

```yaml
logging:
  backend: swanlab
  project: CAIDBench
  mode: cloud
```

For local smoke tests or runs without experiment tracking, set
`logging.backend=none`.

A minimal manifest contains:

```text
path,label,split,task_id,domain,generator,scene
/path/img1.jpg,0,train,0,progan,progan,object
/path/img2.jpg,1,train,0,progan,progan,object
```

`label=0` means real and `label=1` means fake. Image paths are the default data
path; `.npy`, `.pt`, and `.pth` features remain available for explicit
feature-only configs.

## Data Interfaces

CAIDBench supports manifest and AID Arrow sources under `scenario.data`:

```yaml
scenario:
  data:
    backend: manifest
    path: data/manifest.csv
    root: /path/to/images
```

or an AID-style Arrow dataset:

```yaml
scenario:
  data:
    backend: aid_arrow
    path: /data/caid_arrow/deepfakebench_faces
    image_column: image
  protocol: protocols/examples/sur_lid_p3.yaml
  transform:
    size: 224
    preset: imagenet
```

Pack a processed dataset into AID-style Arrow:

```bash
caid-pack-dataset \
  --kind deepfakebench \
  --root /data/caid_processed/deepfakebench_faces \
  --out /data/caid_arrow/deepfakebench_faces \
  --preprocess-profile sur_lid_deepfakebench_v1
```

Inspect existing AID subsets before writing a protocol:

```bash
caid-inspect-aid --root /path/to/AID_arrow_dataset
```

Remote Hugging Face / ModelScope mirrors are configured under
`scenario.data.remote`; see `configs/cddb_arrow_remote.yaml`.

## Methods

Implemented method keys:

```text
finetune, cddb, e3, ca_adapter_cail, hsic_bottleneck, saido,
cored, dfil, prompt2guard, sprompts, hdp, sur_lid
```

These modules are framework implementations of the paper mechanisms, not direct
copies of official repositories. Exact reproduction still requires matching the
official split, backbone, preprocessing, schedule, and checkpoint choices.

## Repository Layout

```text
caidbench/
  cli/          command line entry points
  data/         manifests, AID Arrow loading, protocols, packers
  engine/       continual training loop and outputs
  evaluation/   accuracy, AUC, AA/AF metrics
  losses/       KD, contrastive, HSIC, alignment losses
  memory/       replay buffer and exemplar selectors
  models/       backbones, adapters, LoRA, EKFN, detector heads
  methods/      continual method implementations
configs/        method and dataset configs
protocols/      YAML protocol examples
docs/           compatibility and method-mapping notes
tests/          smoke and packaging tests
```

## More Detail

- `docs/METHOD_MAPPING.md`: method-to-paper mapping and implementation notes.
- `docs/HF_ARROW_COMPAT_PROTOCOL.md`: AID Arrow sidecar schema and YAML protocol interface.
- `docs/OFFICIAL_REPO_INTEGRATION.md`: reproduction and official-code integration checklist.
