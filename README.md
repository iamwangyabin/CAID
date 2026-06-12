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
pip install -e ".[clip]"       # official OpenAI CLIP plus CLIP/OpenCLIP backbones
pip install -e ".[sprompts]"   # S-Prompts/OpenCLIP/timm paths
pip install -e ".[hub]"        # remote Hugging Face / ModelScope mirrors
pip install -e ".[dev]"        # tests
```

## Quick Start

Pack a processed image dataset into AID-style Arrow:

```bash
caid-pack-dataset \
  --kind deepfakebench \
  --root /data/caid_processed/deepfakebench_faces \
  --out /data/caid_arrow/deepfakebench_faces \
  --preprocess-profile sur_lid_deepfakebench_v1
```

Train a method:

```bash
caid-train --config configs/reproduce/dfil.yaml
```

Run the generated CAIDBench continual protocol:

```bash
caid-train --config configs/caidbench/finetune_default.yaml
```

Protocol files only define task order and filters. Dataset paths live in
`scenario.data` inside the training config; see `configs/caidbench/README.md` and
`protocols/README.md`.

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

A minimal AID-style metadata row contains:

```text
path,label,split,dataset,domain,generator,manipulation,video_id,frame_idx,scene,task_hint,preprocess_profile
/path/img1.jpg,0,train,cddb,progan,progan,unknown,,-1,object,task0,
/path/img2.jpg,1,train,cddb,progan,progan,unknown,,-1,object,task0,
```

`label=0` means real and `label=1` means fake. Training data is read from an
AID-style Arrow directory; labels, splits, subsets, and task metadata are
reconstructed from `mapping.json`, split JSON files, and optional
`caid_meta.jsonl`. Continual `task_id` values are derived from the YAML protocol
at load time, not stored as required Arrow sidecar columns.

## Data Interfaces

CAIDBench supports AID-style Arrow sources under `scenario.data`:

```yaml
scenario:
  data:
    backend: aid_arrow
    image_column: image
    remote:
      platform: modelscope
      repo_ids:
        modelscope: yabinnng/CDDB.arrow
      local_dir: data/datasets/CDDB.arrow
  protocol: protocols/examples/cddb_hard_arrow.yaml
  transform:
    trsf:
      - _target_: caidbench.data.transforms.SquareResize
        size: 224
      - _target_: caidbench.data.transforms.ToTensor
      - _target_: caidbench.data.transforms.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
```

Inspect existing AID subsets before writing a protocol:

```bash
caid-inspect-aid --root /path/to/AID_arrow_dataset
```

Remote Hugging Face / ModelScope mirrors are configured under
`scenario.data.remote`; see `configs/_base.yaml`.

## Methods

Implemented method keys:

```text
finetune, cddb, e3, ca_adapter_cail, hsic_bottleneck, saido,
cored, dfil, prompt2guard, sprompts, ranpac, layup, pina,
pina_d, cp_prompt, duct, soyo, loranpac, dce, hdp, sur_lid
```

These modules are framework implementations of the paper mechanisms, not direct
copies of official repositories. Exact reproduction still requires matching the
official split, backbone, preprocessing, schedule, and checkpoint choices.

## Repository Layout

```text
caidbench/
  cli/          command line entry points
  data/         AID Arrow loading, protocols, packers
  engine/       continual training loop and outputs
  evaluation/   accuracy, AUC, AA/AF metrics
  losses/       KD, contrastive, HSIC, alignment losses
  memory/       replay buffer and exemplar selectors
  models/       backbones, adapters, LoRA, EKFN, detector heads
  methods/      continual method implementations
configs/        shared base, CAIDBench entry points, and reproduction configs
  caidbench/    generated CAIDBench protocol experiment entry points
  reproduce/    paper/original-result reproduction configs per method
protocols/      task protocol YAMLs; the current slim set keeps CDDB-Hard Arrow
docs/           compatibility and method-mapping notes
tests/          smoke and packaging tests
```

## More Detail

- `docs/METHOD_MAPPING.md`: method-to-paper mapping and implementation notes.
- `docs/METHOD_PUBLICATIONS.md`: paper titles, venues, years, and source links for method keys.
- `docs/HF_ARROW_COMPAT_PROTOCOL.md`: AID Arrow sidecar schema and YAML protocol interface.
- `docs/OFFICIAL_REPO_INTEGRATION.md`: reproduction and official-code integration checklist.
