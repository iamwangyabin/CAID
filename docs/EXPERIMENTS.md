# Experiment Plan

This document is the experiment ledger for CAIDBench comparison runs. Keep it
focused on runnable commands, exact settings, and post-run results.

Do not write API keys or other credentials into this file. Set credentials in
the shell or the job runner before launching an experiment.

## Shared Setup

Use this shell preamble before every run:

```bash
cd /gemini/code/CAID
export SWANLAB_API_KEY="<set-this-outside-git>"
```

Before launching a formal run:

- Confirm the target machine has the latest committed code.
- Confirm optional dependencies for the target method are installed.
- Confirm `/gemini/data-1/CAIDBench` is readable.
- Record the code commit hash in the post-run fields.

## Result Fields

For every completed experiment, fill:

- Code commit:
- SwanLab URL:
- Output directory:
- Start time:
- End time:
- Exit status:
- Final average accuracy:
- Final average AUC:
- Final average AP:
- Final average F1:
- Final average forgetting:
- Base task metrics:
- Worst tasks:
- Notes:

## Main Baseline

### E001 - Finetune CLIP-L

Status: planned

Purpose: sequential fine-tuning baseline with a frozen CLIP ViT-L/14 feature
backbone and a trainable binary classifier head. This is not full RINE because
it uses the final CLIP image embedding rather than intermediate encoder-block
features.

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
      --config configs/caidbench/finetune.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/finetune_clip_l \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=finetune-clip-l \
        device=auto \
        distributed.backend=nccl \
        method.detector_cfg.backbone.type=clip_vision \
        method.detector_cfg.backbone.backend=open_clip \
        method.detector_cfg.backbone.model_name=ViT-L-14 \
        method.detector_cfg.backbone.pretrained=openai \
        method.detector_cfg.backbone.freeze=true \
        method.detector_cfg.backbone.out_dim=768 \
        method.detector_cfg.backbone.normalize=true \
        train.lr_scheduler=cosine \
        train.optimizer.type=adamw \
        train.optimizer.lr=1.0e-3 \
        train.optimizer.weight_decay=1.0e-4 \
        train.epochs=1 \
        train.batch_size=64 \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

Post-run fields:

- Code commit:
- SwanLab URL:
- Output directory: `/gemini/output/caidbench/finetune_clip_l`
- Exit status:
- Final average accuracy:
- Final average AUC:
- Final average AP:
- Final average F1:
- Final average forgetting:
- Base task metrics:
- Notes:

## Comparison Methods

Only `finetune` and `sprompts` are currently allowed by the trainer's DDP guard.
Run the other comparison methods as single-process jobs unless their method code
is updated for distributed training.

### E002 - MLSB / RINE-style intermediate CLIP features

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/caidbench/mlsb.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/mlsb_clip_l \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=mlsb-clip-l \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E003 - S-Prompts ViT-B-16, 1 epoch

Status: planned

Purpose: S-liPrompts comparison with its default OpenCLIP ViT-B/16 backbone.
This path includes the language side through learned CLIP text prompts for
`real` and `fake`.

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
      --config configs/caidbench/sprompts.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/sprompts_vit_b16_1epoch \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=sprompts-vit-b16-1epoch \
        device=auto \
        distributed.backend=nccl \
        method.backbone.type=open_clip \
        method.backbone.model_name=ViT-B-16 \
        method.backbone.pretrained=openai \
        method.train_backbone=false \
        method.normalize_features=true \
        method.use_official_schedule=true \
        method.init_epoch=1 \
        method.epochs=1 \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E004 - CP-Prompt

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/caidbench/cp_prompt.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/cp_prompt \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=cp-prompt \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E005 - RanPAC

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/caidbench/ranpac.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/ranpac \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=ranpac \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E006 - LoRanPAC

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/caidbench/loranpac.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/loranpac \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=loranpac \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E007 - LayUP

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/caidbench/layup.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/layup \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=layup \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E008 - DCE

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/caidbench/dce.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/dce \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=dce \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E009 - DUCT

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/caidbench/duct.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/duct \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=duct \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E010 - PINA

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/caidbench/pina.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/pina \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=pina \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E011 - SOYO

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/caidbench/soyo.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/soyo \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=soyo \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E012 - Prompt2Guard

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/caidbench/prompt2guard.yaml \
      --override \
        scenario.data.path=/gemini/data-1/CAIDBench \
        output_dir=/gemini/output/caidbench/prompt2guard \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=prompt2guard \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

## Reproduce-config Comparisons

These configs do not have `configs/caidbench/*.yaml` entry points yet. The
commands below explicitly override the dataset backend onto the CAIDBench
protocol. Run them after verifying that the method's data assumptions still
match the generated CAIDBench task order.

### E013 - HSIC Bottleneck

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/reproduce/hsic_bottleneck.yaml \
      --override \
        scenario.data.backend=caidbench \
        scenario.data.path=/gemini/data-1/CAIDBench \
        scenario.data.image_column=image \
        scenario.data.label_column=label \
        scenario.data.generator_column=generator_name \
        scenario.data.dataset_column=source_dataset \
        scenario.data.source_path_column=source_path \
        scenario.data.split_column=split \
        scenario.data.task_hint_mode=dir \
        scenario.data.domain_from=dir_name \
        scenario.data.remote=null \
        scenario.protocol=protocols/caidbench/default_protocol.yaml \
        output_dir=/gemini/output/caidbench/hsic_bottleneck \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=hsic-bottleneck \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

### E014 - DFIL

Status: planned

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/reproduce/dfil.yaml \
      --override \
        scenario.data.backend=caidbench \
        scenario.data.path=/gemini/data-1/CAIDBench \
        scenario.data.image_column=image \
        scenario.data.label_column=label \
        scenario.data.generator_column=generator_name \
        scenario.data.dataset_column=source_dataset \
        scenario.data.source_path_column=source_path \
        scenario.data.split_column=split \
        scenario.data.task_hint_mode=dir \
        scenario.data.domain_from=dir_name \
        scenario.data.remote=null \
        scenario.protocol=protocols/caidbench/default_protocol.yaml \
        output_dir=/gemini/output/caidbench/dfil \
        logging.backend=swanlab \
        logging.project=CAIDBench \
        logging.mode=cloud \
        logging.name=dfil \
        device=auto \
        train.num_workers=16 \
        train.eval_num_workers=8 \
        train.pin_memory=true \
        train.persistent_workers=true \
        train.prefetch_factor=4
```

## Completed Runs

Move finished experiments here after filling their post-run fields.
