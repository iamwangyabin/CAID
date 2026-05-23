# Official-repository integration plan

This file records what must be matched when CAIDBench is used to reproduce paper numbers rather than run framework smoke tests.

## Global reproduction controls

For every paper-method run, record these fields in the YAML config or run metadata:

- Dataset split file and task order.
- Backbone name, initialization, frozen/trainable layers, and checkpoint path.
- Image preprocessing and online backbone configuration.
- Optimizer, learning rate schedule, batch size, epochs, memory size, and random seed.
- Whether evaluation is binary real/fake, generator-incremental, domain-incremental, or OOD.
- Metrics: per-task accuracy/AUC, Average Accuracy, Average Forgetting, and final joint/OOD metrics when required.

## E3

Official reference: `ArefAz/E3-Ensemble-of-Expert-Embedders-CVPRWMF24@9ed8357f9ee5c891dcf028992b7e0d6e469aa0f6`; key files are `models/expert_classifier.py`, `models/mixture_transformer.py`, `models/transformer.py`, and `models/classifier_head.py`.

Use `method.name=e3`. Replace the default backbone with the paper detector backbone. Use the first AID Arrow task as the baseline training corpus, then one task per new generator. Set `memory_size=1000`, rebalance replay by generator, use 200-D expert embeddings, and set EKFN to transformer depth 5, 8 heads, hidden size 64, and dropout 0.5 unless reproducing a different ablation.

## Content-Agnostic Adapter-Based Category-Aware Incremental Learning

Official reference: `theShuai-t/AI-image-incremental-detection@1ee8b4e604e2128dadb849a04b11f7efd3363a3b`; key files are `models/vit_xception_center.py`, `convs/vit_adapter_c.py`, `trainer.py`, and `configs/vit_xception_center.json`.

Use `method.name=ca_adapter_cail`. Replace the default backbone with the ViT/Xception hybrid used in the official code. In CAIDBench's framework-compatible path, freeze the base backbone, train only adapter/head parameters, use replay memory, and keep adapter bottleneck, token-shuffle schedule, replay memory size, initial-task count, and incremental-task count aligned with the official script arguments.

## HSIC Bottleneck

No official repository could be confirmed. The closest public reference is the ICLR 2026 OpenReview submission `msLnKDvhBx`.

Use AID Arrow image data and online CLIP extraction. Set `method.name=hsic_bottleneck`, use a YAML transform list with CLIP mean/std normalization, and set `method.detector_cfg.backbone.type=clip_vision`. The default config uses frozen OpenAI CLIP ViT-L/14 through `open_clip_torch`, so install the optional dependencies with `pip install -e .[clip]`. Put generator IDs in AID JSON metadata or `caid_meta.jsonl` if you extend `HSICBottleneckMethod._nuisance_ids`. The HGR selector combines label-HSIC relevance with k-center coverage and subtracts nuisance alignment when generator/domain IDs are available.

The previous offline feature path is no longer a training data interface; use AID Arrow image data for HSIC reproduction.

## SAIDO

No official repository could be confirmed. The closest public reference is arXiv `2512.00539`.

Use `method.name=saido`. Provide scene labels in AID JSON metadata or implement a VLLM scene-router that fills this field before training. Use CLIP ViT-L/14 for the visual backbone and LoRA ranks/schedules from the paper config. For the scene-aware contrastive term, CAIDBench accepts precomputed `scene_prompt_features` or `scene_text_features` in the batch; those should come from the same scene/content/common prompt construction used by SAIDO's VLLM preprocessing.

## CoReD

Official reference: `alsgkals2/CoReD@596667aab870adabdc4784fc2cca9c58c30d26c9`; key files are `CoReD.py`, `Function_CoReD.py`, `Function_common.py`, and `KD.py`.

Use `method.name=cored`. Match the official source-target domain sequence and backbone family. The CAIDBench module implements the teacher-student KD and the official confidence-bin representation-memory objective: before each target task, the frozen teacher builds real/fake representation bins from correctly predicted target samples, and the student aligns its current batch feature means to those bins.

## CDDB

Use `method.name=cddb`. Build AID Arrow split sidecars for CDDB-Easy, CDDB-Hard, and CDDB-Long by assigning one task/subset per generative source and using the official easy/hard/long order. Select the official binary loss variant with `binary_loss`.

## DFIL

Official reference: `DeepFakeIL/DFIL@f1188db3e3472084faa00255b1850b27d6e76468`; key files are `DFIL/train_CNN_SupCon_and_CE.py`, `DFIL/TaskN_KD.py`, `DFIL/create_memory.py`, `DFIL/get_image_info.py`, and `DFIL/SupConLoss.py`.

Use `method.name=dfil`. Pretrain on the first domain, then assign subsequent datasets to later task IDs. Set replay quota so it matches the official central/hard sample budget: low distance-to-class-mean center samples plus high-entropy hard samples per class. Keep supervised-contrastive, KD, feature-distillation, optimizer, and StepLR weights from the official config.

## Prompt2Guard

Official reference: `laitifranz/Prompt2Guard@a89890203294d4f18051a3052b3a57f2e8d28d80`; key files are `src/methods/prompt2guard.py`, `src/models/slinet.py`, `src/models/clip/prompt_learner.py`, and `src/eval.py`.

Use `method.name=prompt2guard` with raw 224x224 images and `pip install -e .[clip]`. The implementation follows the official SliNet path: frozen CLIP ViT-B/16, per-task text and image prompt learners, top-k object-conditioned text prompts, K-Means prototype keys after each task, and `mix_top_mean` inference aggregation. AID Arrow metadata must provide `object_labels` or `topk_object_labels` for every sample when `topk_classes > 0`.

## S-Prompts

Official reference: `iamwangyabin/S-Prompts@a26e822ae1b139d7bc7663740ea5a951b12be250`; key files are `methods/sprompt.py`, `models/sinet.py`, `models/slinet.py`, and `configs/cddb_slip.json`.

Use `method.name=sprompts` with `implementation=official`. CAIDBench inserts visual prompt tokens into a frozen ViT/OpenCLIP image transformer, freezes old prompt/head pairs after each task, stores K-Means task centers, and routes inference to the nearest center. For paper-number runs, use the official train/test preprocessing and the first-task versus later-task optimizer schedule.

## HDP

Official reference: `skJack/HDP@abe7fcf83c960e08606f4426e56952b426a7bcfd`; key files are `train.py`, `models/image.py`, `configs/train_hdp.yaml`, and `datasets/transforms.py`.

Use `method.name=hdp` with `configs/hdp.yaml`. The CAIDBench method keeps the framework training loop and implements the official reserve/preserve mechanics inside method hooks: generate one UAP after each task, store it in `uap_pool`, replay pseudo-forged `real + UAP` samples on later tasks, and apply feature KD on real and pseudo-forged features. The default HDP config is now the raw-image EfficientNet-B4 setup: `tf_efficientnet_b4_ns`, 224x224 inputs, mean/std `[0.5, 0.5, 0.5]`, single-logit sigmoid/BCELoss (`binary_sigmoid=true` with `detector_cfg.num_classes=1`), clamp range `[-1, 1]`, Adam `lr=0.0002`, `weight_decay=0.00001`, StepLR `step_size=10, gamma=0.1`, `epsilon=0.15`, `uap_alpha=0.0001`, and `uap_success_threshold=0.8`.

## SUR-LID

Use `method.name=sur_lid` with `configs/sur_lid.yaml`. The implementation is CAIDBench-native and uses the public SUR-LID code at `beautyremain/SUR-LID@9a7d228e43a97b75250b5bdbdb79cf193e817317` as a behavioral reference, not as copied project structure. It implements the paper mechanisms inside CAIDBench's method hooks: EfficientNet-B4/timm backbone configuration, sparse robust replay with grid-shuffle consistency, per-task heads, distribution re-filling feature augmentation, supervised-contrastive AFI labels, IDA classifier-weight alignment, teacher KD, and feature-distillation loss.

For paper-number reproduction, use official DeepFakeBench/SUR-LID face preprocessing and split files, 256x256 RGB inputs, normalization mean/std `[0.5, 0.5, 0.5]`, Adam `lr=0.0002`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0.0005`, batch size `32`, StepLR `step_size=10, gamma=0.4`, `nEpochs=10`, `num_pic=504`, `mem_each_batch=3`, and the official EfficientNet-B4 checkpoint. The config keeps CAIDBench's existing AID-style transform components and maps the official augmentation knobs onto them: 256 resize, flip `0.5`, blur `[3, 7]`, JPEG quality `[40, 100]`, and brightness/contrast factors `[0.9, 1.1]`. Use `protocols/examples/sur_lid_p1.yaml`, `sur_lid_p2.yaml`, or `sur_lid_p3.yaml` for the paper task orders and adjust only metadata aliases that map to the same official split membership.

For checking reproduction quality, use the existing CAIDBench `auc` and accuracy matrix; AUC is the core signal for SUR-LID paper-number matching.
