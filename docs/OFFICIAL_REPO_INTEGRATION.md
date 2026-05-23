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

Use `method.name=e3`. Replace the default backbone with the paper detector backbone. Use the first AID Arrow task as the baseline training corpus, then one task per new generator. Set `memory_size=1000` unless reproducing a different ablation. Increase `ekfn_layers` to the official transformer depth when matching paper numbers.

## Content-Agnostic Adapter-Based Category-Aware Incremental Learning

Use `method.name=ca_adapter_cail`. Replace the default backbone with ViT/Xception hybrid used in the official code. Keep the adapter bottleneck, token-shuffle schedule, replay memory size, initial-task count, and incremental-task count aligned with the official script arguments.

## HSIC Bottleneck

Use AID Arrow image data and online CLIP extraction. Set `method.name=hsic_bottleneck`, use a YAML transform list with CLIP mean/std normalization, and set `method.detector_cfg.backbone.type=clip_vision`. The default config uses frozen OpenAI CLIP ViT-L/14 through `open_clip_torch`, so install the optional dependencies with `pip install -e .[clip]`. Put generator IDs in AID JSON metadata or `caid_meta.jsonl` if you extend `HSICBottleneckMethod._nuisance_ids`.

The previous offline feature path is no longer a training data interface; use AID Arrow image data for HSIC reproduction.

## SAIDO

Use `method.name=saido`. Provide scene labels in AID JSON metadata or implement a VLLM scene-router that fills this field before training. Use CLIP ViT-L/14 for the visual backbone and LoRA ranks/schedules from the official config.

## CoReD

Use `method.name=cored`. Match the official source-target domain sequence and backbone family. The current CAIDBench module implements the teacher-student KD and representation-preservation objective; optional reinforcement-learning components from the original code can be added as a plugin loss.

## CDDB

Use `method.name=cddb`. Build AID Arrow split sidecars for CDDB-Easy, CDDB-Hard, and CDDB-Long by assigning one task/subset per generative source and using the official easy/hard/long order. Select the official binary loss variant with `binary_loss`.

## DFIL

Use `method.name=dfil`. Pretrain on the first domain, then assign subsequent datasets to later task IDs. Set replay quota so it matches the official central/hard sample budget and keep the supervised-contrastive and KD weights from the official config.

## Prompt2Guard

Use `method.name=prompt2guard` with raw 224x224 images and `pip install -e .[clip]`. The implementation follows the official SliNet path: frozen CLIP ViT-B/16, per-task text and image prompt learners, top-k object-conditioned text prompts, K-Means prototype keys after each task, and `mix_top_mean` inference aggregation. AID Arrow metadata must provide `object_labels` or `topk_object_labels` for every sample when `topk_classes > 0`.

## HDP

Use `method.name=hdp` with `configs/hdp.yaml`. The CAIDBench method keeps the framework training loop and implements the official reserve/preserve mechanics inside method hooks: generate one UAP after each task, store it in `uap_pool`, replay pseudo-forged `real + UAP` samples on later tasks, and apply feature KD on real and pseudo-forged features. The default HDP config is now the raw-image EfficientNet-B4 setup: `tf_efficientnet_b4_ns`, 224x224 inputs, mean/std `[0.5, 0.5, 0.5]`, clamp range `[-1, 1]`, Adam `lr=0.0002`, `weight_decay=0.00001`, StepLR `step_size=10, gamma=0.1`, `epsilon=0.15`, `uap_alpha=0.0001`, and `uap_success_threshold=0.8`.

## SUR-LID

Use `method.name=sur_lid` with `configs/sur_lid.yaml`. The implementation is CAIDBench-native and uses the public SUR-LID code at `beautyremain/SUR-LID@9a7d228e43a97b75250b5bdbdb79cf193e817317` as a behavioral reference, not as copied project structure. It implements the paper mechanisms inside CAIDBench's method hooks: EfficientNet-B4/timm backbone configuration, sparse robust replay with grid-shuffle consistency, per-task heads, distribution re-filling feature augmentation, supervised-contrastive AFI labels, IDA classifier-weight alignment, teacher KD, and feature-distillation loss.

For paper-number reproduction, use official DeepFakeBench/SUR-LID face preprocessing and split files, 256x256 RGB inputs, normalization mean/std `[0.5, 0.5, 0.5]`, Adam `lr=0.0002`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0.0005`, batch size `32`, StepLR `step_size=10, gamma=0.4`, `nEpochs=10`, `num_pic=504`, `mem_each_batch=3`, and the official EfficientNet-B4 checkpoint. The config keeps CAIDBench's existing AID-style transform components and maps the official augmentation knobs onto them: 256 resize, flip `0.5`, blur `[3, 7]`, JPEG quality `[40, 100]`, and brightness/contrast factors `[0.9, 1.1]`. Use `protocols/examples/sur_lid_p1.yaml`, `sur_lid_p2.yaml`, or `sur_lid_p3.yaml` for the paper task orders and adjust only metadata aliases that map to the same official split membership.

For checking reproduction quality, use the existing CAIDBench `auc` and accuracy matrix; AUC is the core signal for SUR-LID paper-number matching.
