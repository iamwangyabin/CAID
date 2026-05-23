# Official-repository integration plan

This file records what must be matched when CAIDBench is used to reproduce paper numbers rather than run framework smoke tests.

## Global reproduction controls

For every paper-method run, record these fields in the YAML config or run metadata:

- Dataset split file and task order.
- Backbone name, initialization, frozen/trainable layers, and checkpoint path.
- Image preprocessing or feature-extraction recipe.
- Optimizer, learning rate schedule, batch size, epochs, memory size, and random seed.
- Whether evaluation is binary real/fake, generator-incremental, domain-incremental, or OOD.
- Metrics: per-task accuracy/AUC, Average Accuracy, Average Forgetting, and final joint/OOD metrics when required.

## E3

Use `method.name=e3`. Replace the default backbone with the paper detector backbone. Use the first manifest task as the baseline training corpus, then one task per new generator. Set `memory_size=1000` unless reproducing a different ablation. Increase `ekfn_layers` to the official transformer depth when matching paper numbers.

## Content-Agnostic Adapter-Based Category-Aware Incremental Learning

Use `method.name=ca_adapter_cail`. Replace the default backbone with ViT/Xception hybrid used in the official code. Keep the adapter bottleneck, token-shuffle schedule, replay memory size, initial-task count, and incremental-task count aligned with the official script arguments.

## HSIC Bottleneck

Use raw image manifests and online CLIP extraction. Set `method.name=hsic_bottleneck`, use a YAML transform list with CLIP mean/std normalization, and set `method.detector_cfg.backbone.type=clip_vision`. The default config uses frozen OpenAI CLIP ViT-L/14 through `open_clip_torch`, so install the optional dependencies with `pip install -e .[clip]`. Put generator IDs in the manifest `generator` column and caption-alignment/domain IDs in custom columns if you extend `HSICBottleneckMethod._nuisance_ids`.

The previous offline feature path is now only a compatibility fallback: use `backbone.type=identity` only when intentionally benchmarking saved `.npy` tensors, not for the default HSIC reproduction route.

## SAIDO

Use `method.name=saido`. Provide scene labels in the manifest `scene` column or implement a VLLM scene-router that fills this column before training. Use CLIP ViT-L/14 for the visual backbone and LoRA ranks/schedules from the official config.

## CoReD

Use `method.name=cored`. Match the official source-target domain sequence and backbone family. The current CAIDBench module implements the teacher-student KD and representation-preservation objective; optional reinforcement-learning components from the original code can be added as a plugin loss.

## CDDB

Use `method.name=cddb`. Build manifests for CDDB-Easy, CDDB-Hard, and CDDB-Long by assigning one `task_id` per generative source and using the official easy/hard/long order. Select the official binary loss variant with `binary_loss`.

## DFIL

Use `method.name=dfil`. Pretrain on the first domain, then assign subsequent datasets to later task IDs. Set replay quota so it matches the official central/hard sample budget and keep the supervised-contrastive and KD weights from the official config.

## Prompt2Guard

Use `method.name=prompt2guard` with raw 224x224 images and `pip install -e .[clip]`. The implementation follows the official SliNet path: frozen CLIP ViT-B/16, per-task text and image prompt learners, top-k object-conditioned text prompts, K-Means prototype keys after each task, and `mix_top_mean` inference aggregation. Manifests/Arrow metadata must provide `object_labels` or `topk_object_labels` for every sample when `topk_classes > 0`.

## HDP

Use `method.name=hdp`. Set `uap_shape` to the input tensor shape and ensure UAP persistence/checkpointing is enabled. For raw-image reproduction, clamp perturbations in pixel-normalized space consistently with the official training config.

## SUR-LID

Use `method.name=sur_lid`. Match DeepfakeBench preprocessing/backbones and replace the generic sparse-uniform score with the official magnitude/angularity/shuffle consistency score if exact paper-number reproduction is required.
