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

## RanPAC

Official reference: `McDonnell-Research-Lab/RanPAC`; key files are `RanPAC.py`, `inc_net.py`, `trainer.py`, and `args/*_publish.csv`.

Use `method.name=ranpac` with `configs/ranpac.yaml`. CAIDBench follows the official CDDB full RanPAC row (`ID=7` in `args/cddb_publish.csv`): first-task AdaptFormer-style adapter tuning on ViT-B/16 IN21k with the official CosineLinear head for 20 epochs, SGD with `body_lr=head_lr=0.01`, `batch_size=48`, `weight_decay=0.0005`, `min_lr=0.0`, then frozen-feature random projection and cumulative ridge with `M=5000`, no input normalization, and the official ridge grid `1e-8...1e8`. The detector keeps native 768-D ViT features (`detector_cfg.backbone.out_dim=null` so the base config's 512-D projection is disabled).

The first-task adapter tuning uses the official training transform `RandomResizedCrop(224, scale=(0.05, 1.0))`, horizontal flip, `ToTensor`, and AdaptFormer adapter dropout `0.1`. Ridge statistics are collected from the same train split with the test transform `Resize(256)+CenterCrop(224)+ToTensor`, matching the official `train_loader_for_CPs` with `mode="test"`. For the no-PETL Phase-2 ablation (`ID=4`: `model_name=ncm`, `batch_size=128`, `tuned_epoch=0`), use `configs/ranpac_no_petl.yaml`.

The config deliberately uses `protocols/examples/cddb_hard_arrow.yaml`, the same task-wise CDDB-Hard evaluation shape used by the known-good CAIDBench/S-Prompts pipeline: GauGAN test=2000, BigGAN test=800, WildDeepfake test=2063, WhichFaceIsReal test=400, SAN test=83 in the current Arrow mirror. Do not replace each task's `test` split with the complete hard validation set, because that turns the continual matrix into repeated joint evaluation and produces `test=5346` for every task. If a weighted overall hard-validation number is needed, compute it as a separate post-hoc report rather than changing the task tests.

CAIDBench keeps labels binary and uses modulo-safe training targets rather than materializing domain-duplicated class IDs.

## LayUP

Official reference: `ky-ah/LayUP`; key files are `src/layup.py`, `src/modules/intra_layer.py`, `src/modules/ridge.py`, and `src/data/datasets/instances.py`.

Use `method.name=layup` with `configs/layup.yaml`. CAIDBench registers hooks on the last `k=6` ViT blocks when the detector backbone is a timm ViT, concatenates CLS activations, accumulates LayUP's ridge statistics, and solves the shared closed-form classifier. The config records the official CDDB-Hard defaults: `batch_size=48`, `finetune_epochs=20`, `lr=0.003`, `weight_decay=0.0005`, `early_stopping=5`, train `RandomResizedCrop(224, scale=(0.7, 1.0))`, horizontal flip, ColorJitter `0.1`, and test `Resize(224)+CenterCrop(224)`. If no hook-compatible ViT is configured, the method falls back to the detector's final feature vector so smoke checks still exercise the ridge path.

## PINA / PINA-D

Official reference: `qwangcv/PINA`; key files are `methods/pina.py`, `models/pina_vit.py`, `models/pina_clip.py`, `utils/pss.py`, and `configs/cddb_pina_{vit,clip}.yaml`.

Use `method.name=pina` or `method.name=pina_d` with `configs/pina.yaml`. The CAIDBench adaptation preserves the official UC/DSA/PSS contract inside the framework: a frozen feature extractor, one domain adapter per task, a unified classifier trained on the base task then frozen, K-Means task keys, and routed inference. `pina` maps to shallow feature adapters, while `pina_d` maps to the deeper adapter path. The config records the official CDDB-Hard `init_epoch=50`, later `epochs=50`, `lr=0.001`, `weight_decay=0.0002`, `batch_size=128`, `prompt_length=10`, and CDDB image transform. Token-level ViT prompt insertion is not duplicated in this compact CAIDBench module; use the official repo for exact token placement ablations.

## CP-Prompt

Official reference: `dannis97500/CP_Prompt`; key files are `models/clip_prefix_one_prompt_tuning/model.py`, `net.py`, `prompt.py`, `prompt_learner.py`, and `configs/prefix_one_prompt/cddb.json`.

Use `method.name=cp_prompt` with `configs/cp_prompt.yaml`. CAIDBench implements CP-Prompt as a feature-space common prompt plus per-domain personalized prompt, routed by task centers, with frozen CLIP/ViT features. The config records the official CDDB-Hard values from `configs/prefix_one_prompt/cddb.json`: `knn_k=5`, `share_prompt_length=6`, `prefix_prompt_length=10`, `prefix_prompt_layers=[3,5,6,7,8]`, `epochs=50`, `lr/lrate=0.01`, `batch_size=128`, and first-task/later-task weight decay. This keeps the official composition, per-domain prompt isolation, and center-based prompt selection behavior. The official Prefix-One K/V attention injection is structurally invasive to CLIP transformer internals and remains the reference path for exact paper-number reproduction.

## DUCT

Official reference: `Estrella-fugaz/CVPR25-Duct`; key files are `methods/duct.py`, `models/vit_inc.py`, `models/linears.py`, `utils/toolkit.py`, and `configs/Template_CDDB_duct.json`.

Use `method.name=duct` with `configs/duct.yaml`. CAIDBench follows DUCT's two consolidation stages in the existing detector abstraction: train on the current domain, merge backbone task vectors into the initial PTM state with `merge_scalar`, then retrain the head on the merged representation. The config records the official CDDB-Hard `epochs=15`, `lrate=0.1`, `weight_decay=0.0005`, `merge_scalar=0.5`, `head_merge_ratio=0.5`, `lr_re=0.001`, `epc_re=5`, and CDDB image transform. Because CAIDBench uses a shared binary label space, classifier OT transport over domain-duplicated heads is approximated by head interpolation via `head_merge_ratio`.

## SOYO

Official reference: `QWangCV/SOYO`; key files are `methods/soyo.py`, `models/soyo_vit.py`, `models/soyo_clip.py`, `utils/soyo_utils.py`, and `configs/cddb_soyo_{vit,clip}.yaml`.

Use `method.name=soyo` with `configs/soyo.yaml`. CAIDBench implements per-domain parameter isolation through adapter/classifier pools, compresses each domain's frozen features with a `GaussianMixture`, trains a linear SOYO selector from current real features plus old GMM-resampled features, and routes inference to the selected domain parameters. Defaults match the official CDDB DIC settings: `K=2`, `soyo_epoch=30`, `soyo_lr=0.1`, `lr=0.001`, `weight_decay=0.0002`, `batch_size=128`, `prompt_length=10`, and five CDDB-Hard sessions.

## LoRanPAC

Official reference: `liangzu/loranpac`; key files are `models/tsvd.py`, `utils/inc_net.py`, `models/ranpac.py`, `trainer.py`, and `exps/tsvd/cddb.json`.

Use `method.name=loranpac` with `configs/loranpac.yaml`. CAIDBench follows the official low-rank random-feature solver: random features `H = relu(F @ RE)`, cumulative `cov_HY`, an incremental truncated SVD summary of `H^T`, and on-the-fly ridge weights `U @ ((U.T @ cov_HY) / (Sigma^2 + ridge))`. The default config records the official CDDB values `tuned_epoch=20`, `batch_size=48`, `init_lr=0.01`, `weight_decay=0.0005`, no input normalization, train `RandomResizedCrop(224, scale=(0.05, 1.0))`, `E=100000`, `rank=20000`, `truncate_percent=25`, `tsvd_batch_size=1000`, and `ridge=0`; reduce these for CPU smoke runs.

## DCE

Official reference: `Lain810/DCE`; key files are `methods/dce.py`, `models/DceNet.py`, `methods/base.py`, `utils/data.py`, and `configs/cddb.json`.

Use `method.name=dce` with `configs/dce.yaml`. CAIDBench implements the DCE expert group as three per-domain expert heads: naive CE, balanced softmax, and reverse/few-shot CE. The config records the official imbalanced-CDDB defaults: `init_epoch=20`, later `epochs=20`, `lr/lrate=0.01`, `init_weight_decay=0.0005`, later `weight_decay=0.0002`, `batch_size=128`, `prompt_length=10`, and seven sessions. After each task it stores feature mean/covariance statistics, samples balanced synthetic features, trains a dynamic selector, and fuses all historical expert logits at inference. The official code's CDDB-Hard imbalance table differs from the paper appendix; reproduce official-code numbers by following `utils/data.py::make_imb()` and recording that choice in experiment metadata.

## Prompt2Guard

Official reference: `laitifranz/Prompt2Guard@a89890203294d4f18051a3052b3a57f2e8d28d80`; key files are `src/methods/prompt2guard.py`, `src/models/slinet.py`, `src/models/clip/prompt_learner.py`, and `src/eval.py`.

Use `method.name=prompt2guard` with raw 224x224 images and `pip install -e .[clip]`. The implementation follows the official SliNet path: frozen OpenAI CLIP `ViT-B/16`, per-task text and image prompt learners, top-k object-conditioned text prompts, K-Means prototype keys after each task, official prototype-derived task probability weighting, and `mix_top_mean` inference aggregation. AID Arrow metadata must provide `object_labels` or `topk_object_labels` for every sample when `topk_classes > 0`.

The default `configs/prompt2guard.yaml` reads the official object-label sidecar from `data/sidecars/prompt2guard/classes.pkl`. Download it before running Prompt2Guard:

```bash
mkdir -p data/sidecars/prompt2guard
curl -L \
  -o data/sidecars/prompt2guard/classes.pkl \
  https://www.modelscope.cn/models/yabinnng/CAID/resolve/master/classes.pkl
```

## S-Prompts

Official reference: `iamwangyabin/S-Prompts@a26e822ae1b139d7bc7663740ea5a951b12be250`; key files are `methods/sprompt.py`, `models/sinet.py`, `models/slinet.py`, and `configs/cddb_slip.json`.

Use `method.name=sprompts` with `implementation=official`. CAIDBench inserts visual prompt tokens into a frozen ViT/OpenCLIP image transformer, freezes old prompt/head pairs after each task, stores K-Means task centers, and routes inference to the nearest center. For paper-number runs, use the official train/test preprocessing and the first-task versus later-task optimizer schedule.

## HDP

Official reference: `skJack/HDP@abe7fcf83c960e08606f4426e56952b426a7bcfd`; key files are `train.py`, `models/image.py`, `configs/train_hdp.yaml`, and `datasets/transforms.py`.

Use `method.name=hdp` with `configs/hdp.yaml`. The CAIDBench method keeps the framework training loop and implements the official reserve/preserve mechanics inside method hooks: generate one UAP after each task, store it in `uap_pool`, replay pseudo-forged `real + UAP` samples on later tasks, and apply feature KD on real and pseudo-forged features. The default HDP config is now the raw-image EfficientNet-B4 setup: `tf_efficientnet_b4_ns`, 224x224 inputs, mean/std `[0.5, 0.5, 0.5]`, single-logit sigmoid/BCELoss (`binary_sigmoid=true` with `detector_cfg.num_classes=1`), clamp range `[-1, 1]`, Adam `lr=0.0002`, `weight_decay=0.00001`, StepLR `step_size=10, gamma=0.1`, `epsilon=0.15`, `uap_alpha=0.0001`, and `uap_success_threshold=0.8`.

The default `configs/hdp.yaml` sets `detector_cfg.backbone.pretrained=true`, so the first run needs network access or a pre-populated timm/Hugging Face cache for `tf_efficientnet_b4_ns`. For offline smoke runs, override the backbone to `pretrained=false` or point the method at a local checkpoint.

## SUR-LID

Use `method.name=sur_lid` with `configs/sur_lid.yaml`. The implementation is CAIDBench-native and uses the public SUR-LID code at `beautyremain/SUR-LID@9a7d228e43a97b75250b5bdbdb79cf193e817317` as a behavioral reference, not as copied project structure. It implements the paper mechanisms inside CAIDBench's method hooks: EfficientNet-B4/timm backbone configuration, sparse robust replay with grid-shuffle consistency, per-task heads, distribution re-filling feature augmentation, supervised-contrastive AFI labels, IDA classifier-weight alignment, teacher KD, and feature-distillation loss.

For paper-number reproduction, use official DeepFakeBench/SUR-LID face preprocessing and split files, 256x256 RGB inputs, normalization mean/std `[0.5, 0.5, 0.5]`, Adam `lr=0.0002`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0.0005`, batch size `32`, StepLR `step_size=10, gamma=0.4`, `nEpochs=10`, `num_pic=504`, `mem_each_batch=3`, and the official EfficientNet-B4 checkpoint. The slim config set currently ships only `protocols/examples/cddb_hard_arrow.yaml`; add the SUR-LID P1/P2/P3 protocol YAMLs back when reproducing SUR-LID paper task orders.

`configs/sur_lid.yaml` requires the official EfficientNet-B4 checkpoint by default. Place it at `./training/pretrained/efficientnet-b4-6ed6700e.pth`, update `method.detector_cfg.backbone.pretrained_path`, or set `method.require_backbone_checkpoint=false` for non-paper smoke checks.

For checking reproduction quality, use the existing CAIDBench `auc` and accuracy matrix; AUC is the core signal for SUR-LID paper-number matching.
