# Method mapping and reproduction checklist

This document maps each CAIDBench method to the paper-level mechanics implemented in code.

## E3

- Baseline detector is trained before incremental updates.
- For each incoming generator/task, a detector copy is fine-tuned on new fake samples plus real memory samples.
- The classifier is discarded; the embedder is frozen and appended to the expert bank.
- The memory buffer is rebalanced across seen generators.
- The Expert Knowledge Fusion Network consumes the sequence of expert embeddings and is trained from memory.

## Content-Agnostic Adapter CAIL

- Adapter-equipped detector with dual original/shuffled views.
- Token/grid shuffling consistency encourages content-agnostic artifact features.
- Asymmetric alignment: real features are compacted more strongly, fake generator features are allowed to remain diverse while domain shift is regularized.
- Multi-view KD preserves logits and pairwise feature geometry from the previous model.

## HSIC Bottleneck + HGR

- CE for real/fake detection.
- HSIC dependence with real/fake labels can be rewarded while HSIC dependence with nuisance variables such as generator ID, task ID, or caption alignment is penalized.
- HGR keeps compact per-class replay memory using a hybrid score: HSIC relevance plus k-center coverage.

## SAIDO

- Scene router maps each sample to a scene.
- Each scene owns an independent LoRA-style expert branch over a shared backbone.
- IDOM estimates historical importance and transforms gradients for core neurons while allowing non-core neurons to update freely.

## CoReD

- The previous model is kept as teacher.
- Student loss includes current-task CE, logit KD, and feature/representation KD.

## DFIL

- CE plus supervised contrastive learning for domain-invariant forgery representations.
- Label-level and feature-level KD from previous model.
- Replay memory uses central and hard samples.

## Prompt2Guard

- Frozen CLIP ViT-B/16 backbone with one PromptLearner per incremental task.
- Each PromptLearner owns official-style learnable text prompt tokens and visual prompt tokens; only the current task prompt learner is trainable.
- Per-sample top-k object labels condition the real/fake text prompts.
- After each task, K-Means image-feature prototypes are stored for real, fake, and all-image task routing.
- Inference concatenates all seen task prompts and uses prototype-weighted `top1`, `mean`, or `mix_top_mean` aggregation.

## S-Prompts

- Each incoming domain/task gets an independently trained prompt/head pair.
- Historical prompt/head pairs are frozen after their task.
- After each task, K-Means centers over task training features are stored as domain keys.
- Inference routes each sample to the nearest seen task center before applying the corresponding prompt/head.
- Official-style CAIDBench mode inserts learned visual prompt tokens into a frozen ViT/CLIP image transformer, matching S-iPrompts/S-liPrompts mechanics. Feature manifests still use a compatibility fallback because token prompts require raw images.

## HDP

- After each task, a targeted UAP is generated from current real samples and appended to a persistent UAP pool.
- During later tasks, a sampled historical UAP is added to current real samples to create pseudo-forged samples with fake labels.
- Feature-wise KD preserves real and pseudo-forged distributions from the previous detector; optional logit KD follows the official implementation's extra KL-style regularizer.

## SUR-LID

- Sparse Uniform Replay selects stable, high-dimensionally uniform samples.
- Distribution re-filling performs latent mixup around replay centroids.
- Isolation loss separates real/fake and task/domain distributions.
- Incremental decision alignment keeps task-specific binary heads aligned.
