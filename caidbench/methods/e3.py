from __future__ import annotations

import copy
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from ..data.loader import build_dataloader
from ..models.ekfn import ExpertKnowledgeFusionNetwork
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, freeze_module, iter_limited_train_batches


def _local_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return y.long().clamp_min(0).clamp_max(max(int(num_classes) - 1, 0))


@register_method("e3")
class E3Method(ContinualMethod):
    """Official-structure E3 continual detector.

    This implementation follows the released E3 control flow rather than the
    earlier CAIDBench approximation:
      1. train one baseline detector on the first task
      2. fine-tune one expert detector per later task, always starting from the
         frozen baseline detector weights
      3. rebuild and train the official EKFN over all seen expert embedders
    """

    def __init__(
        self,
        memory_size: int = 1000,
        train_dataset_limit_per_class: int = 500,
        train_dataset_limit_real: int = 500,
        baseline_epochs: int | None = None,
        expert_epochs: int | None = None,
        ekfn_epochs: int | None = None,
        baseline_lr: float = 1.0e-4,
        ft_lr: float = 5.0e-5,
        cls_lr: float = 2.5e-4,
        weight_decay: float = 0.01,
        baseline_lr_step_size: int = 3,
        baseline_lr_gamma: float = 0.75,
        lr_step_size: int = 100000,
        lr_decay_rate: float = 0.8,
        loss_weights: Iterable[float] | None = None,
        ekfn_layers: int = 5,
        ekfn_heads: int = 8,
        ekfn_train_classifier_head_only: bool = True,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.memory_size = int(memory_size)
        self.train_dataset_limit_per_class = int(train_dataset_limit_per_class)
        self.train_dataset_limit_real = int(train_dataset_limit_real)
        self.baseline_epochs = baseline_epochs
        self.expert_epochs = None if expert_epochs is None else int(expert_epochs)
        self.ekfn_epochs = None if ekfn_epochs is None else int(ekfn_epochs)
        self.baseline_lr = float(baseline_lr)
        self.ft_lr = float(ft_lr)
        self.cls_lr = float(cls_lr)
        self.weight_decay = float(weight_decay)
        self.baseline_lr_step_size = int(baseline_lr_step_size)
        self.baseline_lr_gamma = float(baseline_lr_gamma)
        self.lr_step_size = int(lr_step_size)
        self.lr_decay_rate = float(lr_decay_rate)
        self.ekfn_layers = int(ekfn_layers)
        self.ekfn_heads = int(ekfn_heads)
        self.ekfn_train_classifier_head_only = bool(ekfn_train_classifier_head_only)
        weights = [1.0, 1.0] if loss_weights is None else [float(x) for x in loss_weights]
        if len(weights) != self.num_classes:
            raise ValueError(f"E3 loss_weights length must equal num_classes={self.num_classes}, got {weights!r}")
        self.loss_weights = tuple(weights)
        self.seed = int(seed if seed is not None else torch.initial_seed())
        self.memory_rows: list[int] = []
        self.experts = nn.ModuleList()
        self.ekfn: ExpertKnowledgeFusionNetwork | None = None

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        if self.ekfn is None or len(self.experts) <= 1:
            return self.detector(x)
        embeddings = self._expert_embeddings(x)
        logits = self.ekfn(embeddings)
        return {"logits": logits, "features": embeddings.mean(dim=1)}

    def classification_loss(self, out: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self._bce_loss(out["logits"], y)

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        task_index = self._find_task_index(trainer, task)
        expert_epochs = int(self.expert_epochs or trainer.max_epochs or 1)
        ekfn_epochs = int(self.ekfn_epochs or trainer.max_epochs or 1)
        if task_index == 0:
            baseline_epochs = int(self.baseline_epochs or trainer.max_epochs or expert_epochs)
            baseline_loader = self._build_dataset_pair_loader(trainer, task_index, split="train", shuffle=True, limit_fake=False)
            baseline_val_loader = self._build_dataset_pair_loader(trainer, task_index, split="val", shuffle=False, limit_fake=False)
            self._train_detector_phase(
                trainer=trainer,
                model=self.detector,
                loader=baseline_loader or train_loader,
                val_loader=baseline_val_loader or val_loader,
                epochs=baseline_epochs,
                lr=self.baseline_lr,
                weight_decay=self.weight_decay,
                step_size=self.baseline_lr_step_size,
                gamma=self.baseline_lr_gamma,
                phase_name="e3_baseline",
                task_name=str(task.name),
                task_index=task_index,
            )
            self.experts.append(freeze_module(copy.deepcopy(self.detector.backbone).to(self.device)))
            self._refresh_memory(trainer, task_index)
            return True

        expert_loader = self._build_dataset_pair_loader(trainer, task_index, split="train", shuffle=True, limit_fake=True)
        expert_val_loader = self._build_dataset_pair_loader(trainer, task_index, split="val", shuffle=False, limit_fake=False)
        expert_detector = copy.deepcopy(self.detector).to(self.device)
        self._train_detector_phase(
            trainer=trainer,
            model=expert_detector,
            loader=expert_loader,
            val_loader=expert_val_loader,
            epochs=expert_epochs,
            lr=self.ft_lr,
            weight_decay=self.weight_decay,
            step_size=self.lr_step_size,
            gamma=self.lr_decay_rate,
            phase_name="e3_expert",
            task_name=str(task.name),
            task_index=task_index,
        )
        self.experts.append(freeze_module(copy.deepcopy(expert_detector.backbone).to(self.device)))

        self._refresh_memory(trainer, task_index, split="train")
        self._rebuild_ekfn()
        ekfn_loader = self._build_memory_loader(trainer, task_index, shuffle=True)
        ekfn_val_loader = self._build_seen_loader(trainer, task_index, split="val", shuffle=False)
        self._train_ekfn_phase(
            trainer=trainer,
            loader=ekfn_loader,
            val_loader=ekfn_val_loader,
            epochs=ekfn_epochs,
            lr=self.cls_lr,
            weight_decay=self.weight_decay,
            step_size=self.lr_step_size,
            gamma=self.lr_decay_rate,
            task_name=str(task.name),
            task_index=task_index,
        )
        return True

    def _find_task_index(self, trainer: Any, task: Any) -> int:
        for index, spec in enumerate(getattr(trainer.scenario, "tasks", [])):
            if spec is task:
                return index
            if getattr(spec, "task_id", None) == getattr(task, "task_id", None) and getattr(spec, "name", None) == getattr(task, "name", None):
                return index
        raise KeyError(f"Could not resolve task index for {task!r}")

    def _loss_weight_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.loss_weights, dtype=torch.float32, device=device).view(1, -1)

    def _bce_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        target = F.one_hot(_local_targets(y, self.num_classes), num_classes=self.num_classes).float()
        return F.binary_cross_entropy_with_logits(logits, target, weight=self._loss_weight_tensor(logits.device))

    @torch.no_grad()
    def _accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        target = _local_targets(y, self.num_classes).to(logits.device)
        pred = logits.argmax(dim=1)
        return float((pred == target).float().mean().item())

    def _build_optimizer(self, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float) -> torch.optim.Optimizer:
        trainable = [p for p in params if p.requires_grad]
        if not trainable:
            raise RuntimeError("E3 phase has no trainable parameters.")
        return torch.optim.AdamW(trainable, lr=float(lr), weight_decay=float(weight_decay))

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, step_size: int, gamma: float) -> torch.optim.lr_scheduler.StepLR:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(int(step_size), 1), gamma=float(gamma))

    def _train_detector_phase(
        self,
        *,
        trainer: Any,
        model: nn.Module,
        loader: Any,
        val_loader: Any | None,
        epochs: int,
        lr: float,
        weight_decay: float,
        step_size: int,
        gamma: float,
        phase_name: str,
        task_name: str,
        task_index: int,
    ) -> None:
        model.to(self.device)
        optimizer = self._build_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = self._build_scheduler(optimizer, step_size=step_size, gamma=gamma)
        best_state: dict[str, Any] | None = None
        best_val_loss = float("inf")

        for epoch in range(max(int(epochs), 1)):
            model.train()
            total_loss = 0.0
            total_acc = 0.0
            total_batches = 0
            for _batch_idx, batch in iter_limited_train_batches(trainer, loader):
                batch = batch_to_device(batch, self.device)
                out = model(batch["x"])
                loss = self._bce_loss(out["logits"], batch["y"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
                optimizer.step()
                trainer.advance_step()
                total_loss += float(loss.detach().cpu())
                total_acc += self._accuracy(out["logits"].detach(), batch["y"])
                total_batches += 1
            scheduler.step()

            avg_loss = total_loss / max(total_batches, 1)
            avg_acc = total_acc / max(total_batches, 1)
            val_loss = float("nan")
            val_acc = float("nan")
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_detector_phase(model, val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
            trainer.log_train_metrics(
                {
                    f"{phase_name}_loss": avg_loss,
                    f"{phase_name}_acc": avg_acc,
                    f"{phase_name}_val_loss": val_loss,
                    f"{phase_name}_val_acc": val_acc,
                },
                task_index=task_index,
                task_name=task_name,
                epoch=epoch + 1,
                epochs=max(int(epochs), 1),
                optimizer=optimizer,
                phase=phase_name,
            )
        if best_state is not None:
            model.load_state_dict(best_state)

    @torch.no_grad()
    def _evaluate_detector_phase(self, model: nn.Module, loader: Any) -> tuple[float, float]:
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            out = model(batch["x"])
            total_loss += float(self._bce_loss(out["logits"], batch["y"]).detach().cpu())
            total_acc += self._accuracy(out["logits"].detach(), batch["y"])
            total_batches += 1
        if was_training:
            model.train()
        if total_batches == 0:
            return float("nan"), float("nan")
        return total_loss / total_batches, total_acc / total_batches

    def _rebuild_ekfn(self) -> None:
        self.ekfn = ExpertKnowledgeFusionNetwork(
            embed_dim=int(self.detector.feature_dim),
            num_experts=len(self.experts),
            num_classes=self.num_classes,
            transformer_layers=self.ekfn_layers,
            nhead=self.ekfn_heads,
        ).to(self.device)

    @torch.no_grad()
    def _expert_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.experts) == 0:
            raise RuntimeError("E3 expert embedding requested before baseline expert was created.")
        embeddings = []
        for expert in self.experts:
            expert.eval()
            embeddings.append(expert(x).detach().unsqueeze(1))
        return torch.cat(embeddings, dim=1)

    def _train_ekfn_phase(
        self,
        *,
        trainer: Any,
        loader: Any,
        val_loader: Any | None,
        epochs: int,
        lr: float,
        weight_decay: float,
        step_size: int,
        gamma: float,
        task_name: str,
        task_index: int,
    ) -> None:
        if self.ekfn is None:
            raise RuntimeError("EKFN must be built before training.")
        if loader is None:
            raise RuntimeError("EKFN training loader is empty.")
        self.ekfn.to(self.device)
        trainable_params = (
            list(self.ekfn.trainable_head.parameters()) if self.ekfn_train_classifier_head_only else list(self.ekfn.parameters())
        )
        optimizer = self._build_optimizer(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = self._build_scheduler(optimizer, step_size=step_size, gamma=gamma)
        best_state: dict[str, Any] | None = None
        best_val_loss = float("inf")

        for epoch in range(max(int(epochs), 1)):
            self.ekfn.train()
            total_loss = 0.0
            total_acc = 0.0
            total_batches = 0
            for _batch_idx, batch in iter_limited_train_batches(trainer, loader):
                batch = batch_to_device(batch, self.device)
                embeddings = self._expert_embeddings(batch["x"])
                logits = self.ekfn(embeddings)
                loss = self._bce_loss(logits, batch["y"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable_params, trainer.grad_clip)
                optimizer.step()
                trainer.advance_step()
                total_loss += float(loss.detach().cpu())
                total_acc += self._accuracy(logits.detach(), batch["y"])
                total_batches += 1
            scheduler.step()

            avg_loss = total_loss / max(total_batches, 1)
            avg_acc = total_acc / max(total_batches, 1)
            val_loss = float("nan")
            val_acc = float("nan")
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_ekfn_phase(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.ekfn.state_dict())
            trainer.log_train_metrics(
                {
                    "e3_ekfn_loss": avg_loss,
                    "e3_ekfn_acc": avg_acc,
                    "e3_ekfn_val_loss": val_loss,
                    "e3_ekfn_val_acc": val_acc,
                },
                task_index=task_index,
                task_name=task_name,
                epoch=epoch + 1,
                epochs=max(int(epochs), 1),
                optimizer=optimizer,
                phase="e3_ekfn",
            )
        if best_state is not None:
            self.ekfn.load_state_dict(best_state)

    @torch.no_grad()
    def _evaluate_ekfn_phase(self, loader: Any) -> tuple[float, float]:
        if self.ekfn is None:
            return float("nan"), float("nan")
        was_training = self.ekfn.training
        self.ekfn.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            embeddings = self._expert_embeddings(batch["x"])
            logits = self.ekfn(embeddings)
            total_loss += float(self._bce_loss(logits, batch["y"]).detach().cpu())
            total_acc += self._accuracy(logits.detach(), batch["y"])
            total_batches += 1
        if was_training:
            self.ekfn.train()
        if total_batches == 0:
            return float("nan"), float("nan")
        return total_loss / total_batches, total_acc / total_batches

    @staticmethod
    def _take_prefix(indices: list[int], limit: int) -> list[int]:
        if limit <= 0 or len(indices) <= limit:
            return list(indices)
        return list(indices[: int(limit)])

    @staticmethod
    def _ordered_unique(indices: Iterable[int]) -> list[int]:
        seen: set[int] = set()
        ordered: list[int] = []
        for index in indices:
            row = int(index)
            if row in seen:
                continue
            seen.add(row)
            ordered.append(row)
        return ordered

    def _refresh_memory(self, trainer: Any, task_index: int, split: str = "train") -> None:
        half_capacity = max(int(self.memory_size) // 2, 0)
        if half_capacity <= 0:
            self.memory_rows = []
            return
        seen_tasks = task_index + 1
        real_limit = half_capacity if self.train_dataset_limit_real <= 0 else min(half_capacity, self.train_dataset_limit_real)
        fake_budget = half_capacity
        if self.train_dataset_limit_per_class > 0:
            fake_budget = min(fake_budget, self.train_dataset_limit_per_class * seen_tasks)
        fake_quota = fake_budget // max(seen_tasks, 1)
        rows: list[int] = []
        rows.extend(self._take_prefix(self._global_real_indices(trainer, split), real_limit))
        for seen_index in range(seen_tasks):
            rows.extend(self._take_prefix(self._task_fake_indices(trainer, seen_index, split), fake_quota))
        self.memory_rows = self._ordered_unique(rows)

    def _split_indices(self, trainer: Any, task_index: int, split: str) -> list[int]:
        return list(getattr(trainer.scenario, "_split_indices", {}).get((task_index, split), []))

    def _filter_label(self, trainer: Any, indices: Iterable[int], label: int) -> list[int]:
        base = self._ordered_unique(indices)
        if not base:
            return []
        df = trainer.scenario.df.iloc[base]
        return [int(i) for i, y in zip(base, df["label"].tolist()) if int(y) == int(label)]

    def _global_real_indices(self, trainer: Any, split: str) -> list[int]:
        rows: list[int] = []
        for task_index in range(len(getattr(trainer.scenario, "tasks", []))):
            rows.extend(self._split_indices(trainer, task_index, split))
        return self._filter_label(trainer, rows, label=0)

    def _task_fake_indices(self, trainer: Any, task_index: int, split: str) -> list[int]:
        return self._filter_label(trainer, self._split_indices(trainer, task_index, split), label=1)

    def _make_loader(
        self,
        trainer: Any,
        indices: Iterable[int],
        *,
        split: str,
        shuffle: bool,
        task_id: int,
        task_name: str,
    ) -> Any | None:
        rows = self._ordered_unique(indices)
        if not rows:
            return None
        dataset = trainer.scenario.source.make_dataset(
            rows,
            transform_cfg=trainer.scenario._transform_for_split(split),
            task_id=task_id,
            task_name=task_name,
        )
        return build_dataloader(
            dataset,
            batch_size=int(getattr(trainer, "batch_size", 32)),
            shuffle=bool(shuffle),
            num_workers=int(getattr(trainer, "num_workers", 0)),
            drop_last=False,
        )

    def _build_memory_loader(self, trainer: Any, task_index: int, shuffle: bool) -> Any | None:
        del task_index
        if not self.memory_rows:
            return None
        return self._make_loader(
            trainer,
            self.memory_rows,
            split="train",
            shuffle=shuffle,
            task_id=-1,
            task_name="e3_fixed_memory",
        )

    def _build_dataset_pair_loader(self, trainer: Any, task_index: int, split: str, shuffle: bool, limit_fake: bool) -> Any | None:
        real_rows = self._global_real_indices(trainer, split)
        if split == "train":
            real_rows = self._take_prefix(real_rows, self.train_dataset_limit_real)
        fake_rows = self._task_fake_indices(trainer, task_index, split)
        if split == "train" and limit_fake:
            fake_rows = self._take_prefix(fake_rows, self.train_dataset_limit_per_class)
        task_name = f"e3_pair_task{task_index}_{split}"
        return self._make_loader(
            trainer,
            [*real_rows, *fake_rows],
            split=split,
            shuffle=shuffle,
            task_id=int(getattr(trainer.scenario.tasks[task_index], "task_id", task_index)),
            task_name=task_name,
        )

    def _build_seen_loader(self, trainer: Any, task_index: int, split: str, shuffle: bool) -> Any | None:
        rows: list[int] = []
        rows.extend(self._global_real_indices(trainer, split))
        for seen_index in range(task_index + 1):
            rows.extend(self._task_fake_indices(trainer, seen_index, split))
        return self._make_loader(
            trainer,
            rows,
            split=split,
            shuffle=shuffle,
            task_id=-1,
            task_name=f"e3_seen_until_{task_index}_{split}",
        )

    def _build_task_loader(self, trainer: Any, task_index: int, split: str, shuffle: bool) -> Any | None:
        indices = self._split_indices(trainer, task_index, split)
        if not indices:
            return None
        task = trainer.scenario.tasks[task_index]
        return self._make_loader(
            trainer,
            indices,
            split=split,
            shuffle=shuffle,
            task_id=int(getattr(task, "task_id", task_index)),
            task_name=str(getattr(task, "name", f"task{task_index}")),
        )

    def _build_expert_loader(self, trainer: Any, task_index: int, shuffle: bool) -> Any:
        train_indices = self._split_indices(trainer, task_index, "train")
        fake_indices = self._take_prefix(self._filter_label(trainer, train_indices, label=1), self.train_dataset_limit_per_class)
        real_indices = self._take_prefix(self._filter_label(trainer, train_indices, label=0), self.train_dataset_limit_real)
        task = trainer.scenario.tasks[task_index]
        loader = self._make_loader(
            trainer,
            [*fake_indices, *real_indices],
            split="train",
            shuffle=shuffle,
            task_id=int(getattr(task, "task_id", task_index)),
            task_name=str(getattr(task, "name", f"task{task_index}")),
        )
        if loader is None:
            raise RuntimeError(f"E3 expert loader for task_index={task_index} is empty.")
        return loader

    def _build_ekfn_loader(self, trainer: Any, task_index: int, split: str, shuffle: bool) -> Any | None:
        if split == "train":
            return self._build_memory_loader(trainer, task_index, shuffle=shuffle)
        return self._build_seen_loader(trainer, task_index, split=split, shuffle=shuffle)
