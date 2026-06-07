from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..losses import feature_distillation_loss, kd_loss
from ..registry import register_method
from .base import ContinualMethod, batch_to_device


@register_method("hdp")
class HDPMethod(ContinualMethod):
    """Historical Distribution Preserving continual face forgery detector.

    This follows the original HDP reserve/preserve implementation:
      - after each task, generate a targeted UAP from real samples that pushes
        the current detector toward the fake class and append it to a UAP pool;
      - while learning later tasks, replay pseudo-forged samples by adding a
        sampled historical UAP to current real images;
      - preserve real and pseudo-forged distributions with feature KD, with
        optional logit KD matching the official implementation's extra KL term.
    """

    def __init__(
        self,
        epsilon: float = 0.15,
        uap_shape: list[int] | tuple[int, ...] | None = None,
        uap_alpha: float = 1e-4,
        uap_iters: int = 5,
        uap_success_threshold: float = 0.8,
        uap_max_steps_per_batch: int = 100,
        uap_init_scale: float = 1e-3,
        feature_kd_weight: float = 1.0,
        logit_kd_weight: float | None = None,
        kd_weight: float | None = None,
        real_kd_weight: float = 1.0,
        adv_kd_weight: float = 1.0,
        temperature: float = 2.0,
        normalize_feature_kd: bool = False,
        clamp_inputs: bool = True,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
        real_label: int = 0,
        fake_label: int = 1,
        binary_sigmoid: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self.uap_alpha = float(uap_alpha)
        self.uap_iters = int(uap_iters)
        self.uap_success_threshold = float(uap_success_threshold)
        self.uap_max_steps_per_batch = int(uap_max_steps_per_batch)
        self.uap_init_scale = float(uap_init_scale)
        self.feature_kd_weight = float(feature_kd_weight)
        if logit_kd_weight is None:
            logit_kd_weight = 1.0 if kd_weight is None else float(kd_weight)
        self.logit_kd_weight = float(logit_kd_weight)
        self.real_kd_weight = float(real_kd_weight)
        self.adv_kd_weight = float(adv_kd_weight)
        self.temperature = float(temperature)
        self.normalize_feature_kd = bool(normalize_feature_kd)
        self.clamp_inputs = bool(clamp_inputs)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.real_label = int(real_label)
        self.fake_label = int(fake_label)
        self.binary_sigmoid = bool(binary_sigmoid)
        self.teacher: nn.Module | None = None
        self.uap_shape = tuple(int(x) for x in uap_shape) if uap_shape is not None else None
        self.register_buffer("uap_pool", torch.empty(0), persistent=True)
        if uap_shape is not None:
            self._init_empty_uap_pool(tuple(int(x) for x in uap_shape))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # The number of stored UAPs grows with the task sequence, so the first
        # dimension is intentionally dynamic across checkpoints.
        key = prefix + "uap_pool"
        if key in state_dict:
            self.uap_pool = state_dict[key].detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def uap_count(self) -> int:
        return int(self.uap_pool.shape[0]) if self.uap_pool.ndim > 1 else 0

    def _init_empty_uap_pool(self, sample_shape: tuple[int, ...]) -> None:
        # Store UAPs with an explicit singleton batch dimension, matching the
        # official implementation's [1, C, H, W] "seal" tensor.
        shape = (0, 1, *sample_shape) if len(sample_shape) >= 1 else (0, 1)
        self.uap_pool = torch.empty(shape, device=self.uap_pool.device)

    def _new_uap(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, *tuple(x.shape[1:]))
        return torch.rand(shape, device=x.device, dtype=x.dtype) * self.uap_init_scale

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        super().before_task(task, train_loader)
        if self.uap_pool.ndim <= 1 and train_loader is not None:
            try:
                first = next(iter(train_loader))
                self._init_empty_uap_pool(tuple(first["x"].shape[1:]))
            except StopIteration:
                pass
        self.uap_pool = self.uap_pool.to(self.device)

    def _sample_uap(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.uap_count == 0:
            return None
        idx = torch.randint(self.uap_count, (1,), device=self.uap_pool.device).item()
        return self.uap_pool[idx].to(device=x.device, dtype=x.dtype)

    def _append_uap(self, uap: torch.Tensor) -> None:
        uap = uap.detach().to(self.device)
        if uap.ndim == 0:
            uap = uap.reshape(1)
        if uap.shape[0] != 1:
            uap = uap.unsqueeze(0)
        new_entry = uap.unsqueeze(0)
        if self.uap_pool.ndim <= 1 or self.uap_pool.numel() == 0:
            if self.uap_pool.ndim > 1 and self.uap_pool.shape[1:] == uap.shape:
                self.uap_pool = torch.cat([self.uap_pool.to(device=uap.device, dtype=uap.dtype), new_entry], dim=0)
            else:
                self.uap_pool = new_entry
            return
        if self.uap_pool.shape[1:] != uap.shape:
            raise ValueError(
                f"Generated UAP shape {tuple(uap.shape)} does not match existing pool shape {tuple(self.uap_pool.shape[1:])}"
            )
        self.uap_pool = torch.cat([self.uap_pool.to(device=uap.device, dtype=uap.dtype), new_entry], dim=0)

    def _add_uap(self, x: torch.Tensor, uap: torch.Tensor) -> torch.Tensor:
        out = x + uap.to(device=x.device, dtype=x.dtype)
        if self.clamp_inputs:
            out = out.clamp(self.clamp_min, self.clamp_max)
        return out

    def _zero_like_loss(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(0.0)

    def _fake_logit(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits[:, self.fake_label] - logits[:, self.real_label]

    def _two_class_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.binary_sigmoid:
            return logits
        fake = self._fake_logit(logits)
        return torch.stack([-fake, fake], dim=-1)

    def _classification_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.binary_sigmoid:
            return F.cross_entropy(logits, y.long())
        target = y.eq(self.fake_label).to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(self._fake_logit(logits), target)

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        out = self.detector(x)
        if self.binary_sigmoid:
            out = dict(out)
            out["logits"] = self._two_class_logits(out["logits"])
        return out

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        x = batch["x"]
        y = batch["y"].long()
        train_x = x
        train_y = y
        real_idx = torch.nonzero(y == self.real_label, as_tuple=False).flatten()
        pseudo_idx: torch.Tensor | None = None
        if self.teacher is not None and real_idx.numel() > 0:
            uap = self._sample_uap(x)
            if uap is not None:
                pseudo_x = self._add_uap(x[real_idx], uap)
                pseudo_y = torch.full((pseudo_x.shape[0],), self.fake_label, device=y.device, dtype=y.dtype)
                pseudo_start = train_x.shape[0]
                train_x = torch.cat([train_x, pseudo_x], dim=0)
                train_y = torch.cat([train_y, pseudo_y], dim=0)
                pseudo_idx = torch.arange(pseudo_start, train_x.shape[0], device=y.device)

        out = self.detector(train_x)
        ce = self._classification_loss(out["logits"], train_y)
        loss = ce
        log: dict[str, torch.Tensor] = {"ce": ce.detach(), "uap_count": ce.new_tensor(float(self.uap_count))}
        if self.teacher is not None and pseudo_idx is not None and pseudo_idx.numel() > 0:
            with torch.no_grad():
                teacher_out = self.teacher(train_x)
            features = out.get("features")
            teacher_features = teacher_out.get("features")
            real_feature_kd = self._zero_like_loss(loss)
            pseudo_feature_kd = self._zero_like_loss(loss)
            if features is not None and teacher_features is not None:
                if real_idx.numel() > 0:
                    real_feature_kd = feature_distillation_loss(
                        features[real_idx],
                        teacher_features[real_idx],
                        normalize=self.normalize_feature_kd,
                    )
                pseudo_feature_kd = feature_distillation_loss(
                    features[pseudo_idx],
                    teacher_features[pseudo_idx],
                    normalize=self.normalize_feature_kd,
                )

            real_logit_kd = (
                kd_loss(self._two_class_logits(out["logits"])[real_idx], self._two_class_logits(teacher_out["logits"])[real_idx], self.temperature)
                if real_idx.numel() > 0
                else self._zero_like_loss(loss)
            )
            pseudo_logit_kd = kd_loss(self._two_class_logits(out["logits"])[pseudo_idx], self._two_class_logits(teacher_out["logits"])[pseudo_idx], self.temperature)
            real_kd = self.feature_kd_weight * real_feature_kd + self.logit_kd_weight * real_logit_kd
            pseudo_kd = self.feature_kd_weight * pseudo_feature_kd + self.logit_kd_weight * pseudo_logit_kd
            loss = loss + self.real_kd_weight * real_kd + self.adv_kd_weight * pseudo_kd
            log.update(
                {
                    "real_feature_kd": real_feature_kd.detach(),
                    "pseudo_feature_kd": pseudo_feature_kd.detach(),
                    "real_logit_kd": real_logit_kd.detach(),
                    "pseudo_logit_kd": pseudo_logit_kd.detach(),
                    "pseudo_count": loss.new_tensor(float(pseudo_idx.numel())),
                }
            )
        log["loss"] = loss
        return log

    @staticmethod
    def _configure_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler | None:
        name = train_cfg.get("lr_scheduler")
        if name is None:
            return None
        if str(name).lower() != "step":
            raise NotImplementedError(f"HDP supports lr_scheduler=step, got {name!r}")
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("step_size", 10)),
            gamma=float(train_cfg.get("gamma", 0.1)),
        )

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        train_cfg = getattr(trainer, "cfg", {}).get("train", {})
        scheduler_name = train_cfg.get("lr_scheduler")
        if scheduler_name is None:
            return False

        self.train()
        optimizer = self.configure_optimizer(getattr(trainer, "optimizer_cfg", None))
        scheduler = self._configure_scheduler(optimizer, train_cfg)
        for epoch in range(trainer.max_epochs):
            totals: dict[str, float] = {}
            n = 0
            for batch in train_loader:
                out = self.observe(batch, task)
                loss = out["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.transform_gradients(task)
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), trainer.grad_clip)
                optimizer.step()
                trainer.advance_step()
                self.after_optimizer_step(task)
                for k, v in self.train_metrics(out).items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                n += 1
            if scheduler is not None:
                scheduler.step()
            if totals:
                metrics = {k: v / max(n, 1) for k, v in totals.items()}
                trainer.log_train_metrics(metrics, task=task, epoch=epoch + 1, epochs=trainer.max_epochs, optimizer=optimizer)
        return True

    def after_optimizer_step(self, task: Any | None = None) -> None:
        return None

    def _generate_uap(self, train_loader: Any | None) -> torch.Tensor | None:
        if train_loader is None:
            return None
        detector_was_training = self.detector.training
        self.detector.eval()
        uap: torch.Tensor | None = None
        try:
            for _epoch in range(max(self.uap_iters, 0)):
                for batch in train_loader:
                    batch = batch_to_device(batch, self.device)
                    x = batch["x"]
                    y = batch["y"].long()
                    real_x = x[y == self.real_label]
                    if real_x.numel() == 0:
                        continue
                    if uap is None:
                        uap = self._new_uap(real_x)
                    for _step in range(max(self.uap_max_steps_per_batch, 1)):
                        uap = uap.detach().to(device=real_x.device, dtype=real_x.dtype)
                        uap.requires_grad_(True)
                        self.detector.zero_grad(set_to_none=True)
                        pseudo_x = self._add_uap(real_x, uap)
                        logits = self.detector(pseudo_x)["logits"]
                        if self.binary_sigmoid:
                            attack_rate = (self._fake_logit(logits) > 0).float().mean()
                        else:
                            attack_rate = (logits.argmax(dim=-1) == self.fake_label).float().mean()
                        if float(attack_rate.detach().cpu()) >= self.uap_success_threshold:
                            uap = uap.detach()
                            break
                        target = torch.full((pseudo_x.shape[0],), self.fake_label, device=real_x.device, dtype=torch.long)
                        loss = self._classification_loss(logits, target)
                        loss.backward()
                        if uap.grad is None:
                            uap = uap.detach()
                            break
                        with torch.no_grad():
                            uap = uap - self.uap_alpha * uap.grad.sign()
                            uap = uap.clamp(-self.epsilon, self.epsilon)
        finally:
            self.detector.zero_grad(set_to_none=True)
            self.detector.train(detector_was_training)
        return None if uap is None else uap.detach()

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        uap = self._generate_uap(train_loader)
        if uap is not None:
            self._append_uap(uap)
        self.teacher = self.frozen_detector_copy()
