from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import register_method
from ..utils.checkpoint import load_checkpoint
from .base import ContinualMethod, batch_to_device, build_optimizer, freeze_module


@dataclass
class _ReplayTask:
    real: torch.Tensor
    fake: torch.Tensor
    real_center: torch.Tensor
    fake_center: torch.Tensor
    real_ptr: int = 0
    fake_ptr: int = 0


def _safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=1e-8)


def _supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Official SUR-LID SupCon loss, adapted only for device safety."""

    features = _safe_normalize(features, dim=1)
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    if labels.shape[0] != batch_size:
        raise ValueError("Num of labels does not match num of features")
    mask = torch.eq(labels, labels.T).float().to(features.device)
    logits = torch.matmul(features, features.T) / float(temperature)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    exp_logits = torch.exp(logits)
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=features.device)
    positives_mask = mask * logits_mask
    negatives_mask = 1.0 - mask
    num_pos = torch.sum(positives_mask, dim=1)
    denominator = torch.sum(exp_logits * negatives_mask, dim=1, keepdim=True) + torch.sum(exp_logits * positives_mask, dim=1, keepdim=True)
    log_probs = logits - torch.log(denominator.clamp_min(1e-12))
    valid = num_pos > 0
    if not valid.any():
        return features.new_tensor(0.0)
    log_probs = torch.sum(log_probs * positives_mask, dim=1)[valid] / num_pos[valid]
    return (-log_probs * float(temperature)).mean()


def _kd_loss(student_logits: torch.Tensor, labels: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 20.0, alpha: float = 0.3) -> torch.Tensor:
    # The official release passes log-softmax teacher scores to KLDivLoss.
    return nn.KLDivLoss(reduction="batchmean", log_target=True)(
        F.log_softmax(student_logits / temperature, dim=1),
        F.log_softmax(teacher_logits / temperature, dim=1),
    ) * (temperature * temperature * 2.0 * alpha) + F.cross_entropy(student_logits, labels.long()) * (1.0 - alpha)


class OfficialLogisticRegression(nn.Module):
    """Official SUR-LID linear head initialization."""

    def __init__(self, input_dim: int, output_dim: int = 2, bn: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.bias.data.zero_()
        self.bn = bool(bn)
        if self.bn:
            self.bn_layer = nn.BatchNorm1d(output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = F.linear(x, self.weight, self.bias)
        return self.bn_layer(logit) if self.bn else logit


def grid_shuffle_tensor(x: torch.Tensor, grid_size: int = 128) -> torch.Tensor:
    """Shuffle image grids for the SUR consistency score."""

    b, c, h, w = x.shape
    if h % grid_size != 0 or w % grid_size != 0:
        raise ValueError(f"grid_size={grid_size} must divide image size {(h, w)}")
    gh, gw = h // grid_size, w // grid_size
    y = x.view(b, c, gh, grid_size, gw, grid_size)
    y = y.permute(0, 1, 2, 4, 3, 5).contiguous()
    y = y.view(b, c, gh * gw, grid_size, grid_size)
    y = y[:, :, torch.randperm(gh * gw, device=x.device)]
    y = y.view(b, c, gh, gw, grid_size, grid_size)
    return y.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h, w)


@register_method("sur_lid")
@register_method("sur-lid")
class SURLIDMethod(ContinualMethod):
    """CAIDBench-native SUR-LID implementation.

    The method follows the paper mechanisms instead of copying the official
    DeepFakeBench implementation: sparse-uniform replay, latent isolation via
    task-aware supervised contrast, distribution re-filling around replay
    centers, incremental decision alignment for task heads, and teacher
    distillation over historical replay.
    """

    def __init__(
        self,
        detector_cfg: dict[str, Any] | None = None,
        num_classes: int = 2,
        max_tasks: int = 4,
        num_pic: int = 504,
        mem_each_batch: int = 3,
        alpha: float = 0.1,
        center_move_avg: bool = True,
        replay_mode: str = "sparse_robust",
        grid_size: int = 128,
        supcon_weight: float = 4.0,
        fd_weight: float = 10.0,
        kd_weight: float = 1.0,
        kd_temperature: float = 20.0,
        kd_alpha: float = 0.3,
        supcon_temperature: float = 0.1,
        sup_task_weight: float = 0.0,
        feat_aug: bool = True,
        feat_aug_v: str = "random",
        align_epoch: int = 1,
        frozen: str | bool = "semi",
        protocol: str = "P1",
        classifier_ensemble: str = "mean",
        backbone_checkpoint: str | None = None,
        require_official_feature_dim: bool = False,
        official_feature_dim: int = 1792,
        require_backbone_checkpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(detector_cfg=detector_cfg, num_classes=num_classes, **kwargs)
        self.detector.head.requires_grad_(False)
        self.feature_dim = self._infer_feature_dim()
        if require_official_feature_dim and self.feature_dim != int(official_feature_dim):
            raise ValueError(
                f"SUR-LID official mode expects EfficientNet-B4 feature_dim={int(official_feature_dim)}, "
                f"got {self.feature_dim}. Check detector_cfg.backbone and projection settings."
            )
        self.max_tasks = int(max_tasks)
        self.heads = nn.ModuleList([OfficialLogisticRegression(self.feature_dim, self.num_classes) for _ in range(self.max_tasks)])
        self.teacher_backbone: nn.Module | None = None
        self.teacher_heads: nn.ModuleList | None = None
        self.replay: list[_ReplayTask] = []

        self.num_pic = int(num_pic)
        self.mem_each_batch = int(mem_each_batch)
        self.alpha = float(alpha)
        self.center_move_avg = bool(center_move_avg)
        self.replay_mode = str(replay_mode)
        self.grid_size = int(grid_size)
        self.supcon_weight = float(supcon_weight)
        self.fd_weight = float(fd_weight)
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)
        self.kd_alpha = float(kd_alpha)
        self.supcon_temperature = float(supcon_temperature)
        self.sup_task_weight = float(sup_task_weight)
        self.feat_aug = bool(feat_aug)
        self.feat_aug_v = str(feat_aug_v)
        self.align_epoch = int(align_epoch)
        self.frozen = frozen
        self.protocol = str(protocol).upper()
        self.classifier_ensemble = str(classifier_ensemble)
        self.current_head = 0
        self.seen_heads = 0

        ckpt = backbone_checkpoint
        if ckpt is None and isinstance(detector_cfg, dict):
            ckpt = (detector_cfg.get("backbone", {}) or {}).get("pretrained_path")
        if require_backbone_checkpoint and not ckpt:
            raise ValueError("SUR-LID official mode requires backbone_checkpoint or detector_cfg.backbone.pretrained_path.")
        if ckpt:
            self._load_backbone_checkpoint(ckpt)

    def _load_backbone_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SUR-LID official checkpoint not found: {path}")
        obj = load_checkpoint(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("state_dict", "model", "backbone", "net"):
                if isinstance(obj.get(key), dict):
                    obj = obj[key]
                    break
        state = {}
        for key, value in obj.items():
            name = str(key)
            for prefix in ("module.", "detector.", "backbone.", "model."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
            state[name] = value
        target = getattr(self.detector.backbone, "model", self.detector.backbone)
        target.load_state_dict(state, strict=False)

    def _infer_feature_dim(self) -> int:
        backbone = self.detector.backbone
        model = getattr(backbone, "model", None)
        if model is not None and hasattr(model, "num_features"):
            return int(model.num_features)
        return int(getattr(self.detector, "feature_dim"))

    def _ensure_heads(self, n: int) -> None:
        while len(self.heads) < n:
            self.heads.append(OfficialLogisticRegression(self.feature_dim, self.num_classes).to(self.device))

    def configure_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        cfg = dict(optimizer_cfg or {})
        name = str(cfg.get("type", "adam")).lower()
        if name != "adam":
            return build_optimizer(self.parameters(), cfg)
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters for optimizer")
        return torch.optim.Adam(
            trainable,
            lr=float(cfg.get("lr", 2.0e-4)),
            betas=(float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.999))),
            eps=float(cfg.get("eps", 1.0e-8)),
            weight_decay=float(cfg.get("weight_decay", 5.0e-4)),
            amsgrad=bool(cfg.get("amsgrad", False)),
        )

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        super().before_task(task, train_loader)
        self.current_head = self.seen_heads
        self._ensure_heads(self.current_head + 1)
        for item in self.replay:
            item.real_ptr = 0
            item.fake_ptr = 0

    def _feature_map_and_vector(self, x: torch.Tensor, teacher: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if teacher:
            if self.teacher_backbone is None:
                raise RuntimeError("SUR-LID teacher is not initialized")
            backbone = self.teacher_backbone
        else:
            backbone = self.detector.backbone
        model = getattr(backbone, "model", None)
        if model is not None and hasattr(model, "forward_features"):
            feature_map = model.forward_features(x.float())
        elif hasattr(backbone, "features"):
            feature_map = backbone.features(x.float())  # type: ignore[attr-defined]
        else:
            feature_map = backbone(x)
        if isinstance(feature_map, (tuple, list)):
            feature_map = feature_map[-1]
        if feature_map.ndim == 2:
            vector = feature_map.float()
            feature_map = vector[:, :, None, None]
        else:
            feature_map = feature_map.float()
            vector = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
        proj = getattr(backbone, "proj", None)
        if proj is not None:
            vector = proj(vector)
        return feature_map, vector

    def _features(self, x: torch.Tensor, teacher: bool = False) -> torch.Tensor:
        return self._feature_map_and_vector(x, teacher=teacher)[1]

    def _ensemble_logits(self, z: torch.Tensor, n_heads: int | None = None) -> torch.Tensor:
        n = max(1, self.seen_heads if n_heads is None else n_heads)
        logits = torch.stack([self.heads[i](z) for i in range(n)], dim=0)
        if self.classifier_ensemble == "mean":
            return logits.mean(dim=0)
        fake_scores = logits[:, :, 1]
        if self.classifier_ensemble == "max":
            _, idx = fake_scores.max(dim=0)
        elif self.classifier_ensemble == "min":
            _, idx = fake_scores.min(dim=0)
        else:
            raise ValueError(f"Unknown classifier_ensemble={self.classifier_ensemble!r}")
        return logits[idx, torch.arange(idx.shape[0], device=idx.device)]

    def _split_head_logits(self, z: torch.Tensor, old_count: int, current_count: int) -> torch.Tensor:
        out = []
        start = 0
        block = self.mem_each_batch * 2
        for i in range(old_count):
            out.append(self.heads[i](z[start : start + block]))
            start += block
        out.append(self.heads[self.current_head](z[start : start + current_count]))
        return torch.cat(out, dim=0)

    def _teacher_logits(self, z: torch.Tensor, old_count: int) -> torch.Tensor:
        if self.teacher_heads is None:
            raise RuntimeError("SUR-LID teacher heads are not initialized")
        out = []
        start = 0
        block = self.mem_each_batch * 2
        for i in range(old_count):
            out.append(self.teacher_heads[i](z[start : start + block]))
            start += block
        return torch.cat(out, dim=0)

    def _next_replay_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.replay:
            return None
        xs, ys, task_ids = [], [], []
        for task_id, item in enumerate(self.replay):
            fake, item.fake_ptr = self._take(item.fake, item.fake_ptr, self.mem_each_batch)
            real, item.real_ptr = self._take(item.real, item.real_ptr, self.mem_each_batch)
            xs.extend([fake, real])
            ys.extend(
                [
                    torch.ones(fake.shape[0], dtype=torch.long),
                    torch.zeros(real.shape[0], dtype=torch.long),
                ]
            )
            task_ids.extend(
                [
                    torch.full((fake.shape[0],), task_id, dtype=torch.long),
                    torch.full((real.shape[0],), task_id, dtype=torch.long),
                ]
            )
        return torch.cat(xs).to(self.device), torch.cat(ys).to(self.device), torch.cat(task_ids).to(self.device)

    @staticmethod
    def _take(bank: torch.Tensor, ptr: int, n: int) -> tuple[torch.Tensor, int]:
        if bank.shape[0] == 0:
            return bank, ptr
        end = ptr + n
        if end <= bank.shape[0]:
            out = bank[ptr:end]
        else:
            out = torch.cat([bank[ptr:], bank[: end % bank.shape[0]]], dim=0)
        return out, end % bank.shape[0]

    def _feature_aug(self, feat: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        if self.feat_aug_v == "v1":
            return self._feature_aug_v1(feat, center)
        if self.feat_aug_v == "v2":
            return self._feature_aug_v2(feat, center)
        if self.feat_aug_v == "v3":
            return self._feature_aug_v3(feat, center)
        if self.feat_aug_v == "random":
            choice = int(torch.randint(0, 3, (1,), device=feat.device).item())
            return [self._feature_aug_v1, self._feature_aug_v2, self._feature_aug_v3][choice](feat, center)
        return feat

    @staticmethod
    def _feature_aug_v1(feat: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        out = feat.clone()
        if out.shape[0] > 0:
            out[int(torch.randint(0, out.shape[0], (1,), device=out.device).item())] = out.mean(dim=0)
        return out

    @staticmethod
    def _feature_aug_v2(feat: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        if feat.shape[0] < 3:
            return feat
        out = feat.clone()
        out[0, :] = (feat[0, :] + feat[1, :]) / 2
        out[1, :] = (feat[1, :] + feat[2, :]) / 2
        out[2, :] = (feat[2, :] + feat[0, :]) / 2
        return out

    @staticmethod
    def _feature_aug_v3(feat: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        if feat.shape[0] < 3:
            return feat
        out = feat.clone()
        center = center.to(feat.device)
        alpha = torch.rand(3, device=feat.device) * 0.5 + 0.5
        out[0, :] = feat[0, :] * alpha[0] + center * (1 - alpha[0])
        out[1, :] = feat[1, :] * alpha[1] + center * (1 - alpha[1])
        out[2, :] = feat[2, :] * alpha[2] + center * (1 - alpha[2])
        return out

    def _distribution_refill(self, z: torch.Tensor, task_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.feat_aug or not self.replay:
            return _safe_normalize(z, dim=1)
        out = _safe_normalize(z, dim=1)
        start = 0
        for item in self.replay:
            fake_slice = slice(start, start + self.mem_each_batch)
            real_slice = slice(start + self.mem_each_batch, start + 2 * self.mem_each_batch)
            out[fake_slice] = self._feature_aug(out[fake_slice], item.fake_center)
            out[real_slice] = self._feature_aug(out[real_slice], item.real_center)
            start += self.mem_each_batch * 2
        return out

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        x_new = batch["x"]
        y_new = batch["y"].long()
        task_new = torch.full_like(y_new, self.current_head)
        replay_batch = self._next_replay_batch()
        if replay_batch is None:
            x = x_new
            y = y_new
            task_ids = task_new
            old_n = 0
        else:
            x_old, y_old, old_task_ids = replay_batch
            x = torch.cat([x_old, x_new], dim=0)
            y = torch.cat([y_old, y_new], dim=0)
            task_ids = torch.cat([old_task_ids, task_new], dim=0)
            old_n = x_old.shape[0]

        feature_map, z = self._feature_map_and_vector(x)
        logits = self._split_head_logits(z, len(self.replay), y_new.shape[0])
        train_all = str(self.frozen).lower() not in {"false", "0", "none"}
        ce = F.cross_entropy(logits, y) if train_all else F.cross_entropy(logits[-y_new.shape[0] :], y_new)

        z_for_lid = self._distribution_refill(z, task_ids, y)
        lid_labels = task_ids * 10 + y
        if self.protocol == "P2":
            lid_labels = lid_labels.clone()
            lid_labels[y == 0] = 0
        supcon = _supervised_contrastive_loss(z_for_lid, lid_labels, temperature=self.supcon_temperature)
        loss = ce + self.supcon_weight * supcon
        log = {"loss": loss, "loss_ce": ce.detach(), "loss_supcon": supcon.detach()}

        if old_n > 0 and self.teacher_backbone is not None:
            with torch.no_grad():
                teacher_map, z_teacher = self._feature_map_and_vector(x[:old_n], teacher=True)
                teacher_logits = self._teacher_logits(z_teacher, len(self.replay))
            student_fd = F.adaptive_avg_pool2d(feature_map[:old_n], (1, 1)).flatten(1)
            teacher_fd = F.adaptive_avg_pool2d(teacher_map, (1, 1)).flatten(1)
            fd = F.mse_loss(_safe_normalize(student_fd, dim=1), _safe_normalize(teacher_fd, dim=1), reduction="mean")
            kd = _kd_loss(logits[:old_n], y[:old_n], teacher_logits, temperature=self.kd_temperature, alpha=self.kd_alpha)
            loss = loss + self.fd_weight * fd + self.kd_weight * kd
            log.update({"loss": loss, "loss_fd": fd.detach(), "loss_kd": kd.detach()})
            if self.protocol == "P1" and self.sup_task_weight > 0:
                task_supcon = _supervised_contrastive_loss(z_for_lid, task_ids, temperature=self.supcon_temperature)
                loss = loss + self.sup_task_weight * task_supcon
                log.update({"loss": loss, "loss_sup_task": task_supcon.detach()})

        if torch.isnan(log["loss"]).any():
            zero = logits.sum() * 0.0
            log = {k: zero if k == "loss" else zero.detach() for k in log}
        return log

    def _align_heads(self) -> torch.Tensor | None:
        if self.current_head == 0:
            return None
        if str(self.frozen).lower() in {"semi", "true", "1"}:
            strengths = [0.99, 0.999, 0.9999, 0.99995]
        else:
            strengths = [0.99, 1.0, 1.0, 1.0]
        n = min(self.current_head + 1, len(strengths))
        weights = [_safe_normalize(self.heads[i].weight.data, dim=1) for i in range(n)]
        sims = []
        for i in range(n):
            others = [(weights[i] * weights[j]).sum().item() for j in range(n) if j != i]
            if not others:
                continue
            farthest_idx = min((j for j in range(n) if j != i), key=lambda j: (weights[i] * weights[j]).sum().item())
            strength = strengths[n - i - 1]
            old_norm = torch.norm(self.heads[i].weight.data, dim=1, keepdim=True)
            mixed = strength * weights[i] + (1.0 - strength) * weights[farthest_idx]
            self.heads[i].weight.data = _safe_normalize(mixed, dim=1) * old_norm
            sims.extend(others)
        return self.heads[0].weight.new_tensor(sum(sims) / max(len(sims), 1))

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        self.train()
        optimizer = self.configure_optimizer(trainer.optimizer_cfg)
        scheduler = self._configure_scheduler(optimizer, getattr(trainer, "cfg", {}).get("train", {}))
        for epoch in range(1, trainer.max_epochs + 1):
            totals: dict[str, float] = {}
            n = 0
            for batch in train_loader:
                out = self.observe(batch, task)
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), trainer.grad_clip)
                optimizer.step()
                aligned = self._align_heads() if epoch > self.align_epoch else None
                trainer.advance_step()
                for key, value in out.items():
                    if torch.is_tensor(value) and value.ndim == 0:
                        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
                if aligned is not None:
                    totals["head_alignment"] = totals.get("head_alignment", 0.0) + float(aligned.detach().cpu())
                n += 1
            if totals:
                metrics = {k: v / max(n, 1) for k, v in totals.items()}
                trainer.logger.info("task=%s epoch=%d/%d %s", task.name, epoch, trainer.max_epochs, ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                trainer.log_metrics({**{f"train/{k}": v for k, v in metrics.items()}, "train/task_index": float(getattr(task, "task_id", 0)), "train/epoch": epoch})
            if scheduler is not None:
                scheduler.step()
        return True

    @staticmethod
    def _configure_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler | None:
        name = train_cfg.get("lr_scheduler")
        if name is None or str(name).lower() in {"none", "null", ""}:
            return None
        if str(name).lower() != "step":
            raise NotImplementedError(f"SUR-LID official reproduction supports lr_scheduler=step, got {name!r}")
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("lr_step", 10)),
            gamma=float(train_cfg.get("lr_gamma", 0.4)),
        )

    @torch.no_grad()
    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        z = self._features(batch["x"])
        logits = self._ensemble_logits(z, n_heads=max(self.seen_heads, self.current_head + 1))
        return {"logits": logits, "features": z}

    @torch.no_grad()
    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is None:
            raise RuntimeError("SUR-LID requires train_loader to build sparse-uniform replay memory")
        self.eval()
        xs, ys, feats, shuffle_feats = [], [], [], []
        for batch in train_loader:
            batch = batch_to_device(batch, self.device)
            x = batch["x"]
            xs.append(x.detach().cpu())
            ys.append(batch["y"].long().detach().cpu())
            feats.append(self._features(x).detach().cpu())
            if self.replay_mode == "sparse_robust" and x.ndim == 4 and x.shape[-1] % self.grid_size == 0 and x.shape[-2] % self.grid_size == 0:
                shuffle_feats.append(self._features(grid_shuffle_tensor(x, self.grid_size)).detach().cpu())
        x_all = torch.cat(xs)
        y_all = torch.cat(ys)
        z_all = torch.cat(feats)
        z_shuf = torch.cat(shuffle_feats) if shuffle_feats else None
        real = y_all == 0
        fake = y_all == 1
        if not real.any() or not fake.any():
            raise RuntimeError("SUR-LID replay requires both real and fake samples in each task")
        real_center = self._center(z_all[real])
        fake_center = self._center(z_all[fake])
        real_idx = self._select_sparse_uniform(z_all[real], real_center, self.num_pic, z_shuf[real] if z_shuf is not None else None)
        fake_idx = self._select_sparse_uniform(z_all[fake], fake_center, self.num_pic, z_shuf[fake] if z_shuf is not None else None)
        self.replay.append(_ReplayTask(real=x_all[real][real_idx].cpu(), fake=x_all[fake][fake_idx].cpu(), real_center=real_center.cpu(), fake_center=fake_center.cpu()))
        self.seen_heads = self.current_head + 1
        self.teacher_backbone = freeze_module(copy.deepcopy(self.detector.backbone).to(self.device))
        self.teacher_heads = nn.ModuleList([copy.deepcopy(self.heads[i]).to(self.device) for i in range(self.seen_heads)])
        for head in self.teacher_heads:
            freeze_module(head)

    def _center(self, z: torch.Tensor) -> torch.Tensor:
        if not self.center_move_avg:
            return z.mean(dim=0)
        center = z[0].clone()
        for row in z[1:]:
            center = (1.0 - self.alpha) * center + self.alpha * row
        return center

    def _select_sparse_uniform(self, z: torch.Tensor, center: torch.Tensor, k: int, z_shuffle: torch.Tensor | None = None) -> torch.Tensor:
        if z.shape[0] <= k:
            return torch.arange(z.shape[0])
        center_col = center.unsqueeze(-1)
        distance = torch.mm(z, center_col).squeeze()
        if self.replay_mode == "center":
            return distance.topk(k).indices.long()
        order = torch.argsort(distance, descending=True)
        step = max(int(len(order) // max(k // 2, 1)), 1)
        if self.replay_mode == "sparse_robust":
            if z_shuffle is None:
                raise RuntimeError("SUR-LID sparse_robust replay requires shuffled features")
            robust_score = torch.diag(torch.mm(z, z_shuffle.T).squeeze()) if z.shape[0] > 1 else torch.mm(z, z_shuffle.T).squeeze().view(1)
        else:
            robust_score = None
        selected: list[int] = []
        for start in range(0, len(order), step):
            if start + step > distance.shape[0] - 1 or len(selected) == k:
                break
            chunk = order[start : min(start + step, distance.shape[0] - 1)]
            if chunk.numel() == 0:
                continue
            if self.replay_mode == "sparse_robust":
                assert robust_score is not None
                anchor = chunk[torch.argsort(robust_score[chunk])[-1]]
                anchor_vec = _safe_normalize(z[anchor], dim=0)
            else:
                anchor = order[min(start + step // 2, (distance.shape[0] - 1 - start) // 2 + start)]
                anchor_vec = _safe_normalize(z[anchor] - center, dim=0)
            selected.append(int(anchor))
            latest_small = z.new_tensor(2.0)
            latest_idx = int(anchor)
            for each in chunk:
                if self.replay_mode == "sparse_robust":
                    each_vec = _safe_normalize(z[each], dim=0)
                else:
                    each_vec = _safe_normalize(z[each] - center, dim=0)
                cosine_similarity = torch.dot(each_vec, anchor_vec)
                if cosine_similarity < latest_small:
                    latest_small = cosine_similarity
                    latest_idx = int(each)
            selected.append(latest_idx)
        return torch.tensor(selected[:k], dtype=torch.long)

    def save_replay(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save([item.real for item in self.replay], path / "real_replay.pt")
        torch.save([item.fake for item in self.replay], path / "fake_replay.pt")
        torch.save([item.real_center for item in self.replay], path / "real_centers.pt")
        torch.save([item.fake_center for item in self.replay], path / "fake_centers.pt")
