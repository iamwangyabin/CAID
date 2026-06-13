from __future__ import annotations

import json
import time
from itertools import islice
from typing import Any, Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import load_config
from ..data.loader import build_dataloader
from ..data.scenario import ContinualScenario, TaskSpec
from ..evaluation import ContinualMetricMatrix, summarize_logits
from ..methods.base import build_optimizer
from ..registry import build_method
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.experiment import build_experiment_logger, compute_experiment_name
from ..utils.logging import attach_file_handler, format_log_value, get_logger
from ..utils.seed import seed_everything
from ..utils.tensor import move_to_device


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "false"}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


class Trainer:
    """Unified continual-detection trainer.

    The trainer owns only the generic protocol: ordered tasks, dataloaders,
    optimization, evaluation matrix, and checkpointing. Method-specific logic
    stays inside `caidbench.methods`.
    """

    def __init__(self, cfg: dict[str, Any] | str | Path) -> None:
        if isinstance(cfg, (str, Path)):
            cfg = load_config(cfg)
        self.cfg = cfg
        seed_everything(int(cfg.get("seed", 0)))
        base_output_dir = Path(cfg.get("output_dir", "outputs/caidbench_run"))
        self.logger = get_logger("caidbench")
        device_cfg = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(device_cfg if device_cfg != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        scfg = cfg.get("scenario", {})
        self.scenario = ContinualScenario.from_config(scfg)
        method_cfg = dict(cfg.get("method", {}))
        method_name = method_cfg.pop("name", cfg.get("method_name", "finetune"))
        train_cfg = cfg.get("train", {})
        eval_cfg = cfg.get("eval", {}) or {}
        if not isinstance(eval_cfg, Mapping):
            raise TypeError("eval config must be a mapping")
        debug_step_limit = train_cfg.get(
            "debug_max_steps_per_epoch",
            cfg.get("debug_max_steps_per_epoch", method_cfg.get("debug_max_steps_per_epoch")),
        )
        self.debug_max_steps_per_epoch = _optional_positive_int(debug_step_limit)
        self.eval_scope = str(eval_cfg.get("scope", "seen")).lower()
        if self.eval_scope not in {"seen", "all", "current"}:
            raise ValueError("eval.scope must be one of: seen, all, current")
        self.eval_max_batches_per_task = _optional_positive_int(eval_cfg.get("max_batches_per_task"))
        self.method = build_method(method_name, **method_cfg).to(self.device)
        self.method._runtime_debug_max_steps_per_epoch = self.debug_max_steps_per_epoch
        self.max_epochs = int(train_cfg.get("epochs", 1))
        self.batch_size = int(train_cfg.get("batch_size", 32))
        self.num_workers = int(train_cfg.get("num_workers", 0))
        self.eval_num_workers = int(train_cfg.get("eval_num_workers", 0))
        self.pin_memory = bool(train_cfg.get("pin_memory", self.device.type == "cuda"))
        self.persistent_workers = bool(train_cfg.get("persistent_workers", self.num_workers > 0))
        self.prefetch_factor = train_cfg.get("prefetch_factor", 2 if self.num_workers > 0 else None)
        self.non_blocking = bool(train_cfg.get("non_blocking", self.pin_memory and self.device.type == "cuda"))
        self.drop_last = bool(train_cfg.get("drop_last", False))
        checkpoint_cfg = cfg.get("checkpoint", {}) or {}
        if not isinstance(checkpoint_cfg, Mapping):
            raise TypeError("checkpoint config must be a mapping")
        self.save_last_checkpoint = bool(checkpoint_cfg.get("save_last", True))
        self.save_base_checkpoint = bool(checkpoint_cfg.get("save_base", True))
        self.save_task_checkpoints = bool(
            checkpoint_cfg.get(
                "save_each_task",
                checkpoint_cfg.get("save_task_checkpoints", checkpoint_cfg.get("save_intermediate", True)),
            )
        )
        self.train_log_interval = int(train_cfg.get("log_interval", 50))
        if self.train_log_interval <= 0:
            self.train_log_interval = 0
        self.grad_clip = train_cfg.get("grad_clip")
        self.grad_clip = None if self.grad_clip is None else float(self.grad_clip)
        self.optimizer_cfg = dict(train_cfg.get("optimizer", {"type": "adamw", "lr": 1e-4}))
        self.metric_matrix = ContinualMetricMatrix([t.name for t in self.scenario.tasks])
        self.eval_records: list[dict[str, Any]] = []
        self.global_step = 0
        self.resume_task_index = -1
        self._active_train_task_index: float | None = None
        experiment_name = compute_experiment_name(self.cfg, base_output_dir, str(method_name))
        self.output_dir = base_output_dir / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        attach_file_handler(self.logger, self.output_dir / "train.log")
        self.experiment = build_experiment_logger(self.cfg, self.output_dir, str(method_name), experiment_name=experiment_name)
        resume_from = cfg.get("resume_from")
        if resume_from:
            self._resume_from_checkpoint(resume_from)

    def effective_train_batches(self, train_loader: Any) -> int:
        num_batches = len(train_loader)
        if self.debug_max_steps_per_epoch is None:
            return num_batches
        return min(num_batches, self.debug_max_steps_per_epoch)

    def iter_train_batches(self, train_loader: Any) -> Iterable[tuple[int, Any]]:
        batches = enumerate(train_loader, start=1)
        if self.debug_max_steps_per_epoch is None:
            return batches
        return islice(batches, self.debug_max_steps_per_epoch)

    def make_optimizer(self, params: Iterable[torch.nn.Parameter] | None = None) -> torch.optim.Optimizer:
        return build_optimizer(self.method.parameters() if params is None else params, self.optimizer_cfg)

    def dataloader(
        self,
        task_index: int,
        split: str,
        shuffle: bool = False,
        transform_split: str | None = None,
        drop_last: bool | None = None,
    ):
        ds = self.scenario.task_dataset(split, task_index, transform_split=transform_split)
        effective_drop_last = self.drop_last and split == "train" if drop_last is None else bool(drop_last)
        effective_num_workers = self.num_workers if split == "train" else self.eval_num_workers
        effective_persistent_workers = self.persistent_workers and split == "train" and effective_num_workers > 0
        effective_prefetch_factor = self.prefetch_factor if effective_num_workers > 0 else None
        loader = build_dataloader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=effective_num_workers,
            drop_last=effective_drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=effective_persistent_workers,
            prefetch_factor=effective_prefetch_factor,
        )
        return loader

    @staticmethod
    def _scalar_train_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in metrics.items():
            if key == "logits":
                continue
            if torch.is_tensor(value):
                if value.ndim != 0:
                    continue
                value = value.detach().cpu().item()
            if isinstance(value, (int, float)):
                out[str(key)] = float(value)
        return out

    def log_train_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        task: TaskSpec | Any | None = None,
        task_index: int | float | None = None,
        task_name: str | None = None,
        epoch: int | float | None = None,
        epochs: int | float | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float | None = None,
        phase: str | None = None,
        step: int | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        started_at: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {}
        for key, value in self._scalar_train_metrics(metrics).items():
            metric_key = key if key.startswith("train/") else f"train/{key}"
            payload[metric_key] = value
        if task is not None:
            payload["train/task_index"] = float(getattr(task, "task_id", 0))
            if task_name is None:
                task_name = getattr(task, "name", None)
        elif task_index is not None:
            payload["train/task_index"] = float(task_index)
        if epoch is not None:
            payload["train/epoch"] = float(epoch)
        if optimizer is not None and optimizer.param_groups:
            payload["train/lr"] = float(optimizer.param_groups[0].get("lr", 0.0))
        if lr is not None:
            payload["train/lr"] = float(lr)
        if not payload:
            return
        self._print_train_metrics(
            payload,
            task=task,
            task_name=task_name,
            phase=phase,
            epochs=epochs,
            batch_idx=batch_idx,
            num_batches=num_batches,
            started_at=started_at,
        )
        self.log_metrics(payload, step=step)

    def _print_train_metrics(
        self,
        payload: Mapping[str, Any],
        *,
        task: TaskSpec | Any | None,
        task_name: str | None,
        phase: str | None,
        epochs: int | float | None,
        batch_idx: int | None,
        num_batches: int | None,
        started_at: float | None,
    ) -> None:
        metrics = {
            str(key)[len("train/") :]: float(value)
            for key, value in payload.items()
            if str(key).startswith("train/") and key not in {"train/task_index", "train/epoch", "train/global_step", "train/lr"}
        }
        if not metrics:
            return
        task_name = task_name or getattr(task, "name", None)
        task_index = payload.get("train/task_index", self._active_train_task_index)
        parts = ["train"]
        if task_name is not None:
            parts.append(f"task={format_log_value(task_name)}")
        elif task_index is not None:
            parts.append(f"task_index={int(float(task_index))}")
        if phase:
            parts.append(f"phase={phase}")
        if "train/epoch" in payload:
            epoch = int(float(payload["train/epoch"]))
            if epochs is not None:
                parts.append(f"epoch={epoch}/{int(float(epochs))}")
            else:
                parts.append(f"epoch={epoch}")
        if batch_idx is not None and num_batches is not None and num_batches > 0:
            batch_idx = min(max(int(batch_idx), 0), int(num_batches))
            num_batches = int(num_batches)
            progress = 100.0 * batch_idx / max(num_batches, 1)
            parts.append(f"step={batch_idx}/{num_batches}")
            parts.append(f"progress={progress:.2f}%")
            if started_at is not None and batch_idx > 0:
                elapsed = max(time.monotonic() - float(started_at), 0.0)
                remaining = max(num_batches - batch_idx, 0)
                eta = elapsed / batch_idx * remaining
                parts.append(f"eta={_format_duration(eta)}")
        if "train/lr" in payload:
            parts.append(f"lr={float(payload['train/lr']):.6g}")
        parts.extend(f"{key}={value:.4f}" for key, value in metrics.items())
        self.logger.info(" ".join(parts))

    def _normalize_train_payload(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(metrics)
        if not any(str(key).startswith("train/") for key in payload):
            return payload
        if self._active_train_task_index is not None:
            payload.setdefault("train/task_index", float(self._active_train_task_index))
        payload.setdefault("train/epoch", 0.0)
        payload.setdefault("train/global_step", float(self.global_step))
        return payload

    def default_train_loop(self, method, task: TaskSpec, train_loader, optimizer: torch.optim.Optimizer | None = None) -> None:
        method.train()
        optimizer = optimizer or method.configure_optimizer(self.optimizer_cfg)
        num_batches = self.effective_train_batches(train_loader)
        for epoch in range(self.max_epochs):
            totals: dict[str, float] = {}
            n = 0
            epoch_started_at = time.monotonic()
            for batch_idx, batch in self.iter_train_batches(train_loader):
                batch = move_to_device(batch, self.device, non_blocking=self.non_blocking)
                out = method.observe(batch, task)
                loss = out["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                method.transform_gradients(task)
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(method.parameters(), self.grad_clip)
                optimizer.step()
                self.advance_step()
                method.after_optimizer_step(task)
                for k, v in method.train_metrics(out).items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                n += 1

                if self.train_log_interval > 0 and n % self.train_log_interval == 0:
                    metrics = {k: v / max(n, 1) for k, v in totals.items()}
                    self.log_train_metrics(
                        metrics,
                        task=task,
                        epoch=epoch + 1,
                        epochs=self.max_epochs,
                        optimizer=optimizer,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        started_at=epoch_started_at,
                    )

            if totals:
                metrics = {k: v / max(n, 1) for k, v in totals.items()}
                self.log_train_metrics(
                    metrics,
                    task=task,
                    epoch=epoch + 1,
                    epochs=self.max_epochs,
                    optimizer=optimizer,
                    batch_idx=n,
                    num_batches=num_batches,
                    started_at=epoch_started_at,
                )

    def advance_step(self, n: int = 1) -> None:
        self.global_step += int(n)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        payload = self._normalize_train_payload(metrics)
        self.experiment.log(payload, step=self.global_step if step is None else step)

    @torch.no_grad()
    def evaluate_loader(self, loader) -> dict[str, float | int]:
        self.method.eval()
        logits_list: list[torch.Tensor] = []
        y_list: list[torch.Tensor] = []
        iterator = iter(loader)
        try:
            batches = iterator if self.eval_max_batches_per_task is None else islice(iterator, self.eval_max_batches_per_task)
            for batch in batches:
                batch = move_to_device(batch, self.device, non_blocking=self.non_blocking)
                out = self.method.predict(batch)
                logits_list.append(out["logits"].detach().cpu())
                y_list.append(batch["y"].detach().cpu())
        finally:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
        if not logits_list:
            return {"acc": float("nan"), "auc": float("nan"), "ap": float("nan"), "f1": float("nan"), "ece": float("nan"), "num_samples": 0}
        logits = torch.cat(logits_list, dim=0)
        y = torch.cat(y_list, dim=0)
        return {**summarize_logits(logits, y), "num_samples": int(y.numel())}

    def _eval_task_indices(self, train_index: int) -> range:
        if self.eval_scope == "all":
            return range(len(self.scenario.tasks))
        if self.eval_scope == "current":
            return range(train_index, train_index + 1)
        return range(train_index + 1)

    @staticmethod
    def _nanmean(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        valid = values[~np.isnan(values)]
        return float(valid.mean()) if valid.size else float("nan")

    def _seen_row_average(self, train_index: int, kind: str) -> float:
        matrix = self.metric_matrix._matrix(kind)
        return self._nanmean(matrix[train_index, : train_index + 1])

    def _future_row_average(self, train_index: int, kind: str) -> float:
        matrix = self.metric_matrix._matrix(kind)
        return self._nanmean(matrix[train_index, train_index + 1 :])

    @staticmethod
    def _weighted_records(records: Iterable[Mapping[str, Any]], key: str) -> float:
        total = 0.0
        denom = 0.0
        for record in records:
            value = float(record.get(key, float("nan")))
            weight = float(record.get("num_samples", 0) or 0)
            if np.isnan(value) or weight <= 0:
                continue
            total += value * weight
            denom += weight
        return float(total / denom) if denom > 0 else float("nan")

    def run(self) -> dict[str, Any]:
        try:
            self.logger.info("Starting CAIDBench run on %s with method=%s", self.device, self.method.__class__.__name__)
            if self.debug_max_steps_per_epoch is not None:
                self.logger.info("Debug train step limit enabled: max_steps_per_epoch=%d", self.debug_max_steps_per_epoch)
            if self.eval_max_batches_per_task is not None:
                self.logger.info("Debug eval batch limit enabled: max_batches_per_task=%d", self.eval_max_batches_per_task)
            start_task_index = min(max(self.resume_task_index + 1, 0), len(self.scenario.tasks))
            if start_task_index > 0:
                self.logger.info("Resuming from completed task_index=%d; next_task_index=%d", self.resume_task_index, start_task_index)
            for i, task in enumerate(self.scenario.tasks[start_task_index:], start=start_task_index):
                self.logger.info("=== Task %d/%d: %s train=%d test=%d ===", i + 1, len(self.scenario.tasks), task.name, task.num_train, task.num_test)
                self._active_train_task_index = float(getattr(task, "task_id", i))
                train_loader = self.dataloader(i, "train", shuffle=True)
                val_loader = self.dataloader(i, "val", shuffle=False) if task.num_val > 0 else None
                self.method.before_task(task, train_loader)
                self.method.to(self.device)
                handled = self.method.fit_task(self, task, train_loader, val_loader)
                if not handled:
                    self.default_train_loop(self.method, task, train_loader)
                self.method.after_task(task, train_loader)
                eval_payload: dict[str, float | int] = {}
                eval_rows: list[dict[str, Any]] = []
                for j in self._eval_task_indices(i):
                    test_loader = self.dataloader(j, "test", shuffle=False)
                    try:
                        metrics = self.evaluate_loader(test_loader)
                    finally:
                        del test_loader
                    self.metric_matrix.update(i, j, metrics["acc"], metrics["auc"], metrics["ap"], metrics["f1"])
                    record = {
                        "after_task": i,
                        "after_task_name": task.name,
                        "eval_task": j,
                        "eval_task_name": self.scenario.tasks[j].name,
                        "acc": metrics["acc"],
                        "auc": metrics["auc"],
                        "ap": metrics["ap"],
                        "f1": metrics["f1"],
                        "ece": metrics["ece"],
                        "num_samples": metrics["num_samples"],
                    }
                    self.eval_records.append(record)
                    eval_rows.append(record)
                seen_eval_rows = [row for row in eval_rows if int(row["eval_task"]) <= i]
                future_eval_rows = [row for row in eval_rows if int(row["eval_task"]) > i]
                eval_payload.update(
                    {
                        "eval/average_accuracy": self._seen_row_average(i, "acc"),
                        "eval/average_auc": self._seen_row_average(i, "auc"),
                        "eval/average_ap": self._seen_row_average(i, "ap"),
                        "eval/average_f1": self._seen_row_average(i, "f1"),
                        "eval/official_weighted_accuracy": self._weighted_records(seen_eval_rows, "acc"),
                        "eval/official_weighted_auc": self._weighted_records(seen_eval_rows, "auc"),
                        "eval/official_weighted_ap": self._weighted_records(seen_eval_rows, "ap"),
                        "eval/official_weighted_f1": self._weighted_records(seen_eval_rows, "f1"),
                        "eval/after_task": i,
                    }
                )
                if future_eval_rows:
                    eval_payload.update(
                        {
                            "eval/future_average_accuracy": self._future_row_average(i, "acc"),
                            "eval/future_average_auc": self._future_row_average(i, "auc"),
                            "eval/future_average_ap": self._future_row_average(i, "ap"),
                            "eval/future_average_f1": self._future_row_average(i, "f1"),
                            "eval/future_weighted_accuracy": self._weighted_records(future_eval_rows, "acc"),
                            "eval/future_weighted_auc": self._weighted_records(future_eval_rows, "auc"),
                            "eval/future_weighted_ap": self._weighted_records(future_eval_rows, "ap"),
                            "eval/future_weighted_f1": self._weighted_records(future_eval_rows, "f1"),
                        }
                    )
                self._log_eval_console_table(eval_rows, eval_payload)
                self._log_eval_table(eval_rows, step=i)
                self.log_metrics(eval_payload, step=i)
                self._save_intermediate(i)
                self._write_outputs(log_tables=False)
                self._active_train_task_index = None
            summary = self._write_outputs(log_tables=True)
            summary_payload = {
                "summary/average_accuracy": summary["average_accuracy"],
                "summary/average_forgetting": summary["average_forgetting"],
                "summary/average_auc": summary["average_auc"],
                "summary/auc_forgetting": summary["auc_forgetting"],
                "summary/average_ap": summary["average_ap"],
                "summary/ap_forgetting": summary["ap_forgetting"],
                "summary/average_f1": summary["average_f1"],
                "summary/f1_forgetting": summary["f1_forgetting"],
                "summary/official_average_accuracy": summary["official_average_accuracy"],
                "summary/official_last_accuracy": summary["official_last_accuracy"],
            }
            if self.eval_scope == "all":
                summary_payload.update(
                    {
                        "summary/future_average_accuracy": summary["future_average_accuracy"],
                        "summary/future_average_auc": summary["future_average_auc"],
                        "summary/future_average_ap": summary["future_average_ap"],
                        "summary/future_average_f1": summary["future_average_f1"],
                    }
                )
            self.log_metrics(summary_payload)
            self.logger.info(
                "Finished: AA=%.4f AF=%.4f AUC_AA=%.4f AP_AA=%.4f F1_AA=%.4f OfficialAbar=%.4f OfficialAB=%.4f",
                summary["average_accuracy"],
                summary["average_forgetting"],
                summary["average_auc"],
                summary["average_ap"],
                summary["average_f1"],
                summary["official_average_accuracy"],
                summary["official_last_accuracy"],
            )
            return summary
        except BaseException:
            self._write_partial_outputs_after_error()
            raise
        finally:
            self.experiment.finish()

    def _save_intermediate(self, task_index: int) -> None:
        payload = {
            "checkpoint_version": 1,
            "model": self.method.state_dict(),
            "auxiliary": self.method.auxiliary_state_dict(),
            "cfg": self.cfg,
            "task_index": task_index,
            "completed_tasks": task_index + 1,
            "global_step": self.global_step,
            "metric_tables": self.metric_matrix.to_tables(),
            "eval_records": self.eval_records,
        }
        if self.save_task_checkpoints:
            save_checkpoint(self.output_dir / f"task_{task_index}.pt", **payload)
        if self.save_last_checkpoint:
            save_checkpoint(self.output_dir / "last.pt", **payload)
        if task_index == 0 and self.save_base_checkpoint:
            save_checkpoint(self.output_dir / "base.pt", **payload)

    def _resume_from_checkpoint(self, path: str | Path) -> None:
        ckpt = load_checkpoint(path, map_location=self.device)
        result = self.method.load_state_dict(ckpt["model"], strict=False)
        if result.missing_keys:
            self.logger.info("resume_from: missing keys %s", result.missing_keys)
        if result.unexpected_keys:
            self.logger.info("resume_from: unexpected keys %s", result.unexpected_keys)
        self.method.load_auxiliary_state_dict(ckpt.get("auxiliary"))
        self.global_step = int(ckpt.get("global_step", 0))
        self.resume_task_index = int(ckpt.get("task_index", -1))
        self._load_checkpoint_metrics(ckpt)
        self.logger.info("resume_from: loaded checkpoint %s (task_index=%s, global_step=%s)", path, ckpt.get("task_index"), self.global_step)

    def _load_checkpoint_metrics(self, ckpt: Mapping[str, Any]) -> None:
        raw_records = ckpt.get("eval_records")
        if isinstance(raw_records, list):
            self.eval_records = [dict(record) for record in raw_records if isinstance(record, Mapping)]
        raw_tables = ckpt.get("metric_tables")
        if not isinstance(raw_tables, Mapping):
            return
        n = len(self.scenario.tasks)
        for kind in ("acc", "auc", "ap", "f1"):
            table = raw_tables.get(kind)
            if not isinstance(table, list):
                continue
            matrix = np.full((n, n), np.nan, dtype=float)
            for row_idx, row in enumerate(table[:n]):
                if not isinstance(row, list):
                    continue
                for col_idx, value in enumerate(row[:n]):
                    if value is None:
                        continue
                    matrix[row_idx, col_idx] = float(value)
            setattr(self.metric_matrix, kind, matrix)

    def _completed_task_count(self) -> int:
        if not self.eval_records:
            return 0
        return max(int(record["after_task"]) for record in self.eval_records) + 1

    def _average_forgetting_until(self, kind: str, completed_task_count: int) -> float:
        matrix = self.metric_matrix._matrix(kind)[:completed_task_count, :completed_task_count]
        n = matrix.shape[0]
        vals: list[float] = []
        for j in range(max(n - 1, 0)):
            best_before = np.nanmax(matrix[: n - 1, j])
            final = matrix[n - 1, j]
            if not (np.isnan(best_before) or np.isnan(final)):
                vals.append(float(best_before - final))
        return float(np.mean(vals)) if vals else float("nan")

    def _write_partial_outputs_after_error(self) -> None:
        if not self.eval_records:
            return
        try:
            summary = self._write_outputs(log_tables=False)
        except Exception:
            self.logger.exception("Failed to write partial outputs after run error")
            return
        self.logger.info(
            "Wrote partial outputs after run error: completed_tasks=%d output_dir=%s",
            int(summary["completed_tasks"]),
            self.output_dir,
        )

    def _write_outputs(self, *, log_tables: bool = True) -> dict[str, Any]:
        completed_task_count = self._completed_task_count()
        tables = self.metric_matrix.to_tables()
        output_task_count = len(self.scenario.tasks) if self.eval_scope == "all" else completed_task_count
        output_task_names = [t.name for t in self.scenario.tasks[:output_task_count]]
        output_tables = {
            kind: [row[:output_task_count] for row in tables[kind][:completed_task_count]]
            for kind in ("acc", "auc", "ap", "f1")
        }
        for kind in ("acc", "auc", "ap", "f1"):
            pd.DataFrame(output_tables[kind], columns=output_task_names).to_csv(
                self.output_dir / f"{kind}_matrix.csv",
                index_label="after_task",
            )
        pd.DataFrame(self.eval_records).to_csv(self.output_dir / "eval_details.csv", index=False)
        task_rows = [t.__dict__ for t in self.scenario.tasks]
        pd.DataFrame(task_rows).to_csv(self.output_dir / "task_details.csv", index=False)
        official_weighted_curves = {}
        for kind in ("acc", "auc", "ap", "f1"):
            official_weighted_curves[kind] = [
                self._weighted_records(
                    (
                        record
                        for record in self.eval_records
                        if int(record["after_task"]) == i and int(record["eval_task"]) <= i
                    ),
                    kind,
                )
                for i in range(completed_task_count)
            ]
        future_weighted_curves = {}
        if self.eval_scope == "all":
            for kind in ("acc", "auc", "ap", "f1"):
                future_weighted_curves[kind] = [
                    self._weighted_records(
                        (
                            record
                            for record in self.eval_records
                            if int(record["after_task"]) == i and int(record["eval_task"]) > i
                        ),
                        kind,
                    )
                    for i in range(completed_task_count)
                ]
        pd.DataFrame(
            [
                {"after_task": i, **{kind: official_weighted_curves[kind][i] for kind in ("acc", "auc", "ap", "f1")}}
                for i in range(completed_task_count)
            ]
        ).to_csv(self.output_dir / "official_weighted_curves.csv", index=False)
        if future_weighted_curves:
            pd.DataFrame(
                [
                    {"after_task": i, **{kind: future_weighted_curves[kind][i] for kind in ("acc", "auc", "ap", "f1")}}
                    for i in range(completed_task_count)
                ]
            ).to_csv(self.output_dir / "future_weighted_curves.csv", index=False)
        last_index = completed_task_count - 1
        summary = {
            "tasks": [t.__dict__ for t in self.scenario.tasks],
            "completed_tasks": completed_task_count,
            "eval_scope": self.eval_scope,
            "eval_max_batches_per_task": self.eval_max_batches_per_task,
            "average_accuracy": self._seen_row_average(last_index, "acc") if completed_task_count else float("nan"),
            "average_forgetting": self._average_forgetting_until("acc", completed_task_count) if completed_task_count else float("nan"),
            "average_auc": self._seen_row_average(last_index, "auc") if completed_task_count else float("nan"),
            "auc_forgetting": self._average_forgetting_until("auc", completed_task_count) if completed_task_count else float("nan"),
            "average_ap": self._seen_row_average(last_index, "ap") if completed_task_count else float("nan"),
            "ap_forgetting": self._average_forgetting_until("ap", completed_task_count) if completed_task_count else float("nan"),
            "average_f1": self._seen_row_average(last_index, "f1") if completed_task_count else float("nan"),
            "f1_forgetting": self._average_forgetting_until("f1", completed_task_count) if completed_task_count else float("nan"),
            "official_average_accuracy": float(np.nanmean(official_weighted_curves["acc"])) if completed_task_count else float("nan"),
            "official_last_accuracy": float(official_weighted_curves["acc"][-1]) if official_weighted_curves["acc"] else float("nan"),
            "official_weighted_curves": official_weighted_curves,
            "future_average_accuracy": self._future_row_average(last_index, "acc") if self.eval_scope == "all" and completed_task_count else float("nan"),
            "future_average_auc": self._future_row_average(last_index, "auc") if self.eval_scope == "all" and completed_task_count else float("nan"),
            "future_average_ap": self._future_row_average(last_index, "ap") if self.eval_scope == "all" and completed_task_count else float("nan"),
            "future_average_f1": self._future_row_average(last_index, "f1") if self.eval_scope == "all" and completed_task_count else float("nan"),
            "future_weighted_curves": future_weighted_curves,
            "tables": output_tables,
            "full_tables": tables,
        }
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        if log_tables:
            self._log_summary_tables(summary)
        return summary

    @staticmethod
    def _format_console_metric(value: Any) -> str:
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return ""
        return "" if np.isnan(value_float) else f"{value_float:.4f}"

    @staticmethod
    def _format_console_table(headers: list[str], rows: list[list[Any]], *, right_align: set[str] | None = None) -> str:
        right_align = right_align or set()
        str_rows = [[str(value) for value in row] for row in rows]
        widths = [
            max(len(headers[col]), *(len(row[col]) for row in str_rows)) if str_rows else len(headers[col])
            for col in range(len(headers))
        ]

        def format_row(row: list[str]) -> str:
            cells = []
            for header, value, width in zip(headers, row, widths):
                cells.append(value.rjust(width) if header in right_align else value.ljust(width))
            return "  ".join(cells)

        separator = "  ".join("-" * width for width in widths)
        return "\n".join([format_row(headers), separator, *(format_row(row) for row in str_rows)])

    def _log_eval_console_table(self, records: list[Mapping[str, Any]], payload: Mapping[str, Any]) -> None:
        if not records:
            return
        num_samples = sum(int(record.get("num_samples", 0) or 0) for record in records)
        headers = ["on_task", "task", "n", "acc", "auc", "ap", "f1"]
        rows: list[list[Any]] = [
            [
                int(record["eval_task"]),
                record["eval_task_name"],
                int(record.get("num_samples", 0) or 0),
                self._format_console_metric(record.get("acc")),
                self._format_console_metric(record.get("auc")),
                self._format_console_metric(record.get("ap")),
                self._format_console_metric(record.get("f1")),
            ]
            for record in records
        ]
        rows.extend(
            [
                [
                    "mean",
                    "-",
                    num_samples,
                    self._format_console_metric(payload.get("eval/average_accuracy")),
                    self._format_console_metric(payload.get("eval/average_auc")),
                    self._format_console_metric(payload.get("eval/average_ap")),
                    self._format_console_metric(payload.get("eval/average_f1")),
                ],
                [
                    "weighted",
                    "-",
                    num_samples,
                    self._format_console_metric(payload.get("eval/official_weighted_accuracy")),
                    self._format_console_metric(payload.get("eval/official_weighted_auc")),
                    self._format_console_metric(payload.get("eval/official_weighted_ap")),
                    self._format_console_metric(payload.get("eval/official_weighted_f1")),
                ],
            ]
        )
        self.logger.info(
            "eval after_task=%d task=%s seen_tasks=%d\n%s",
            int(payload.get("eval/after_task", records[-1].get("after_task", 0))),
            format_log_value(records[-1].get("after_task_name", "")),
            len(records),
            self._format_console_table(headers, rows, right_align={"on_task", "n", "acc", "auc", "ap", "f1"}),
        )

    @staticmethod
    def _table_value(value: Any) -> str | int | float:
        if value is None:
            return ""
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return str(value)
        if np.isnan(value_float):
            return ""
        return round(value_float, 6)

    def _log_eval_table(self, records: list[Mapping[str, Any]], step: int | None = None) -> None:
        rows = [
            [
                record["after_task"],
                record["after_task_name"],
                record["eval_task"],
                record["eval_task_name"],
                self._table_value(record["num_samples"]),
                self._table_value(record["acc"]),
                self._table_value(record["ap"]),
                self._table_value(record["f1"]),
            ]
            for record in records
        ]
        self.experiment.log_table(
            "eval/task_metrics",
            ["after_task", "after_task_name", "eval_task", "eval_task_name", "num_samples", "acc", "ap", "f1"],
            rows,
            step=step,
        )

    def _log_summary_tables(self, summary: Mapping[str, Any]) -> None:
        table_width = len(summary["tables"]["acc"][0]) if summary.get("tables", {}).get("acc") else 0
        task_names = [t.name for t in self.scenario.tasks[:table_width]]
        headers = ["after_task"] + task_names
        for kind in ("acc", "auc", "ap", "f1"):
            rows = [
                [row_idx] + [self._table_value(value) for value in row]
                for row_idx, row in enumerate(summary["tables"][kind])
            ]
            self.experiment.log_table(f"summary/{kind}_matrix", headers, rows)

        detail_rows = [
            [
                record["after_task"],
                record["after_task_name"],
                record["eval_task"],
                record["eval_task_name"],
                self._table_value(record["num_samples"]),
                self._table_value(record["acc"]),
                self._table_value(record["auc"]),
                self._table_value(record["ap"]),
                self._table_value(record["f1"]),
                self._table_value(record["ece"]),
            ]
            for record in self.eval_records
        ]
        self.experiment.log_table(
            "summary/eval_details",
            ["after_task", "after_task_name", "eval_task", "eval_task_name", "num_samples", "acc", "auc", "ap", "f1", "ece"],
            detail_rows,
        )

        task_rows = [
            [
                idx,
                task.task_id,
                task.name,
                ", ".join(task.domains),
                ", ".join(task.generators),
                ", ".join(task.scenes),
                task.num_train,
                task.num_val,
                task.num_test,
            ]
            for idx, task in enumerate(self.scenario.tasks)
        ]
        self.experiment.log_table(
            "summary/task_details",
            ["index", "task_id", "name", "domains", "generators", "scenes", "train", "val", "test"],
            task_rows,
        )
