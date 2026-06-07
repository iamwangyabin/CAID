from __future__ import annotations

import json
from typing import Any, Iterable, Mapping
from pathlib import Path

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

import numpy as np
import pandas as pd
import torch

from ..config import load_config
from ..data.loader import build_dataloader
from ..data.scenario import ContinualScenario, TaskSpec
from ..evaluation import ContinualMetricMatrix, summarize_logits
from ..methods.base import build_optimizer
from ..registry import build_method
from ..utils.checkpoint import save_checkpoint
from ..utils.experiment import build_experiment_logger
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from ..utils.tensor import move_to_device


class _ProgressDataLoader:
    def __init__(
        self,
        loader,
        *,
        enabled: bool,
        desc: str,
        leave: bool = False,
        unit: str = "batch",
    ) -> None:
        self._loader = loader
        self._enabled = bool(enabled and tqdm is not None)
        self._desc = desc
        self._leave = leave
        self._unit = unit
        self._passes = 0

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        self._passes += 1
        desc = self._desc if self._passes == 1 else f"{self._desc} | pass {self._passes}"
        return self.iter_with_progress(desc=desc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def iter_with_progress(self, *, desc: str | None = None, leave: bool | None = None, unit: str | None = None):
        iterator = iter(self._loader)
        if not self._enabled:
            return iterator
        return tqdm(
            iterator,
            total=len(self._loader),
            desc=desc or self._desc,
            dynamic_ncols=True,
            leave=self._leave if leave is None else bool(leave),
            unit=unit or self._unit,
        )


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
        self.output_dir = Path(cfg.get("output_dir", "outputs/caidbench_run"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("caidbench")
        device_cfg = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(device_cfg if device_cfg != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        scfg = cfg.get("scenario", {})
        self.scenario = ContinualScenario.from_config(scfg)
        method_cfg = dict(cfg.get("method", {}))
        method_name = method_cfg.pop("name", cfg.get("method_name", "finetune"))
        self.method = build_method(method_name, **method_cfg).to(self.device)
        train_cfg = cfg.get("train", {})
        self.max_epochs = int(train_cfg.get("epochs", 1))
        self.batch_size = int(train_cfg.get("batch_size", 32))
        self.num_workers = int(train_cfg.get("num_workers", 0))
        self.pin_memory = bool(train_cfg.get("pin_memory", self.device.type == "cuda"))
        self.persistent_workers = bool(train_cfg.get("persistent_workers", self.num_workers > 0))
        self.prefetch_factor = train_cfg.get("prefetch_factor", 2 if self.num_workers > 0 else None)
        self.non_blocking = bool(train_cfg.get("non_blocking", self.pin_memory and self.device.type == "cuda"))
        self.drop_last = bool(train_cfg.get("drop_last", False))
        self.train_log_interval = int(train_cfg.get("log_interval", 50))
        if self.train_log_interval <= 0:
            self.train_log_interval = 0
        self.show_tqdm = bool(train_cfg.get("tqdm", True))
        self.grad_clip = train_cfg.get("grad_clip")
        self.grad_clip = None if self.grad_clip is None else float(self.grad_clip)
        self.optimizer_cfg = dict(train_cfg.get("optimizer", {"type": "adamw", "lr": 1e-4}))
        self.metric_matrix = ContinualMetricMatrix([t.name for t in self.scenario.tasks])
        self.eval_records: list[dict[str, Any]] = []
        self.global_step = 0
        self._active_train_task_index: float | None = None
        self.experiment = build_experiment_logger(self.cfg, self.output_dir, str(method_name))

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
        loader = build_dataloader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=effective_drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )
        task = self.scenario.tasks[task_index]
        return _ProgressDataLoader(
            loader,
            enabled=self.show_tqdm,
            desc=f"Task {task.name} | {split}",
            leave=False,
            unit="batch",
        )

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
        self._print_train_metrics(payload, task=task, task_name=task_name, phase=phase, epochs=epochs)
        self.log_metrics(payload, step=step)

    def _print_train_metrics(
        self,
        payload: Mapping[str, Any],
        *,
        task: TaskSpec | Any | None,
        task_name: str | None,
        phase: str | None,
        epochs: int | float | None,
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
            parts.append(f"task={task_name}")
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
        parts.append(f"step={self.global_step}")
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
        use_tqdm = bool(self.show_tqdm and tqdm is not None)
        for epoch in range(self.max_epochs):
            totals: dict[str, float] = {}
            n = 0
            if hasattr(train_loader, "iter_with_progress"):
                bar = train_loader.iter_with_progress(desc=f"Task {task.name} | epoch {epoch + 1}/{self.max_epochs}")
            elif use_tqdm:
                bar = tqdm(
                    train_loader,
                    desc=f"Task {task.name} | epoch {epoch + 1}/{self.max_epochs}",
                    dynamic_ncols=True,
                    leave=False,
                    unit="batch",
                )
            else:
                bar = train_loader
            for batch in bar:
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
                    self.log_train_metrics(metrics, task=task, epoch=epoch + 1, epochs=self.max_epochs, optimizer=optimizer)
                    if hasattr(bar, "set_postfix") and "loss" in metrics:
                        bar.set_postfix(loss=f"{metrics['loss']:.4f}", step=self.global_step, refresh=False)

            if totals:
                metrics = {k: v / max(n, 1) for k, v in totals.items()}
                self.log_train_metrics(metrics, task=task, epoch=epoch + 1, epochs=self.max_epochs, optimizer=optimizer)

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
        for batch in loader:
            batch = move_to_device(batch, self.device, non_blocking=self.non_blocking)
            out = self.method.predict(batch)
            logits_list.append(out["logits"].detach().cpu())
            y_list.append(batch["y"].detach().cpu())
        if not logits_list:
            return {"acc": float("nan"), "auc": float("nan"), "ap": float("nan"), "f1": float("nan"), "ece": float("nan"), "num_samples": 0}
        logits = torch.cat(logits_list, dim=0)
        y = torch.cat(y_list, dim=0)
        return {**summarize_logits(logits, y), "num_samples": int(y.numel())}

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
            for i, task in enumerate(self.scenario.tasks):
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
                for j in range(i + 1):
                    test_loader = self.dataloader(j, "test", shuffle=False)
                    metrics = self.evaluate_loader(test_loader)
                    self.metric_matrix.update(i, j, metrics["acc"], metrics["auc"], metrics["ap"], metrics["f1"])
                    self.logger.info(
                        "eval after_task=%d on_task=%d acc=%.4f auc=%.4f ap=%.4f f1=%.4f",
                        i,
                        j,
                        metrics["acc"],
                        metrics["auc"],
                        metrics["ap"],
                        metrics["f1"],
                    )
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
                eval_payload.update(
                    {
                        "eval/average_accuracy": self.metric_matrix.average_accuracy(train_index=i, kind="acc"),
                        "eval/average_auc": self.metric_matrix.average_accuracy(train_index=i, kind="auc"),
                        "eval/average_ap": self.metric_matrix.average_accuracy(train_index=i, kind="ap"),
                        "eval/average_f1": self.metric_matrix.average_accuracy(train_index=i, kind="f1"),
                        "eval/official_weighted_accuracy": self._weighted_records(eval_rows, "acc"),
                        "eval/official_weighted_auc": self._weighted_records(eval_rows, "auc"),
                        "eval/official_weighted_ap": self._weighted_records(eval_rows, "ap"),
                        "eval/official_weighted_f1": self._weighted_records(eval_rows, "f1"),
                        "eval/after_task": i,
                    }
                )
                self._log_eval_table(eval_rows, step=i)
                self.log_metrics(eval_payload, step=i)
                self._save_intermediate(i)
                self._active_train_task_index = None
            summary = self._write_outputs()
            self.log_metrics(
                {
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
            )
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
        finally:
            self.experiment.finish()

    def _save_intermediate(self, task_index: int) -> None:
        payload = {
            "model": self.method.state_dict(),
            "auxiliary": self.method.auxiliary_state_dict(),
            "cfg": self.cfg,
            "task_index": task_index,
            "global_step": self.global_step,
        }
        save_checkpoint(self.output_dir / f"task_{task_index}.pt", **payload)
        save_checkpoint(self.output_dir / "last.pt", **payload)

    def _write_outputs(self) -> dict[str, Any]:
        tables = self.metric_matrix.to_tables()
        for kind in ("acc", "auc", "ap", "f1"):
            pd.DataFrame(tables[kind], columns=[t.name for t in self.scenario.tasks]).to_csv(self.output_dir / f"{kind}_matrix.csv", index=False)
        official_weighted_curves = {}
        for kind in ("acc", "auc", "ap", "f1"):
            official_weighted_curves[kind] = [
                self._weighted_records((record for record in self.eval_records if int(record["after_task"]) == i), kind)
                for i in range(len(self.scenario.tasks))
            ]
        summary = {
            "tasks": [t.__dict__ for t in self.scenario.tasks],
            "average_accuracy": self.metric_matrix.average_accuracy(kind="acc"),
            "average_forgetting": self.metric_matrix.average_forgetting(kind="acc"),
            "average_auc": self.metric_matrix.average_accuracy(kind="auc"),
            "auc_forgetting": self.metric_matrix.average_forgetting(kind="auc"),
            "average_ap": self.metric_matrix.average_accuracy(kind="ap"),
            "ap_forgetting": self.metric_matrix.average_forgetting(kind="ap"),
            "average_f1": self.metric_matrix.average_accuracy(kind="f1"),
            "f1_forgetting": self.metric_matrix.average_forgetting(kind="f1"),
            "official_average_accuracy": float(np.nanmean(official_weighted_curves["acc"])),
            "official_last_accuracy": float(official_weighted_curves["acc"][-1]) if official_weighted_curves["acc"] else float("nan"),
            "official_weighted_curves": official_weighted_curves,
            "tables": tables,
        }
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self._log_summary_tables(summary)
        return summary

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
        task_names = [t.name for t in self.scenario.tasks]
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
