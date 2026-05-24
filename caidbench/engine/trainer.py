from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

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
        self.drop_last = bool(train_cfg.get("drop_last", False))
        self.grad_clip = train_cfg.get("grad_clip")
        self.grad_clip = None if self.grad_clip is None else float(self.grad_clip)
        self.optimizer_cfg = dict(train_cfg.get("optimizer", {"type": "adamw", "lr": 1e-4}))
        self.metric_matrix = ContinualMetricMatrix([t.name for t in self.scenario.tasks])
        self.eval_records: list[dict[str, Any]] = []
        self.global_step = 0
        self.experiment = build_experiment_logger(self.cfg, self.output_dir, str(method_name))

    def make_optimizer(self, params: Iterable[torch.nn.Parameter] | None = None) -> torch.optim.Optimizer:
        return build_optimizer(self.method.parameters() if params is None else params, self.optimizer_cfg)

    def dataloader(self, task_index: int, split: str, shuffle: bool = False):
        ds = self.scenario.task_dataset(split, task_index)
        return build_dataloader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, drop_last=self.drop_last and split == "train")

    def default_train_loop(self, method, task: TaskSpec, train_loader, optimizer: torch.optim.Optimizer | None = None) -> None:
        method.train()
        optimizer = optimizer or method.configure_optimizer(self.optimizer_cfg)
        for epoch in range(self.max_epochs):
            totals: dict[str, float] = {}
            n = 0
            for batch in train_loader:
                batch = move_to_device(batch, self.device)
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
                for k, v in out.items():
                    if k == "logits":
                        continue
                    if torch.is_tensor(v) and v.ndim == 0:
                        totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())
                n += 1
            if totals:
                metrics = {k: v / max(n, 1) for k, v in totals.items()}
                msg = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                self.logger.info("task=%s epoch=%d/%d %s", task.name, epoch + 1, self.max_epochs, msg)
                self.log_metrics(
                    {
                        **{f"train/{k}": v for k, v in metrics.items()},
                        "train/task_index": float(getattr(task, "task_id", 0)),
                        "train/epoch": epoch + 1,
                    }
                )

    def advance_step(self, n: int = 1) -> None:
        self.global_step += int(n)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        self.experiment.log(metrics, step=self.global_step if step is None else step)

    @torch.no_grad()
    def evaluate_loader(self, loader) -> dict[str, float]:
        self.method.eval()
        logits_list: list[torch.Tensor] = []
        y_list: list[torch.Tensor] = []
        for batch in loader:
            batch = move_to_device(batch, self.device)
            out = self.method.predict(batch)
            logits_list.append(out["logits"].detach().cpu())
            y_list.append(batch["y"].detach().cpu())
        if not logits_list:
            return {"acc": float("nan"), "auc": float("nan"), "ap": float("nan"), "f1": float("nan"), "ece": float("nan")}
        logits = torch.cat(logits_list, dim=0)
        y = torch.cat(y_list, dim=0)
        return summarize_logits(logits, y)

    def run(self) -> dict[str, Any]:
        try:
            self.logger.info("Starting CAIDBench run on %s with method=%s", self.device, self.method.__class__.__name__)
            for i, task in enumerate(self.scenario.tasks):
                self.logger.info("=== Task %d/%d: %s train=%d test=%d ===", i + 1, len(self.scenario.tasks), task.name, task.num_train, task.num_test)
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
                    }
                    self.eval_records.append(record)
                    eval_rows.append(record)
                eval_payload.update(
                    {
                        "eval/average_accuracy": self.metric_matrix.average_accuracy(train_index=i, kind="acc"),
                        "eval/average_auc": self.metric_matrix.average_accuracy(train_index=i, kind="auc"),
                        "eval/average_ap": self.metric_matrix.average_accuracy(train_index=i, kind="ap"),
                        "eval/average_f1": self.metric_matrix.average_accuracy(train_index=i, kind="f1"),
                        "eval/after_task": i,
                    }
                )
                self._log_eval_table(eval_rows, step=i)
                self.log_metrics(eval_payload, step=i)
                self._save_intermediate(i)
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
                }
            )
            self.logger.info(
                "Finished: AA=%.4f AF=%.4f AUC_AA=%.4f AP_AA=%.4f F1_AA=%.4f",
                summary["average_accuracy"],
                summary["average_forgetting"],
                summary["average_auc"],
                summary["average_ap"],
                summary["average_f1"],
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
                self._table_value(record["acc"]),
                self._table_value(record["ap"]),
                self._table_value(record["f1"]),
            ]
            for record in records
        ]
        self.experiment.log_table(
            "eval/task_metrics",
            ["after_task", "after_task_name", "eval_task", "eval_task_name", "acc", "ap", "f1"],
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
            ["after_task", "after_task_name", "eval_task", "eval_task_name", "acc", "auc", "ap", "f1", "ece"],
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
