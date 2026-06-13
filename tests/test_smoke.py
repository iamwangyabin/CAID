from __future__ import annotations

import copy
import importlib.util
import logging
import json
import re
import sys
import types
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch

import caidbench.engine.trainer as trainer_mod
from caidbench.data.scenario import ContinualScenario
from caidbench.engine import Trainer
from caidbench.engine.trainer import _format_duration
from caidbench.methods.base import effective_train_batches, iter_limited_train_batches
from caidbench.models.backbones import build_backbone
from caidbench.registry import list_methods
from caidbench.utils.checkpoint import save_checkpoint
from caidbench.utils.logging import get_logger


def write_aid_arrow_dataset(root: Path, rows: list[dict], payload_column: str) -> Path:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    root.mkdir(parents=True, exist_ok=True)
    payload = []
    for row in rows:
        value = row[payload_column]
        payload.append(value.tolist() if isinstance(value, np.ndarray) else value)
    table = pa.Table.from_pydict({payload_column: payload})
    with pa.OSFile(str(root / "data.arrow"), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    mapping = {str(row["path"]): i for i, row in enumerate(rows)}
    with open(root / "mapping.json", "w", encoding="utf-8") as fp:
        json.dump(mapping, fp)

    split_payloads: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        split = str(row["split"])
        path = str(row["path"])
        label = int(row["label"])
        payloads = split_payloads.setdefault(split, {})
        subsets = ["all", "real" if label == 0 else "fake", row.get("domain"), row.get("generator"), row.get("task_hint")]
        for subset in subsets:
            if subset is not None and str(subset) not in {"", "unknown"}:
                payloads.setdefault(str(subset), {})[path] = label
    for split, payloads in split_payloads.items():
        with open(root / f"{split}.json", "w", encoding="utf-8") as fp:
            json.dump(payloads, fp)

    with open(root / "caid_meta.jsonl", "w", encoding="utf-8") as fp:
        for row in rows:
            meta = {key: value for key, value in row.items() if key != payload_column}
            fp.write(json.dumps(meta) + "\n")
    return root


def task_protocol(tasks: int) -> dict:
    return {
        "tasks": [
            {"id": f"task{task}", "name": f"task{task}", "numeric_id": task, "filter": {"include": {"task_hint": f"task{task}"}}}
            for task in range(tasks)
        ]
    }


def image_transform(size: int) -> dict:
    return {
        "trsf": [
            {"_target_": "caidbench.data.transforms.SquareResize", "size": size},
            {"_target_": "caidbench.data.transforms.ToTensor"},
            {
                "_target_": "caidbench.data.transforms.Normalize",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        ]
    }


def base_cfg(tmp_path: Path, method: str) -> dict:
    data_root = make_image_arrow(tmp_path, tasks=2)
    return {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / f"out_{method}"),
        "logging": {"backend": "none"},
        "scenario": {
            "data": {"backend": "aid_arrow", "path": str(data_root), "image_column": "image"},
            "protocol": task_protocol(2),
            "transform": image_transform(16),
        },
        "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "optimizer": {"type": "adamw", "lr": 1e-3}},
        "method": {
            "name": method,
            "num_classes": 2,
            "memory_size": 2,
            "memory_batch_size": 1,
            "uap_shape": [3, 16, 16],
            "detector_cfg": {"num_classes": 2, "backbone": {"type": "small_conv", "out_dim": 8}},
        },
    }


def test_registry_contains_methods():
    assert {
        "e3",
        "dfil",
        "hsic_bottleneck",
        "saido",
        "prompt2guard",
        "sprompts",
        "s-prompts",
        "ranpac",
        "layup",
        "pina",
        "pina-d",
        "cp-prompt",
        "duct",
        "soyo",
        "loranpac",
        "dce",
        "hdp",
        "sur_lid",
    }.issubset(set(list_methods()))


def test_train_eta_duration_formatting():
    assert _format_duration(9.4) == "9s"
    assert _format_duration(65) == "1m05s"
    assert _format_duration(3661) == "1h01m01s"


def test_train_logging_uses_step_instead_of_batch(caplog):
    trainer = Trainer.__new__(Trainer)
    trainer.logger = logging.getLogger("caidbench.test.train_log")
    trainer.logger.setLevel(logging.INFO)
    trainer.global_step = 123
    trainer._active_train_task_index = None

    payload = {
        "train/loss": 0.9272,
        "train/acc": 0.5722,
        "train/lr": 1e-5,
    }

    with caplog.at_level(logging.INFO, logger="caidbench.test.train_log"):
        trainer._print_train_metrics(
            payload,
            task=None,
            task_name="ProGAN",
            phase=None,
            epochs=20,
            batch_idx=50,
            num_batches=11252,
            started_at=None,
        )

    msg = caplog.records[-1].getMessage()
    assert "batch=" not in msg
    assert msg.count("step=") == 1
    assert "step=50/11252" in msg
    assert "step=123" not in msg


def test_default_logger_does_not_propagate_to_root():
    logger = get_logger("caidbench.test.no_propagate")
    assert logger.propagate is False


def test_eval_console_logging_uses_table(caplog):
    trainer = Trainer.__new__(Trainer)
    trainer.logger = logging.getLogger("caidbench.test.eval_table")
    trainer.logger.setLevel(logging.INFO)
    records = [
        {
            "after_task": 1,
            "after_task_name": "task 1",
            "eval_task": 0,
            "eval_task_name": "task 0",
            "num_samples": 2,
            "acc": 0.6,
            "auc": 0.7,
            "ap": 0.65,
            "f1": 0.55,
        },
        {
            "after_task": 1,
            "after_task_name": "task 1",
            "eval_task": 1,
            "eval_task_name": "task 1",
            "num_samples": 3,
            "acc": 0.8,
            "auc": 0.9,
            "ap": 0.85,
            "f1": 0.75,
        },
    ]
    payload = {
        "eval/after_task": 1,
        "eval/average_accuracy": 0.7,
        "eval/average_auc": 0.8,
        "eval/average_ap": 0.75,
        "eval/average_f1": 0.65,
        "eval/official_weighted_accuracy": 0.72,
        "eval/official_weighted_auc": 0.81,
        "eval/official_weighted_ap": 0.76,
        "eval/official_weighted_f1": 0.66,
    }

    with caplog.at_level(logging.INFO, logger="caidbench.test.eval_table"):
        trainer._log_eval_console_table(records, payload)

    assert len(caplog.records) == 1
    msg = caplog.records[-1].getMessage()
    assert 'task="task 1"' in msg
    assert "seen_tasks=2" in msg
    assert "on_task  task" in msg
    assert "task 0" in msg
    assert "0.6000" in msg
    assert "mean" in msg
    assert "weighted" in msg
    assert "on_task=" not in msg


def test_finetune_smoke(tmp_path):
    cfg = base_cfg(tmp_path, "finetune")
    trainer = Trainer(cfg)
    summary = trainer.run()
    assert "average_accuracy" in summary
    assert "average_ap" in summary
    assert "average_f1" in summary
    assert summary["completed_tasks"] == 2
    out_dir = trainer.output_dir
    assert (out_dir / "ap_matrix.csv").exists()
    assert (out_dir / "f1_matrix.csv").exists()
    assert (out_dir / "eval_details.csv").exists()
    assert (out_dir / "official_weighted_curves.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "train.log").exists()
    assert (out_dir / "base.pt").exists()
    assert (out_dir / "last.pt").exists()
    assert (out_dir / "task_0.pt").exists()
    assert (out_dir / "task_1.pt").exists()
    with open(out_dir / "f1_matrix.csv", "r", encoding="utf-8") as fp:
        rows = [line.strip() for line in fp if line.strip()]
    assert rows[0] == "after_task,task0,task1"
    assert len(rows) == 3


def test_per_task_checkpoints_can_be_disabled(tmp_path):
    cfg = base_cfg(tmp_path, "finetune")
    cfg["checkpoint"] = {"save_each_task": False}
    trainer = Trainer(cfg)
    trainer.run()

    assert not list(trainer.output_dir.glob("task_*.pt"))
    assert (trainer.output_dir / "base.pt").exists()
    assert (trainer.output_dir / "last.pt").exists()


def test_checkpoints_are_weights_only_loadable(tmp_path):
    cfg = base_cfg(tmp_path, "finetune")
    trainer = Trainer(cfg)
    trainer.run()

    checkpoint = torch.load(trainer.output_dir / "last.pt", map_location="cpu", weights_only=True)
    assert isinstance(checkpoint["model"], dict)
    assert checkpoint["checkpoint_version"] == 1
    assert checkpoint["completed_tasks"] == 2
    assert checkpoint["metric_tables"]["acc"][0][0] is not None

    class UnsafeObject:
        pass

    with pytest.raises(TypeError, match="unsupported object"):
        save_checkpoint(tmp_path / "bad.pt", model=UnsafeObject())


def test_resume_from_checkpoint_continues_after_completed_task(tmp_path):
    cfg = base_cfg(tmp_path, "finetune")
    first = Trainer(cfg)
    first.run()

    resumed_cfg = copy.deepcopy(cfg)
    resumed_cfg["output_dir"] = str(tmp_path / "out_resume")
    resumed_cfg["resume_from"] = str(first.output_dir / "base.pt")
    resumed = Trainer(resumed_cfg)
    summary = resumed.run()

    assert resumed.resume_task_index == 0
    assert resumed.global_step == 2
    assert summary["completed_tasks"] == 2
    assert len(resumed.eval_records) == 3
    assert resumed.eval_records[0]["after_task"] == 0
    assert resumed.eval_records[-1]["after_task"] == 1


def test_partial_outputs_are_written_after_run_error(tmp_path, monkeypatch):
    cfg = base_cfg(tmp_path, "finetune")
    trainer = Trainer(cfg)

    def fail_save_intermediate(task_index: int) -> None:
        raise RuntimeError(f"checkpoint failed at task {task_index}")

    monkeypatch.setattr(trainer, "_save_intermediate", fail_save_intermediate)
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        trainer.run()

    with open(trainer.output_dir / "summary.json", "r", encoding="utf-8") as fp:
        summary = json.load(fp)
    assert summary["completed_tasks"] == 1
    assert (trainer.output_dir / "acc_matrix.csv").exists()
    assert (trainer.output_dir / "eval_details.csv").exists()


def test_eval_dataloaders_do_not_use_persistent_workers(tmp_path, monkeypatch):
    cfg = base_cfg(tmp_path, "finetune")
    cfg["train"]["num_workers"] = 2
    cfg["train"]["persistent_workers"] = True
    trainer = Trainer(cfg)
    calls = []

    def fake_build_dataloader(dataset, **kwargs):
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr(trainer_mod, "build_dataloader", fake_build_dataloader)

    trainer.dataloader(0, "train", shuffle=True)
    trainer.dataloader(0, "test", shuffle=False)
    trainer.dataloader(0, "val", shuffle=False)

    assert [call["num_workers"] for call in calls] == [2, 0, 0]
    assert [call["persistent_workers"] for call in calls] == [True, False, False]


def test_eval_num_workers_override_enables_eval_workers(tmp_path, monkeypatch):
    cfg = base_cfg(tmp_path, "finetune")
    cfg["train"]["num_workers"] = 2
    cfg["train"]["eval_num_workers"] = 1
    cfg["train"]["persistent_workers"] = True
    trainer = Trainer(cfg)
    calls = []

    def fake_build_dataloader(dataset, **kwargs):
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr(trainer_mod, "build_dataloader", fake_build_dataloader)

    trainer.dataloader(0, "train", shuffle=True)
    trainer.dataloader(0, "test", shuffle=False)
    trainer.dataloader(0, "val", shuffle=False)

    assert [call["num_workers"] for call in calls] == [2, 1, 1]
    assert [call["persistent_workers"] for call in calls] == [True, False, False]


def test_caidbench_arrow_dataset_close_releases_reader_cache():
    from caidbench.data.caidbench_arrow import CAIDBenchArrowImageDataset

    class FakeClosable:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    source = FakeClosable()
    reader = FakeClosable()
    dataset = CAIDBenchArrowImageDataset.__new__(CAIDBenchArrowImageDataset)
    dataset._reader_cache = {0: (source, reader)}

    dataset.close()

    assert source.closed == 1
    assert reader.closed == 1
    assert dataset._reader_cache == {}
    dataset.close()


def test_evaluate_loader_closes_dataset_after_use():
    class FakeMethod:
        def eval(self):
            return None

    class FakeDataset:
        def __init__(self):
            self.closes = 0

        def close(self):
            self.closes += 1

    class FakeLoader:
        def __init__(self):
            self.dataset = FakeDataset()

        def __iter__(self):
            return iter(())

    trainer = Trainer.__new__(Trainer)
    trainer.method = FakeMethod()
    trainer.eval_max_batches_per_task = None

    loader = FakeLoader()
    trainer.evaluate_loader(loader)

    assert loader.dataset.closes == 1


def test_eval_scope_all_populates_future_task_columns(tmp_path):
    cfg = base_cfg(tmp_path, "finetune")
    cfg["eval"] = {"scope": "all"}
    trainer = Trainer(cfg)
    summary = trainer.run()

    assert len(trainer.eval_records) == 4
    assert not np.isnan(trainer.metric_matrix.f1[0, 1])
    assert summary["eval_scope"] == "all"
    assert summary["tables"]["f1"][0][1] is not None
    assert "future_weighted_curves" in summary
    with open(trainer.output_dir / "f1_matrix.csv", "r", encoding="utf-8") as fp:
        rows = [line.strip() for line in fp if line.strip()]
    assert rows[0] == "after_task,task0,task1"
    assert rows[1].split(",")[2] != ""


def test_eval_scope_current_only_populates_diagonal(tmp_path):
    cfg = base_cfg(tmp_path, "finetune")
    cfg["eval"] = {"scope": "current"}
    trainer = Trainer(cfg)
    trainer.run()

    assert len(trainer.eval_records) == 2
    assert np.isnan(trainer.metric_matrix.acc[1, 0])
    assert not np.isnan(trainer.metric_matrix.acc[1, 1])


def test_eval_max_batches_per_task_limits_eval_samples(tmp_path):
    data_root = make_protocol_arrow(tmp_path)
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "out_eval_limit"),
        "logging": {"backend": "none"},
        "scenario": {
            "data": {"backend": "aid_arrow", "path": str(data_root), "image_column": "image"},
            "protocol": {
                "tasks": [{"id": "d0", "name": "D0", "numeric_id": 0, "filter": {"include": {"domain": "d0"}}}],
            },
            "transform": image_transform(16),
        },
        "train": {"epochs": 1, "batch_size": 2, "num_workers": 0, "optimizer": {"type": "adamw", "lr": 1e-3}},
        "method": {
            "name": "finetune",
            "num_classes": 2,
            "detector_cfg": {"num_classes": 2, "backbone": {"type": "small_conv", "out_dim": 8}},
        },
    }
    cfg["eval"] = {"max_batches_per_task": 1}
    trainer = Trainer(cfg)
    trainer.run()

    assert trainer.scenario.tasks[0].num_test == 4
    assert {record["num_samples"] for record in trainer.eval_records} == {2}


def test_preextracted_feature_interfaces_are_rejected(tmp_path):
    data_root = make_image_arrow(tmp_path)
    with pytest.raises(ValueError, match="Pre-extracted feature columns"):
        ContinualScenario.from_config(
            {
                "data": {"backend": "aid_arrow", "path": str(data_root), "feature_column": "feature"},
                "protocol": task_protocol(1),
            }
        )

    with pytest.raises(KeyError):
        build_backbone({"type": "identity", "in_dim": 16})


def test_default_logging_uses_swanlab(tmp_path, monkeypatch):
    calls = {"init": [], "log": [], "finish": 0}

    class FakeRun:
        def finish(self):
            calls["finish"] += 1

    class FakeTable:
        def __init__(self):
            self.headers = None
            self.rows = None

        def add(self, headers, rows):
            self.headers = headers
            self.rows = rows

    fake_swanlab = types.SimpleNamespace(
        init=lambda **kwargs: calls["init"].append(kwargs) or FakeRun(),
        log=lambda data, step=None: calls["log"].append((data, step)),
        finish=lambda: calls.__setitem__("finish", calls["finish"] + 1),
        echarts=types.SimpleNamespace(Table=FakeTable),
    )
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)

    cfg = base_cfg(tmp_path, "finetune")
    cfg.pop("logging")
    summary = Trainer(cfg).run()

    assert "average_accuracy" in summary
    assert calls["init"]
    assert calls["init"][0]["project"] == "CAIDBench"
    assert calls["init"][0]["mode"] == "cloud"
    assert re.fullmatch(r"finetune-images_aid-\d{8}-\d{6}-\d{3}", calls["init"][0]["experiment_name"])
    assert not any("run/started" in data for data, _ in calls["log"])
    assert not any(any(key.startswith("task/") for key in data) for data, _ in calls["log"])
    assert not any(any(key.startswith("eval/task_") and key.count("/") > 1 for key in data) for data, _ in calls["log"])
    assert any("summary/average_accuracy" in data for data, _ in calls["log"])
    eval_curves = [
        (
            data["eval/after_task"],
            step,
        )
        for data, step in calls["log"]
        if "eval/average_accuracy" in data
    ]
    assert eval_curves == [
        (0, 0),
        (1, 1),
    ]
    table_logs = {next(iter(data)): next(iter(data.values())) for data, _ in calls["log"] if len(data) == 1}
    assert table_logs["eval/task_metrics"].headers == ["after_task", "after_task_name", "eval_task", "eval_task_name", "num_samples", "acc", "ap", "f1"]
    assert len(table_logs["eval/task_metrics"].rows) == 2
    assert [row[4] for row in table_logs["eval/task_metrics"].rows] == [2, 2]
    assert table_logs["summary/acc_matrix"].headers == ["after_task", "task0", "task1"]
    assert len(table_logs["summary/acc_matrix"].rows) == 2
    assert table_logs["summary/acc_matrix"].rows[0][0] == 0
    assert table_logs["summary/acc_matrix"].rows[1][0] == 1
    assert table_logs["summary/auc_matrix"].headers == ["after_task", "task0", "task1"]
    assert table_logs["summary/ap_matrix"].headers == ["after_task", "task0", "task1"]
    assert table_logs["summary/f1_matrix"].headers == ["after_task", "task0", "task1"]
    assert table_logs["summary/eval_details"].headers == [
        "after_task",
        "after_task_name",
        "eval_task",
        "eval_task_name",
        "num_samples",
        "acc",
        "auc",
        "ap",
        "f1",
        "ece",
    ]
    assert len(table_logs["summary/eval_details"].rows) == 3
    assert [row[4] for row in table_logs["summary/eval_details"].rows] == [2, 2, 2]
    assert table_logs["summary/task_details"].headers == [
        "index",
        "task_id",
        "name",
        "domains",
        "generators",
        "scenes",
        "train",
        "val",
        "test",
    ]
    assert calls["finish"] == 1


def test_sprompts_smoke(tmp_path):
    if importlib.util.find_spec("timm") is None:
        return
    cfg = base_cfg(tmp_path, "sprompts")
    cfg["method"].pop("detector_cfg", None)
    cfg["method"]["net_type"] = "sip"
    cfg["method"]["prompt_length"] = 2
    cfg["method"]["num_centers"] = 2
    cfg["method"]["backbone"] = {"type": "timm", "name": "vit_tiny_patch16_224", "pretrained": False, "img_size": 16}
    cfg["method"]["use_official_schedule"] = True
    cfg["method"]["init_epoch"] = 1
    cfg["method"]["init_milestones"] = [1]
    cfg["method"]["init_lr_decay"] = 0.5
    cfg["method"]["epochs"] = 1
    cfg["method"]["milestones"] = [1]
    cfg["method"]["lrate_decay"] = 0.5
    summary = Trainer(cfg).run()
    assert "average_accuracy" in summary


def test_sprompts_prompt_token_sip_smoke(tmp_path):
    if importlib.util.find_spec("timm") is None:
        return
    data_root = make_image_arrow(tmp_path)
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "out_sprompts_sip"),
        "logging": {"backend": "none"},
        "scenario": {
            "data": {"backend": "aid_arrow", "path": str(data_root), "image_column": "image"},
            "protocol": task_protocol(1),
            "transform": image_transform(16),
        },
        "train": {"epochs": 1, "batch_size": 2, "num_workers": 0, "optimizer": {"type": "sgd", "lr": 1e-2}},
        "method": {
            "name": "sprompts",
            "num_classes": 2,
            "net_type": "sip",
            "prompt_length": 2,
            "num_centers": 2,
            "backbone": {"type": "timm", "name": "vit_tiny_patch16_224", "pretrained": False, "img_size": 16},
        },
    }
    summary = Trainer(cfg).run()
    assert "average_accuracy" in summary


def test_soyo_official_vit_smoke(tmp_path):
    if importlib.util.find_spec("timm") is None:
        return
    cfg = base_cfg(tmp_path, "soyo")
    cfg["method"].pop("detector_cfg", None)
    cfg["train"]["optimizer"] = {"type": "sgd", "lr": 1e-2, "momentum": 0.9}
    cfg["method"].update(
        {
            "implementation": "official",
            "net_type": "soyo_vit",
            "total_sessions": 2,
            "prompt_length": 2,
            "hidden_dim": 4,
            "gmm_components": 1,
            "soyo_epoch": 1,
            "soyo_lr": 1e-2,
            "init_epoch": 1,
            "epochs": 1,
            "backbone": {"type": "timm", "name": "vit_tiny_patch16_224", "pretrained": False, "img_size": 16, "out_dim": 8},
        }
    )

    summary = Trainer(cfg).run()
    assert "average_accuracy" in summary


def make_image_arrow(root: Path, tasks: int = 1) -> Path:
    from PIL import Image
    rows = []
    rng = np.random.default_rng(1)
    for task in range(tasks):
        for split, n in [("train", 2), ("test", 2)]:
            for i in range(n):
                y = i % 2
                base = 180 if y else 60
                arr = np.clip(base + task * 10 + rng.normal(0, 10, size=(16, 16, 3)), 0, 255).astype("uint8")
                buf = BytesIO()
                Image.fromarray(arr).save(buf, format="PNG")
                rows.append(
                    {
                        "path": f"img_t{task}_{split}_{i}.png",
                        "image": buf.getvalue(),
                        "label": y,
                        "split": split,
                        "task_id": task,
                        "task_hint": f"task{task}",
                        "domain": f"d{task}",
                        "generator": f"g{task}",
                        "scene": "s",
                    }
                )
    return write_aid_arrow_dataset(root / "images_aid", rows, "image")


def test_hsic_online_image_smoke(tmp_path):
    data_root = make_image_arrow(tmp_path)
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "out_hsic_online"),
        "logging": {"backend": "none"},
        "scenario": {
            "data": {"backend": "aid_arrow", "path": str(data_root), "image_column": "image"},
            "protocol": task_protocol(1),
            "transform": image_transform(16),
        },
        "train": {"epochs": 1, "batch_size": 2, "num_workers": 0, "optimizer": {"type": "adamw", "lr": 1e-3}},
        "method": {
            "name": "hsic_bottleneck",
            "num_classes": 2,
            "hsic_weight": 0.1,
            "memory_size": 2,
            "memory_batch_size": 1,
            "detector_cfg": {"num_classes": 2, "backbone": {"type": "small_conv", "out_dim": 8}},
        },
    }
    summary = Trainer(cfg).run()
    assert "average_accuracy" in summary


def test_hsic_official_online_image_smoke(tmp_path):
    data_root = make_image_arrow(tmp_path)
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "out_hsic_official_online"),
        "logging": {"backend": "none"},
        "scenario": {
            "data": {"backend": "aid_arrow", "path": str(data_root), "image_column": "image"},
            "protocol": task_protocol(1),
            "transform": image_transform(16),
        },
        "train": {"epochs": 1, "batch_size": 2, "num_workers": 0, "optimizer": {"type": "sgd", "lr": 1e-4}},
        "method": {
            "name": "hsic_bottleneck",
            "objective": "official",
            "num_classes": 2,
            "lambda_x": 1.0,
            "lambda_y": 1.0,
            "bottleneck_dim": 4,
            "hgr_keep_frac": 0.5,
            "memory_size": 4,
            "memory_batch_size": 1,
            "detector_cfg": {"num_classes": 2, "backbone": {"type": "small_conv", "out_dim": 8}},
        },
    }
    summary = Trainer(cfg).run()
    assert "average_accuracy" in summary


@pytest.mark.parametrize(
    ("method_name", "overrides"),
    [
        ("pina", {"init_epoch": 1, "epochs": 1, "num_centers": 1, "hidden_dim": 4}),
        ("cp_prompt", {"init_epoch": 1, "epochs": 1, "num_centers": 1, "hidden_dim": 4}),
        ("duct", {"lrate": 1e-2, "epc_re": 1, "retrain_epochs": 1, "increment": 2, "total_sessions": 2}),
        ("soyo", {"init_epoch": 1, "epochs": 1, "soyo_epoch": 1, "gmm_components": 1, "resample_per_domain": 2, "selector_batch_size": 2}),
        ("loranpac", {"E": 8, "rank": 4, "tsvd_batch_size": 2}),
        ("dce", {"init_epoch": 1, "epochs": 1, "bal_epoch": 1, "selector_epoch": 1, "num_sampled_pcls": 2}),
    ],
)
def test_compact_domain_incremental_methods_smoke(tmp_path, method_name, overrides):
    cfg = base_cfg(tmp_path, method_name)
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 2
    cfg["method"].update(overrides)

    summary = Trainer(cfg).run()

    assert "average_accuracy" in summary



def make_protocol_arrow(root: Path) -> Path:
    from PIL import Image
    rows = []
    rng = np.random.default_rng(2)
    for domain in ["d0", "d1"]:
        for split, n in [("train", 6), ("test", 4)]:
            for i in range(n):
                y = i % 2
                base = 180 if y else 60
                arr = np.clip(base + (20 if domain == "d1" else 0) + rng.normal(0, 10, size=(16, 16, 3)), 0, 255).astype("uint8")
                buf = BytesIO()
                Image.fromarray(arr).save(buf, format="PNG")
                rows.append(
                    {
                        "path": f"proto_{domain}_{split}_{i}.png",
                        "image": buf.getvalue(),
                        "label": y,
                        "split": split,
                        "task_hint": domain,
                        "domain": domain,
                        "generator": domain,
                        "scene": "s",
                    }
                )
    return write_aid_arrow_dataset(root / "protocol_aid", rows, "image")


def test_debug_max_steps_per_epoch_limits_default_train_loop(tmp_path):
    data_root = make_protocol_arrow(tmp_path)
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "out_debug_steps"),
        "logging": {"backend": "none"},
        "scenario": {
            "data": {"backend": "aid_arrow", "path": str(data_root), "image_column": "image"},
            "protocol": {
                "tasks": [{"id": "d0", "name": "D0", "numeric_id": 0, "filter": {"include": {"domain": "d0"}}}],
            },
            "transform": image_transform(16),
        },
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "debug_max_steps_per_epoch": 2,
            "optimizer": {"type": "adamw", "lr": 1e-3},
        },
        "method": {
            "name": "finetune",
            "num_classes": 2,
            "detector_cfg": {"num_classes": 2, "backbone": {"type": "small_conv", "out_dim": 8}},
        },
    }

    trainer = Trainer(cfg)
    assert trainer.scenario.tasks[0].num_train == 6
    trainer.run()

    assert trainer.global_step == 2


def test_runtime_debug_limit_controls_method_side_loader_helpers():
    owner = types.SimpleNamespace(_runtime_debug_max_steps_per_epoch=2)
    loader = ["a", "b", "c", "d"]

    assert effective_train_batches(owner, loader) == 2
    assert list(iter_limited_train_batches(owner, loader)) == [(1, "a"), (2, "b")]


def test_yaml_protocol_decouples_task_sequence_from_storage(tmp_path):
    data_root = make_protocol_arrow(tmp_path)
    protocol = {
        "name": "two_domain_protocol",
        "tasks": [
            {"id": "second_first", "name": "D1 first", "numeric_id": 0, "filter": {"include": {"domain": "d1"}}},
            {"id": "first_second", "name": "D0 second", "numeric_id": 1, "filter": {"include": {"domain": "d0"}}},
        ],
    }
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "out_protocol"),
        "logging": {"backend": "none"},
        "scenario": {
            "data": {"backend": "aid_arrow", "path": str(data_root), "image_column": "image"},
            "protocol": protocol,
            "transform": image_transform(16),
        },
        "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "optimizer": {"type": "adamw", "lr": 1e-3}},
        "method": {
            "name": "finetune",
            "num_classes": 2,
            "detector_cfg": {"num_classes": 2, "backbone": {"type": "small_conv", "out_dim": 8}},
        },
    }
    tr = Trainer(cfg)
    assert [t.name for t in tr.scenario.tasks] == ["D1 first", "D0 second"]
    assert [t.num_train for t in tr.scenario.tasks] == [6, 6]
    summary = tr.run()
    assert "average_accuracy" in summary
