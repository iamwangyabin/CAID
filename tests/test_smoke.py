from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from caidbench.data.scenario import ContinualScenario
from caidbench.engine import Trainer
from caidbench.models.backbones import build_backbone
from caidbench.registry import list_methods


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


def test_finetune_smoke(tmp_path):
    cfg = base_cfg(tmp_path, "finetune")
    summary = Trainer(cfg).run()
    assert "average_accuracy" in summary
    assert "average_ap" in summary
    assert "average_f1" in summary
    out_dir = Path(cfg["output_dir"])
    assert (out_dir / "ap_matrix.csv").exists()
    assert (out_dir / "f1_matrix.csv").exists()


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
    assert re.fullmatch(r"finetune-out_finetune-\d{8}-\d{6}-\d{3}", calls["init"][0]["experiment_name"])
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
def test_compact_official_dil_methods_smoke(tmp_path, method_name, overrides):
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


def test_yaml_protocol_decouples_task_order_from_storage(tmp_path):
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
