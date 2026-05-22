from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np

from caidbench.engine import Trainer
from caidbench.registry import list_methods


def make_manifest(root: Path, dim: int = 16, tasks: int = 2) -> Path:
    rows = []
    rng = np.random.default_rng(0)
    for task in range(tasks):
        for split, n in [("train", 8), ("test", 4)]:
            for i in range(n):
                y = i % 2
                x = rng.normal(task + (2 if y else -2), 1, size=(dim,)).astype("float32")
                p = root / f"t{task}_{split}_{i}.npy"
                np.save(p, x)
                rows.append({"path": p.name, "label": y, "split": split, "task_id": task, "domain": f"d{task}", "generator": f"g{task}", "scene": "s"})
    manifest = root / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["path", "label", "split", "task_id", "domain", "generator", "scene"])
        w.writeheader(); w.writerows(rows)
    return manifest


def base_cfg(tmp_path: Path, method: str) -> dict:
    manifest = make_manifest(tmp_path)
    return {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / f"out_{method}"),
        "scenario": {"data": {"backend": "manifest", "path": str(manifest), "root": str(tmp_path)}},
        "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "optimizer": {"type": "adamw", "lr": 1e-3}},
        "method": {
            "name": method,
            "num_classes": 2,
            "memory_size": 2,
            "memory_batch_size": 1,
            "uap_shape": [16],
            "detector_cfg": {"num_classes": 2, "backbone": {"type": "identity", "in_dim": 16}},
        },
    }


def test_registry_contains_methods():
    assert {"e3", "dfil", "hsic_bottleneck", "saido", "prompt2guard", "sprompts", "sur_lid"}.issubset(set(list_methods()))


def test_finetune_smoke(tmp_path):
    summary = Trainer(base_cfg(tmp_path, "finetune")).run()
    assert "average_accuracy" in summary


def test_sprompts_smoke(tmp_path):
    cfg = base_cfg(tmp_path, "sprompts")
    cfg["method"]["prompt_length"] = 2
    cfg["method"]["num_centers"] = 2
    cfg["method"]["head_type"] = "linear"
    summary = Trainer(cfg).run()
    assert "average_accuracy" in summary


def test_sprompts_prompt_token_sip_smoke(tmp_path):
    if importlib.util.find_spec("timm") is None:
        return
    manifest = make_image_manifest(tmp_path)
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "out_sprompts_sip"),
        "scenario": {"data": {"backend": "manifest", "path": str(manifest), "root": str(tmp_path)}, "transform": {"size": 16}},
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


def make_image_manifest(root: Path, tasks: int = 1) -> Path:
    from PIL import Image
    rows = []
    rng = np.random.default_rng(1)
    for task in range(tasks):
        for split, n in [("train", 2), ("test", 2)]:
            for i in range(n):
                y = i % 2
                base = 180 if y else 60
                arr = np.clip(base + task * 10 + rng.normal(0, 10, size=(16, 16, 3)), 0, 255).astype("uint8")
                p = root / f"img_t{task}_{split}_{i}.png"
                Image.fromarray(arr).save(p)
                rows.append({"path": p.name, "label": y, "split": split, "task_id": task, "domain": f"d{task}", "generator": f"g{task}", "scene": "s"})
    manifest = root / "image_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["path", "label", "split", "task_id", "domain", "generator", "scene"])
        w.writeheader(); w.writerows(rows)
    return manifest


def test_hsic_online_image_smoke(tmp_path):
    manifest = make_image_manifest(tmp_path)
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "out_hsic_online"),
        "scenario": {"data": {"backend": "manifest", "path": str(manifest), "root": str(tmp_path)}, "transform": {"size": 16}},
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



def make_protocol_manifest(root: Path, dim: int = 16) -> Path:
    rows = []
    rng = np.random.default_rng(2)
    for domain in ["d0", "d1"]:
        for split, n in [("train", 6), ("test", 4)]:
            for i in range(n):
                y = i % 2
                x = rng.normal((1 if domain == "d1" else 0) + (2 if y else -2), 1, size=(dim,)).astype("float32")
                p = root / f"proto_{domain}_{split}_{i}.npy"
                np.save(p, x)
                rows.append({"path": p.name, "label": y, "split": split, "domain": domain, "generator": domain, "scene": "s"})
    manifest = root / "protocol_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["path", "label", "split", "domain", "generator", "scene"])
        w.writeheader(); w.writerows(rows)
    return manifest


def test_yaml_protocol_decouples_task_order_from_storage(tmp_path):
    manifest = make_protocol_manifest(tmp_path)
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
        "scenario": {
            "data": {"backend": "manifest", "path": str(manifest), "root": str(tmp_path)},
            "protocol": protocol,
        },
        "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "optimizer": {"type": "adamw", "lr": 1e-3}},
        "method": {
            "name": "finetune",
            "num_classes": 2,
            "detector_cfg": {"num_classes": 2, "backbone": {"type": "identity", "in_dim": 16}},
        },
    }
    tr = Trainer(cfg)
    assert [t.name for t in tr.scenario.tasks] == ["D1 first", "D0 second"]
    assert [t.num_train for t in tr.scenario.tasks] == [6, 6]
    summary = tr.run()
    assert "average_accuracy" in summary
