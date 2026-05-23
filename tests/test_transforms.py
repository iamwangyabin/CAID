from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from caidbench.data.scenario import ContinualScenario
from caidbench.data.transforms import build_transform


def test_yaml_transform_pipeline_outputs_tensor() -> None:
    transform = build_transform(
        {
            "trsf": [
                {"_target_": "data.DataAugment", "blur_prob": 0.0, "jpg_prob": 0.0},
                {"_target_": "torchvision.transforms.Resize", "size": 20},
                {"_target_": "torchvision.transforms.CenterCrop", "size": 16},
                {"_target_": "torchvision.transforms.ToTensor"},
                {
                    "_target_": "torchvision.transforms.Normalize",
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                },
            ]
        }
    )
    img = Image.fromarray(np.full((18, 24, 3), 128, dtype=np.uint8))

    out = transform(img)

    assert torch.is_tensor(out)
    assert tuple(out.shape) == (3, 16, 16)


def test_scenario_uses_split_specific_transforms(tmp_path: Path) -> None:
    image = Image.fromarray(np.full((24, 24, 3), 128, dtype=np.uint8))
    rows = []
    for split in ("train", "test"):
        path = tmp_path / f"{split}.png"
        image.save(path)
        rows.append({"path": path.name, "label": 0, "split": split, "task_id": 0})
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["path", "label", "split", "task_id"])
        writer.writeheader()
        writer.writerows(rows)

    scenario = ContinualScenario.from_config(
        {
            "data": {"backend": "manifest", "path": str(manifest), "root": str(tmp_path)},
            "transform": {
                "train": {
                    "trsf": [
                        {"_target_": "caidbench.data.transforms.SquareResize", "size": 8},
                        {"_target_": "caidbench.data.transforms.ToTensor"},
                    ]
                },
                "test": {
                    "trsf": [
                        {"_target_": "caidbench.data.transforms.SquareResize", "size": 12},
                        {"_target_": "caidbench.data.transforms.ToTensor"},
                    ]
                },
            },
        }
    )

    train_sample = scenario.task_dataset("train", 0)[0]
    test_sample = scenario.task_dataset("test", 0)[0]

    assert tuple(train_sample["x"].shape) == (3, 8, 8)
    assert tuple(test_sample["x"].shape) == (3, 12, 12)
