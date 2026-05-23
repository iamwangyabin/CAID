from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from caidbench.data.scenario import ContinualScenario
from caidbench.data.transforms import build_transform


def write_aid_image_dataset(root: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    root.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict({"image": [row["image"] for row in rows]})
    with pa.OSFile(str(root / "data.arrow"), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    with open(root / "mapping.json", "w", encoding="utf-8") as fp:
        json.dump({str(row["path"]): i for i, row in enumerate(rows)}, fp)
    split_payloads: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        split_payloads.setdefault(str(row["split"]), {}).setdefault("all", {})[str(row["path"])] = int(row["label"])
    for split, payload in split_payloads.items():
        with open(root / f"{split}.json", "w", encoding="utf-8") as fp:
            json.dump(payload, fp)
    with open(root / "caid_meta.jsonl", "w", encoding="utf-8") as fp:
        for row in rows:
            meta = {key: value for key, value in row.items() if key != "image"}
            fp.write(json.dumps(meta) + "\n")
    return root


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
        buf = BytesIO()
        image.save(buf, format="PNG")
        rows.append({"path": f"{split}.png", "image": buf.getvalue(), "label": 0, "split": split, "task_hint": "task0"})
    data_root = write_aid_image_dataset(tmp_path / "aid_images", rows)

    scenario = ContinualScenario.from_config(
        {
            "data": {"backend": "aid_arrow", "path": str(data_root), "image_column": "image"},
            "protocol": {"tasks": [{"id": "task0", "name": "task0", "filter": {"include": {"task_hint": "task0"}}}]},
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
