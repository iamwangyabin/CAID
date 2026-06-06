from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

import caidbench.data.scenario as scenario_mod
from caidbench.data.scenario import ContinualScenario


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _write_split(path: Path, generator: str, split: str, n: int = 2) -> None:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {
            "image": [_png_bytes((i * 20, 10, 30)) for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "generator_name": [generator] * n,
            "source_dataset": ["UnitSet"] * n,
            "source_path": [f"UnitSet/{generator}/{split}/{i}.png" for i in range(n)],
            "split": [split] * n,
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def test_stitched_arrow_backend_builds_protocol_tasks(tmp_path):
    root = tmp_path / "stitched"
    _write_split(root / "ADM" / "train.arrow", "ADM", "train")
    _write_split(root / "ADM" / "test.arrow", "ADM", "test")
    _write_split(root / "DALL_E_3" / "train.arrow", "DALL-E 3", "train")
    _write_split(root / "DALL_E_3" / "test.arrow", "DALL-E 3", "test")

    scenario = ContinualScenario.from_config(
        {
            "data": {"backend": "stitched_arrow", "path": str(root), "image_column": "image"},
            "protocol": {
                "tasks": [
                    {"id": "adm", "name": "ADM", "numeric_id": 0, "task_hint": "ADM"},
                    {"id": "dalle", "name": "DALL-E 3", "numeric_id": 1, "task_hint": "DALL_E_3"},
                ]
            },
            "transform": {"default": {"trsf": [{"_target_": "caidbench.data.transforms.ToTensor"}]}},
        }
    )

    assert [t.name for t in scenario.tasks] == ["ADM", "DALL-E 3"]
    assert [t.num_train for t in scenario.tasks] == [2, 2]
    assert [t.num_test for t in scenario.tasks] == [2, 2]

    sample = scenario.task_dataset("train", 1)[0]
    assert sample["x"].shape[0] == 3
    assert int(sample["y"]) in {0, 1}
    assert sample["task_hint"] == "DALL_E_3"
    assert sample["generator"] == "DALL-E 3"
    assert sample["dir_name"] == "DALL_E_3"


def test_stitched_arrow_protocol_uses_fast_index_path(tmp_path, monkeypatch):
    root = tmp_path / "stitched"
    _write_split(root / "ADM" / "train.arrow", "ADM", "train")
    _write_split(root / "ADM" / "test.arrow", "ADM", "test")

    def fail_apply_filter(*args, **kwargs):
        raise AssertionError("stitched task_hint filters should use source.select_indices")

    monkeypatch.setattr(scenario_mod, "apply_filter", fail_apply_filter)

    scenario = ContinualScenario.from_config(
        {
            "data": {"backend": "stitched_arrow", "path": str(root), "image_column": "image"},
            "protocol": {"tasks": [{"id": "adm", "name": "ADM", "numeric_id": 0, "task_hint": "ADM"}]},
            "transform": {"default": {"trsf": [{"_target_": "caidbench.data.transforms.ToTensor"}]}},
        }
    )

    assert scenario.tasks[0].num_train == 2
    assert scenario.tasks[0].num_test == 2
