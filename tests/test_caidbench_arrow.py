from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

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


def test_caidbench_backend_builds_protocol_tasks(tmp_path):
    root = tmp_path / "caidbench"
    _write_split(root / "ADM" / "train.arrow", "ADM", "train")
    _write_split(root / "ADM" / "test.arrow", "ADM", "test")
    _write_split(root / "DALL_E_3" / "train.arrow", "DALL-E 3", "train")
    _write_split(root / "DALL_E_3" / "test.arrow", "DALL-E 3", "test")

    scenario = ContinualScenario.from_config(
        {
            "data": {"backend": "caidbench", "path": str(root), "image_column": "image"},
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


def test_caidbench_protocol_uses_fast_index_path(tmp_path, monkeypatch):
    root = tmp_path / "caidbench"
    _write_split(root / "ADM" / "train.arrow", "ADM", "train")
    _write_split(root / "ADM" / "test.arrow", "ADM", "test")

    def fail_apply_filter(*args, **kwargs):
        raise AssertionError("caidbench task_hint filters should use source.select_indices")

    monkeypatch.setattr(scenario_mod, "apply_filter", fail_apply_filter)

    scenario = ContinualScenario.from_config(
        {
            "data": {"backend": "caidbench", "path": str(root), "image_column": "image"},
            "protocol": {"tasks": [{"id": "adm", "name": "ADM", "numeric_id": 0, "task_hint": "ADM"}]},
            "transform": {"default": {"trsf": [{"_target_": "caidbench.data.transforms.ToTensor"}]}},
        }
    )

    assert scenario.tasks[0].num_train == 2
    assert scenario.tasks[0].num_test == 2


def test_caidbench_index_path_loads_selected_rows_for_shuffle_training(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "caidbench"
    _write_split(root / "Raw_A" / "train.arrow", "Raw_A", "train", n=4)
    _write_split(root / "Raw_A" / "test.arrow", "Raw_A", "test", n=4)
    _write_split(root / "Raw_B" / "train.arrow", "Raw_B", "train", n=4)
    _write_split(root / "Raw_B" / "test.arrow", "Raw_B", "test", n=4)

    rows = [
        {"task_id": 0, "generator_name": "Canonical A", "raw_generator_name": "Raw_A", "split": "train", "label": 0, "arrow_path": "Raw_A/train.arrow", "batch_id": 0, "row_in_batch": 0},
        {"task_id": 0, "generator_name": "Canonical A", "raw_generator_name": "Raw_A", "split": "train", "label": 1, "arrow_path": "Raw_A/train.arrow", "batch_id": 0, "row_in_batch": 1},
        {"task_id": 0, "generator_name": "Canonical A", "raw_generator_name": "Raw_A", "split": "test", "label": 0, "arrow_path": "Raw_A/test.arrow", "batch_id": 0, "row_in_batch": 0},
        {"task_id": 0, "generator_name": "Canonical A", "raw_generator_name": "Raw_A", "split": "test", "label": 1, "arrow_path": "Raw_A/test.arrow", "batch_id": 0, "row_in_batch": 1},
        {"task_id": 1, "generator_name": "Canonical B", "raw_generator_name": "Raw_B", "split": "train", "label": 0, "arrow_path": "Raw_B/train.arrow", "batch_id": 0, "row_in_batch": 0},
        {"task_id": 1, "generator_name": "Canonical B", "raw_generator_name": "Raw_B", "split": "train", "label": 1, "arrow_path": "Raw_B/train.arrow", "batch_id": 0, "row_in_batch": 1},
        {"task_id": 1, "generator_name": "Canonical B", "raw_generator_name": "Raw_B", "split": "test", "label": 0, "arrow_path": "Raw_B/test.arrow", "batch_id": 0, "row_in_batch": 0},
        {"task_id": 1, "generator_name": "Canonical B", "raw_generator_name": "Raw_B", "split": "test", "label": 1, "arrow_path": "Raw_B/test.arrow", "batch_id": 0, "row_in_batch": 1},
    ]
    index_path = tmp_path / "selected_index.parquet"
    pq.write_table(pa.Table.from_pylist(rows), index_path)

    scenario = ContinualScenario.from_config(
        {
            "data": {"backend": "caidbench", "path": str(root), "index_path": str(index_path), "image_column": "image"},
            "protocol": {
                "tasks": [
                    {"id": "canonical_b", "name": "Canonical B", "numeric_id": 0, "filter": {"include": {"task_id": 1}}},
                    {"id": "canonical_a", "name": "Canonical A", "numeric_id": 1, "filter": {"include": {"task_id": 0}}},
                ]
            },
            "transform": {"default": {"trsf": [{"_target_": "caidbench.data.transforms.ToTensor"}]}},
        }
    )

    assert [t.task_id for t in scenario.tasks] == [0, 1]
    assert [t.num_train for t in scenario.tasks] == [2, 2]
    assert [t.num_test for t in scenario.tasks] == [2, 2]
    assert [t.generators for t in scenario.tasks] == [("Canonical B",), ("Canonical A",)]

    dataset = scenario.task_dataset("train", 0)
    sample = dataset[0]
    assert sample["x"].shape == (3, 8, 8)
    assert sample["generator_name"] == "Canonical B"
    assert sample["generator"] == "Canonical B"
    assert sample["raw_generator_name"] == "Raw_B"
    assert int(sample["task_id"]) == 0

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)))
    assert batch["x"].shape == (2, 3, 8, 8)
    assert set(batch["generator_name"]) == {"Canonical B"}


def test_caidbench_protocol_local_index_path(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "caidbench"
    _write_split(root / "Raw_A" / "train.arrow", "Raw_A", "train", n=2)
    _write_split(root / "Raw_A" / "test.arrow", "Raw_A", "test", n=2)

    protocol_dir = tmp_path / "protocols"
    protocol_dir.mkdir()
    index_path = protocol_dir / "selected_index.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"task_id": 0, "generator_name": "Canonical A", "raw_generator_name": "Raw_A", "split": "train", "label": 0, "arrow_path": "Raw_A/train.arrow", "batch_id": 0, "row_in_batch": 0},
                {"task_id": 0, "generator_name": "Canonical A", "raw_generator_name": "Raw_A", "split": "test", "label": 1, "arrow_path": "Raw_A/test.arrow", "batch_id": 0, "row_in_batch": 0},
            ]
        ),
        index_path,
    )
    protocol_path = protocol_dir / "caidbench.yaml"
    protocol_path.write_text(
        """
index_path: selected_index.parquet
tasks:
  - id: canonical_a
    name: Canonical A
    numeric_id: 0
    filter:
      include:
        task_id: 0
""".lstrip(),
        encoding="utf-8",
    )

    scenario = ContinualScenario.from_config(
        {
            "data": {"backend": "caidbench", "path": str(root), "image_column": "image"},
            "protocol": str(protocol_path),
            "transform": {"default": {"trsf": [{"_target_": "caidbench.data.transforms.ToTensor"}]}},
        }
    )

    assert len(scenario.tasks) == 1
    assert scenario.tasks[0].num_train == 1
    assert scenario.tasks[0].num_test == 1
