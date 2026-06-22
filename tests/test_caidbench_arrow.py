from __future__ import annotations

import pandas as pd
import pytest
import torch

from caidbench.data.caidbench_arrow import (
    CAIDBenchArrowDataSource,
    CAIDBenchArrowImageDataset,
    LoadedCAIDBenchArrow,
    _CorruptImageError,
)


def _loaded() -> LoadedCAIDBenchArrow:
    metadata = pd.DataFrame(
        {
            "_file_id": [0, 0],
            "_batch_index": [3, 3],
            "_batch_row": [10, 11],
            "label": [0, 1],
            "task_hint": ["Imagen-4", "Imagen-4"],
            "split": ["train", "train"],
            "generator_name": ["Imagen-4", "Imagen-4"],
            "source_path": ["bad.png", "ok.png"],
            "arrow_file": ["Imagen-4/train.arrow", "Imagen-4/train.arrow"],
        }
    )
    return LoadedCAIDBenchArrow(
        files=[],
        metadata=metadata,
        image_column="image",
        label_column="label",
        generator_column="generator_name",
        dataset_column="source_dataset",
        source_path_column="source_path",
        split_column="split",
    )


def _sample(label: int) -> dict[str, torch.Tensor]:
    return {
        "x": torch.zeros(3, 4, 4),
        "y": torch.tensor(label),
        "task_id": torch.tensor(0),
    }


def test_caidbench_arrow_dataset_skips_corrupt_samples_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = CAIDBenchArrowImageDataset(_loaded(), indices=[0, 1])
    calls: list[int] = []

    def fake_load_sample(meta_pos: int) -> dict[str, torch.Tensor]:
        calls.append(meta_pos)
        if meta_pos == 0:
            raise _CorruptImageError("image file is truncated")
        return _sample(meta_pos)

    monkeypatch.setattr(dataset, "_load_sample", fake_load_sample)

    with pytest.warns(UserWarning, match="Skipping corrupted sample"):
        sample = dataset[0]

    assert calls == [0, 1]
    assert sample["y"].item() == 1


def test_caidbench_arrow_dataset_fails_clearly_when_all_candidates_are_corrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = CAIDBenchArrowImageDataset(_loaded(), indices=[0, 1])
    calls: list[int] = []

    def fake_load_sample(meta_pos: int) -> dict[str, torch.Tensor]:
        calls.append(meta_pos)
        raise _CorruptImageError("image file is truncated")

    monkeypatch.setattr(dataset, "_load_sample", fake_load_sample)

    with pytest.warns(UserWarning, match="Skipping corrupted sample"):
        with pytest.raises(RuntimeError, match="No valid sample found"):
            dataset[0]

    assert calls == [0, 1]


def test_caidbench_arrow_data_source_builds_image_dataset() -> None:
    source = CAIDBenchArrowDataSource(_loaded())
    dataset = source.make_dataset([0, 1])

    assert isinstance(dataset, CAIDBenchArrowImageDataset)
