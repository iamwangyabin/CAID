from __future__ import annotations

from pathlib import Path

from PIL import Image

from caidbench.data.dataset_packers import build_cddb, build_deepfakebench, build_genimage
from caidbench.data.arrow_schema import CANONICAL_COLUMNS, _split_payload


def write_img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(128, 128, 128)).save(path)


def test_cddb_packer_normalizes_aid_sidecar_schema(tmp_path):
    write_img(tmp_path / "gaugan" / "train" / "real" / "a.png")
    write_img(tmp_path / "gaugan" / "train" / "fake" / "b.png")
    df = build_cddb(tmp_path)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) == 2
    assert set(df["label"].tolist()) == {0, 1}
    assert set(df["split"].tolist()) == {"train"}
    assert set(df["generator"].tolist()) == {"GauGAN"}


def test_genimage_packer_detects_ai_nature(tmp_path):
    write_img(tmp_path / "imagenet_ai_0419_sdv4" / "train" / "nature" / "a.png")
    write_img(tmp_path / "imagenet_ai_0419_sdv4" / "train" / "ai" / "b.png")
    df = build_genimage(tmp_path)
    assert set(df["label"].tolist()) == {0, 1}
    assert set(df["generator"].tolist()) == {"SDv4"}


def test_deepfakebench_packer_preserves_video_metadata(tmp_path):
    write_img(tmp_path / "FaceForensics++" / "c23" / "Deepfakes" / "train" / "vid001" / "000032.png")
    write_img(tmp_path / "FaceForensics++" / "c23" / "original" / "train" / "vid002" / "000032.png")
    df = build_deepfakebench(tmp_path)
    assert set(df["dataset"].tolist()) == {"FaceForensics++"}
    assert set(df["label"].tolist()) == {0, 1}
    assert set(df["video_id"].tolist()) == {"vid001", "vid002"}
    assert set(df["frame_idx"].tolist()) == {32}
    assert set(df["preprocess_profile"].tolist()) == {"sur_lid_deepfakebench_v1"}


def test_aid_split_payload_has_aid_subsets(tmp_path):
    write_img(tmp_path / "gaugan" / "train" / "real" / "a.png")
    write_img(tmp_path / "gaugan" / "train" / "fake" / "b.png")
    df = build_cddb(tmp_path)
    payload = _split_payload(df, "train")
    assert "all" in payload
    assert "real" in payload
    assert "fake" in payload
    assert "generator:GauGAN" in payload
    assert len(payload["all"]) == 2


def test_read_existing_aid_sidecars_exposes_subset_membership(tmp_path):
    from caidbench.data.arrow_schema import read_aid_split_sidecars
    import json

    (tmp_path / "mapping.json").write_text(json.dumps({"a.png": 0, "b.png": 1}), encoding="utf-8")
    (tmp_path / "train.json").write_text(
        json.dumps({"all": {"a.png": 0, "b.png": 1}, "sd15": {"b.png": 1}, "real": {"a.png": 0}}),
        encoding="utf-8",
    )
    (tmp_path / "test.json").write_text(json.dumps({"sd15": {"b.png": 1}}), encoding="utf-8")

    df = read_aid_split_sidecars(tmp_path)
    assert {"path", "label", "split", "subset", "_rowid"}.issubset(set(df.columns))
    assert len(df) == 3  # (train,a), (train,b), (test,b)
    train_b = df[(df["split"] == "train") & (df["path"] == "b.png")].iloc[0]
    assert "sd15" in train_b["subset"].split(";")
    assert int(train_b["_rowid"]) == 1


def test_protocol_subset_filter_matches_aid_membership_strings():
    from caidbench.data.protocol import apply_filter
    import pandas as pd

    df = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "subset": ["all;real", "all;fake;sd15", "all;fake;sd3"],
            "label": [0, 1, 1],
        }
    )
    out = apply_filter(df, {"split": "train", "include": {"subset": "sd15"}})
    assert len(out) == 1
    assert out.iloc[0]["label"] == 1
