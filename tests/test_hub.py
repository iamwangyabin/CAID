from __future__ import annotations

from pathlib import Path

from caidbench.data import hub


def test_remote_resolver_selects_huggingface_mirror(monkeypatch, tmp_path):
    calls = []

    def fake_download(repo_id: str, params: dict) -> Path:
        calls.append((repo_id, params))
        return tmp_path / "cddb"

    monkeypatch.setattr(hub, "_download_huggingface", fake_download)

    out = hub.resolve_data_path(
        {
            "backend": "aid_arrow",
            "remote": {
                "platform": "huggingface",
                "repo_ids": {"huggingface": "nebula/CDDB.arrow", "modelscope": "yabinnng/CDDB.arrow"},
                "local_dir": str(tmp_path / "local"),
                "path_in_repo": ".",
            },
        }
    )

    assert out == tmp_path / "cddb"
    assert calls[0][0] == "nebula/CDDB.arrow"
    assert calls[0][1]["local_dir"] == str(tmp_path / "local")


def test_remote_resolver_selects_modelscope_mirror_and_subdir(monkeypatch, tmp_path):
    root = tmp_path / "snapshot"
    (root / "CDDB").mkdir(parents=True)
    calls = []

    def fake_download(repo_id: str, params: dict) -> Path:
        calls.append((repo_id, params))
        return root

    monkeypatch.setattr(hub, "_download_modelscope", fake_download)

    out = hub.resolve_data_path(
        {
            "backend": "aid_arrow",
            "remote": {
                "platform": "modelscope",
                "repo_ids": {"huggingface": "nebula/DF-arrow", "modelscope": "yabinnng/DF-arrow"},
                "path_in_repo": "CDDB",
            },
        }
    )

    assert out == root / "CDDB"
    assert calls[0][0] == "yabinnng/DF-arrow"
    assert calls[0][1]["allow_patterns"] == ["CDDB/*"]


def test_local_path_stays_local():
    assert hub.resolve_data_path({"backend": "aid_arrow", "path": "data/CDDB"}) == Path("data/CDDB")
