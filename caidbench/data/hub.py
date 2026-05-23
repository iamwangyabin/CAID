from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Mapping


_PROVIDER_ALIASES = {
    "hf": "huggingface",
    "huggingface": "huggingface",
    "hugging_face": "huggingface",
    "modelscope": "modelscope",
    "ms": "modelscope",
}


def _normalize_provider(value: Any) -> str:
    provider = _PROVIDER_ALIASES.get(str(value or "huggingface").lower())
    if provider is None:
        raise ValueError("remote provider must be one of: huggingface, hf, modelscope, ms")
    return provider


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _select_repo_id(remote: Mapping[str, Any], provider: str) -> str:
    for key in ("repo_id", "dataset_id", "id"):
        if remote.get(key):
            return str(remote[key])

    mirrors = remote.get("repo_ids", remote.get("mirrors", remote.get("repositories")))
    if isinstance(mirrors, Mapping):
        for key in (provider, "hf" if provider == "huggingface" else "ms"):
            if mirrors.get(key):
                return str(mirrors[key])
    if remote.get(provider):
        return str(remote[provider])
    raise ValueError(f"remote dataset config has no repo_id for provider={provider!r}")


def _filter_supported_kwargs(func: Callable[..., Any], params: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(func)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return {k: v for k, v in params.items() if v is not None}
    return {k: v for k, v in params.items() if v is not None and k in sig.parameters}


def _download_huggingface(repo_id: str, params: dict[str, Any]) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover - exercised only without optional dep
        raise ImportError("Hugging Face remote datasets require: pip install huggingface_hub") from e

    kwargs = _filter_supported_kwargs(
        snapshot_download,
        {
            "repo_id": repo_id,
            "repo_type": params.get("repo_type", "dataset"),
            "revision": params.get("revision"),
            "cache_dir": params.get("cache_dir"),
            "local_dir": params.get("local_dir"),
            "local_files_only": params.get("local_files_only"),
            "allow_patterns": params.get("allow_patterns"),
            "ignore_patterns": params.get("ignore_patterns"),
            "token": params.get("token"),
        },
    )
    return Path(snapshot_download(**kwargs))


def _download_modelscope(repo_id: str, params: dict[str, Any]) -> Path:
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download
    except Exception as e:  # pragma: no cover - exercised only without optional dep
        raise ImportError("ModelScope remote datasets require: pip install modelscope") from e

    kwargs = _filter_supported_kwargs(
        dataset_snapshot_download,
        {
            "dataset_id": repo_id,
            "revision": params.get("revision"),
            "cache_dir": params.get("cache_dir"),
            "local_dir": params.get("local_dir"),
            "local_files_only": params.get("local_files_only"),
            "allow_patterns": params.get("allow_patterns"),
            "ignore_patterns": params.get("ignore_patterns"),
            "token": params.get("token"),
            "endpoint": params.get("endpoint"),
            "max_workers": params.get("max_workers"),
        },
    )
    return Path(dataset_snapshot_download(**kwargs))


def resolve_data_path(cfg: Mapping[str, Any]) -> Path | None:
    """Resolve a local or mirrored remote dataset path.

    Local configs keep using ``path``/``root``. Remote configs may either place
    fields directly under ``scenario.data`` or inside ``scenario.data.remote``:

      remote:
        platform: huggingface
        repo_ids:
          huggingface: nebula/CDDB.arrow
          modelscope: yabinnng/CDDB.arrow
        local_dir: data/datasets/CDDB.arrow
    """
    cfg = dict(cfg)
    remote_raw = cfg.get("remote", cfg.get("hub"))
    has_inline_remote = any(k in cfg for k in ("repo_id", "dataset_id", "repo_ids", "mirrors", "repositories", "provider", "platform"))
    if remote_raw is None and not has_inline_remote:
        path = cfg.get("path", cfg.get("root"))
        return Path(path) if path is not None else None

    remote = dict(remote_raw or {})
    for key, value in cfg.items():
        if key not in {"remote", "hub", "backend", "type", "path", "root", "image_column", "path_column", "root_dir"}:
            remote.setdefault(key, value)

    provider = _normalize_provider(remote.get("provider", remote.get("platform", cfg.get("provider", cfg.get("platform")))))
    repo_id = _select_repo_id(remote, provider)
    path_in_repo = str(remote.get("path_in_repo", remote.get("subdir", ""))).strip("/")

    allow_patterns = _as_list(remote.get("allow_patterns", remote.get("allow_file_pattern")))
    if path_in_repo and path_in_repo != "." and allow_patterns is None:
        allow_patterns = [f"{path_in_repo}/*"]

    params = {
        "repo_type": remote.get("repo_type", "dataset"),
        "revision": remote.get("revision"),
        "cache_dir": remote.get("cache_dir"),
        "local_dir": remote.get("local_dir"),
        "local_files_only": remote.get("local_files_only"),
        "allow_patterns": allow_patterns,
        "ignore_patterns": _as_list(remote.get("ignore_patterns", remote.get("ignore_file_pattern"))),
        "token": remote.get("token"),
        "endpoint": remote.get("endpoint"),
        "max_workers": remote.get("max_workers"),
    }

    root = _download_huggingface(repo_id, params) if provider == "huggingface" else _download_modelscope(repo_id, params)
    if not path_in_repo or path_in_repo == ".":
        return root

    candidate = root / path_in_repo
    if candidate.exists():
        return candidate
    if (root / "dataset_info.json").exists():
        return root
    raise FileNotFoundError(f"Downloaded remote dataset but subdirectory was not found: {candidate}")
