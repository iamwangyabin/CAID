from __future__ import annotations

import json
import logging
from pathlib import Path


def format_log_value(value: object) -> str:
    text = str(value)
    if not text or any(char.isspace() for char in text) or "=" in text:
        return json.dumps(text, ensure_ascii=False)
    return text


def get_logger(name: str = "caidbench", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def attach_file_handler(logger: logging.Logger, path: str | Path, level: int = logging.INFO) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    for handler in list(logger.handlers):
        if getattr(handler, "_caidbench_file_handler", False):
            if getattr(handler, "_caidbench_file_handler_path", None) == resolved:
                return
            logger.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    handler.setLevel(level)
    handler._caidbench_file_handler = True  # type: ignore[attr-defined]
    handler._caidbench_file_handler_path = resolved  # type: ignore[attr-defined]
    logger.addHandler(handler)
