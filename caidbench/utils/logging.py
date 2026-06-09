from __future__ import annotations

import json
import logging


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
