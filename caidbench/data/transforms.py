from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class BasicImageTransform:
    size: int | tuple[int, int] = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if isinstance(self.size, int):
            size = (self.size, self.size)
        else:
            size = self.size
        img = img.convert("RGB").resize(size, Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor(self.mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std


def build_transform(cfg: dict[str, Any] | None = None) -> Callable[[Image.Image], torch.Tensor]:
    cfg = cfg or {}
    preset = str(cfg.get("preset", "imagenet")).lower()
    mean = cfg.get("mean")
    std = cfg.get("std")
    if mean is None or std is None:
        if preset in {"clip", "openai_clip", "open_clip"}:
            mean = CLIP_MEAN
            std = CLIP_STD
        else:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
    return BasicImageTransform(
        size=cfg.get("size", 224),
        mean=tuple(mean),
        std=tuple(std),
    )
