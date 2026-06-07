from __future__ import annotations

import io
import math
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_RESAMPLING = getattr(Image, "Resampling", Image)
_TRANSPOSE = getattr(Image, "Transpose", Image)


def _resampling(name: Any | None = None) -> int:
    if isinstance(name, int):
        return name
    value = "bicubic" if name is None else str(name).split(".")[-1].lower()
    mapping = {
        "nearest": _RESAMPLING.NEAREST,
        "bilinear": _RESAMPLING.BILINEAR,
        "bicubic": _RESAMPLING.BICUBIC,
        "box": _RESAMPLING.BOX,
        "hamming": _RESAMPLING.HAMMING,
        "lanczos": _RESAMPLING.LANCZOS,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported interpolation={name!r}; expected one of {sorted(mapping)}.")
    return mapping[value]


def _size2(size: Any) -> tuple[int, int]:
    if isinstance(size, int):
        return int(size), int(size)
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        values = list(size)
        if len(values) == 1:
            return int(values[0]), int(values[0])
        if len(values) == 2:
            return int(values[0]), int(values[1])
    raise ValueError(f"size must be an int or length-2 sequence, got {size!r}.")


def _as_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def _to_tensor(img: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(img):
        return img.float()
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    tensor = torch.from_numpy(np.array(arr, copy=True)).permute(2, 0, 1)
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    return tensor.float()


def _sample_range(value: Any, *, discrete: bool = False) -> float | int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if not values:
            raise ValueError("Cannot sample from an empty sequence.")
        if discrete or len(values) != 2:
            return random.choice(values)
        return random.uniform(float(values[0]), float(values[1]))
    return value


def _pad_if_needed(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_h, target_w = size
    width, height = img.size
    if width >= target_w and height >= target_h:
        return img
    out = Image.new(img.mode, (max(width, target_w), max(height, target_h)))
    left = (out.size[0] - width) // 2
    top = (out.size[1] - height) // 2
    out.paste(img, (left, top))
    return out


class Compose:
    def __init__(self, transforms: Sequence[Callable[[Any], Any]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, img: Any) -> Any:
        for transform in self.transforms:
            img = transform(img)
        return img


class Resize:
    """Torchvision-like resize.

    Integer sizes resize the shorter edge while preserving aspect ratio. A
    length-2 size resizes exactly to ``(height, width)``.
    """

    def __init__(self, size: int | Sequence[int], interpolation: Any | None = None) -> None:
        self.size = size
        self.interpolation = _resampling(interpolation)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = _as_rgb(img)
        width, height = img.size
        if isinstance(self.size, int):
            short = int(self.size)
            if min(width, height) == short:
                return img
            if width < height:
                out_w = short
                out_h = int(round(short * height / width))
            else:
                out_h = short
                out_w = int(round(short * width / height))
            return img.resize((out_w, out_h), self.interpolation)
        target_h, target_w = _size2(self.size)
        return img.resize((target_w, target_h), self.interpolation)


class ResizeIfSmaller:
    def __init__(self, size: int | Sequence[int], interpolation: Any | None = None) -> None:
        self.size = _size2(size)
        self.interpolation = _resampling(interpolation)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = _as_rgb(img)
        target_h, target_w = self.size
        width, height = img.size
        if width >= target_w and height >= target_h:
            return img
        scale = max(target_w / max(width, 1), target_h / max(height, 1))
        out_w = int(round(width * scale))
        out_h = int(round(height * scale))
        out_w = max(out_w, target_w)
        out_h = max(out_h, target_h)
        return img.resize((out_w, out_h), self.interpolation)


class SquareResize:
    def __init__(self, size: int | Sequence[int], interpolation: Any | None = None) -> None:
        self.size = _size2(size)
        self.interpolation = _resampling(interpolation)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = _as_rgb(img)
        target_h, target_w = self.size
        return img.resize((target_w, target_h), self.interpolation)


class CenterCrop:
    def __init__(self, size: int | Sequence[int]) -> None:
        self.size = _size2(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = _pad_if_needed(_as_rgb(img), self.size)
        target_h, target_w = self.size
        width, height = img.size
        left = int(round((width - target_w) / 2.0))
        top = int(round((height - target_h) / 2.0))
        return img.crop((left, top, left + target_w, top + target_h))


class RandomCrop:
    def __init__(self, size: int | Sequence[int]) -> None:
        self.size = _size2(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = _pad_if_needed(_as_rgb(img), self.size)
        target_h, target_w = self.size
        width, height = img.size
        left = random.randint(0, width - target_w) if width > target_w else 0
        top = random.randint(0, height - target_h) if height > target_h else 0
        return img.crop((left, top, left + target_w, top + target_h))


class RandomResizedCrop:
    def __init__(
        self,
        size: int | Sequence[int],
        scale: Sequence[float] = (0.08, 1.0),
        ratio: Sequence[float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: Any | None = None,
    ) -> None:
        self.size = _size2(size)
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        self.interpolation = _resampling(interpolation)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = _as_rgb(img)
        width, height = img.size
        area = width * height

        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            aspect_ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_w <= width and 0 < crop_h <= height:
                left = random.randint(0, width - crop_w)
                top = random.randint(0, height - crop_h)
                cropped = img.crop((left, top, left + crop_w, top + crop_h))
                target_h, target_w = self.size
                return cropped.resize((target_w, target_h), self.interpolation)

        in_ratio = width / height
        if in_ratio < self.ratio[0]:
            crop_w = width
            crop_h = int(round(crop_w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            crop_h = height
            crop_w = int(round(crop_h * self.ratio[1]))
        else:
            crop_w, crop_h = width, height
        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        target_h, target_w = self.size
        return cropped.resize((target_w, target_h), self.interpolation)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = _as_rgb(img)
        return img.transpose(_TRANSPOSE.FLIP_LEFT_RIGHT) if random.random() < self.p else img


class ColorJitter:
    def __init__(
        self,
        brightness: float | Sequence[float] = 0.0,
        contrast: float | Sequence[float] = 0.0,
        saturation: float | Sequence[float] = 0.0,
        hue: float | Sequence[float] = 0.0,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _factor(self, value: float | Sequence[float]) -> float | None:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            vals = list(value)
            if len(vals) == 2:
                return random.uniform(float(vals[0]), float(vals[1]))
        delta = float(value)
        if delta <= 0:
            return None
        return random.uniform(max(0.0, 1.0 - delta), 1.0 + delta)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = _as_rgb(img)
        ops = []
        brightness = self._factor(self.brightness)
        contrast = self._factor(self.contrast)
        saturation = self._factor(self.saturation)
        hue = self._hue_factor(self.hue)
        if brightness is not None:
            ops.append(lambda x: ImageEnhance.Brightness(x).enhance(brightness))
        if contrast is not None:
            ops.append(lambda x: ImageEnhance.Contrast(x).enhance(contrast))
        if saturation is not None:
            ops.append(lambda x: ImageEnhance.Color(x).enhance(saturation))
        if hue is not None:
            ops.append(lambda x: _adjust_hue(x, hue))
        random.shuffle(ops)
        for op in ops:
            img = op(img)
        return img

    def _hue_factor(self, value: float | Sequence[float]) -> float | None:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            vals = list(value)
            if len(vals) == 2:
                return random.uniform(float(vals[0]), float(vals[1]))
        delta = float(value)
        if delta <= 0:
            return None
        return random.uniform(-delta, delta)


class RandomInterpolationResize:
    def __init__(self, size: int | Sequence[int]) -> None:
        self.size = size
        self.interpolations = [
            Image.Resampling.NEAREST,
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.BOX,
            Image.Resampling.HAMMING,
            Image.Resampling.LANCZOS,
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        resize = Resize(self.size)
        resize.interpolation = random.choice(self.interpolations)
        return resize(img)


class DataAugment:
    """AID-style blur/JPEG augmentation configured from YAML."""

    def __init__(
        self,
        blur_prob: float = 0.0,
        blur_sig: Sequence[float] = (0.0, 3.0),
        jpg_prob: float = 0.0,
        jpg_method: Sequence[str] = ("pil",),
        jpg_qual: Sequence[int] = (30, 100),
    ) -> None:
        self.blur_prob = float(blur_prob)
        self.blur_sig = blur_sig
        self.jpg_prob = float(jpg_prob)
        self.jpg_method = tuple(str(x).lower() for x in jpg_method)
        self.jpg_qual = jpg_qual

    def __call__(self, img: Image.Image) -> Image.Image:
        img = _as_rgb(img)
        if random.random() < self.blur_prob:
            sigma = float(_sample_range(self.blur_sig))
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        if random.random() < self.jpg_prob:
            _method = random.choice(self.jpg_method) if self.jpg_method else "pil"
            quality = int(_sample_range(self.jpg_qual, discrete=True))
            img = _jpeg_roundtrip(img, quality)
        return img


class RandomCompress:
    def __init__(self, method: str = "JPEG", qf: Sequence[int] = (60, 100)) -> None:
        self.method = method
        self.qf = qf

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = _sample_range(self.qf)
        if isinstance(quality, float):
            quality = random.randint(int(self.qf[0]), int(self.qf[1])) if isinstance(self.qf, Sequence) and len(self.qf) == 2 else int(quality)
        return _jpeg_roundtrip(_as_rgb(img), int(quality), method=self.method)


class Compress:
    def __init__(self, method: str = "JPEG", qf: int = 100) -> None:
        self.method = method
        self.qf = int(qf)

    def __call__(self, img: Image.Image) -> Image.Image:
        return _jpeg_roundtrip(_as_rgb(img), self.qf, method=self.method)


class ToTensor:
    def __call__(self, img: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        return _to_tensor(img)


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = torch.tensor(tuple(mean), dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(tuple(std), dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3:
            raise ValueError(f"Normalize expects a CHW tensor, got shape={tuple(tensor.shape)}.")
        return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)


def _jpeg_roundtrip(img: Image.Image, quality: int, method: str = "JPEG") -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format=method if method.upper() != "CV2" else "JPEG", quality=int(quality))
    buffer.seek(0)
    out = Image.open(buffer).convert("RGB").copy()
    buffer.close()
    return out


def _adjust_hue(img: Image.Image, hue_factor: float) -> Image.Image:
    hsv = _as_rgb(img).convert("HSV")
    arr = np.asarray(hsv, dtype=np.uint8).copy()
    shift = int(round(float(hue_factor) * 255.0))
    arr[..., 0] = (arr[..., 0].astype(np.int16) + shift) % 256
    return Image.fromarray(arr, mode="HSV").convert("RGB")


_TARGETS: dict[str, type] = {
    "resize": Resize,
    "resizeifsmaller": ResizeIfSmaller,
    "squareresize": SquareResize,
    "centercrop": CenterCrop,
    "randomcrop": RandomCrop,
    "randomresizedcrop": RandomResizedCrop,
    "randomhorizontalflip": RandomHorizontalFlip,
    "colorjitter": ColorJitter,
    "randominterpolationresize": RandomInterpolationResize,
    "dataaugment": DataAugment,
    "randomcompress": RandomCompress,
    "compress": Compress,
    "totensor": ToTensor,
    "normalize": Normalize,
}


def _target_name(raw: str) -> str:
    return raw.split(".")[-1].replace("_", "").lower()


def _build_step(spec: str | Mapping[str, Any]) -> Callable[[Any], Any]:
    if isinstance(spec, str):
        target = spec
        kwargs: dict[str, Any] = {}
    else:
        kwargs = dict(spec)
        target = str(kwargs.pop("_target_", kwargs.pop("target", kwargs.pop("type", kwargs.pop("name", "")))))
        kwargs.pop("_partial_", None)
    if not target:
        raise ValueError(f"Transform step is missing _target_/type/name: {spec!r}.")
    name = _target_name(target)
    cls = _TARGETS.get(name)
    if cls is None:
        raise ValueError(f"Unsupported transform target={target!r}.")
    return cls(**kwargs)


def _steps_from_cfg(cfg: Any) -> list[Any] | None:
    if isinstance(cfg, list):
        return cfg
    if isinstance(cfg, Mapping):
        for key in ("trsf", "transforms", "steps", "pipeline"):
            if key in cfg:
                return list(cfg[key])
    return None


def build_transform(cfg: dict[str, Any] | list[Any] | None = None) -> Callable[[Image.Image], torch.Tensor]:
    if cfg is None or cfg == {}:
        return Compose([ToTensor()])
    steps = _steps_from_cfg(cfg)
    if steps is not None:
        return Compose([_build_step(step) for step in steps])

    if isinstance(cfg, Mapping) and any(k in cfg for k in ("_target_", "target", "type", "name")):
        return Compose([_build_step(cfg)])

    raise ValueError(
        "scenario.transform must be a YAML transform list, e.g. "
        "{trsf: [{_target_: caidbench.data.transforms.SquareResize, size: 224}, "
        "{_target_: caidbench.data.transforms.ToTensor}, "
        "{_target_: caidbench.data.transforms.Normalize, mean: [...], std: [...]}]}."
    )
