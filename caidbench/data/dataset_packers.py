from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .arrow_schema import IMAGE_EXTENSIONS, normalize_records

REAL_TOKENS = {
    "real", "0_real", "0-real", "nature", "natural", "authentic", "original", "pristine", "negative", "0", "live"
}
FAKE_TOKENS = {
    "fake", "1_fake", "1-fake", "ai", "synthetic", "generated", "manipulated", "deepfake", "positive", "1", "swap", "reenactment"
}
SPLIT_ALIASES = {
    "train": "train", "training": "train", "trn": "train",
    "val": "val", "valid": "val", "validation": "val", "dev": "val",
    "test": "test", "testing": "test", "eval": "test",
}
KNOWN_GENERATORS = {
    "adm": "ADM", "glide": "GLIDE", "biggan": "BigGAN", "progan": "ProGAN", "stylegan": "StyleGAN",
    "stylegan2": "StyleGAN2", "stylegan-xl": "StyleGAN-XL", "styleganxl": "StyleGAN-XL",
    "cyclegan": "CycleGAN", "gaugan": "GauGAN", "crn": "CRN", "imle": "IMLE", "san": "SAN",
    "stargan": "StarGAN", "glow": "GLOW", "whichfaceisreal": "WhichFaceReal", "whichface": "WhichFaceReal",
    "wilddeepfake": "WildDeepfake", "faceforensics++": "FaceForensics++", "ff++": "FF++",
    "midjourney": "Midjourney", "midjourney-v5": "Midjourney-V5", "midjourney-v6": "Midjourney-V6",
    "stable_diffusion": "StableDiffusion", "stable-diffusion": "StableDiffusion", "sdv4": "SDv4", "sd15": "SD1.5", "sd1.5": "SD1.5",
    "sdv21": "SDv21", "sd3": "SD3", "wukong": "Wukong", "vqdm": "VQDM", "sagan": "SAGAN",
    "imagen3": "Imagen3", "flux1-dev": "FLUX1-dev", "r3gan": "R3GAN",
    "dfdc": "DFDC", "dfdcp": "DFDCP", "dfd": "DFD", "deepfakedetection": "DeepFakeDetection",
    "celeb-df": "Celeb-DF", "celeb-df-v2": "Celeb-DF-v2", "cdf": "Celeb-DF", "celeba": "CelebA",
}
KNOWN_DEEPFAKE_DATASETS = {
    "FaceForensics++", "FF++", "Celeb-DF", "Celeb-DF-v2", "CDF", "DFDC", "DFDCP", "DFD", "DeepFakeDetection", "WildDeepfake"
}
KNOWN_MANIPULATIONS = {
    "deepfakes": "Deepfakes", "deepfake": "Deepfakes", "faceswap": "FaceSwap", "face2face": "Face2Face",
    "neuraltextures": "NeuralTextures", "neural_textures": "NeuralTextures", "faceshifter": "FaceShifter",
    "original": "real", "real": "real", "youtube": "real", "real_youtube": "real",
    "mcnet": "MCNet", "blendface": "BlendFace", "stylegan3": "StyleGAN3", "hybrid": "Hybrid",
}


def norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9+._-]+", "", s.strip().lower())


def iter_image_files(root: str | Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> Iterable[Path]:
    root = Path(root)
    exts = {e.lower() for e in extensions}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def rel_parts(path: Path, root: Path) -> list[str]:
    try:
        return list(path.relative_to(root).parts)
    except Exception:
        return list(path.parts)


def infer_split(parts: list[str], default: str = "train") -> str:
    for part in parts:
        key = norm_token(part)
        if key in SPLIT_ALIASES:
            return SPLIT_ALIASES[key]
    return default


def infer_label(parts: list[str], default: int = -1) -> int:
    toks = [norm_token(p) for p in parts]
    for t in reversed(toks):
        if t in REAL_TOKENS:
            return 0
        if t in FAKE_TOKENS:
            return 1
    joined = "/".join(toks)
    if any(tok in joined for tok in ["/0_real/", "/real/", "/nature/", "/original/"]):
        return 0
    if any(tok in joined for tok in ["/1_fake/", "/fake/", "/ai/", "/synthetic/", "/manipulated/"]):
        return 1
    return default


def infer_known_name(parts: list[str], mapping: Mapping[str, str]) -> str | None:
    for part in parts:
        key = norm_token(part)
        if key in mapping:
            return mapping[key]
    # also try compact forms for folders such as imagenet_ai_0419_sdv4
    joined = " ".join(norm_token(p) for p in parts)
    for key, val in mapping.items():
        if key and key in joined:
            return val
    return None


def infer_generator(parts: list[str], default: str = "unknown") -> str:
    known = infer_known_name(parts, KNOWN_GENERATORS)
    if known:
        return known
    # fallback: first non-split/non-label path segment near root
    for part in parts:
        key = norm_token(part)
        if key not in SPLIT_ALIASES and key not in REAL_TOKENS and key not in FAKE_TOKENS and key not in {"images", "imgs", "frames", "face", "faces", "c23", "c40", "raw"}:
            return part
    return default


def infer_dataset(parts: list[str], default: str = "unknown") -> str:
    known = infer_known_name(parts, KNOWN_GENERATORS)
    if known in KNOWN_DEEPFAKE_DATASETS:
        return known
    return default


def infer_manipulation(parts: list[str], label: int, generator: str) -> str:
    known = infer_known_name(parts, KNOWN_MANIPULATIONS)
    if known:
        return known
    return "real" if label == 0 else generator


def infer_video_id(path: Path, parts: list[str]) -> str:
    stem = path.stem
    # DeepfakeBench often stores frames as .../<video_id>/<frame>.png
    if len(parts) >= 2 and norm_token(parts[-2]) not in REAL_TOKENS | FAKE_TOKENS | set(SPLIT_ALIASES):
        parent = Path(parts[-2]).stem
        if parent and parent.lower() not in {"frames", "images", "faces", "face", "0_real", "1_fake", "real", "fake"}:
            return parent
    # fallback: remove frame-like suffixes from file stem
    m = re.match(r"(.+?)(?:[_-]?(?:frame)?\d{1,8})$", stem, flags=re.IGNORECASE)
    return m.group(1) if m else stem


def infer_frame_idx(path: Path) -> int:
    m = re.search(r"(\d{1,8})$", path.stem)
    return int(m.group(1)) if m else -1


def scan_generic(
    root: str | Path,
    *,
    dataset_name: str,
    source: str,
    default_split: str = "train",
    preprocess_profile: str = "",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    root = Path(root)
    records = []
    for i, p in enumerate(iter_image_files(root)):
        if max_samples is not None and i >= max_samples:
            break
        parts = rel_parts(p, root)
        split = infer_split(parts, default=default_split)
        label = infer_label(parts)
        generator = infer_generator(parts)
        dataset = dataset_name
        domain = generator if dataset_name in {"CDDB", "CNNDetection", "GenImage", "TIFS-CAIL"} else dataset
        manipulation = infer_manipulation(parts, label, generator)
        records.append({
            "path": p,
            "label": label,
            "split": split,
            "dataset": dataset,
            "domain": domain,
            "generator": generator,
            "manipulation": manipulation,
            "source": source,
            "video_id": infer_video_id(p, parts) if dataset_name not in {"CDDB", "CNNDetection", "GenImage", "TIFS-CAIL"} else "",
            "frame_idx": infer_frame_idx(p),
            "preprocess_profile": preprocess_profile,
        })
    return records


def build_cddb(root: str | Path, **kwargs: Any) -> pd.DataFrame:
    records = scan_generic(root, dataset_name="CDDB", source="cddb", **_scan_kwargs(kwargs))
    for r in records:
        r["domain"] = r["generator"]
    return normalize_records(records, root=root, **_norm_kwargs(kwargs))


def build_cnn_detection(root: str | Path, **kwargs: Any) -> pd.DataFrame:
    records = scan_generic(root, dataset_name="CNNDetection", source="cnn_detection", **_scan_kwargs(kwargs))
    return normalize_records(records, root=root, **_norm_kwargs(kwargs))


def _genimage_name(parts: list[str]) -> str:
    # GenImage folders often look like imagenet_ai_0419_sdv4.
    joined = "_".join(norm_token(p) for p in parts)
    aliases = {
        "sdv4": "SDv4", "sdv5": "SDv5", "stable_diffusion": "StableDiffusion", "midjourney": "Midjourney",
        "adm": "ADM", "glide": "GLIDE", "wukong": "Wukong", "vqdm": "VQDM", "biggan": "BigGAN",
    }
    for k, v in aliases.items():
        if k in joined:
            return v
    return infer_generator(parts)


def build_genimage(root: str | Path, **kwargs: Any) -> pd.DataFrame:
    root = Path(root)
    max_samples = kwargs.get("max_samples")
    records = []
    for i, p in enumerate(iter_image_files(root)):
        if max_samples is not None and i >= max_samples:
            break
        parts = rel_parts(p, root)
        label = infer_label(parts)
        # GenImage uses nature/ai in many releases.
        if label == -1:
            toks = [norm_token(x) for x in parts]
            label = 0 if "nature" in toks else (1 if "ai" in toks else -1)
        gen = _genimage_name(parts)
        records.append({
            "path": p,
            "label": label,
            "split": infer_split(parts, default=kwargs.get("default_split", "train")),
            "dataset": "GenImage",
            "domain": gen,
            "generator": gen,
            "manipulation": "real" if label == 0 else gen,
            "source": "genimage",
            "preprocess_profile": kwargs.get("preprocess_profile", ""),
        })
    return normalize_records(records, root=root, **_norm_kwargs(kwargs))


def build_deepfakebench(root: str | Path, **kwargs: Any) -> pd.DataFrame:
    root = Path(root)
    max_samples = kwargs.get("max_samples")
    profile = kwargs.get("preprocess_profile", "sur_lid_deepfakebench_v1")
    records = []
    for i, p in enumerate(iter_image_files(root)):
        if max_samples is not None and i >= max_samples:
            break
        parts = rel_parts(p, root)
        dataset = infer_dataset(parts, default="DeepfakeBench")
        gen = infer_generator(parts, default=dataset)
        label = infer_label(parts)
        manipulation = infer_manipulation(parts, label, gen)
        # If a known fake manipulation appears and no explicit label was found.
        if label == -1 and manipulation not in {"unknown", "real"}:
            label = 1
        if label == -1 and dataset in {"WildDeepfake"}:
            label = 1
        records.append({
            "path": p,
            "label": label,
            "split": infer_split(parts, default=kwargs.get("default_split", "train")),
            "dataset": dataset,
            "domain": dataset,
            "generator": gen if label == 1 else "real",
            "manipulation": manipulation,
            "source": "deepfakebench",
            "video_id": infer_video_id(p, parts),
            "frame_idx": infer_frame_idx(p),
            "preprocess_profile": profile,
        })
    return normalize_records(records, root=root, **_norm_kwargs(kwargs))


def build_tifs_cail(root: str | Path, **kwargs: Any) -> pd.DataFrame:
    records = scan_generic(root, dataset_name="TIFS-CAIL", source="tifs_cail_protocol2", **_scan_kwargs(kwargs))
    return normalize_records(records, root=root, **_norm_kwargs(kwargs))


def _scan_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "default_split": kwargs.get("default_split", "train"),
        "preprocess_profile": kwargs.get("preprocess_profile", ""),
        "max_samples": kwargs.get("max_samples"),
    }


def _norm_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    # AID-style Arrow stores only image bytes in the Arrow table and keeps all
    # metadata in sidecars; these options are accepted for CLI compatibility.
    return {"strict_images": bool(kwargs.get("strict_images", False))}

PACKERS = {
    "cddb": build_cddb,
    "cnn_detection": build_cnn_detection,
    "cnnspot": build_cnn_detection,
    "forensynths": build_cnn_detection,
    "genimage": build_genimage,
    "deepfakebench": build_deepfakebench,
    "sur_lid": build_deepfakebench,
    "tifs_cail": build_tifs_cail,
}
