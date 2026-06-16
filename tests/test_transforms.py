from __future__ import annotations

import io

from PIL import Image

from caidbench.data.transforms import ConditionalJPEGCompress, _estimate_jpeg_quality


def _jpeg_image(quality: int) -> Image.Image:
    img = Image.new("RGB", (32, 32), (96, 128, 160))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    out = Image.open(buffer)
    out.load()
    return out


def test_conditional_jpeg_compress_recompresses_non_jpeg() -> None:
    img = Image.new("RGB", (32, 32), (96, 128, 160))
    out = ConditionalJPEGCompress(quality=80, recompress_if_jpeg_quality_above=80)(img)

    assert out is not img
    assert out.format == "JPEG"
    assert _estimate_jpeg_quality(out) == 80


def test_conditional_jpeg_compress_skips_jpeg_below_target_quality() -> None:
    img = _jpeg_image(79)
    assert _estimate_jpeg_quality(img) == 79

    out = ConditionalJPEGCompress(quality=80, recompress_if_jpeg_quality_above=80)(img)

    assert out is img


def test_conditional_jpeg_compress_skips_jpeg_at_target_quality() -> None:
    img = _jpeg_image(80)
    assert _estimate_jpeg_quality(img) == 80

    out = ConditionalJPEGCompress(quality=80, recompress_if_jpeg_quality_above=80)(img)

    assert out is img


def test_conditional_jpeg_compress_recompresses_jpeg_above_target_quality() -> None:
    img = _jpeg_image(81)
    assert _estimate_jpeg_quality(img) == 81

    out = ConditionalJPEGCompress(quality=80, recompress_if_jpeg_quality_above=80)(img)

    assert out is not img
    assert out.format == "JPEG"
    assert _estimate_jpeg_quality(out) == 80
