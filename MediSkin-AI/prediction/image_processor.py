"""Image preprocessing shared by training (via training/preprocess.py) and inference.

Contract: the model receives float32 RGB in [0, 1], shape (1, H, W, 3). Any model-specific
rescaling (e.g. to [-1, 1] for MobileNetV2) lives *inside* the saved model.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from config import constants


def resize_image(image: Image.Image, size: tuple[int, int] = constants.MODEL_INPUT_SIZE) -> Image.Image:
    """Orient (EXIF), convert to RGB and resize to ``size`` = (width, height)."""
    oriented = ImageOps.exif_transpose(image)
    return oriented.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def normalize(image: Image.Image) -> np.ndarray:
    """Pixel values to float32 in [0, 1]; returns shape (H, W, 3)."""
    return np.asarray(image, dtype=np.float32) / 255.0


def preprocess_image(
    image: Image.Image, size: tuple[int, int] = constants.MODEL_INPUT_SIZE
) -> np.ndarray:
    """Full inference preprocessing; returns a batch of one: (1, H, W, 3)."""
    return normalize(resize_image(image, size))[np.newaxis, ...]
