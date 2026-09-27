"""Synthetic optical-image degradations, not a storm or damage simulator.

Versioned, bounded transforms preserve image size and labels. Evaluation RNGs
depend on image identity, not iteration order, worker count or model seed.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CONDITIONS = ("cloud_haze", "illumination", "resolution", "sensor_noise")
VERSION = "optical-proxies-v1"
PROFILES = ("baseline", "weather_robust")


def condition_seed(seed: int, identity: str, condition: str) -> int:
    key = f"{VERSION}|{seed}|{identity}|{condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def degrade(image: Image.Image, condition: str, severity: int, seed: int) -> Image.Image:
    """Apply one proxy at severity 1/2/3; severity 0 is a clean copy.

    Cloud/haze: smooth partial veil, not physical cloud coverage or a cyclone.
    Illumination: exposure/contrast proxy, not a day/night or season label.
    Resolution: resampling/blur, not a calibrated ground sampling distance.
    Sensor noise: additive RGB noise, not a particular sensor or SAR speckle.
    """
    if condition not in ("clean", *CONDITIONS):
        raise ValueError(f"Unknown condition: {condition}")
    if severity not in (0, 1, 2, 3) or (condition == "clean" and severity != 0):
        raise ValueError("severity must be 0..3; clean requires 0")
    if severity == 0:
        return image.convert("RGB").copy()
    image = image.convert("RGB")
    rng = np.random.default_rng(seed)
    index = severity - 1
    if condition == "cloud_haze":
        # Keep the spatial pattern fixed across severity levels for paired tests.
        field = rng.integers(0, 256, size=(6, 6), dtype=np.uint8)
        mask = Image.fromarray(field).resize(image.size, Image.Resampling.BICUBIC)
        mask = mask.filter(ImageFilter.GaussianBlur(min(image.size) * .035))
        texture = np.asarray(mask, dtype=np.float32) / 255.0
        opacity = (.10, .20, .30)[index] + (.25, .40, .55)[index] * texture
        pixels = np.asarray(image, dtype=np.float32)
        result = pixels * (1 - opacity[..., None]) + 245 * opacity[..., None]
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    if condition == "illumination":
        # A given sample remains on the same bright/dark branch at all levels.
        bright = bool(rng.integers(2))
        factor = ((1.15, 1.35, 1.60) if bright else (.80, .60, .40))[index]
        result = ImageEnhance.Brightness(image).enhance(factor)
        return ImageEnhance.Contrast(result).enhance((.95, .85, .75)[index])
    if condition == "resolution":
        divisor = (2, 4, 8)[index]
        size = tuple(max(1, dimension // divisor) for dimension in image.size)
        result = image.resize(size, Image.Resampling.BOX).resize(image.size, Image.Resampling.BILINEAR)
        return result.filter(ImageFilter.GaussianBlur(min(image.size) / 224 * (.3, .6, 1.0)[index]))
    pixels = np.asarray(image, dtype=np.float32)
    noise = rng.normal(0, (4, 10, 20)[index], size=pixels.shape)
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))


class RandomDegradation:
    """Keep half the training samples clean; apply one mild/moderate proxy otherwise."""

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < .5:
            return image
        return degrade(image, random.choice(CONDITIONS), random.choice((1, 2)), random.getrandbits(64))


class EvaluationDegradation:
    def __init__(self, condition: str, severity: int, seed: int = 2026):
        if condition not in ("clean", *CONDITIONS):
            raise ValueError(f"Unknown condition: {condition}")
        if (condition == "clean" and severity != 0) or (condition != "clean" and severity not in (1, 2, 3)):
            raise ValueError("Use severity 0 for clean, 1..3 for degraded images")
        self.condition, self.severity, self.seed = condition, severity, seed

    def __call__(self, image: Image.Image, identity: str) -> Image.Image:
        return degrade(image, self.condition, self.severity,
                       condition_seed(self.seed, identity, self.condition))


def profile_metadata(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unknown augmentation profile: {profile}")
    return {
        "profile": profile, "version": VERSION,
        "conditions": list(CONDITIONS) if profile == "weather_robust" else [],
        "clean_probability": .5 if profile == "weather_robust" else 1.,
        "training_severities": [1, 2] if profile == "weather_robust" else [],
        "synthetic_only": True,
    }
