"""
Noise inconsistency analysis.

Idea: every camera sensor leaves a faint, fairly consistent noise pattern
across a real photo. When a region is copied from elsewhere or edited,
its noise pattern usually doesn't match the rest of the image.

Approach: extract a high-frequency "noise residual" (image minus a
denoised version of itself), split into blocks, and check how much the
noise level varies from block to block. Real photos -> fairly even
noise. Forged photos -> some blocks look noticeably different.
"""

import numpy as np
import cv2
from PIL import Image


def noise_residual(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB")).astype("float32")
    denoised = cv2.GaussianBlur(arr, (5, 5), 0)
    residual = arr - denoised
    return residual


def block_noise_map(residual: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Std-dev of the noise residual per block -> a coarse noise map."""
    h, w, _ = residual.shape
    gray_res = residual.mean(axis=2)

    bh, bw = h // block_size, w // block_size
    noise_map = np.zeros((bh, bw), dtype="float32")

    for i in range(bh):
        for j in range(bw):
            block = gray_res[i * block_size:(i + 1) * block_size,
                              j * block_size:(j + 1) * block_size]
            noise_map[i, j] = block.std()

    return noise_map


def noise_score(img: Image.Image) -> float:
    residual = noise_residual(img)
    noise_map = block_noise_map(residual)

    if noise_map.size == 0 or noise_map.mean() < 1e-6:
        return 0.0

    # Coefficient of variation across blocks: how inconsistent the
    # noise level is across the image. High = suspicious.
    cv_score = noise_map.std() / (noise_map.mean() + 1e-6)

    score = 1 / (1 + np.exp(-(cv_score - 1.2)))
    return float(np.clip(score, 0.0, 1.0))


def run(img: Image.Image) -> dict:
    residual = noise_residual(img)
    noise_map = block_noise_map(residual)
    return {
        "technique": "noise",
        "score": noise_score(img),
        "visual": noise_map,  # small 2D array, used for localization
    }
