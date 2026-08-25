"""
Error Level Analysis (ELA).

Idea: re-save the image at a known JPEG quality, then compare it to the
original pixel-by-pixel. Untouched regions compress fairly evenly.
Edited regions (pasted-in, retouched, spliced) often show a different,
uneven error pattern because they were compressed a different number of
times before being combined into this image.

Returns:
    score  -> float 0-1, higher means more likely forged
    ela_map -> the amplified difference image, used later for localization
"""

import io
import numpy as np
from PIL import Image, ImageChops


def compute_ela(img: Image.Image, quality: int = 90, amplify: int = 20):
    # Re-save at a known quality in memory (no disk write needed)
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(img.convert("RGB"), resaved)
    diff_arr = np.array(diff).astype("float32")

    # Amplify so differences are visible / usable as a score
    amplified = np.clip(diff_arr * amplify, 0, 255).astype("uint8")
    ela_map = Image.fromarray(amplified)

    return diff_arr, ela_map


def ela_score(img: Image.Image) -> float:
    """
    Turn the raw difference map into a single 0-1 forgery score.
    Approach: look at how *uneven* the error is across the image.
    A real, untouched photo tends to have fairly uniform compression noise.
    A forged image tends to have patches of much higher error than the rest.
    """
    diff_arr, _ = compute_ela(img)
    gray = diff_arr.mean(axis=2)  # average over R,G,B

    # Unevenness = how much the error varies compared to its own average.
    mean_err = gray.mean()
    std_err = gray.std()
    if mean_err < 1e-6:
        return 0.0

    unevenness = std_err / (mean_err + 1e-6)

    # Squash into 0-1 with a simple, tunable curve.
    # These constants are a reasonable starting point -- retune them
    # once you run this against your labeled CASIA 2.0 set.
    score = 1 / (1 + np.exp(-(unevenness - 1.5)))
    return float(np.clip(score, 0.0, 1.0))


def run(img: Image.Image) -> dict:
    diff_arr, ela_map = compute_ela(img)
    return {
        "technique": "ela",
        "score": ela_score(img),
        "visual": ela_map,  # PIL Image, usable for localization/report
    }
