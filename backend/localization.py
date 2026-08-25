"""
Localization: turn a small heatmap (from Grad-CAM, ELA, or noise
analysis) into a highlighted version of the original image, so the
user can see *where* the forgery is, not just *that* it exists.
"""

import numpy as np
import cv2
from PIL import Image


def heatmap_to_overlay(original: Image.Image, heatmap_2d: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    heatmap_2d: any small 2D array of values (roughly 0-1), e.g. the
    (32,32) Grad-CAM output. Resized to match the original image and
    blended on top in a red/yellow "hot" color scheme.
    """
    orig_arr = np.array(original.convert("RGB"))
    h, w = orig_arr.shape[:2]

    heat = heatmap_2d.astype("float32")
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_resized = cv2.resize(heat, (w, h))
    heat_uint8 = np.uint8(255 * heat_resized)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = (orig_arr * (1 - alpha) + heat_color * alpha).astype("uint8")
    return Image.fromarray(overlay)


def build_localization(original: Image.Image, technique_visuals: dict) -> Image.Image:
    """
    Pick the best available heatmap to show. Priority: CNN Grad-CAM
    (most reliable), then noise map, then ELA map -- whichever is
    present in technique_visuals.
    """
    if "splicing_cnn" in technique_visuals:
        return heatmap_to_overlay(original, technique_visuals["splicing_cnn"])
    if "noise" in technique_visuals:
        return heatmap_to_overlay(original, technique_visuals["noise"])
    if "ela" in technique_visuals:
        ela_img = technique_visuals["ela"].convert("L")
        return heatmap_to_overlay(original, np.array(ela_img).astype("float32"))
    return original
