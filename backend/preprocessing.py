"""
Preprocessing module.
Matches the pipeline already completed in Objective 1 (Review-1 slides):
resize to 128x128, convert to RGB, normalize pixel values to 0-1.
"""

import numpy as np
from PIL import Image

TARGET_SIZE = (128, 128)


def load_image(file_bytes_or_path) -> Image.Image:
    """Load an image from a file path or in-memory bytes, force RGB."""
    img = Image.open(file_bytes_or_path)
    return img.convert("RGB")


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Resize to 128x128, keep RGB, normalize to 0-1.
    Returns a float32 numpy array of shape (128, 128, 3).
    This is the array used as input to the CNN.
    """
    resized = img.resize(TARGET_SIZE)
    arr = np.array(resized).astype("float32") / 255.0
    return arr


def preprocess_for_display(img: Image.Image, max_dim: int = 512) -> Image.Image:
    """
    A *separate* copy kept at higher resolution, only for showing the
    highlighted/marked result to the user (128x128 is too small to look at).
    Does not affect the model input.
    """
    img = img.copy()
    img.thumbnail((max_dim, max_dim))
    return img
