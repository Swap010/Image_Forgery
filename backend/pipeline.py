"""
Ties preprocessing -> techniques -> fusion -> localization into one
function. Used by main.py (the API) and can also be imported directly
for testing/evaluation scripts.
"""

import base64
import io
from PIL import Image

from backend.preprocessing import preprocess, preprocess_for_display
from backend.techniques import ela, noise_cfa, splicing_cnn
from backend import fusion_model
from backend.localization import build_localization

_model_cache = {}


def get_cnn_model():
    if "cnn" not in _model_cache:
        _model_cache["cnn"] = splicing_cnn.load_model()
    return _model_cache["cnn"]


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_pipeline(img: Image.Image) -> dict:
    arr = preprocess(img)  # (128,128,3) float32 0-1, feeds the CNN

    cnn_out = splicing_cnn.run(get_cnn_model(), arr)
    ela_out = ela.run(img)
    noise_out = noise_cfa.run(img)

    scores = {
        "splicing_cnn": cnn_out["score"],
        "ela": ela_out["score"],
        "noise": noise_out["score"],
    }
    visuals = {
        "splicing_cnn": cnn_out["visual"],
        "ela": ela_out["visual"],
        "noise": noise_out["visual"],
    }

    decision = fusion_model.predict(scores)

    display_img = preprocess_for_display(img)
    highlighted = build_localization(display_img, visuals)

    return {
        "verdict": decision["verdict"],
        "confidence": decision["confidence"],
        "technique_scores": {k: round(v, 4) for k, v in scores.items()},
        "technique_contributions": decision["technique_contributions"],
        "highlighted_image_base64": image_to_base64(highlighted),
    }
