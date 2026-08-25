"""
Fusion layer.

Takes the scores from the 3 techniques (CNN, ELA, noise) and combines
them into one final verdict + confidence, using a small Logistic
Regression model. This is the "AI/ML model that takes inputs from all
detection techniques and gives a final decision" from the project
objectives.

Why Logistic Regression and not another deep model: it's fast to train
on a small feature table (3 numbers per image), and -- importantly for
your viva -- it's easy to explain. You can literally show the learned
weight for each technique and say "this is how much each one
contributed."
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURE_ORDER = ["splicing_cnn", "ela", "noise"]
MODEL_PATH = "models/fusion_model.pkl"


def scores_to_vector(scores: dict) -> np.ndarray:
    """scores: {'splicing_cnn': 0.8, 'ela': 0.3, 'noise': 0.6} -> ordered array."""
    return np.array([[scores[name] for name in FEATURE_ORDER]])


def train_fusion(X: np.ndarray, y: np.ndarray, out_path: str = MODEL_PATH) -> LogisticRegression:
    """
    X: (n_samples, 3) - technique scores for each training image
    y: (n_samples,)   - 1 = forged, 0 = authentic
    Produce these by running all 3 techniques over your CASIA 2.0 set
    once (see notebooks/03_train_fusion_model.py) and saving the scores.
    """
    clf = LogisticRegression()
    clf.fit(X, y)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)

    return clf


def load_fusion(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(scores: dict, model_path: str = MODEL_PATH) -> dict:
    """
    Returns the final verdict + confidence.
    If no trained fusion model exists yet, falls back to a simple
    average of the 3 scores -- so the pipeline is testable end-to-end
    before the fusion model has been trained.
    """
    clf = load_fusion(model_path)
    x = scores_to_vector(scores)

    if clf is not None:
        proba = clf.predict_proba(x)[0][1]  # probability of "forged" class
        contributions = dict(zip(FEATURE_ORDER, clf.coef_[0] * x[0]))
    else:
        proba = float(np.mean(x))
        contributions = dict(zip(FEATURE_ORDER, x[0]))

    verdict = "Forged" if proba >= 0.5 else "Authentic"
    return {
        "verdict": verdict,
        "confidence": round(float(proba if verdict == "Forged" else 1 - proba), 4),
        "raw_forgery_probability": round(float(proba), 4),
        "technique_contributions": {k: round(float(v), 4) for k, v in contributions.items()},
    }
