"""
Build the feature table (run all 3 techniques over CASIA 2.0) and
train the fusion model on top of it.

Run this AFTER 02_train_splicing_cnn.py has produced models/splicing_cnn.pth.

    python notebooks/03_train_fusion_model.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from backend.preprocessing import preprocess, load_image  # noqa: E402
from backend.techniques import ela, noise_cfa, splicing_cnn  # noqa: E402
from backend.fusion_model import train_fusion, FEATURE_ORDER  # noqa: E402

AU_DIR = "data/raw/CASIA2/Au"
TP_DIR = "data/raw/CASIA2/Tp"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_scores():
    model = splicing_cnn.load_model()

    rows, labels = [], []
    for label, folder in [(0, AU_DIR), (1, TP_DIR)]:
        if not os.path.isdir(folder):
            print(f"Missing folder: {folder}")
            continue
        for fname in sorted(os.listdir(folder)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                continue
            path = os.path.join(folder, fname)
            try:
                img = load_image(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

            arr = preprocess(img)
            cnn_score = splicing_cnn.predict(model, arr)
            ela_s = ela.ela_score(img)
            noise_s = noise_cfa.noise_score(img)

            rows.append([cnn_score, ela_s, noise_s])  # must match FEATURE_ORDER
            labels.append(label)

    return np.array(rows), np.array(labels)


def main():
    X, y = collect_scores()
    if len(X) == 0:
        print("No data found. Point AU_DIR/TP_DIR at your CASIA 2.0 folders and re-run.")
        return

    print(f"Collected {len(X)} samples. Feature order: {FEATURE_ORDER}")
    clf = train_fusion(X, y)
    print("Fusion model trained and saved to models/fusion_model.pkl")
    print("Learned weights:", dict(zip(FEATURE_ORDER, clf.coef_[0])))


if __name__ == "__main__":
    main()
