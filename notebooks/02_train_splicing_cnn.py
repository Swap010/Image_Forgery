"""
Fine-tune the CNN on CASIA 2.0.

This is TRANSFER LEARNING, not training from scratch: the model starts
from MobileNetV2 pretrained on ImageNet, with most layers frozen. Only
the small classifier head (and the last few backbone layers) actually
get trained here, which is why the learning rate is lower and fewer
epochs are usually needed compared to training a network from zero.

Expected folder layout (matches what your team already built in
Objective 1 -- Authentic/Forged split):

    data/raw/CASIA2/Au/       <- authentic images
    data/raw/CASIA2/Tp/       <- forged/tampered images

Run from the project root:
    python notebooks/02_train_splicing_cnn.py

Requires internet access on your machine (downloads ImageNet weights
the first time it runs -- this is normal and only happens once, cached
afterward). Saves the fine-tuned weights to models/splicing_cnn.pth,
which backend/techniques/splicing_cnn.py loads automatically.

NOTE: This script needs the actual CASIA 2.0 dataset on disk, which
isn't part of this upload. Point AU_DIR / TP_DIR at wherever your team
has it (locally or in Colab) before running.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from backend.preprocessing import preprocess, load_image  # noqa: E402
from backend.techniques.splicing_cnn import ForgeryCNN, DEVICE  # noqa: E402

AU_DIR = "data/raw/CASIA2/Au"
TP_DIR = "data/raw/CASIA2/Tp"
WEIGHTS_OUT = "models/splicing_cnn.pth"
EPOCHS = 8          # fewer needed than training from scratch -- backbone is already pretrained
BATCH_SIZE = 32
LR = 1e-4            # lower LR: we're fine-tuning, not learning from zero
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class CasiaDataset(Dataset):
    def __init__(self, au_dir, tp_dir):
        self.paths = []
        self.labels = []
        for fname in sorted(os.listdir(au_dir)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                continue
            self.paths.append(os.path.join(au_dir, fname))
            self.labels.append(0.0)  # authentic
        for fname in sorted(os.listdir(tp_dir)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                continue
            self.paths.append(os.path.join(tp_dir, fname))
            self.labels.append(1.0)  # forged

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = load_image(self.paths[idx])
        except Exception:
            arr = np.zeros((128, 128, 3), dtype=np.float32)
            x = torch.from_numpy(arr).permute(2, 0, 1).float()
            y = torch.tensor([self.labels[idx]], dtype=torch.float32)
            return x, y
        arr = preprocess(img)  # (128,128,3) float32 0-1
        x = torch.from_numpy(arr).permute(2, 0, 1).float()
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y


def train():
    if not (os.path.isdir(AU_DIR) and os.path.isdir(TP_DIR)):
        print(f"Dataset not found at {AU_DIR} / {TP_DIR}.")
        print("Point AU_DIR/TP_DIR at your CASIA 2.0 folders and re-run.")
        return

    dataset = CasiaDataset(AU_DIR, TP_DIR)
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ForgeryCNN(pretrained=True, freeze_backbone=True).to(DEVICE)
    # Only optimize the parameters that are actually unfrozen (classifier
    # head + last few backbone layers) -- frozen layers don't need gradients.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    criterion = nn.BCELoss()

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                preds = (out > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(1, total)

        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(train_ds):.4f} "
              f"- val_acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_OUT)

    print(f"Best val accuracy: {best_val_acc:.4f}. Weights saved to {WEIGHTS_OUT}")


if __name__ == "__main__":
    train()
