"""
CNN-based forgery detector -- now using TRANSFER LEARNING.

Why this changed from the first version:
The original ForgeryCNN was a custom architecture trained from zero,
which needs a lot of labeled data to learn well. Nobody has pretrained
weights for a custom architecture, so "from scratch" was the only option
for that design.

This version instead starts from MobileNetV2, pretrained on ImageNet
(1.4 million general photos). That pretrained network already knows how
to recognize edges, textures, and shapes -- skills that transfer well to
spotting forgery artifacts. We freeze most of it and only train a small
new classifier head on top, using YOUR CASIA 2.0 data. This needs far
less data and far less training time than starting from zero, which
fits a student dataset and deadline much better.

Grad-CAM still works the same way, targeting MobileNetV2's last
convolutional layer instead of a custom one.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization stats -- required because the pretrained
# backbone was trained on images normalized this way.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BACKBONE_INPUT_SIZE = 224  # MobileNetV2's expected input size


class ForgeryCNN(nn.Module):
    """
    MobileNetV2 backbone (pretrained on ImageNet) + a small new
    classifier head, trained on CASIA 2.0.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()

        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        try:
            backbone = models.mobilenet_v2(weights=weights)
            self.using_pretrained_weights = pretrained
        except Exception as e:
            # No internet access to download weights (e.g. this sandbox).
            # Falls back to the same architecture with random weights, so
            # the pipeline still runs end-to-end. On your own machine,
            # with normal internet access, the download will succeed.
            print(f"[splicing_cnn] Could not download pretrained weights ({e}). "
                  f"Using random init instead -- fine for testing the pipeline, "
                  f"but train before using real results.")
            backbone = models.mobilenet_v2(weights=None)
            self.using_pretrained_weights = False

        self.features = backbone.features   # convolutional layers (kept)
        self.last_conv_channels = 1280       # MobileNetV2's feature output channels

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            # Unfreeze just the last few layers so the model can still
            # adapt to forgery-specific textures, not just generic ones.
            for param in self.features[-3:].parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.last_conv_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._gradients = None
        self._activations = None

    def activations_hook(self, grad):
        self._gradients = grad

    def forward(self, x):
        feat = self.features(x)  # (B, 1280, 7, 7) for a 224x224 input
        if feat.requires_grad:
            feat.register_hook(self.activations_hook)
        self._activations = feat
        out = self.classifier(feat)
        return torch.sigmoid(out)


def load_model(weights_path: str = "models/splicing_cnn.pth") -> ForgeryCNN:
    model = ForgeryCNN(pretrained=True, freeze_backbone=True).to(DEVICE)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"[splicing_cnn] Loaded fine-tuned weights from {weights_path}")
    except FileNotFoundError:
        # Not fine-tuned on CASIA 2.0 yet -- see notebooks/02_train_splicing_cnn.py.
        # Backbone is still ImageNet-pretrained even without this file,
        # so predictions are meaningful sooner than a from-scratch model
        # would be, but the classifier head is still untrained until you
        # run the training script.
        print(f"[splicing_cnn] No fine-tuned weights at {weights_path} yet. "
              f"Run notebooks/02_train_splicing_cnn.py to train the classifier head.")
    model.eval()
    return model


def to_tensor(arr_hwc: np.ndarray) -> torch.Tensor:
    """
    arr_hwc: (128,128,3) float32 0-1 -- this is your team's existing
    Objective-1 preprocessed format.
    Internally resized to 224x224 and ImageNet-normalized, because
    that's what the pretrained backbone expects. Your 128x128 pipeline
    stays the "official" preprocessing step for the report; this resize
    is just an implementation detail of feeding the pretrained model.
    """
    t = torch.from_numpy(arr_hwc).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,128,128)
    t = F.interpolate(t, size=(BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE),
                       mode="bilinear", align_corners=False)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.to(DEVICE)


def predict(model: ForgeryCNN, arr_hwc: np.ndarray) -> float:
    x = to_tensor(arr_hwc)
    with torch.no_grad():
        out = model(x)
    return float(out.item())


def grad_cam(model: ForgeryCNN, arr_hwc: np.ndarray) -> np.ndarray:
    """
    Returns a (7,7) heatmap (MobileNetV2's final feature map resolution)
    showing which regions most influenced the decision. Upscaled onto
    the original image in localization.py.
    """
    model.zero_grad()
    x = to_tensor(arr_hwc)
    x.requires_grad_(True)
    out = model(x)
    out.backward()

    gradients = model._gradients[0]      # (1280, 7, 7)
    activations = model._activations[0]  # (1280, 7, 7)

    weights = gradients.mean(dim=(1, 2))  # (1280,)
    cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=DEVICE)
    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.detach().cpu().numpy()


def run(model: ForgeryCNN, arr_hwc: np.ndarray) -> dict:
    score = predict(model, arr_hwc)
    heatmap = grad_cam(model, arr_hwc)
    return {
        "technique": "splicing_cnn",
        "score": score,
        "visual": heatmap,
    }
