# 🚀 Image Forgery Detection — Standalone Run Guide

A complete AI pipeline that detects image tampering using **three combined techniques**
(CNN + Error Level Analysis + Noise CFA Inconsistency), fused by a trained Logistic Regression,
exposed via a **FastAPI backend** and **Streamlit frontend**.

```
Upload Image  →  { ELA, Noise-CFA, MobileNetV2-CNN }  →  Logistic Regression Fusion
              →  Verdict + Confidence % + Per-technique scores + Heatmap overlay
```

---

## 📋 Contents

1. [Prerequisites (Install these first)](#1-prerequisites)
2. [Get the project](#2-get-the-project)
3. [Download the CASIA 2.0 dataset](#3-download-the-casia-20-dataset)
4. [Arrange the dataset folders](#4-arrange-the-dataset-folders)
5. [Install Python dependencies](#5-install-python-dependencies)
6. [Train the models (run once)](#6-train-the-models-run-once)
7. [Start the backend + frontend](#7-start-the-backend--frontend)
8. [Use the app](#8-use-the-app)
9. [Folder structure](#9-folder-structure)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Install these on your machine **before** starting:

| Tool | Minimum | Get it |
|---|---|---|
| **Python** | 3.10+ | https://www.python.org/downloads/ (check "Add Python to PATH" in installer) |
| **pip** | latest | Bundled with Python (run `python -m pip install --upgrade pip` after install) |
| **Git** (optional) | any | https://git-scm.com/ |
| **Web browser** | any | Chrome / Edge / Firefox |

💡 **GPU users (optional but recommended):** If you have an NVIDIA GPU, install
[PyTorch with CUDA](https://pytorch.org/get-started/locally/) *before* step 5.
Training will be 10–50× faster. Default CPU training still works, just slower.

Verify install:
```bash
python --version     # should print 3.10.x, 3.11.x, or 3.12.x
pip --version
```

---

## 2. Get the project

If you already have the project folder, skip to step 3. Otherwise:

```bash
# Option A — clone with Git (if it lives in a repo)
git clone <YOUR-REPO-URL> image-forgery-detection
cd image-forgery-detection

# Option B — just copy/extract the project folder
```

Project root (the folder this guide lives in) looks like this:
```
image-forgery-detection/
├── backend/
├── frontend/
├── notebooks/
├── tests/
├── requirements.txt
└── (this guide)
```

---

## 3. Download the CASIA 2.0 dataset

The project needs **CASIA 2.0** (authentic + tampered JPEG images) to train the models.
This is NOT included in the project — download it here (~2.6 GB):

### ✅ Recommended — click the revised full-dataset link:
> **https://bit.ly/2QazgkG**  ← click this bit.ly link (hosted by dataset maintainer Nam Pham)

### Alternative mirrors (if above is down):
| Source | Link |
|---|---|
| GitHub (groundtruth + full data link) | https://github.com/namtpham/casia2groundtruth  (look for *"Revised dataset"* link) |
| Kaggle mirror | https://www.kaggle.com/code/animeshshukla20/casia-2-0-dataset-for-image-forgery-detecion/input  (free Kaggle account required) |

### ❌ What NOT to download
- Do NOT download only `CASIA2.0_Groundtruth.zip` — that contains **black/white mask PNGs**,
  not the real tampered/genuine JPEGs. You need the big ~2.6 GB zip.

Extract the downloaded zip anywhere on your disk (e.g. `C:\Downloads\CASIA2\`).
Inside you'll see thousands of files:
```
Au_ani_00001.jpg       ← starts with "Au_"  → AUTHENTIC
Tp_D_CRN_..._11818.jpg ← starts with "Tp_"  → FORGED / TAMPERED
Tp_..._gt.png          ← ends with "_gt.png"  → MASKS (IGNORE THESE)
```

---

## 4. Arrange the dataset folders

The project expects **this exact folder structure inside the project root**.
Create empty folders manually, or run the one-liner commands below (copy-paste into your terminal).

### 🖥️ If you use Windows PowerShell:
```powershell
cd image-forgery-detection
New-Item -ItemType Directory -Force -Path "data\raw\CASIA2\Au", "data\raw\CASIA2\Tp", "models"
```

### 🍎 If you use macOS / Linux (Bash):
```bash
cd image-forgery-detection
mkdir -p data/raw/CASIA2/Au data/raw/CASIA2/Tp models
```

You now have:
```
image-forgery-detection/
├── data/
│   └── raw/
│       └── CASIA2/
│           ├── Au/    ← EMPTY — put authentic images here
│           └── Tp/    ← EMPTY — put forged images here
├── models/            ← EMPTY — trained weights save here
...
```

### Copy the images in

#### Option 1 — File Explorer (easiest for beginners)
1. Open the folder where you extracted CASIA 2.0 in step 3.
2. In the search bar type `Au_*.jpg` → Select ALL results → **Copy** (`Ctrl+C`) →
   **Paste** into `image-forgery-detection\data\raw\CASIA2\Au\`
3. Search for `Tp_*.jpg` → Select ALL → **Copy** →
   **Paste** into `image-forgery-detection\data\raw\CASIA2\Tp\`
4. **Do NOT copy any `*_gt.png` mask files** — skip them.

#### Option 2 — One-liner commands (fast for ~12k files)
**Windows PowerShell** (replace `C:\PATH\TO\EXTRACTED\CASIA` with the folder you extracted to in step 3):
```powershell
$src = "C:\PATH\TO\EXTRACTED\CASIA"
Get-ChildItem -Path $src -Recurse -Filter "Au_*.jpg" | Copy-Item -Destination ".\data\raw\CASIA2\Au\"
Get-ChildItem -Path $src -Recurse -Filter "Tp_*.jpg" | Copy-Item -Destination ".\data\raw\CASIA2\Tp\"
```

**macOS / Linux Bash** (replace with your extracted path):
```bash
SRC="/path/to/extracted/CASIA2"
find "$SRC" -type f -name "Au_*.jpg" -exec cp {} ./data/raw/CASIA2/Au/ \;
find "$SRC" -type f -name "Tp_*.jpg" -exec cp {} ./data/raw/CASIA2/Tp/ \;
```

### Verify the counts
```powershell
# Windows
(dir data\raw\CASIA2\Au -File | Measure-Object).Count
(dir data\raw\CASIA2\Tp -File | Measure-Object).Count

# macOS / Linux
ls data/raw/CASIA2/Au | wc -l
ls data/raw/CASIA2/Tp | wc -l
```

Expected (or close to it — any subset works for training):
```
Au ≈ 7,491 images
Tp ≈ 5,123 images
```

---

## 5. Install Python dependencies

From the **project root** (`cd image-forgery-detection`):

```bash
# (Recommended) Create a virtual environment first
# ——— Windows ———
python -m venv .venv
.venv\Scripts\activate

# ——— macOS / Linux ———
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (same for all OS)
pip install --upgrade pip
pip install -r requirements.txt
```

⚠️ **Heads up (first run only):** The first time any script loads MobileNetV2,
PyTorch will automatically download ImageNet pretrained weights (~14 MB) from the internet.
This is normal and only happens once. Your machine needs internet access for that.

---

## 6. Train the models (run once)

Run these **in order** from the project root. Outputs save to `models/`.

### Step 6.1 — Train the Splicing CNN
```bash
python notebooks/02_train_splicing_cnn.py
```

**What it does:** Takes MobileNetV2 (pretrained on ImageNet), swaps the classifier head,
and fine-tunes it on your CASIA 2.0 authentic/forged split.

**Typical runtime** (approximate):
| Hardware | Time |
|---|---|
| Laptop CPU only | 2–4 hours |
| NVIDIA RTX 3060 (or better) w/ CUDA | 5–15 minutes |
| Google Colab (free GPU) | 10–30 minutes → recommended 👍 |

**Expected output:**
```
Epoch 1/8 - loss: 0.6272 - val_acc: 0.6739
...
Epoch 8/8 - loss: 0.3746 - val_acc: 0.6892
Best val accuracy: 0.6982. Weights saved to models/splicing_cnn.pth
```
→ Produces: `models/splicing_cnn.pth` (~9 MB)

### Step 6.2 — Train the Fusion Model
```bash
python notebooks/03_train_fusion_model.py
```

**What it does:** Runs ALL 3 detection techniques (CNN + ELA + Noise) on EVERY image
in CASIA 2.0 → builds a score table → trains a Logistic Regression classifier on top.
This learns **which technique to trust most**.

**Runtime:** ~20–90 minutes (CPU) — slower than training, because it evaluates 3 techniques
on 12k+ images one by one. The CNN weights file is loaded automatically from step 6.1.

**Expected output:**
```
[splicing_cnn] Loaded fine-tuned weights from models/splicing_cnn.pth
Collected 12614 samples. Feature order: ['splicing_cnn', 'ela', 'noise']
Fusion model trained and saved to models/fusion_model.pkl
Learned weights: {'splicing_cnn': 4.39, 'ela': -3.85, 'noise': 2.24}
```
→ Produces: `models/fusion_model.pkl` (<1 KB, tiny pickle)

---

## 7. Start the backend + frontend

You need **two separate terminal windows** (both in project root, virtual env activated).

### Terminal 1 — FastAPI Backend
```bash
cd image-forgery-detection
# Windows:   .venv\Scripts\activate
# macOS:     source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Wait for this line:
```
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Terminal 2 — Streamlit Frontend
```bash
cd image-forgery-detection
# Windows:   .venv\Scripts\activate
# macOS:     source .venv/bin/activate
streamlit run frontend/app.py --server.port 8501
```

Wait for this line:
```
Local URL: http://localhost:8501
```

That's it! Both apps are running.

---

## 8. Use the app

1. Open your browser and go to: **http://localhost:8501**
2. Click **Browse files** and upload a JPG / JPEG / PNG (max 10 MB).
   - Try the samples inside `tests/`: `sample_authentic.jpg` or `sample_forged.jpg`
3. Click the **Analyze** button (appears after upload).
4. Wait 5–30 seconds (first run loads models into memory; subsequent runs are faster).

### What you'll see

| Panel | Description |
|---|---|
| **Uploaded image** (left column) | Your original image |
| **Highlighted result** (right column) | Heatmap overlay — red = suspected tampered regions (computed via Grad-CAM + ELA + noise residual fusion) |
| **Verdict badge** | `Authentic` (green) or `Forged` (red) + a **Confidence %** from the fusion model |
| **Per-technique scores** | JSON object with raw `splicing_cnn`, `ela`, `noise` scores (0.0 = authentic, 1.0 = forged for CNN/noise; inverse for ELA) |
| **Contribution weights** | JSON of how much each technique tilted the final verdict (uses the fusion LogisticRegression coefficients) |

### Also try the backend directly
FastAPI auto-generates docs. While the backend is running, open:
```
http://127.0.0.1:8000/docs
```
You can call `/health` or upload a file directly to the `/detect` endpoint from the browser.
Or via curl:
```bash
curl -X POST -F "file=@tests/sample_forged.jpg" http://127.0.0.1:8000/detect
```

---

## 9. Folder structure

```
image-forgery-detection/
├── backend/
│   ├── main.py                  # FastAPI server (/health, /detect)
│   ├── pipeline.py              # Wires preprocessing + 3 techniques + fusion + localization
│   ├── preprocessing.py         # Resize → RGB → Normalize (shared)
│   ├── fusion_model.py          # Logistic Regression fusion: train, predict, save/load
│   ├── localization.py          # Heatmap overlay on forged regions
│   └── techniques/
│       ├── ela.py               # Error Level Analysis score
│       ├── noise_cfa.py         # Noise / CFA inconsistency score
│       └── splicing_cnn.py      # MobileNetV2 CNN model + Grad-CAM extractor + save/load
│
├── frontend/
│   └── app.py                   # Streamlit UI: uploads image → hits backend → shows results
│
├── notebooks/
│   ├── 02_train_splicing_cnn.py # Run 1st: fine-tunes CNN → models/splicing_cnn.pth
│   └── 03_train_fusion_model.py # Run 2nd: score table → fusion → models/fusion_model.pkl
│
├── tests/
│   ├── sample_authentic.jpg     # Quick test image (synthetic)
│   └── sample_forged.jpg        # Quick test image (synthetic)
│
├── data/                        # NOT committed — created by you in step 4
│   └── raw/
│       └── CASIA2/
│           ├── Au/              # 7,491 authentic CASIA 2.0 JPGs
│           └── Tp/              # 5,123 tampered CASIA 2.0 JPGs
│
├── models/                      # NOT committed — created by training
│   ├── splicing_cnn.pth         # CNN weights (output of notebook 02)
│   └── fusion_model.pkl         # Fusion weights (output of notebook 03)
│
├── requirements.txt             # All Python dependencies
└── README.md                    # Original project notes
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| **`ModuleNotFoundError: No module named 'fastapi'`** | You forgot to activate the venv OR you never ran `pip install -r requirements.txt`. Re-run step 5 in the project root. |
| **`Dataset not found at data/raw/CASIA2/Au / Tp`** when training | Folders are empty or misnamed. Re-do step 4 — make sure `data/raw/CASIA2/Au/` has actual `Au_*.jpg` files *directly inside* (no subfolders). |
| **`PIL.UnidentifiedImageError: cannot identify image file '...Thumbs.db'`** | Windows auto-created a `Thumbs.db` cache file. Delete it in both `Au/` and `Tp/`, or re-run step 4 copy commands. The latest training scripts also filter these out automatically. |
| **"Can't reach the backend" error in Streamlit** | Backend isn't running or crashed. Check Terminal 1 is still on. Re-run `uvicorn backend.main:app --port 8000`. |
| **CNN val accuracy jumps to 99% in epoch 1** | Likely **overfitting**. Reduce epochs to 2–3, or increase the 15% validation split in `02_train_splicing_cnn.py`, or add image augmentation. |
| **`OSError: [WinError 10013]` port already in use** | Kill whatever's using port 8000 / 8501 (or choose new ones: `--port 8001` / `--server.port 8502`). If frontend port changes, remember to update `API_URL` in `frontend/app.py`. |
| **Streamlit shows `Failed to scan component manifests: 'NoneType' object has no attribute 'lower'`** | Harmless warning — it's a Streamlit internal bug in some versions. Ignore it; the UI still works perfectly. |
| **Training is very slow on laptop CPU** | Upload the project to **Google Colab**, point dataset at Google Drive mounted content, and enable Runtime → Change Runtime Type → **GPU** (free T4). Training is ~20× faster. |
| **Backend says `Loaded default ImageNet weights` instead of fine-tuned** | Training either didn't run or the `.pth` file isn't in `models/`. Re-run step 6.1 and confirm `models/splicing_cnn.pth` exists (should be ~9 MB, not 0 KB). |
| **`RuntimeError: CUDA out of memory`** during training | Reduce `BATCH_SIZE` from 32 to 8 (or 4) in `notebooks/02_train_splicing_cnn.py` and run again. |

---

**Good luck!** 🎓 For training tips and interpretation of the fusion weights for your
Phase-II report / viva presentation, see the original `README.md` in the project root.
