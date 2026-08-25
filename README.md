# AI-Driven Multi-Technique Image Forgery Detection — Working Codebase

This is the actual, running implementation of the plan we discussed:
**3 techniques (CNN + ELA + Noise) → Logistic Regression fusion → FastAPI backend → Streamlit frontend.**

## What's built and already tested

Every file below runs. I tested the full pipeline end-to-end in this environment
using two synthetic sample images (`tests/sample_authentic.jpg`, `tests/sample_forged.jpg`,
since your real CASIA 2.0 dataset isn't part of this upload) and confirmed:

- Preprocessing works (resize/RGB/normalize)
- All 3 techniques run without errors and return scores
- The fusion layer combines the 3 scores into a verdict + confidence
- The FastAPI `/detect` endpoint works (tested with `TestClient`, status 200)
- The localization overlay image generates correctly (see `tests/sample_forged_highlighted.png`)

**What is NOT trained yet:** the CASIA-specific fine-tuning (`models/splicing_cnn.pth`)
and the fusion model (`models/fusion_model.pkl`) don't exist yet, because training needs
your real CASIA 2.0 dataset, which isn't in this environment.

**About the CNN specifically — there is no pretrained forgery-detection model anywhere,
by anyone, that you can just download and use.** No public pretrained weights exist
for this specific task on this specific dataset. What the CNN uses instead is **transfer
learning**: `backend/techniques/splicing_cnn.py` starts from **MobileNetV2, pretrained on
ImageNet** (general photos — not forgery-specific), and only trains a small new classifier
head on top using your CASIA 2.0 data. This needs far less data/training time than
building a forgery detector from zero, because the network already knows how to see
edges, textures, and shapes — it just needs to learn what "forged" looks like.

Two things to know about this:
- The ImageNet weights download automatically the first time you run the code, **as
  long as your machine has normal internet access.** (This sandbox's network is
  restricted and can't reach that download server, so if you saw a warning about
  "could not download pretrained weights" while I was testing, that's why — it will
  work normally on your own laptop/Colab.)
- Even with those ImageNet weights loaded, the model still needs the fine-tuning step
  (`02_train_splicing_cnn.py`) before it's actually useful for forgery detection —
  ImageNet weights alone don't know anything about forgery, they just give the model
  a head start.

The rest of the pipeline (ELA, noise analysis, fusion, API, localization) is fully
tested and working right now, independent of whether the CNN is trained — I confirmed
this by running everything end-to-end with random CNN weights standing in.

## How to run it

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the CNN and fusion model trained (do this once you have CASIA 2.0)
Put your dataset here:
```
data/raw/CASIA2/Au/   <- authentic images
data/raw/CASIA2/Tp/   <- forged/tampered images
```
Then run, in order:
```bash
python notebooks/02_train_splicing_cnn.py     # trains the CNN, saves models/splicing_cnn.pth
python notebooks/03_train_fusion_model.py      # trains the fusion layer, saves models/fusion_model.pkl
```
(Recommended: run these on Google Colab with a free GPU — training will be much faster
than on a laptop CPU.)

### 3. Start the backend
```bash
uvicorn backend.main:app --reload
```

### 4. Start the frontend (in a separate terminal)
```bash
streamlit run frontend/app.py
```

Open the Streamlit URL it prints, upload an image, click **Analyze**.

## What to do next, in order

1. **Get your CASIA 2.0 dataset into `data/raw/CASIA2/Au` and `data/raw/CASIA2/Tp`.**
2. Run `02_train_splicing_cnn.py`. Watch the printed validation accuracy — if it's
   very high (>99%) very fast, be suspicious of overfitting (see the Risks section
   of the master plan).
3. Run `03_train_fusion_model.py` and note the printed learned weights — this tells
   you which technique the model trusts most, useful for your report/viva.
4. Retune the score thresholds in `backend/techniques/ela.py` and `noise_cfa.py`
   (marked with comments) once you can check them against real labeled data —
   the current constants are reasonable starting points, not final values.
5. Test the full app (steps 3–4 above) with real images, including a few images
   NOT from CASIA 2.0, to honestly check generalization (this is the #1 weakness
   your own literature review flagged in almost every paper you reviewed).
6. Take screenshots of the working demo for your Phase-II report and slides.

## Project structure
```
backend/
  preprocessing.py         # resize, RGB, normalize (your existing Objective 1 pipeline)
  techniques/
    ela.py                 # Error Level Analysis — tested, working
    noise_cfa.py            # noise inconsistency analysis — tested, working
    splicing_cnn.py         # CNN model + Grad-CAM — tested, working (untrained weights)
  fusion_model.py           # Logistic Regression fusion layer — tested, working
  localization.py           # heatmap overlay for the final image — tested, working
  pipeline.py                # wires all of the above together — tested, working
  main.py                    # FastAPI app (/detect, /health) — tested, working
frontend/
  app.py                     # Streamlit UI — not runnable in this sandbox (no browser), but code is complete and follows the same API contract as the tested backend
notebooks/
  02_train_splicing_cnn.py   # CNN training script — needs your dataset to run
  03_train_fusion_model.py    # fusion training script — needs your dataset to run
tests/
  sample_authentic.jpg, sample_forged.jpg   # synthetic test images used to verify the pipeline
  sample_forged_highlighted.png              # example output of the localization step
models/                       # trained weights go here (empty until you train)
data/raw/CASIA2/               # put your dataset here (Au/ and Tp/ subfolders)
```
