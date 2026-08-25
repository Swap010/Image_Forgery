"""
FastAPI backend.

Run with:
    uvicorn backend.main:app --reload

Then POST an image to http://127.0.0.1:8000/detect
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from backend.pipeline import run_pipeline

app = FastAPI(title="AI-Driven Multi-Technique Image Forgery Detection")

MAX_FILE_SIZE_MB = 10
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large (max {MAX_FILE_SIZE_MB}MB).")

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this file as an image.")

    result = run_pipeline(img)
    return result
