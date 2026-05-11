"""
api.py — FastAPI REST endpoint for digit classification
MNIST Digit Recognizer Project

Run with:
    uvicorn src.api:app --reload
"""

import sys
import base64
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dataset import preprocess_image_tensor
from train import load_model

app = FastAPI(
    title="MNIST Digit Recognizer API",
    description="CNN-powered handwritten digit classification.",
    version="1.0.0",
)

# Serve static files (the drawing app)
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Resolve device once at startup — uses GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model at startup
@app.on_event("startup")
async def load():
    global model
    try:
        model = load_model(device=DEVICE)
        model.eval()
        print(f"Model loaded on: {DEVICE}")
        if DEVICE.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except FileNotFoundError:
        model = None
        print("WARNING: No trained model found. Run train.py first.")


# ── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    image: str  # base64-encoded PNG or JPEG (28×28 grayscale)


class PredictResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]  # all 10 class probabilities


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Serve the drawing web app."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "ok", "message": "MNIST Digit Recognizer API. Draw app at /static/index.html"}


@app.get("/health")
def health():
    return {"model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Classify a handwritten digit image.

    Request body:
        { "image": "<base64-encoded PNG>" }

    Response:
        { "digit": 7, "confidence": 0.9982, "probabilities": [...] }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    # Decode base64 image
    try:
        img_bytes = base64.b64decode(req.image)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")  # force grayscale
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Preprocess and run inference
    try:
        tensor = preprocess_image_tensor(pil_img).to(DEVICE)  # move to GPU if available
        with torch.no_grad():
            log_probs = model(tensor)
            probs = torch.exp(log_probs).squeeze().tolist()   # list of 10

        digit = int(np.argmax(probs))
        confidence = round(float(probs[digit]), 6)

        return PredictResponse(
            digit=digit,
            confidence=confidence,
            probabilities=[round(p, 6) for p in probs],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
