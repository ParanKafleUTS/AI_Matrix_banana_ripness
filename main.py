"""
main.py – FastAPI inference server for banana ripeness classification.

Endpoints:
  GET  /            → serves static/index.html
  GET  /health      → health check
  POST /predict     → accepts an image file, returns predicted class & confidence
"""

import io
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "saved_models/EfficientNetB0_banana_ripeness.pth")
NUM_CLASSES = 6
CLASS_NAMES = ["freshripe", "freshunripe", "overripe", "ripe", "rotten", "unripe"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Image pre-processing (identical to val/test transforms used during training)
# ---------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(path: str) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


_model: nn.Module | None = None


def get_model() -> nn.Module:
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(f"Model weights not found at '{MODEL_PATH}'. Run train.py first.")
        _model = load_model(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Banana Ripeness Classifier", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Banana Ripeness Classifier</h1>", status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    model = get_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, pred_idx = torch.max(probs, dim=0)

    return JSONResponse({
        "predicted_class": CLASS_NAMES[pred_idx.item()],
        "confidence": round(confidence.item(), 4),
        "all_probabilities": {
            cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASS_NAMES)
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
