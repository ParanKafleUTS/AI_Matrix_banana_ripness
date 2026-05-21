"""
main.py  —  Banana Ripeness Classification API
===============================================
FastAPI backend — fully self-contained, no train.py required.

Features:
  POST /predict        Single image → class, confidence scores, Grad-CAM overlay
  POST /compare        Same image through all 4 models side-by-side
  GET  /history        Inference history (SQLite)
  DELETE /history      Clear history
  GET  /models         Loaded model status
  GET  /health         Health check
  GET  /              Serves the web UI (static/index.html)

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Requirements:
    pip install fastapi uvicorn python-multipart torch torchvision pillow
                numpy matplotlib aiofiles
"""

from __future__ import annotations

import base64
import io
import json
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.cm as mpl_cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["overripe", "ripe", "rotten", "unripe"]
NUM_CLASSES   = len(CLASS_NAMES)
INPUT_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
MODELS_DIR    = Path("models")
DB_PATH       = Path("inference_history.db")
STATIC_DIR    = Path("static")

MODEL_NAMES   = ["efficientnet", "efficientnetv2", "mobilenet", "resnet"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Recipes per class
# ─────────────────────────────────────────────────────────────────────────────
RECIPES: dict[str, dict] = {
    "overripe": {
        "status":      "Overripe",
        "emoji":       "🍌",
        "color":       "#f59e0b",
        "description": "Very sweet and soft — ideal for baking. Do not eat raw.",
        "tips":        "Freeze overripe bananas for up to 3 months for future baking.",
        "recipes": [
            {"name": "Banana Bread",        "time": "65 min",  "difficulty": "Easy"},
            {"name": "Banana Muffins",       "time": "30 min",  "difficulty": "Easy"},
            {"name": "Banana Smoothie",      "time": "5 min",   "difficulty": "Easy"},
            {"name": "Banana Nice Cream",    "time": "10 min",  "difficulty": "Easy"},
            {"name": "Banana Oat Cookies",   "time": "25 min",  "difficulty": "Easy"},
            {"name": "Banana Pudding",       "time": "20 min",  "difficulty": "Easy"},
        ],
    },
    "ripe": {
        "status":      "Perfectly Ripe",
        "emoji":       "✅",
        "color":       "#22c55e",
        "description": "At peak sweetness and flavor. Best eaten fresh or used immediately.",
        "tips":        "Refrigerate to slow ripening by 3–5 days. The peel darkens but the fruit stays fresh.",
        "recipes": [
            {"name": "Eat Fresh",            "time": "0 min",   "difficulty": "Very Easy"},
            {"name": "Banana Split",         "time": "10 min",  "difficulty": "Easy"},
            {"name": "Banana Foster",        "time": "15 min",  "difficulty": "Medium"},
            {"name": "Banana Pancakes",      "time": "20 min",  "difficulty": "Easy"},
            {"name": "Fruit Salad",          "time": "10 min",  "difficulty": "Easy"},
            {"name": "Banana Crepes",        "time": "30 min",  "difficulty": "Medium"},
        ],
    },
    "rotten": {
        "status":      "Rotten",
        "emoji":       "❌",
        "color":       "#ef4444",
        "description": "Fermented or mouldy — not safe to eat. Discard immediately.",
        "tips":        "Add to compost. Banana peels are rich in potassium and great for garden soil.",
        "recipes": [
            {"name": "Compost it 🌱",        "time": "—",       "difficulty": "—"},
            {"name": "Garden fertiliser",    "time": "—",       "difficulty": "—"},
        ],
    },
    "unripe": {
        "status":      "Unripe",
        "emoji":       "🟢",
        "color":       "#84cc16",
        "description": "Starchy and firm. Leave at room temperature for 2–4 days.",
        "tips":        "Speed up ripening by placing in a paper bag with an apple.",
        "recipes": [
            {"name": "Wait 2–4 days",        "time": "—",       "difficulty": "—"},
            {"name": "Fried Green Bananas",  "time": "15 min",  "difficulty": "Easy"},
            {"name": "Green Banana Curry",   "time": "35 min",  "difficulty": "Medium"},
            {"name": "Banana Chips",         "time": "25 min",  "difficulty": "Easy"},
            {"name": "Green Banana Salad",   "time": "15 min",  "difficulty": "Easy"},
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Transform
# ─────────────────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────
def build_model(name: str, dropout: float = 0.3) -> nn.Module:
    if name == "efficientnet":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
        )
    elif name == "efficientnetv2":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
        )
    elif name == "mobilenet":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[2] = nn.Dropout(p=dropout, inplace=True)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    elif name == "resnet":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(m.fc.in_features, NUM_CLASSES),
        )
    else:
        raise ValueError(f"Unknown model: {name!r}")
    return m

def get_gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    if name in ("efficientnet", "efficientnetv2", "mobilenet"):
        return model.features[-1]
    if name == "resnet":
        return model.layer4[-1]
    raise ValueError(name)

# ─────────────────────────────────────────────────────────────────────────────
# Model registry (loaded at startup)
# ─────────────────────────────────────────────────────────────────────────────
loaded_models: dict[str, nn.Module] = {}
model_status:  dict[str, str]       = {}

def load_all_models() -> None:
    for name in MODEL_NAMES:
        ckpt = MODELS_DIR / f"{name}_best.pth"
        try:
            m = build_model(name).to(device)
            if ckpt.exists():
                sd = torch.load(str(ckpt), map_location=device, weights_only=True)
                m.load_state_dict(sd, strict=False)
                model_status[name] = "loaded"
                print(f"  ✓  {name:14}  loaded from {ckpt.name}")
            else:
                # No checkpoint — keep random weights so API still responds
                model_status[name] = "no_checkpoint"
                print(f"  ⚠  {name:14}  checkpoint not found ({ckpt}) — using random weights")
            m.eval()
            loaded_models[name] = m
        except Exception as e:
            model_status[name] = f"error: {e}"
            print(f"  ✗  {name:14}  failed to load: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
def compute_gradcam_overlay(
    pil_img: Image.Image,
    model: nn.Module,
    model_name: str,
    tensor: torch.Tensor,
    class_idx: int,
) -> str:
    """Run Grad-CAM for class_idx and return base64-encoded PNG overlay."""
    target_layer = get_gradcam_layer(model, model_name)
    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    fwd = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o))

    # Use register_backward_hook (works on all platforms including Windows/CPU)
    bwd = target_layer.register_backward_hook(
        lambda m, gi, go: gradients.append(go[0]))

    try:
        # Need a fresh tensor with grad enabled for Grad-CAM
        inp = _transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
        inp.requires_grad_(True)
        model.zero_grad()
        out = model(inp)
        score = out[0, class_idx]
        score.backward()
    except Exception as e:
        fwd.remove(); bwd.remove()
        raise e
    finally:
        fwd.remove()
        bwd.remove()

    if not activations or not gradients:
        buf = io.BytesIO()
        pil_img.resize((INPUT_SIZE, INPUT_SIZE)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    act  = activations[0].detach().squeeze(0)       # [C, H, W]
    grad = gradients[0].detach().squeeze(0)          # [C, H, W]

    # Handle potential sign flip from backward_hook vs full_backward_hook
    if grad.dim() == 3:
        weights = grad.mean(dim=(1, 2))
    else:
        weights = grad.mean()

    cam = F.relu((weights[:, None, None] * act).sum(0))  # [H, W]

    cam_np = cam.cpu().float().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    # Resize cam to INPUT_SIZE × INPUT_SIZE
    cam_pil   = Image.fromarray(np.uint8(cam_np * 255)).resize(
                    (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    cam_arr   = np.array(cam_pil) / 255.0  # [H,W] 0-1

    # Apply colormap (jet) using matplotlib
    colormap  = mpl_cm.jet(cam_arr)[:, :, :3]  # [H,W,3] 0-1
    heatmap   = (colormap * 255).astype(np.uint8)

    # Overlay
    orig      = np.array(pil_img.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB"))
    overlay   = np.uint8(0.55 * orig + 0.45 * heatmap)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    pil_img: Image.Image,
    model_name: str,
    gradcam: bool = True,
) -> dict:
    """Run inference + optional Grad-CAM on a single PIL image."""
    if model_name not in loaded_models:
        raise HTTPException(404, f"Model '{model_name}' not loaded")

    model  = loaded_models[model_name]
    tensor = _transform(pil_img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits.float(), dim=-1)[0]

    pred_idx    = probs.argmax().item()
    pred_class  = CLASS_NAMES[pred_idx]
    confidences = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}

    gradcam_b64 = None
    if gradcam:
        try:
            gradcam_b64 = compute_gradcam_overlay(
                pil_img, model, model_name, tensor, pred_idx)
        except Exception as e:
            print(f"  Grad-CAM failed for {model_name}: {e}")

    return {
        "model":       model_name,
        "class":       pred_class,
        "confidence":  float(probs[pred_idx]),
        "confidences": confidences,
        "gradcam_b64": gradcam_b64,
        "recipe":      RECIPES[pred_class],
        "status":      model_status.get(model_name, "unknown"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SQLite history
# ─────────────────────────────────────────────────────────────────────────────
def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id          TEXT PRIMARY KEY,
            timestamp   TEXT NOT NULL,
            filename    TEXT,
            model       TEXT,
            predicted   TEXT,
            confidence  REAL,
            confidences TEXT,
            gradcam_b64 TEXT
        )
    """)
    con.commit()
    con.close()

def save_history(
    filename: str,
    model: str,
    predicted: str,
    confidence: float,
    confidences: dict,
    gradcam_b64: Optional[str],
) -> str:
    rec_id = str(uuid.uuid4())[:8]
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO history VALUES (?,?,?,?,?,?,?,?)",
        (rec_id, datetime.utcnow().isoformat(),
         filename, model, predicted, confidence,
         json.dumps(confidences), gradcam_b64),
    )
    con.commit()
    con.close()
    return rec_id

def get_history(limit: int = 50) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id,timestamp,filename,model,predicted,confidence,confidences,gradcam_b64 "
        "FROM history ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [
        {
            "id": r[0], "timestamp": r[1], "filename": r[2],
            "model": r[3], "predicted": r[4], "confidence": r[5],
            "confidences": json.loads(r[6]),
            "gradcam_b64": r[7],
        }
        for r in rows
    ]

def clear_history() -> int:
    con = sqlite3.connect(DB_PATH)
    n   = con.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    con.execute("DELETE FROM history")
    con.commit()
    con.close()
    return n

# ─────────────────────────────────────────────────────────────────────────────
# App lifespan
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 50)
    print("  Banana Ripeness API  starting up")
    print(f"  Device  : {device}")
    print(f"  Models  : {MODELS_DIR.resolve()}")
    print("=" * 50)
    load_all_models()
    init_db()
    print("=" * 50 + "\n")
    yield
    print("API shutting down.")

app = FastAPI(
    title       = "Banana Ripeness API",
    description = "Classify banana ripeness with Grad-CAM explainability",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Serve static files (frontend)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Banana Ripeness API — place index.html in static/</h2>")

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "device":  str(device),
        "models":  model_status,
        "classes": CLASS_NAMES,
    }

@app.get("/models")
async def list_models():
    return {
        name: {
            "status":     model_status.get(name, "not loaded"),
            "checkpoint": str(MODELS_DIR / f"{name}_best.pth"),
        }
        for name in MODEL_NAMES
    }

@app.post("/predict")
async def predict(
    file:        UploadFile = File(...),
    model_name:  str        = Form(default="mobilenet"),
    use_gradcam: bool       = Form(default=True),
):
    """Classify a single image with optional Grad-CAM."""
    if model_name not in MODEL_NAMES:
        raise HTTPException(400, f"model_name must be one of {MODEL_NAMES}")

    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    t0     = time.perf_counter()
    result = run_inference(pil_img, model_name, gradcam=use_gradcam)
    ms     = (time.perf_counter() - t0) * 1000

    result["filename"]  = file.filename or "upload.jpg"
    result["ms"]        = round(ms, 1)

    # Persist to history
    save_history(
        filename    = result["filename"],
        model       = model_name,
        predicted   = result["class"],
        confidence  = result["confidence"],
        confidences = result["confidences"],
        gradcam_b64 = result.get("gradcam_b64"),
    )

    return result

@app.post("/compare")
async def compare(file: UploadFile = File(...)):
    """Run all loaded models on the same image and return side-by-side results."""
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    results = []
    for name in MODEL_NAMES:
        if name not in loaded_models:
            results.append({"model": name, "error": "not loaded"})
            continue
        t0  = time.perf_counter()
        res = run_inference(pil_img, name, gradcam=True)
        res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        results.append(res)

    return {"filename": file.filename, "results": results}

@app.get("/history")
async def history(limit: int = 50):
    return {"history": get_history(limit)}

@app.delete("/history")
async def delete_history():
    n = clear_history()
    return {"deleted": n}
