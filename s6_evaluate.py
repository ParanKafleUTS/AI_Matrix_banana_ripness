"""
s6_evaluate.py — Detailed Model Evaluation & Comparison
=========================================================
Evaluates all 4 trained models on the test set and produces a
complete set of visualisation and comparison artefacts.

Outputs (saved to results_evaluation/):
    per_model/
        <model>_confusion_matrix.png     — normalised + raw confusion matrix
        <model>_roc_curves.png           — per-class ROC curves + AUC
        <model>_pr_curves.png            — precision-recall curves
        <model>_confidence_hist.png      — confidence distribution (correct vs wrong)
        <model>_per_class_metrics.png    — bar chart of F1/precision/recall per class
        <model>_gradcam_samples.png      — Grad-CAM overlays on 16 test images
        <model>_misclassified.png        — worst misclassified examples
    comparison/
        accuracy_comparison.png          — bar chart all models side by side
        f1_comparison.png                — per-class F1 heatmap across models
        roc_overlay.png                  — ROC curves for all models on one plot
        speed_comparison.png             — inference speed (ms/image)
        confidence_calibration.png       — reliability diagram (calibration)
        error_analysis.png               — confusion heatmap for each model
    report/
        evaluation_report.csv            — all metrics in one CSV
        evaluation_summary.txt           — human-readable summary

Run:
    python s6_evaluate.py

Requirements:
    pip install torch torchvision scikit-learn matplotlib seaborn numpy pillow
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, accuracy_score,
)
from torchvision import datasets, models, transforms

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
OUT_DIR     = Path("results_evaluation")
CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE  = 224
BATCH_SIZE  = 32
NUM_WORKERS = 2
SEED        = 42
GRADCAM_SAMPLES = 16   # images to show in Grad-CAM grid

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_NAMES = ["efficientnet", "efficientnetv2", "mobilenet", "resnet"]

PALETTE = {
    "overripe": "#f59e0b",
    "ripe":     "#22c55e",
    "rotten":   "#ef4444",
    "unripe":   "#84cc16",
}
MODEL_COLORS = {
    "efficientnet":   "#60a5fa",
    "efficientnetv2": "#a78bfa",
    "mobilenet":      "#34d399",
    "resnet":         "#fb923c",
}
STYLE = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "border":  "#30363d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# Create output directories
for d in ["per_model", "comparison", "report"]:
    (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  MODEL EVALUATION — Banana Ripeness Classification")
print("=" * 65)
print(f"  Device  : {device}")
if torch.cuda.is_available():
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
print(f"  Data    : {DATA_DIR.resolve()}")
print(f"  Models  : {MODELS_DIR.resolve()}")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Plot style helpers
# ─────────────────────────────────────────────────────────────────────────────
def apply_dark_style(fig, axes_list):
    fig.patch.set_facecolor(STYLE["bg"])
    for ax in (axes_list if hasattr(axes_list, '__iter__') else [axes_list]):
        ax.set_facecolor(STYLE["surface"])
        ax.tick_params(colors=STYLE["text"])
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        ax.title.set_color(STYLE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["border"])
        ax.grid(alpha=0.15, color=STYLE["border"])

def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Transform & dataset
# ─────────────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_split = "test" if (DATA_DIR / "test").exists() else "val"
test_ds = datasets.ImageFolder(DATA_DIR / test_split, transform)

assert list(test_ds.classes) == CLASS_NAMES, (
    f"Class mismatch: {test_ds.classes} vs {CLASS_NAMES}")

test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    multiprocessing_context="fork" if NUM_WORKERS > 0 and device.type=="cuda" else None,
)
print(f"\n  Test set : {len(test_ds)} images  ({test_split} split)")
print(f"  Classes  : {CLASS_NAMES}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────
def build_model(name: str) -> nn.Module:
    if name == "efficientnet":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
        )
    elif name == "efficientnetv2":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
        )
    elif name == "mobilenet":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[2] = nn.Dropout(p=0.3, inplace=True)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    elif name == "resnet":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(m.fc.in_features, NUM_CLASSES),
        )
    return m

def get_gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    if name in ("efficientnet", "efficientnetv2", "mobilenet"):
        return model.features[-1]
    return model.layer4[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Load all models
# ─────────────────────────────────────────────────────────────────────────────
loaded: dict[str, nn.Module] = {}
for name in MODEL_NAMES:
    ckpt = MODELS_DIR / f"{name}_best.pth"
    if not ckpt.exists():
        print(f"  ⚠  {name}: checkpoint not found at {ckpt} — skipping")
        continue
    m = build_model(name).to(device)
    m.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
    m.eval()
    loaded[name] = m
    print(f"  ✓  {name} loaded")

if not loaded:
    raise RuntimeError("No model checkpoints found in models/")

available = list(loaded.keys())
print(f"\n  Evaluating {len(available)} models: {available}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Run inference — collect all predictions, probs, labels
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model: nn.Module) -> dict:
    all_labels, all_preds, all_probs = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits.float(), dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    elapsed = time.perf_counter() - t0
    return {
        "labels":    np.array(all_labels),
        "preds":     np.array(all_preds),
        "probs":     np.array(all_probs),
        "ms_per_img": elapsed * 1000 / len(test_ds),
    }

print("  Running inference on all models...")
results: dict[str, dict] = {}
for name, model in loaded.items():
    print(f"    {name}...", end=" ", flush=True)
    results[name] = run_inference(model)
    acc = accuracy_score(results[name]["labels"], results[name]["preds"])
    print(f"acc={acc:.4f}  {results[name]['ms_per_img']:.1f}ms/img")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute full metrics for one model
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(r: dict) -> dict:
    labels, preds, probs = r["labels"], r["preds"], r["probs"]
    report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    cm     = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Per-class ROC
    roc, pr_auc = {}, {}
    for i, cls in enumerate(CLASS_NAMES):
        y_bin = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        roc[cls] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
        precision, recall, _ = precision_recall_curve(y_bin, probs[:, i])
        pr_auc[cls] = {
            "precision": precision, "recall": recall,
            "auc": average_precision_score(y_bin, probs[:, i])
        }

    # Confidence of correct vs wrong
    correct_mask = (preds == labels)
    top_conf     = probs.max(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "report":   report,
        "cm":       cm,
        "cm_norm":  cm_norm,
        "roc":      roc,
        "pr_auc":   pr_auc,
        "conf_correct": top_conf[correct_mask],
        "conf_wrong":   top_conf[~correct_mask],
        "ms_per_img":   r["ms_per_img"],
    }

metrics: dict[str, dict] = {
    name: compute_metrics(results[name]) for name in available
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFUSION MATRIX (per model)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating confusion matrices...")
for name in available:
    m = metrics[name]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    apply_dark_style(fig, axes)
    fig.suptitle(f"Confusion Matrix — {name.upper()}  (acc={m['accuracy']:.4f})",
                 color=STYLE["text"], fontsize=13, fontweight="bold")

    for ax, data, title in zip(axes,
                                [m["cm_norm"], m["cm"]],
                                ["Normalised", "Raw Counts"]):
        fmt   = ".2f" if "Norm" in title else "d"
        vmax  = 1.0 if "Norm" in title else None
        cmap  = "Blues"
        im    = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title, color=STYLE["text"], fontsize=11)
        plt.colorbar(im, ax=ax)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val    = data[i, j]
                txt    = f"{val:{fmt}}" if fmt == ".2f" else str(int(val))
                thresh = (data.max() * 0.6) if "Norm" not in title else 0.6
                color  = "white" if val > thresh else STYLE["text"]
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    plt.tight_layout()
    save_fig(fig, OUT_DIR / "per_model" / f"{name}_confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROC CURVES (per model)
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating ROC curves...")
for name in available:
    m   = metrics[name]
    fig, ax = plt.subplots(figsize=(8, 6))
    apply_dark_style(fig, [ax])
    ax.set_title(f"ROC Curves — {name.upper()}", color=STYLE["text"],
                 fontsize=12, fontweight="bold")

    mean_auc = 0.0
    for cls in CLASS_NAMES:
        r   = m["roc"][cls]
        col = PALETTE[cls]
        ax.plot(r["fpr"], r["tpr"], color=col, lw=2,
                label=f"{cls}  (AUC={r['auc']:.3f})")
        mean_auc += r["auc"]

    ax.plot([0, 1], [0, 1], "--", color=STYLE["muted"], lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right", facecolor=STYLE["surface"],
              labelcolor=STYLE["text"], fontsize=9)
    ax.text(0.62, 0.08, f"Mean AUC = {mean_auc/NUM_CLASSES:.4f}",
            transform=ax.transAxes, color=STYLE["text"],
            fontsize=10, bbox=dict(boxstyle="round", facecolor=STYLE["surface"],
                                   edgecolor=STYLE["border"]))
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "per_model" / f"{name}_roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRECISION-RECALL CURVES (per model)
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating PR curves...")
for name in available:
    m   = metrics[name]
    fig, ax = plt.subplots(figsize=(8, 6))
    apply_dark_style(fig, [ax])
    ax.set_title(f"Precision-Recall — {name.upper()}", color=STYLE["text"],
                 fontsize=12, fontweight="bold")

    for cls in CLASS_NAMES:
        pr  = m["pr_auc"][cls]
        col = PALETTE[cls]
        ax.plot(pr["recall"], pr["precision"], color=col, lw=2,
                label=f"{cls}  (AP={pr['auc']:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.legend(loc="lower left", facecolor=STYLE["surface"],
              labelcolor=STYLE["text"], fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "per_model" / f"{name}_pr_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONFIDENCE HISTOGRAM (per model)
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating confidence histograms...")
for name in available:
    m   = metrics[name]
    fig, ax = plt.subplots(figsize=(8, 5))
    apply_dark_style(fig, [ax])
    ax.set_title(f"Confidence Distribution — {name.upper()}",
                 color=STYLE["text"], fontsize=12, fontweight="bold")

    bins = np.linspace(0, 1, 25)
    ax.hist(m["conf_correct"], bins=bins, alpha=0.7, color="#22c55e",
            label=f"Correct ({len(m['conf_correct'])})", density=True)
    ax.hist(m["conf_wrong"],   bins=bins, alpha=0.7, color="#ef4444",
            label=f"Wrong ({len(m['conf_wrong'])})", density=True)

    ax.axvline(m["conf_correct"].mean(), color="#22c55e", ls="--", lw=1.5,
               label=f"Mean correct={m['conf_correct'].mean():.3f}")
    ax.axvline(m["conf_wrong"].mean(),   color="#ef4444", ls="--", lw=1.5,
               label=f"Mean wrong={m['conf_wrong'].mean():.3f}")

    ax.set_xlabel("Prediction Confidence", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"], fontsize=9)
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "per_model" / f"{name}_confidence_hist.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PER-CLASS METRICS BAR CHART (per model)
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating per-class metric charts...")
for name in available:
    m   = metrics[name]
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_dark_style(fig, [ax])
    ax.set_title(f"Per-Class Metrics — {name.upper()}",
                 color=STYLE["text"], fontsize=12, fontweight="bold")

    x      = np.arange(NUM_CLASSES)
    width  = 0.25
    metric_names = ["precision", "recall", "f1-score"]
    colors = ["#60a5fa", "#34d399", "#f472b6"]

    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        vals = [m["report"][cls][metric] for cls in CLASS_NAMES]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=color, alpha=0.85, edgecolor=STYLE["surface"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=STYLE["text"])

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim([0, 1.12])
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"])
    ax.set_title(f"Per-Class Metrics — {name.upper()}  (acc={m['accuracy']:.4f})",
                 color=STYLE["text"], fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "per_model" / f"{name}_per_class_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRAD-CAM SAMPLES (per model)
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating Grad-CAM sample grids...")
import matplotlib.cm as mpl_cm

def compute_gradcam(model, name, img_tensor, class_idx):
    target_layer  = get_gradcam_layer(model, name)
    activations, gradients = [], []
    fwd = target_layer.register_forward_hook(lambda m,i,o: activations.append(o))
    bwd = target_layer.register_backward_hook(lambda m,gi,go: gradients.append(go[0]))
    try:
        inp = img_tensor.clone().requires_grad_(True)
        out = model(inp)
        model.zero_grad()
        out[0, class_idx].backward()
    finally:
        fwd.remove(); bwd.remove()

    if not activations or not gradients:
        return None
    act  = activations[0].detach().squeeze(0)
    grad = gradients[0].detach().squeeze(0)
    w    = grad.mean(dim=(1, 2))
    cam  = F.relu((w[:, None, None] * act).sum(0)).cpu().float().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam

def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(3,1,1)
    return (tensor.cpu() * std + mean).clamp(0,1).permute(1,2,0).numpy()

def overlay_cam(img_np, cam_np):
    cam_resized = np.array(Image.fromarray(np.uint8(cam_np*255))
                           .resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)) / 255.0
    heat = (mpl_cm.jet(cam_resized)[:,:,:3] * 255).astype(np.uint8)
    orig = (img_np * 255).astype(np.uint8)
    return np.uint8(0.55 * orig + 0.45 * heat)

# Collect sample indices — 4 per class
sample_indices: list[int] = []
for cls_idx in range(NUM_CLASSES):
    idxs = [i for i, (_, label) in enumerate(test_ds.samples) if label == cls_idx]
    np.random.shuffle(idxs)
    sample_indices.extend(idxs[:4])

for name, model in loaded.items():
    ncols = 4
    nrows = NUM_CLASSES
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 5, nrows * 3))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle(f"Grad-CAM Samples — {name.upper()}",
                 color=STYLE["text"], fontsize=13, fontweight="bold", y=1.01)

    sample_ptr = 0
    for row, cls_idx in enumerate(range(NUM_CLASSES)):
        idxs = [i for i, (_, l) in enumerate(test_ds.samples) if l == cls_idx][:4]
        for col, img_idx in enumerate(idxs):
            img_pil, true_label = test_ds[img_idx]
            img_tensor = img_pil.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_tensor)
                pred_idx  = logits.argmax(1).item()
                conf      = F.softmax(logits.float(), dim=-1)[0, pred_idx].item()

            cam     = compute_gradcam(model, name, img_tensor, pred_idx)
            img_np  = denormalize(img_pil)

            ax_orig = axes[row, col * 2]
            ax_cam  = axes[row, col * 2 + 1]

            ax_orig.imshow(img_np)
            ax_orig.axis("off")
            correct = pred_idx == true_label
            color   = "#22c55e" if correct else "#ef4444"
            mark    = "✓" if correct else "✗"
            ax_orig.set_title(
                f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_idx]} {mark} {conf:.0%}",
                fontsize=7, color=color, pad=2)

            if cam is not None:
                ax_cam.imshow(overlay_cam(img_np, cam))
            else:
                ax_cam.imshow(img_np)
            ax_cam.axis("off")
            ax_cam.set_title("Grad-CAM", fontsize=7,
                             color=STYLE["muted"], pad=2)

    plt.tight_layout()
    save_fig(fig, OUT_DIR / "per_model" / f"{name}_gradcam_samples.png", dpi=120)


# ─────────────────────────────────────────────────────────────────────────────
# 7. WORST MISCLASSIFICATIONS (per model)
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating misclassification panels...")
for name in available:
    r   = results[name]
    labels, preds, probs = r["labels"], r["preds"], r["probs"]
    wrong_idxs = np.where(labels != preds)[0]

    if len(wrong_idxs) == 0:
        print(f"    {name}: no misclassifications!")
        continue

    # Sort by confidence of wrong prediction (most confident wrong = worst)
    wrong_conf   = probs[wrong_idxs].max(axis=1)
    sorted_wrong = wrong_idxs[np.argsort(-wrong_conf)][:16]

    ncols = 4
    nrows = (len(sorted_wrong) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle(f"Worst Misclassifications — {name.upper()} (most confident wrong first)",
                 color=STYLE["text"], fontsize=12, fontweight="bold")
    axes_flat = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for ax, idx in zip(axes_flat, sorted_wrong):
        img_tensor, _ = test_ds[idx]
        img_np = denormalize(img_tensor)
        true_cls = CLASS_NAMES[labels[idx]]
        pred_cls = CLASS_NAMES[preds[idx]]
        conf     = probs[idx, preds[idx]]
        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(
            f"True: {true_cls}\nPred: {pred_cls} ({conf:.0%})",
            fontsize=8, color="#ef4444", pad=3)

    for ax in axes_flat[len(sorted_wrong):]:
        ax.axis("off")

    plt.tight_layout()
    save_fig(fig, OUT_DIR / "per_model" / f"{name}_misclassified.png", dpi=120)


# ─────────────────────────────────────────────────────────────────────────────
# 8. ACCURACY COMPARISON (all models)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating comparison charts...")
fig, ax = plt.subplots(figsize=(10, 5))
apply_dark_style(fig, [ax])
ax.set_title("Model Accuracy Comparison", color=STYLE["text"],
             fontsize=13, fontweight="bold")

names  = available
accs   = [metrics[n]["accuracy"] for n in names]
colors = [MODEL_COLORS[n] for n in names]
bars   = ax.bar(names, accs, color=colors, edgecolor=STYLE["surface"],
                width=0.5, alpha=0.9)

best_idx = np.argmax(accs)
for i, (bar, acc) in enumerate(zip(bars, accs)):
    mark = " ★" if i == best_idx else ""
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{acc:.4f}{mark}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=STYLE["text"])

ax.set_ylim([min(accs) * 0.95, 1.02])
ax.set_ylabel("Accuracy", fontsize=11)
ax.axhline(np.mean(accs), color=STYLE["muted"], ls="--", lw=1.5,
           label=f"Mean = {np.mean(accs):.4f}")
ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"])
plt.tight_layout()
save_fig(fig, OUT_DIR / "comparison" / "accuracy_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. F1 HEATMAP (all models × all classes)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
apply_dark_style(fig, [ax])
ax.set_title("F1 Score Heatmap — All Models × All Classes",
             color=STYLE["text"], fontsize=13, fontweight="bold")

f1_matrix = np.array([
    [metrics[n]["report"][cls]["f1-score"] for cls in CLASS_NAMES]
    for n in available
])
im = ax.imshow(f1_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0)
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_yticks(range(len(available))); ax.set_yticklabels(available, fontsize=11)
ax.set_xlabel("Class", fontsize=11); ax.set_ylabel("Model", fontsize=11)
plt.colorbar(im, ax=ax, label="F1 Score")

for i in range(len(available)):
    for j in range(NUM_CLASSES):
        v = f1_matrix[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="black" if v > 0.75 else "white")

plt.tight_layout()
save_fig(fig, OUT_DIR / "comparison" / "f1_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. ROC OVERLAY (all models, macro average)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
apply_dark_style(fig, [ax])
ax.set_title("ROC Curves — All Models (Macro Average)",
             color=STYLE["text"], fontsize=13, fontweight="bold")

for name in available:
    m     = metrics[name]
    color = MODEL_COLORS[name]
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros(200)
    for cls in CLASS_NAMES:
        mean_tpr += np.interp(all_fpr, m["roc"][cls]["fpr"], m["roc"][cls]["tpr"])
    mean_tpr /= NUM_CLASSES
    mean_auc  = np.mean([m["roc"][cls]["auc"] for cls in CLASS_NAMES])
    ax.plot(all_fpr, mean_tpr, color=color, lw=2.5,
            label=f"{name}  (AUC={mean_auc:.4f})")

ax.plot([0,1],[0,1], "--", color=STYLE["muted"], lw=1)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"], fontsize=10)
plt.tight_layout()
save_fig(fig, OUT_DIR / "comparison" / "roc_overlay.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. INFERENCE SPEED COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
apply_dark_style(fig, [ax])
ax.set_title("Inference Speed (ms / image)", color=STYLE["text"],
             fontsize=13, fontweight="bold")

speeds = [metrics[n]["ms_per_img"] for n in available]
bars   = ax.barh(available, speeds, color=[MODEL_COLORS[n] for n in available],
                 alpha=0.9, edgecolor=STYLE["surface"])
for bar, v in zip(bars, speeds):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{v:.1f} ms", va="center", fontsize=11, color=STYLE["text"])

ax.set_xlabel("ms per image", fontsize=11)
ax.invert_yaxis()
plt.tight_layout()
save_fig(fig, OUT_DIR / "comparison" / "speed_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12. CONFIDENCE CALIBRATION (reliability diagram)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
if len(available) == 1:
    axes = [axes]
apply_dark_style(fig, axes)
fig.suptitle("Confidence Calibration (Reliability Diagram)",
             color=STYLE["text"], fontsize=13, fontweight="bold")

for ax, name in zip(axes, available):
    r      = results[name]
    probs  = r["probs"].max(axis=1)
    labels = (r["preds"] == r["labels"]).astype(int)
    bins   = np.linspace(0, 1, 11)
    bin_acc, bin_conf, bin_size = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            bin_acc.append(labels[mask].mean())
            bin_conf.append(probs[mask].mean())
            bin_size.append(mask.sum())

    ax.plot([0, 1], [0, 1], "--", color=STYLE["muted"], lw=1.5, label="Perfect calibration")
    ax.plot(bin_conf, bin_acc, "o-", color=MODEL_COLORS[name], lw=2,
            markersize=7, label=name)
    ax.fill_between(bin_conf, bin_acc, bin_conf,
                    alpha=0.15, color=MODEL_COLORS[name])
    ax.set_xlabel("Mean Confidence", fontsize=10)
    ax.set_ylabel("Fraction Correct", fontsize=10)
    ax.set_title(name.upper(), color=STYLE["text"], fontsize=11)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"], fontsize=8)

plt.tight_layout()
save_fig(fig, OUT_DIR / "comparison" / "confidence_calibration.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13. FULL ERROR ANALYSIS GRID
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5))
if len(available) == 1:
    axes = [axes]
apply_dark_style(fig, axes)
fig.suptitle("Normalised Confusion Matrix — All Models",
             color=STYLE["text"], fontsize=13, fontweight="bold")

for ax, name in zip(axes, available):
    cm = metrics[name]["cm_norm"]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([c[:4] for c in CLASS_NAMES], fontsize=9, rotation=30)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels([c[:4] for c in CLASS_NAMES], fontsize=9)
    ax.set_title(f"{name.upper()}\nacc={metrics[name]['accuracy']:.4f}",
                 color=STYLE["text"], fontsize=10)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if v > 0.6 else STYLE["text"])
    plt.colorbar(im, ax=ax)

plt.tight_layout()
save_fig(fig, OUT_DIR / "comparison" / "error_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14. CSV REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Saving CSV report...")
import csv
rows = []
for name in available:
    m = metrics[name]
    row = {
        "model":        name,
        "accuracy":     round(m["accuracy"], 6),
        "ms_per_img":   round(m["ms_per_img"], 2),
        "mean_auc":     round(np.mean([m["roc"][c]["auc"] for c in CLASS_NAMES]), 6),
        "mean_ap":      round(np.mean([m["pr_auc"][c]["auc"] for c in CLASS_NAMES]), 6),
    }
    for cls in CLASS_NAMES:
        row[f"{cls}_precision"] = round(m["report"][cls]["precision"], 6)
        row[f"{cls}_recall"]    = round(m["report"][cls]["recall"],    6)
        row[f"{cls}_f1"]        = round(m["report"][cls]["f1-score"],  6)
        row[f"{cls}_auc"]       = round(m["roc"][cls]["auc"],          6)
    rows.append(row)

csv_path = OUT_DIR / "report" / "evaluation_report.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"  Saved: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 15. TEXT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
best_name = max(available, key=lambda n: metrics[n]["accuracy"])
sep = "─" * 65

lines = [
    sep,
    "  EVALUATION SUMMARY — Banana Ripeness Classification",
    sep,
    f"  Test split   : {test_split}  ({len(test_ds)} images)",
    f"  Device       : {device}",
    f"  Models       : {', '.join(available)}",
    "",
    f"  {'Model':<16}  {'Accuracy':>9}  {'Mean AUC':>9}  {'Mean AP':>9}  {'ms/img':>8}",
    "  " + "-" * 57,
]
for name in available:
    m    = metrics[name]
    mauc = np.mean([m["roc"][c]["auc"]    for c in CLASS_NAMES])
    map_ = np.mean([m["pr_auc"][c]["auc"] for c in CLASS_NAMES])
    mark = "  ★ BEST" if name == best_name else ""
    lines.append(
        f"  {name:<16}  {m['accuracy']:>9.4f}  {mauc:>9.4f}  {map_:>9.4f}  {m['ms_per_img']:>7.1f}ms{mark}"
    )
lines += [
    "  " + "-" * 57, "",
    f"  Best model : {best_name.upper()}  (accuracy={metrics[best_name]['accuracy']:.4f})",
    "",
    "  Per-class F1 scores:",
    f"  {'Class':<14}" + "".join(f"  {n:>14}" for n in available),
    "  " + "-" * (14 + 16 * len(available)),
]
for cls in CLASS_NAMES:
    row = f"  {cls:<14}" + "".join(
        f"  {metrics[n]['report'][cls]['f1-score']:>14.4f}" for n in available)
    lines.append(row)
lines += ["", sep]

summary = "\n".join(lines)
print("\n" + summary)

txt_path = OUT_DIR / "report" / "evaluation_summary.txt"
txt_path.write_text(summary)
print(f"\n  Saved: {txt_path}")

print(f"\n  All outputs saved to: {OUT_DIR.resolve()}")
print("=" * 65)
print("  EVALUATION COMPLETE ✓")
print("=" * 65)
