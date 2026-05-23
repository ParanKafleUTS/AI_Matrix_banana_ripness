"""
s3_train_model.py  —  Pipeline Stage 3: Train All 4 Models
===========================================================
Self-contained — does NOT require train.py to be present.

Trains all 4 architectures by default:
    efficientnet   — EfficientNet-B0
    efficientnetv2 — EfficientNet-V2-S
    mobilenet      — MobileNet-V3-Large
    resnet         — ResNet-50

To train only specific models, edit the MODELS_TO_TRAIN list below,
or override via ClearML parameter "General/models_to_train".

Each model gets its own ClearML Task with full logging:
    - Per-epoch loss, accuracy, LR (Scalars tab)
    - Confusion matrix, ROC curves, F1 chart (Plots tab)
    - Training curves (Plots tab)
    - Checkpoint uploaded to ClearML (Models tab)

Hyperparameter notes (conservative for 30-epoch baseline):
    aug_strength  0.4  — mild; lets model see clear images in short runs
    mixup_alpha   0.0  — disabled; needs ≥50 epochs to help
    unfreeze_after 3   — unfreeze backbone quickly for more fine-tuning epochs
    num_workers   2    — safe for SageMaker (often only 2 vCPUs available)

TF/CUDA warnings at startup are HARMLESS — TensorFlow is pre-installed on
SageMaker and logs noise on import. It does not affect PyTorch training.

Run:
    python s3_train_model.py

Requirements:
    pip install clearml torch torchvision scikit-learn matplotlib numpy
"""

from __future__ import annotations

import json
import platform
import random
import shutil
import time
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc, classification_report, confusion_matrix, roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

warnings.filterwarnings("ignore")

from clearml import OutputModel, Task

# ─────────────────────────────────────────────────────────────────────────────
# Stage-level ClearML Task (orchestrator — one task for the whole stage)
# Each model also creates its own child task inside train_one_model().
# ─────────────────────────────────────────────────────────────────────────────
stage_task = Task.init(
    project_name       = "Banana Ripeness",
    task_name          = "Pipeline step 3 train model",
    task_type          = Task.TaskTypes.training,
    reuse_last_task_id = False,
)
stage_task.add_tags(["baseline", "pipeline", "all-models"])

# ─────────────────────────────────────────────────────────────────────────────
# Global parameters — shared across all models
# ─────────────────────────────────────────────────────────────────────────────
args = {
    # ── Which models to train ──────────────────────────────────────────────
    # Remove any you don't want. Order determines training sequence.
    "models_to_train":  ["efficientnet", "efficientnetv2", "mobilenet", "resnet"],

    # ── Paths ──────────────────────────────────────────────────────────────
    "data_path":        "data",
    "output_dir":       "results_baseline",

    # ── Training ───────────────────────────────────────────────────────────
    "epochs":           30,
    "batch_size":       32,
    "lr":               3e-4,
    "weight_decay":     1e-4,
    "label_smoothing":  0.05,
    "dropout":          0.3,
    "mixup_alpha":      0.0,    # disabled — needs ≥50 epochs to help
    "aug_strength":     0.4,    # mild augmentation for 30-epoch baseline
    "warmup_epochs":    3,
    "patience":         10,
    "min_delta":        0.001,
    "grad_clip":        1.0,
    "unfreeze_after":   3,      # unfreeze backbone after this many epochs
    "tta_n":            5,
    "num_workers":      2,      # safe for SageMaker (2 vCPUs)
    "seed":             42,
    "optimizer":        "adamw",
}
stage_task.connect(args, name="General")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["overripe", "ripe", "rotten", "unripe"]
NUM_CLASSES   = len(CLASS_NAMES)
INPUT_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
COLORS        = ["#4ade80", "#60a5fa", "#f87171", "#fbbf24"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out = Path(args["output_dir"])
out.mkdir(parents=True, exist_ok=True)
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("=" * 60)
print("  STAGE 3: Baseline Training — All 4 Models")
print("=" * 60)
print(f"  Device        : {device}")
if torch.cuda.is_available():
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")
print(f"  Models        : {args['models_to_train']}")
print(f"  Epochs        : {args['epochs']}")
print(f"  Batch size    : {args['batch_size']}")
print(f"  LR            : {args['lr']}")
print(f"  Aug strength  : {args['aug_strength']}  (0.4 = mild, safe for 30 epochs)")
print(f"  Mixup         : {'off' if args['mixup_alpha'] <= 0 else args['mixup_alpha']}")
print(f"  Unfreeze after: epoch {args['unfreeze_after']}")
print()
print("  NOTE: TF/CUDA warnings at startup are harmless (TF pre-installed on SageMaker)")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(args["seed"])


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms(s: float) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE,
                                     scale=(max(0.6, 0.85 - 0.25 * s), 1.0),
                                     ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=int(10 + 15 * s)),
        transforms.ColorJitter(brightness=0.3*s, contrast=0.3*s,
                               saturation=0.2*s, hue=0.05*s),
        transforms.RandomGrayscale(p=0.03),
        transforms.RandomAutocontrast(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_tta_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (loaded once, shared by all models)
# ─────────────────────────────────────────────────────────────────────────────
root = Path(args["data_path"])
if not (root / "train").exists():
    raise FileNotFoundError(
        f"Dataset not found at {root.resolve()}\n"
        "Run Stage 1 first:  python s1_dataset.py"
    )

train_ds = datasets.ImageFolder(root / "train", get_train_transforms(args["aug_strength"]))
val_ds   = datasets.ImageFolder(root / "val",   get_val_transforms())
test_ds  = datasets.ImageFolder(root / "test",  get_val_transforms())

print(f"\n  Dataset  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
print(f"  Classes  : {train_ds.classes}")

# ── Class counts and weights (must come before sampler) ──────────────────────
class_counts  = np.bincount(train_ds.targets, minlength=NUM_CLASSES)
counts_f      = class_counts.astype(float)
w             = 1.0 / np.maximum(counts_f, 1)
w             = w / w.sum() * NUM_CLASSES
class_weights = torch.tensor(w, dtype=torch.float, device=device)

print("\n  Training class distribution (imbalanced → fixed by WeightedRandomSampler):")
for cls, cnt, wt in zip(CLASS_NAMES, class_counts, w):
    pct = cnt / len(train_ds) * 100
    bar = "█" * int(pct / 3)
    print(f"    {cls:<12}  {cnt:>5}  ({pct:5.1f}%)  loss_weight={wt:.3f}  {bar}")
print()
print("  With sampler: each batch will contain ~equal numbers of all 4 classes.")

kw = dict(num_workers=args["num_workers"], pin_memory=True)

# ── WeightedRandomSampler ─────────────────────────────────────────────────────
# Problem: "ripe" is 57% of the dataset. Without balancing, every batch is
# ~57% ripe and the model learns to predict "ripe" for everything (~56% acc).
#
# Fix: assign each sample a weight = 1 / count_of_its_class, then sample with
# replacement so every batch sees ~equal numbers of all 4 classes.
# This is combined with class weights in the loss function for double coverage.
#
_sample_weights = [1.0 / class_counts[t] for t in train_ds.targets]
_sampler        = WeightedRandomSampler(
    weights     = _sample_weights,
    num_samples = len(train_ds),
    replacement = True,
)
# NOTE: sampler and shuffle are mutually exclusive — shuffle=True is removed
train_loader = DataLoader(train_ds, batch_size=args["batch_size"],
                          sampler=_sampler, **kw)
val_loader   = DataLoader(val_ds,   batch_size=args["batch_size"], shuffle=False, **kw)
test_loader  = DataLoader(test_ds,  batch_size=args["batch_size"], shuffle=False, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────
def build_model(name: str, dropout: float) -> nn.Module:
    """
    Build a pretrained model with a new classification head for NUM_CLASSES outputs.
    The head architecture matches train.py exactly so checkpoints are interchangeable.
    """
    if name == "efficientnet":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Replace full classifier: Dropout + Linear
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
        )
    elif name == "efficientnetv2":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
        )
    elif name == "mobilenet":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        # MobileNetV3 classifier: [Linear(960→1280), Hardswish, Dropout, Linear(1280→1000)]
        # Only replace the final linear
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    elif name == "resnet":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(m.fc.in_features, NUM_CLASSES),
        )
    else:
        raise ValueError(f"Unknown model: {name!r}. "
                         f"Choose from: efficientnet, efficientnetv2, mobilenet, resnet")
    return m


def get_head_param_names(name: str) -> tuple[str, ...]:
    """Return the parameter name prefixes that belong to the classification head."""
    return {
        "efficientnet":   ("classifier.",),
        "efficientnetv2": ("classifier.",),
        "mobilenet":      ("classifier.3.",),   # only the final linear, not classifier.0
        "resnet":         ("fc.",),
    }[name]


def freeze_backbone(model: nn.Module, name: str) -> None:
    head_prefixes = get_head_param_names(name)
    for pname, p in model.named_parameters():
        p.requires_grad = any(pname.startswith(pfx) for pfx in head_prefixes)

def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosine(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, opt, warmup: int, total: int, min_ratio: float = 0.01):
        def fn(ep):
            if ep < warmup:
                return (ep + 1) / max(warmup, 1)
            p = (ep - warmup) / max(1, total - warmup)
            return min_ratio + 0.5 * (1 - min_ratio) * (1 + np.cos(np.pi * p))
        super().__init__(opt, fn)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, path: str):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.best_score = None
        self.counter    = 0
        self.best_epoch = 0

    def __call__(self, val_acc: float, model: nn.Module, epoch: int) -> bool:
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.path)
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args["lr"], weight_decay=args["weight_decay"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main training function (called once per model)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_model(model_name: str) -> dict:
    """
    Train a single architecture end-to-end and return a results dict.
    Creates its own ClearML Task so each model has separate plots in the UI.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*60}")

    seed_everything(args["seed"])

    # ── Per-model ClearML Task ────────────────────────────────────────────────
    # Use Task.create() — NOT Task.init() — because Task.init() can only be
    # called once per process. Task.create() creates a sibling task without
    # replacing the current stage task context.
    model_task = Task.create(
        project_name = "Banana Ripeness",
        task_name    = f"train/{model_name}",
        task_type    = Task.TaskTypes.training,
    )
    model_task.add_tags([model_name, "banana-ripeness", "baseline",
                         f"epochs-{args['epochs']}"])
    model_task.connect(args, name="Training")
    model_task.connect({
        "python":      platform.python_version(),
        "torch":       torch.__version__,
        "cuda":        torch.cuda.is_available(),
        "gpu":         torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "num_classes": NUM_CLASSES,
        "class_names": str(CLASS_NAMES),
    }, name="Environment")
    model_task.mark_started()

    mlog = model_task.get_logger()

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(model_name, args["dropout"]).to(device)
    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    freeze_backbone(model, model_name)
    n_head_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total params : {n_params_M:.1f}M  |  Head (frozen backbone): {n_head_M:.4f}M")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion     = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=args["label_smoothing"])
    val_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = make_optimizer(model)
    scheduler = WarmupCosine(optimizer, args["warmup_epochs"], args["epochs"])
    scaler    = GradScaler(enabled=torch.cuda.is_available())

    ckpt_path = str(out / f"{model_name}_best.pth")
    stopper   = EarlyStopping(args["patience"], args["min_delta"], ckpt_path)
    history: dict[str, list] = defaultdict(list)

    # Log dataset sizes
    mlog.report_scalar("Dataset", "train", len(train_ds), 0)
    mlog.report_scalar("Dataset", "val",   len(val_ds),   0)
    mlog.report_scalar("Dataset", "test",  len(test_ds),  0)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  {'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>8}  "
          f"{'Train Acc':>9}  {'Val Acc':>7}  {'LR':>9}  {'Time':>6}")
    print(f"  {'─'*68}")

    for epoch in range(1, args["epochs"] + 1):

        # Unfreeze backbone
        if args["unfreeze_after"] > 0 and epoch == args["unfreeze_after"] + 1:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
            scheduler = WarmupCosine(optimizer, 0, args["epochs"] - epoch)
            scaler    = GradScaler(enabled=torch.cuda.is_available())
            n_now = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            print(f"  [epoch {epoch}] Backbone unfrozen — full fine-tuning ({n_now:.1f}M params)")

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        total_loss = correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            correct    += (logits.detach().argmax(1) == labels).sum().item()
            total      += images.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(enabled=torch.cuda.is_available()):
                    logits = model(images)
                    # val_criterion MUST be inside autocast so logits (float16)
                    # and the loss function use the same dtype. Calling it outside
                    # autocast causes "expected Half but found Float" on CUDA.
                    v_loss += val_criterion(logits, labels).item() * images.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += images.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        mlog.report_scalar("Loss",          "train", train_loss, epoch)
        mlog.report_scalar("Loss",          "val",   val_loss,   epoch)
        mlog.report_scalar("Accuracy",      "train", train_acc,  epoch)
        mlog.report_scalar("Accuracy",      "val",   val_acc,    epoch)
        mlog.report_scalar("Learning Rate", "lr",    lr,         epoch)

        mark = " ✓" if (stopper.best_score is None or
                        val_acc > stopper.best_score + args["min_delta"]) else ""
        print(f"  {epoch:>6}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
              f"{train_acc:>9.4f}  {val_acc:>7.4f}  {lr:>9.2e}  "
              f"{time.time()-t0:>5.0f}s{mark}")

        if stopper(val_acc, model, epoch):
            print(f"\n  Early stopping — best epoch {stopper.best_epoch} "
                  f"(val_acc={stopper.best_score:.4f})")
            break

    # ── Test evaluation ────────────────────────────────────────────────────────
    print(f"\n  Loading best checkpoint (epoch {stopper.best_epoch})")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
            probs  = torch.softmax(logits.float(), dim=1)
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    preds    = np.array(all_preds)
    labels   = np.array(all_labels)
    probs_np = np.array(all_probs)
    test_acc = float((preds == labels).mean())
    print(f"\n  Test accuracy (standard) : {test_acc:.4f}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    # ── TTA ────────────────────────────────────────────────────────────────────
    print(f"  Running TTA (×{args['tta_n']}) …")
    orig_tf           = test_ds.transform
    test_ds.transform = get_tta_transforms()
    tta_loader        = DataLoader(test_ds, batch_size=args["batch_size"],
                                   shuffle=False, num_workers=args["num_workers"],
                                   pin_memory=True)
    probs_accum = tta_labels = None
    for _ in range(args["tta_n"]):
        bp, bl = [], []
        with torch.no_grad():
            for images, lbs in tta_loader:
                with autocast(enabled=torch.cuda.is_available()):
                    logits = model(images.to(device))
                bp.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
                bl.extend(lbs.numpy())
        bp          = np.array(bp)
        probs_accum = bp if probs_accum is None else probs_accum + bp
        if tta_labels is None:
            tta_labels = np.array(bl)
    test_ds.transform = orig_tf
    probs_accum /= args["tta_n"]
    tta_preds = probs_accum.argmax(axis=1)
    tta_acc   = float((tta_preds == tta_labels).mean())
    print(f"  TTA accuracy (×{args['tta_n']})  : {tta_acc:.4f}")

    # ── Inference speed ────────────────────────────────────────────────────────
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        for _ in range(10): model(dummy)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100): model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_image = (time.time() - t0) / 100 * 1000
    print(f"  Inference speed          : {ms_per_image:.2f} ms/image")

    # ── ClearML plots ──────────────────────────────────────────────────────────
    # Confusion matrix
    cm      = confusion_matrix(labels, tta_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name} (TTA)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > 0.6 else "black")
    plt.tight_layout()
    mlog.report_matplotlib_figure("Evaluation", "Confusion Matrix (TTA)", fig, 0)
    plt.savefig(str(out / f"{model_name}_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ROC curves
    y_bin   = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_accum[:, i])
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curves — {model_name}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    mlog.report_matplotlib_figure("Evaluation", "ROC Curves", fig, 0)
    plt.close()

    # Per-class metrics
    report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                    output_dict=True, digits=4)
    for cls in CLASS_NAMES:
        mlog.report_scalar("Per-Class F1",        cls, report[cls]["f1-score"],  0)
        mlog.report_scalar("Per-Class Precision",  cls, report[cls]["precision"], 0)
        mlog.report_scalar("Per-Class Recall",     cls, report[cls]["recall"],    0)

    x = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.25, [report[c]["precision"] for c in CLASS_NAMES],
           0.25, label="Precision", color="#60a5fa", alpha=0.85)
    ax.bar(x,        [report[c]["recall"]    for c in CLASS_NAMES],
           0.25, label="Recall",    color="#4ade80", alpha=0.85)
    ax.bar(x + 0.25, [report[c]["f1-score"]  for c in CLASS_NAMES],
           0.25, label="F1",        color="#fbbf24", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.1); ax.set_title(f"Per-Class Metrics — {model_name}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    mlog.report_matplotlib_figure("Evaluation", "Per-Class Metrics", fig, 0)
    plt.close()

    # Training curves
    eps = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(eps, history["train_acc"],  label="Train", color="#60a5fa")
    ax1.plot(eps, history["val_acc"],    label="Val",   color="#4ade80")
    ax1.set_title(f"Accuracy — {model_name}"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(eps, history["train_loss"], label="Train", color="#60a5fa")
    ax2.plot(eps, history["val_loss"],   label="Val",   color="#4ade80")
    ax2.set_title(f"Loss — {model_name}"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    mlog.report_matplotlib_figure("Training Curves", model_name, fig, 0)
    plt.savefig(str(out / f"{model_name}_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Summary scalars
    mlog.report_scalar("Summary", "best_val_acc",  stopper.best_score, 0)
    mlog.report_scalar("Summary", "test_acc",      test_acc,           0)
    mlog.report_scalar("Summary", "tta_acc",       tta_acc,            0)
    mlog.report_scalar("Summary", "ms_per_image",  ms_per_image,       0)
    mlog.report_scalar("Summary", "params_M",      n_params_M,         0)

    # Register model in ClearML
    om = OutputModel(task=model_task, name=f"{model_name}_banana_ripeness")
    om.update_weights(weights_filename=ckpt_path, auto_delete_file=False)
    om.update_design(config_dict={
        "architecture": model_name,
        "num_classes":  NUM_CLASSES,
        "class_names":  CLASS_NAMES,
        "input_size":   INPUT_SIZE,
        "test_acc":     round(test_acc, 4),
        "tta_acc":      round(tta_acc,  4),
    })
    om.publish()
    model_task.upload_artifact(f"{model_name}_checkpoint", ckpt_path)
    model_task.upload_artifact("best_val_acc",  stopper.best_score)
    model_task.upload_artifact("test_acc",      test_acc)
    model_task.upload_artifact("tta_acc",       tta_acc)

    # Copy checkpoint to models/
    dest = models_dir / f"{model_name}_best.pth"
    shutil.copy2(ckpt_path, dest)
    print(f"  Checkpoint saved to: {dest}")

    model_task.close()

    return {
        "model_name":   model_name,
        "best_val_acc": stopper.best_score,
        "test_acc":     test_acc,
        "tta_acc":      tta_acc,
        "ms_per_image": ms_per_image,
        "n_params_M":   n_params_M,
        "history":      {k: [float(v) for v in vs] for k, vs in history.items()},
        "report":       report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train all models
# ─────────────────────────────────────────────────────────────────────────────
all_results = []
for model_name in args["models_to_train"]:
    result = train_one_model(model_name)
    all_results.append(result)


# ─────────────────────────────────────────────────────────────────────────────
# Save combined results.json
# ─────────────────────────────────────────────────────────────────────────────
results_json = out / "results.json"
results_json.write_text(json.dumps({"models": all_results}, indent=2,
                                    default=lambda x: float(x) if hasattr(x, '__float__') else str(x)))
stage_task.upload_artifact("results_json", str(results_json))
stage_task.upload_artifact("data_path",    args["data_path"])
print(f"\n  results.json saved: {results_json}")

# ─────────────────────────────────────────────────────────────────────────────
# Stage-level comparison plots logged to Pipeline step 3 task
# ─────────────────────────────────────────────────────────────────────────────
slog = stage_task.get_logger()

# ── 1. Accuracy comparison bar chart ─────────────────────────────────────────
names   = [r["model_name"]    for r in all_results]
val_acc = [r["best_val_acc"]  for r in all_results]
tst_acc = [r["test_acc"]      for r in all_results]
tta_acc_vals = [r["tta_acc"]  for r in all_results]

x      = np.arange(len(names))
width  = 0.25
colors = ["#60a5fa", "#4ade80", "#fbbf24"]
fig, ax = plt.subplots(figsize=(10, 5))
for i, (data, label, col) in enumerate(zip(
        [val_acc, tst_acc, tta_acc_vals],
        ["Val Acc", "Test Acc", "TTA Acc"], colors)):
    bars = ax.bar(x + i * width, data, width, label=label, color=col, alpha=0.85)
    for bar, v in zip(bars, data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + width); ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, 1.12); ax.set_ylabel("Accuracy"); ax.legend()
ax.set_title("Model Accuracy Comparison — Stage 3 Baseline")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
slog.report_matplotlib_figure("Comparison", "Accuracy Comparison", fig, 0)
plt.savefig(str(out / "accuracy_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 2. Per-class F1 heatmap ───────────────────────────────────────────────────
f1_matrix = np.array([
    [r["report"][cls]["f1-score"] for cls in CLASS_NAMES]
    for r in all_results
])
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(f1_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10)
ax.set_title("F1 Score Heatmap — All Models")
plt.colorbar(im, ax=ax, label="F1 Score")
for i in range(len(names)):
    for j in range(NUM_CLASSES):
        v = f1_matrix[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="black" if v > 0.75 else "white")
plt.tight_layout()
slog.report_matplotlib_figure("Comparison", "F1 Heatmap", fig, 0)
plt.savefig(str(out / "f1_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 3. Speed vs accuracy scatter ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for r, col in zip(all_results, ["#60a5fa", "#a78bfa", "#34d399", "#fb923c"]):
    ax.scatter(r["ms_per_image"], r["tta_acc"],
               s=200, color=col, zorder=5, label=r["model_name"])
    ax.annotate(r["model_name"],
                (r["ms_per_image"], r["tta_acc"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.set_xlabel("Inference Speed (ms/image)"); ax.set_ylabel("TTA Accuracy")
ax.set_title("Speed vs Accuracy Trade-off")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
slog.report_matplotlib_figure("Comparison", "Speed vs Accuracy", fig, 0)
plt.savefig(str(out / "speed_vs_accuracy.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 4. Summary scalars on stage task ────────────────────────────────────────
for r in all_results:
    slog.report_scalar("Stage3 Val Acc",  r["model_name"], r["best_val_acc"], 0)
    slog.report_scalar("Stage3 Test Acc", r["model_name"], r["test_acc"],     0)
    slog.report_scalar("Stage3 TTA Acc",  r["model_name"], r["tta_acc"],      0)
    slog.report_scalar("Stage3 Speed",    r["model_name"], r["ms_per_image"], 0)

print("  Stage-level comparison plots logged to ClearML ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Final comparison summary
# ─────────────────────────────────────────────────────────────────────────────
sep  = "─" * 72
best = max(all_results, key=lambda r: r["tta_acc"])

print(f"\n{sep}")
print("  STAGE 3 — FINAL RESULTS SUMMARY")
print(sep)
print(f"  {'Model':<16}  {'Val Acc':>8}  {'Test Acc':>9}  {'TTA Acc':>8}  "
      f"{'ms/img':>7}  {'Params':>7}")
print(sep)
for r in all_results:
    mark = "  ◀ BEST" if r is best else ""
    print(f"  {r['model_name']:<16}  {r['best_val_acc']:>8.4f}  "
          f"{r['test_acc']:>9.4f}  {r['tta_acc']:>8.4f}  "
          f"{r['ms_per_image']:>6.1f}ms  {r['n_params_M']:>5.1f}M{mark}")
print(sep)
print(f"\n  Best model : {best['model_name'].upper()}")
print(f"  Best TTA   : {best['tta_acc']:.4f}")
print(f"\n  Per-class F1 ({best['model_name']}):")
for cls in CLASS_NAMES:
    f1  = best["report"][cls]["f1-score"]
    bar = "█" * int(f1 * 30)
    print(f"    {cls:<12}  {f1:.4f}  {bar}")
print(sep)

stage_task.close()
print("\nStage 3 COMPLETE ✓")