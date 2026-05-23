"""
s5_final_model.py  —  Pipeline Stage 5: Final Model Training
=============================================================
Self-contained — does NOT require train.py to be present.

What this file does:
  1.  Auto-loads best hyperparameters from results_hpo/hpo_results.json
      (produced by Stage 4). Falls back to defaults if not found.
  2.  Validates CLASS_NAMES against dataset folder order.
  3.  Trains the best model for 50 epochs with:
        - Frozen backbone for first 5 epochs (head warmup)
        - Full fine-tuning with discriminative LR after epoch 5
        - WarmupCosine LR schedule
        - WeightedRandomSampler (class imbalance)
        - LabelSmoothingCrossEntropy
        - Mixup augmentation
        - Per-epoch train loss + val acc + per-class val acc → ClearML
        - Majority-class collapse detection
        - Early stopping (patience=10)
  4.  Best checkpoint saved to results_final/<model>_best.pth
  5.  Test-set evaluation: accuracy, per-class accuracy, F1, confusion matrix
  6.  TTA (Test-Time Augmentation) evaluation
  7.  Inference speed benchmark (ms/image)
  8.  All metrics + plots → ClearML
  9.  Registers model in ClearML Model Registry
  10. Copies checkpoint to models/ for API use

Run:
    python s5_final_model.py

Requirements:
    pip install clearml torch torchvision scikit-learn numpy matplotlib
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

from clearml import OutputModel, Task

# ─────────────────────────────────────────────────────────────────────────────
# ClearML Task
# ─────────────────────────────────────────────────────────────────────────────
task = Task.init(
    project_name       = "Banana Ripeness",
    task_name          = "Pipeline step 5 final model",
    task_type          = Task.TaskTypes.training,
    reuse_last_task_id = False,
)
task.add_tags(["final", "pipeline", "production"])

# ─────────────────────────────────────────────────────────────────────────────
# Auto-load best HPO params from Stage 4 output
# ─────────────────────────────────────────────────────────────────────────────
HPO_RESULTS = Path("results_hpo/hpo_results.json")
_hpo_best: dict = {}

if HPO_RESULTS.exists():
    try:
        _all = json.loads(HPO_RESULTS.read_text())
        _hpo_best = max(_all, key=lambda r: r.get("best_val_acc", 0.0))
        print(f"  [s4→s5] Loaded HPO results: {HPO_RESULTS}")
        print(f"          Best model : {_hpo_best.get('model_name')}")
        print(f"          Val acc    : {_hpo_best.get('best_val_acc', 0):.4f}")
    except Exception as e:
        print(f"  [s4→s5] Could not parse {HPO_RESULTS}: {e} — using defaults")
else:
    print(f"  [s4→s5] {HPO_RESULTS} not found — using default params")

_bp = _hpo_best.get("best_params", {})

# ─────────────────────────────────────────────────────────────────────────────
# Parameters  (HPO values injected automatically; defaults are safe fallbacks)
# ─────────────────────────────────────────────────────────────────────────────
args = {
    "data_path":        "data",
    "output_dir":       "results_final",

    # Model — taken from HPO best, defaulting to mobilenet
    "model":            _hpo_best.get("model_name", "mobilenet"),

    # Training schedule
    "epochs":           50,
    "unfreeze_after":   5,    # epochs of head-only warmup before unfreezing backbone
    "patience":         10,   # early-stop if val acc doesn't improve for this many epochs

    # Hyperparameters — from HPO or safe defaults
    "batch_size":       int(_bp.get("batch_size",       32)),
    "lr":               float(_bp.get("lr",             3e-4)),
    "weight_decay":     float(_bp.get("weight_decay",   1e-4)),
    "dropout":          float(_bp.get("dropout",        0.3)),
    "label_smoothing":  float(_bp.get("label_smoothing",0.1)),
    "mixup_alpha":      float(_bp.get("mixup_alpha",    0.3)),
    "aug_strength":     float(_bp.get("aug_strength",   0.5)),
    "optimizer":        str(_bp.get("optimizer",        "adamw")),

    # TTA — number of augmented forward passes at test time (1 = disabled)
    "tta_passes":       5,

    # Misc
    "num_workers":      2,
    "seed":             42,
}
task.connect(args, name="General")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["overripe", "ripe", "rotten", "unripe"]
NUM_CLASSES   = len(CLASS_NAMES)
INPUT_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
PALETTE       = ["#4ade80", "#60a5fa", "#f87171", "#fbbf24"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out    = Path(args["output_dir"])
out.mkdir(parents=True, exist_ok=True)

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
# torch.amp compatibility shim (PyTorch 1.x and 2.x)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE = None

def amp_autocast():
    if _AMP_DEVICE:
        return autocast(device_type=_AMP_DEVICE, enabled=torch.cuda.is_available())
    return autocast(enabled=torch.cuda.is_available())

# ─────────────────────────────────────────────────────────────────────────────
# Print header
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("  STAGE 5: Final Model Training")
print("=" * 62)
print(f"  Device         : {device}")
if torch.cuda.is_available():
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
print(f"  Model          : {args['model']}")
print(f"  Epochs         : {args['epochs']}  (unfreeze backbone after {args['unfreeze_after']})")
print(f"  LR             : {args['lr']:.2e}")
print(f"  Batch size     : {args['batch_size']}")
print(f"  Aug strength   : {args['aug_strength']:.3f}")
print(f"  Optimizer      : {args['optimizer']}")
print(f"  TTA passes     : {args['tta_passes']}")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms(aug_strength: float) -> transforms.Compose:
    s = aug_strength
    return transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE,
                                     scale=(max(0.6, 0.85 - 0.25 * s), 1.0),
                                     ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
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
    """Lightweight random augmentation for TTA — weaker than training."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────
root = Path(args["data_path"])
if not (root / "train").exists():
    raise FileNotFoundError(
        f"Dataset not found at {root.resolve()}\n"
        "Run s1_rebuild.py first to build the clean dataset."
    )

train_ds = datasets.ImageFolder(root / "train", get_train_transforms(args["aug_strength"]))
val_ds   = datasets.ImageFolder(root / "val",   get_val_transforms())
test_ds  = datasets.ImageFolder(root / "test",  get_val_transforms()) \
           if (root / "test").exists() else None

# CLASS_NAMES validation — crashes early if folder order doesn't match
assert list(train_ds.classes) == CLASS_NAMES, (
    f"CLASS_NAMES mismatch!\n"
    f"  Expected : {CLASS_NAMES}\n"
    f"  Got      : {list(train_ds.classes)}\n"
    f"  Fix: update CLASS_NAMES to match the folder names above."
)

# Class weights for loss
class_counts  = np.bincount(train_ds.targets, minlength=NUM_CLASSES)
w             = 1.0 / np.maximum(class_counts.astype(float), 1)
w             = w / w.sum() * NUM_CLASSES
class_weights = torch.tensor(w, dtype=torch.float, device=device)

# Per-sample weights for WeightedRandomSampler
sample_weights = [1.0 / np.sqrt(class_counts[t]) for t in train_ds.targets]
sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(train_ds),
    replacement = True,
)

print(f"\n  Dataset  train={len(train_ds)}  val={len(val_ds)}"
      + (f"  test={len(test_ds)}" if test_ds else "  test=none"))
print(f"  Classes  {train_ds.classes}  ✓ matches CLASS_NAMES")
print(f"\n  Class distribution + loss weights:")
for cls, cnt, wt in zip(CLASS_NAMES, class_counts, w):
    pct = cnt / len(train_ds) * 100
    print(f"    {cls:<12}  {cnt:>5}  ({pct:5.1f}%)  weight={wt:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
nw = args["num_workers"]
_loader_kwargs = dict(
    num_workers             = nw,
    pin_memory              = True,
    persistent_workers      = False,
    multiprocessing_context = "fork" if nw > 0 else None,
)

train_loader = DataLoader(train_ds, batch_size=args["batch_size"],
                          sampler=sampler, **_loader_kwargs)
val_loader   = DataLoader(val_ds,   batch_size=args["batch_size"],
                          shuffle=False, **_loader_kwargs)
test_loader  = DataLoader(test_ds,  batch_size=args["batch_size"],
                          shuffle=False, **_loader_kwargs) if test_ds else None

# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────
def build_model(name: str, dropout: float) -> nn.Module:
    if name == "efficientnet":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
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
        m.classifier[2] = nn.Dropout(p=dropout, inplace=True)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    elif name == "resnet":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(m.fc.in_features, NUM_CLASSES),
        )
    else:
        raise ValueError(f"Unknown model: {name!r}")
    return m

def get_head_prefixes(name: str) -> tuple[str, ...]:
    return {
        "efficientnet":   ("classifier.",),
        "efficientnetv2": ("classifier.",),
        "mobilenet":      ("classifier.2.", "classifier.3."),
        "resnet":         ("fc.",),
    }[name]

def freeze_backbone(model: nn.Module, name: str) -> None:
    head = get_head_prefixes(name)
    for pname, p in model.named_parameters():
        p.requires_grad = any(pname.startswith(h) for h in head)

def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True

# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosine(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, opt, warmup: int, total: int, min_ratio: float = 0.01):
        def fn(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            p = (epoch - warmup) / max(1, total - warmup)
            return min_ratio + 0.5 * (1 - min_ratio) * (1 + np.cos(np.pi * p))
        super().__init__(opt, fn)

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        n      = logits.size(-1)
        smooth = torch.full_like(log_probs, self.smoothing / (n - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth * log_probs).sum(dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()

# ─────────────────────────────────────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────────────────────────────────────
model_name = args["model"]
model      = build_model(model_name, args["dropout"]).to(device)

# Warm-start from s4 HPO best checkpoint if available
hpo_ckpt = Path("results_hpo") / f"{model_name}_hpo_best.pth"
checkpoint_loaded = False
if hpo_ckpt.exists():
    try:
        sd = torch.load(str(hpo_ckpt), map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        checkpoint_loaded = True
        print(f"\n  [warm-start] Loaded HPO checkpoint: {hpo_ckpt}")
    except Exception as e:
        print(f"\n  [warm-start] Could not load {hpo_ckpt}: {e} — using ImageNet init")

# Freeze/unfreeze strategy — same logic as s4
if checkpoint_loaded:
    unfreeze_all(model)
    unfreeze_epoch = 0
else:
    freeze_backbone(model, model_name)
    unfreeze_epoch = max(1, args["unfreeze_after"])
    print(f"  Backbone frozen — will unfreeze at epoch {unfreeze_epoch}")

# ─────────────────────────────────────────────────────────────────────────────
# Loss, optimiser, scheduler, scaler
# ─────────────────────────────────────────────────────────────────────────────
criterion = LabelSmoothingCE(args["label_smoothing"], class_weights)

def make_optimizer(model: nn.Module, unfreeze_done: bool) -> torch.optim.Optimizer:
    """
    Discriminative LR: backbone gets 10x lower LR than the head.
    When backbone is still frozen, only head params are passed.
    """
    head_pfx = get_head_prefixes(model_name)
    if not unfreeze_done:
        params = [p for n, p in model.named_parameters() if p.requires_grad]
        param_groups = [{"params": params, "lr": args["lr"]}]
    else:
        backbone_params = [p for n, p in model.named_parameters()
                           if not any(n.startswith(h) for h in head_pfx)]
        head_params     = [p for n, p in model.named_parameters()
                           if any(n.startswith(h) for h in head_pfx)]
        param_groups = [
            {"params": backbone_params, "lr": args["lr"] * 0.1},
            {"params": head_params,     "lr": args["lr"]},
        ]

    if args["optimizer"] == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=args["weight_decay"])
    return torch.optim.SGD(param_groups, momentum=0.9,
                           weight_decay=args["weight_decay"], nesterov=True)

opt       = make_optimizer(model, checkpoint_loaded)
scheduler = WarmupCosine(opt, warmup=3, total=args["epochs"])
if _AMP_DEVICE:
    scaler = GradScaler(_AMP_DEVICE, enabled=torch.cuda.is_available())
else:
    scaler = GradScaler(enabled=torch.cuda.is_available())

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
logger         = task.get_logger()
best_val_acc   = 0.0
best_epoch     = 0
best_state     = None
no_improve     = 0

history: dict[str, list] = {"train_loss": [], "val_acc": [], "lr": []}

print(f"\n{'─'*62}")
print(f"  Training  ({args['epochs']} epochs, early stop patience={args['patience']})")
print(f"{'─'*62}")

for epoch in range(1, args["epochs"] + 1):

    # ── Unfreeze backbone after warmup ────────────────────────────────────────
    if not checkpoint_loaded and epoch == unfreeze_epoch:
        unfreeze_all(model)
        opt       = make_optimizer(model, unfreeze_done=True)
        scheduler = WarmupCosine(opt, warmup=2,
                                 total=args["epochs"] - unfreeze_epoch + 1)
        print(f"  Epoch {epoch}: backbone unfrozen — discriminative LR active")

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    running_loss  = 0.0
    running_steps = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        alpha = args["mixup_alpha"]
        if alpha > 0:
            lam = float(np.random.beta(alpha, alpha))
            idx = torch.randperm(images.size(0), device=device)
            images = lam * images + (1 - lam) * images[idx]
            ya, yb = labels, labels[idx]
        else:
            lam, ya, yb = 1.0, labels, labels

        opt.zero_grad()
        with amp_autocast():
            logits = model(images)
            loss   = (lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                      if lam < 1.0 else criterion(logits, ya))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        running_loss  += loss.item()
        running_steps += 1

    mean_loss = running_loss / max(running_steps, 1)
    current_lr = opt.param_groups[-1]["lr"]
    scheduler.step()

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    correct       = 0
    total         = 0
    class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    class_total   = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    all_preds     = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with amp_autocast():
                logits = model(images)
            preds   = logits.float().argmax(1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)
            for c in range(NUM_CLASSES):
                mask = (labels == c)
                class_correct[c] += (preds[mask] == c).sum()
                class_total[c]   += mask.sum()
            all_preds.append(preds.cpu())

    val_acc = correct / total

    # ── Collapse detection ────────────────────────────────────────────────────
    if epoch >= 3:
        preds_cat     = torch.cat(all_preds)
        pred_counts   = torch.bincount(preds_cat, minlength=NUM_CLASSES)
        top_frac      = pred_counts.max().item() / len(preds_cat)
        if top_frac > 0.95:
            top_cls = pred_counts.argmax().item()
            print(f"  ⚠  Epoch {epoch}: collapse — {100*top_frac:.1f}% predicted "
                  f"as '{CLASS_NAMES[top_cls]}'. Stopping early.")
            break

    # ── ClearML logging ───────────────────────────────────────────────────────
    logger.report_scalar("Loss",  "train_loss", mean_loss,   epoch)
    logger.report_scalar("Acc",   "val_acc",    val_acc,     epoch)
    logger.report_scalar("LR",    "lr",         current_lr,  epoch)
    for c, cls_name in enumerate(CLASS_NAMES):
        cls_acc = (class_correct[c] / class_total[c].clamp(min=1)).item()
        logger.report_scalar("Val Acc Per Class", cls_name, cls_acc, epoch)

    # ── History + console ─────────────────────────────────────────────────────
    history["train_loss"].append(mean_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    per_class_str = "  ".join(
        f"{CLASS_NAMES[c][:4]}={class_correct[c].item()/max(class_total[c].item(),1):.2f}"
        for c in range(NUM_CLASSES)
    )
    print(f"  Ep {epoch:>3}/{args['epochs']}  loss={mean_loss:.4f}  "
          f"val={val_acc:.4f}  [{per_class_str}]  lr={current_lr:.2e}")

    # ── Checkpoint + early stop ───────────────────────────────────────────────
    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        best_epoch   = epoch
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve   = 0
        # Save immediately so a crash doesn't lose the best weights
        ckpt_path = out / f"{model_name}_best.pth"
        torch.save(best_state, ckpt_path)
    else:
        no_improve += 1
        if no_improve >= args["patience"]:
            print(f"\n  Early stop at epoch {epoch} "
                  f"(no improvement for {args['patience']} epochs)")
            break

print(f"\n  Best val acc: {best_val_acc:.4f}  (epoch {best_epoch})")

# ─────────────────────────────────────────────────────────────────────────────
# Training curves chart
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs_x = range(1, len(history["train_loss"]) + 1)

ax1.plot(epochs_x, history["train_loss"], color="#60a5fa", lw=2)
ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.axvline(best_epoch, color="#f87171", ls="--", lw=1, label=f"best ep={best_epoch}")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs_x, history["val_acc"], color="#4ade80", lw=2)
ax2.axhline(best_val_acc, color="#f87171", ls="--", lw=1,
            label=f"best={best_val_acc:.4f}")
ax2.set_title("Val Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle(f"Training — {model_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
curves_path = out / "training_curves.png"
fig.savefig(curves_path, dpi=150, bbox_inches="tight")
logger.report_matplotlib_figure("Training", "Curves", fig, 0)   # FIX: log fig directly
plt.close()
task.upload_artifact("training_curves", str(curves_path))

# ─────────────────────────────────────────────────────────────────────────────
# Test-set evaluation (standard)
# ─────────────────────────────────────────────────────────────────────────────
test_acc   = 0.0
tta_acc    = 0.0
ms_per_img = 0.0

if test_loader is not None and best_state is not None:
    print(f"\n{'─'*62}")
    print("  TEST SET EVALUATION")
    print(f"{'─'*62}")

    # Reload best weights
    model.load_state_dict(best_state)
    model.eval()

    all_labels: list[int] = []
    all_preds_test: list[int] = []
    class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    class_total   = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    correct = total = 0

    t0 = time.perf_counter()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with amp_autocast():
                logits = model(images)
            preds = logits.float().argmax(1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)
            for c in range(NUM_CLASSES):
                mask = (labels == c)
                class_correct[c] += (preds[mask] == c).sum()
                class_total[c]   += mask.sum()
            all_labels.extend(labels.cpu().tolist())
            all_preds_test.extend(preds.cpu().tolist())
    t1 = time.perf_counter()

    test_acc   = correct / total
    ms_per_img = (t1 - t0) * 1000 / max(total, 1)

    print(f"\n  Test accuracy : {test_acc:.4f}")
    print(f"  ms / image    : {ms_per_img:.2f}")
    print(f"\n  Per-class test accuracy:")
    for c, cls_name in enumerate(CLASS_NAMES):
        cls_acc = (class_correct[c] / class_total[c].clamp(min=1)).item()
        bar     = "█" * int(cls_acc * 20)
        print(f"    {cls_name:<12}  {cls_acc:.4f}  {bar}")

    # sklearn classification report
    report_str  = classification_report(all_labels, all_preds_test,
                                        target_names=CLASS_NAMES)
    report_dict = classification_report(all_labels, all_preds_test,
                                        target_names=CLASS_NAMES, output_dict=True)
    print(f"\n{report_str}")
    (out / "classification_report.txt").write_text(report_str)

    # Per-class F1 → ClearML
    for cls_name in CLASS_NAMES:
        f1 = report_dict.get(cls_name, {}).get("f1-score", 0.0)
        logger.report_scalar("Test F1 Per Class", cls_name, f1, 0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}  (test acc={test_acc:.4f})")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                    fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = out / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    logger.report_matplotlib_figure("Test", "Confusion Matrix", fig, 0)  # FIX: log fig directly
    plt.close()
    task.upload_artifact("confusion_matrix", str(cm_path))

    # ── TTA evaluation ────────────────────────────────────────────────────────
    n_passes = args["tta_passes"]
    if n_passes > 1:
        print(f"\n  TTA evaluation ({n_passes} passes per image) ...")

        # Reload val_ds raw images for TTA (PIL access)
        tta_ds = datasets.ImageFolder(root / "test", get_tta_transforms())

        tta_correct = 0
        tta_total   = 0
        model.eval()

        with torch.no_grad():
            for idx in range(len(test_ds)):
                label = test_ds.targets[idx]
                # n_passes forward passes with different augmentations
                aug_logits = []
                for _ in range(n_passes):
                    img_tensor, _ = tta_ds[idx]
                    img_tensor = img_tensor.unsqueeze(0).to(device)
                    with amp_autocast():
                        logits = model(img_tensor)
                    aug_logits.append(F.softmax(logits.float(), dim=-1))
                mean_probs = torch.stack(aug_logits).mean(0)
                pred = mean_probs.argmax(1).item()
                if pred == label:
                    tta_correct += 1
                tta_total += 1

        tta_acc = tta_correct / max(tta_total, 1)
        print(f"  TTA accuracy  : {tta_acc:.4f}  (vs standard {test_acc:.4f})")
        logger.report_scalar("Final Results", "tta_acc", tta_acc, 0)

    # ClearML: log all final scalars
    logger.report_scalar("Final Results", "best_val_acc",  best_val_acc,  0)
    logger.report_scalar("Final Results", "test_acc",      test_acc,      0)
    logger.report_scalar("Final Results", "ms_per_image",  ms_per_img,    0)

# ─────────────────────────────────────────────────────────────────────────────
# Save results JSON
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "model_name":   model_name,
    "best_val_acc": best_val_acc,
    "best_epoch":   best_epoch,
    "test_acc":     test_acc,
    "tta_acc":      tta_acc,
    "ms_per_image": ms_per_img,
    "epochs_run":   len(history["train_loss"]),
    "args":         args,
}
results_json = out / "results.json"
results_json.write_text(json.dumps(results, indent=2))
print(f"\n  Results saved: {results_json}")

# ─────────────────────────────────────────────────────────────────────────────
# Upload checkpoint + register in ClearML Model Registry
# ─────────────────────────────────────────────────────────────────────────────
ckpt_path = out / f"{model_name}_best.pth"
if ckpt_path.exists():
    # ClearML artifacts
    task.upload_artifact(f"{model_name}_checkpoint", str(ckpt_path))
    task.upload_artifact("final_model_path",         str(ckpt_path))
    task.upload_artifact("final_test_acc",           test_acc)
    task.upload_artifact("final_tta_acc",            tta_acc)

    # ClearML Model Registry
    output_model = OutputModel(
        task        = task,
        name        = f"banana-ripeness-{model_name}",
        tags        = [model_name, "production", f"val={best_val_acc:.4f}",
                       f"test={test_acc:.4f}"],
        framework   = "PyTorch",
    )
    output_model.update_weights(str(ckpt_path), auto_delete_file=False)
    output_model.update_design(config_dict={
        "model":        model_name,
        "num_classes":  NUM_CLASSES,
        "class_names":  CLASS_NAMES,
        "input_size":   INPUT_SIZE,
        "best_val_acc": best_val_acc,
        "test_acc":     test_acc,
        "tta_acc":      tta_acc,
        "args":         args,
    })
    print(f"\n  ✓  Checkpoint uploaded: {ckpt_path.name}")
    print(f"  ✓  Registered in ClearML Model Registry as 'banana-ripeness-{model_name}'")

    # Copy to models/ for API use
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / f"{model_name}_best.pth"
    shutil.copy2(ckpt_path, dest)
    print(f"  ✓  Copied to: {dest}")
else:
    print(f"\n  WARNING: checkpoint not found at {ckpt_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  STAGE 5 — FINAL RESULTS")
print(f"{'='*62}")
print(f"  Model          : {model_name.upper()}")
print(f"  Best val acc   : {best_val_acc:.4f}  (epoch {best_epoch})")
print(f"  Test acc       : {test_acc:.4f}")
if tta_acc:
    print(f"  TTA acc        : {tta_acc:.4f}")
print(f"  ms / image     : {ms_per_img:.2f}")
print(f"  Checkpoint     : {ckpt_path}")
print(f"\n  Model is ready for API inference.")
print(f"  Run: uvicorn main:app --host 0.0.0.0 --port 8000")
print(f"{'='*62}")

task.close()
print("\nStage 5 COMPLETE ✓")
