"""
s4_hpo.py  —  Pipeline Stage 4: Hyperparameter Optimisation (All 4 Models)
===========================================================================
Self-contained — does NOT require train.py to be present.

Runs Optuna HPO for all 4 architectures and finds the best hyperparameters
for each one. Each model gets its own ClearML Task with full trial logging.

Search space:
    lr              log-uniform  [1e-5, 1e-2]
    weight_decay    log-uniform  [1e-5, 1e-2]
    dropout         uniform      [0.1,  0.5]
    label_smoothing uniform      [0.05, 0.2]
    mixup_alpha     uniform      [0.0,  0.6]
    aug_strength    uniform      [0.4,  1.0]
    batch_size      categorical  [32, 64]
    optimizer       categorical  [adamw, sgd]

Fixes applied vs original:
    - WeightedRandomSampler       → fixes class imbalance
    - autocast + .float()         → fixes Half/Float dtype crash on CUDA
    - Task.create() per model     → fixes ClearML "task already created" error
    - No subprocess calls         → fully self-contained
    - Conditional unfreeze        → backbone frozen at epoch 1 unless warm-start
                                    checkpoint loaded; prevents head from corrupting
                                    pretrained features (root cause of stagnant val acc)
    - Per-epoch trial.report()    → MedianPruner fires mid-trial as intended
    - mobilenet dropout param     → searched dropout value is now applied
    - Removed dead code           → unused global _sampler and val_crit removed

Additions vs previous fixed version:
    - CLASS_NAMES assert          → crashes early if folder names don't match
                                    CLASS_NAMES (prevents silent weight misalignment)
    - Collapse detector           → prunes trial immediately if >95% of val
                                    predictions are a single class after epoch 2
    - Per-class val accuracy      → logged to ClearML every epoch so per-class
                                    failures are visible on the dashboard
    - Train loss logging          → logged to ClearML every epoch; needed to
                                    diagnose overfitting vs underfitting
    - Best-trial checkpoint save  → best epoch state_dict saved to
                                    results_hpo/<model>_hpo_best.pth; survives crashes
    - Test-set evaluation         → best model evaluated on held-out test split
                                    after HPO; unbiased accuracy logged to ClearML
    - torch.amp compatibility     → try/except shim supports PyTorch 1.x and 2.x
                                    without DeprecationWarning

Run:
    python s4_hpo.py

Requirements:
    pip install clearml optuna torch torchvision scikit-learn numpy matplotlib
"""

from __future__ import annotations

import json
import random
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Compatibility shim: torch.amp is preferred in PyTorch 2.x;
# torch.cuda.amp still works but raises DeprecationWarning.
try:
    from torch.amp import GradScaler, autocast   # PyTorch 2.x
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch 1.x
    _AMP_DEVICE = None   # unused in 1.x — autocast() takes no device arg
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: optuna not installed — run: pip install optuna")

from clearml import Task

# ─────────────────────────────────────────────────────────────────────────────
# Stage-level ClearML Task
# ─────────────────────────────────────────────────────────────────────────────
stage_task = Task.init(
    project_name       = "Banana Ripeness",
    task_name          = "Pipeline step 4 hpo",
    task_type          = Task.TaskTypes.optimizer,
    reuse_last_task_id = False,
)
stage_task.add_tags(["hpo", "optuna", "pipeline", "all-models"])

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
args = {
    # Which models to run HPO on
    "models_to_tune":   ["efficientnet", "efficientnetv2", "mobilenet", "resnet"],

    # Paths
    "data_path":        "data",
    "output_dir":       "results_hpo",

    # HPO settings
    "hpo_trials":       15,   # 15 trials × ~12 min/trial on T4
    "hpo_epochs":       8,    # enough epochs to compare configs
    "hpo_patience":     4,    # early-stop a trial if no improvement for this many epochs

    # Warm-start: set to the folder containing <model>_best.pth files from s3.
    # Leave empty ("") to start from ImageNet weights (safe default).
    "warm_start_dir":   "",

    # HPO subset: fraction of training data per trial (0.5 = 50%)
    # Hyperparameter ranking is stable on subsets — halves epoch time.
    "hpo_subset":       0.5,

    # Fixed settings
    "num_workers":      2,    # 2 workers with fork context — safe on SageMaker Linux
    "seed":             42,

    # FIX 1 — unfreeze_after: number of epochs to train head-only before unfreezing
    # the full backbone. Only used when NO warm-start checkpoint is found.
    # Set to 0 only when a warm-start checkpoint IS loaded (already fine-tuned).
    "unfreeze_after":   3,    # was 0 — caused backbone corruption on ImageNet init
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

print("=" * 60)
print("  STAGE 4: Hyperparameter Optimisation — All 4 Models")
print("=" * 60)
print(f"  Device         : {device}")
if torch.cuda.is_available():
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
print(f"  Models         : {args['models_to_tune']}")
print(f"  Trials / model : {args['hpo_trials']}")
print(f"  Epochs / trial : {args['hpo_epochs']}")
est_mins = len(args["models_to_tune"]) * args["hpo_trials"] * 8 * 36 / 60
est_hrs  = est_mins / 60
print(f"  Total trials   : {len(args['models_to_tune']) * args['hpo_trials']}")
print(f"  Est. time      : ~{est_mins:.0f} min (~{est_hrs:.1f} hrs) on Tesla T4")
print(f"  Warm-start     : {args['warm_start_dir'] or 'disabled (ImageNet init)'}")
print(f"  Unfreeze after : {args['unfreeze_after']} epochs (when no warm-start checkpoint)")
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
def get_train_transforms(aug_strength: float) -> transforms.Compose:
    s = aug_strength
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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (loaded once, transforms swapped per-trial inside run_trial)
# ─────────────────────────────────────────────────────────────────────────────
root = Path(args["data_path"])
if not (root / "train").exists():
    raise FileNotFoundError(
        f"Dataset not found at {root.resolve()}\n"
        "Run Stage 1 first:  python s1_dataset.py"
    )

_train_base = datasets.ImageFolder(root / "train", get_val_transforms())
val_ds       = datasets.ImageFolder(root / "val",   get_val_transforms())
test_ds      = datasets.ImageFolder(root / "test",  get_val_transforms()) \
               if (root / "test").exists() else None

# FIX 1 — Validate CLASS_NAMES against the actual folder order.
# ImageFolder sorts classes alphabetically, so CLASS_NAMES must match exactly.
# A mismatch silently assigns loss weights to the wrong class.
assert list(_train_base.classes) == CLASS_NAMES, (
    f"CLASS_NAMES mismatch!\n"
    f"  Expected : {CLASS_NAMES}\n"
    f"  Got      : {list(_train_base.classes)}\n"
    f"  Fix: update CLASS_NAMES at the top of this file to match the folder names."
)

# Class weights for loss (computed once from the full training set)
class_counts  = np.bincount(_train_base.targets, minlength=NUM_CLASSES)
counts_f      = class_counts.astype(float)
w             = 1.0 / np.maximum(counts_f, 1)
w             = w / w.sum() * NUM_CLASSES
class_weights = torch.tensor(w, dtype=torch.float, device=device)

# Per-sample weights for WeightedRandomSampler (sqrt — gentle rebalance)
# FIX 4 — global _sampler removed; per-trial sub_sampler is built inside run_trial
_sample_weights = [1.0 / np.sqrt(class_counts[t]) for t in _train_base.targets]

print(f"\n  Dataset   train={len(_train_base)}  val={len(val_ds)}"
      + (f"  test={len(test_ds)}" if test_ds else "  test=none"))
print(f"  Classes   {_train_base.classes}  ✓ matches CLASS_NAMES")
print(f"\n  Class distribution + loss weights:")
for cls, cnt, wt in zip(CLASS_NAMES, class_counts, w):
    pct = cnt / len(_train_base) * 100
    print(f"    {cls:<12}  {cnt:>5}  ({pct:5.1f}%)  weight={wt:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
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
        # FIX 5 — searched dropout value is now applied; was silently ignored before.
        # MobileNetV3-Large classifier: [Linear(960,1280), Hardswish, Dropout, Linear(1280,1000)]
        # We replace indices 2 (Dropout) and 3 (head Linear) so dropout is tunable.
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
    """Freeze all layers except the classification head."""
    head = get_head_prefixes(name)
    for pname, p in model.named_parameters():
        p.requires_grad = any(pname.startswith(h) for h in head)

def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter in the model."""
    for p in model.parameters():
        p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosine(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by cosine annealing."""
    def __init__(self, opt, warmup: int, total: int, min_ratio: float = 0.01):
        def fn(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            p = (epoch - warmup) / max(1, total - warmup)
            return min_ratio + 0.5 * (1 - min_ratio) * (1 + np.cos(np.pi * p))
        super().__init__(opt, fn)


def amp_autocast():
    """Returns an autocast context that works on both PyTorch 1.x and 2.x."""
    if _AMP_DEVICE:
        return autocast(device_type=_AMP_DEVICE, enabled=torch.cuda.is_available())
    return autocast(enabled=torch.cuda.is_available())

def make_loader(dataset, sampler=None, batch_size: int = 32, shuffle: bool = False):
    nw = args["num_workers"]
    return DataLoader(
        dataset,
        batch_size              = batch_size,
        sampler                 = sampler,
        shuffle                 = shuffle if sampler is None else False,
        num_workers             = nw,
        pin_memory              = True,
        persistent_workers      = False,
        # 'fork' avoids the SageMaker deadlock that 'spawn' causes with
        # WeightedRandomSampler. Safe on Linux (SageMaker = Linux).
        multiprocessing_context = "fork" if nw > 0 else None,
    )


class LabelSmoothingCE(nn.Module):
    """
    Per-sample label-smoothing cross-entropy with optional class weights.
    Applies class weights per-sample, giving a stronger gradient signal on
    minority classes compared to standard nn.CrossEntropyLoss.
    """
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
# Single trial
# ─────────────────────────────────────────────────────────────────────────────
def run_trial(
    model_name: str,
    trial_cfg:  dict,
    trial_num:  int,
    hpo_logger,
    optuna_trial,          # FIX 2 — passed in so we can report per-epoch + prune mid-trial
) -> float:
    """
    Train one Optuna trial for model_name with trial_cfg hyperparameters.
    Returns best val_acc achieved across all epochs.
    """
    seed_everything(args["seed"] + trial_num)

    # Swap training transforms for this trial's aug_strength
    _train_base.transform = get_train_transforms(trial_cfg["aug_strength"])

    # Subsample the training set for faster trials
    subset_frac = args.get("hpo_subset", 1.0)
    n_samples   = max(1, int(len(_train_base) * subset_frac))
    sub_sampler = WeightedRandomSampler(
        weights     = _sample_weights,
        num_samples = n_samples,
        replacement = True,
    )

    train_loader = make_loader(_train_base, sampler=sub_sampler,
                               batch_size=trial_cfg["batch_size"])
    val_loader   = make_loader(val_ds, shuffle=False,
                               batch_size=trial_cfg["batch_size"])

    # ── Build model ────────────────────────────────────────────────────────────
    model = build_model(model_name, trial_cfg["dropout"]).to(device)

    # ── Warm-start from s3 checkpoint if available ─────────────────────────────
    warm_start_dir    = args.get("warm_start_dir", "")
    ckpt_path         = Path(warm_start_dir) / f"{model_name}_best.pth" if warm_start_dir else None
    checkpoint_loaded = False

    if ckpt_path and ckpt_path.exists():
        try:
            sd = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            missing, _ = model.load_state_dict(sd, strict=False)
            # Reinitialise only the head so backbone warm-start is preserved
            head_prefixes = get_head_prefixes(model_name)
            for pname, p in model.named_parameters():
                if any(pname.startswith(h) for h in head_prefixes):
                    nn.init.xavier_uniform_(p.data) if p.dim() >= 2 else nn.init.zeros_(p.data)
            checkpoint_loaded = True
            print(f"    [warm-start] Loaded {ckpt_path.name}")
        except Exception as e:
            print(f"    [warm-start] Could not load {ckpt_path.name}: {e} — using ImageNet init")

    # FIX 1 — Conditional freeze/unfreeze.
    # If a warm-start checkpoint was loaded the model already has banana-domain
    # weights, so it is safe to fine-tune all layers from epoch 1.
    # If starting from ImageNet init the randomly-initialised head would send
    # large gradients into the pretrained backbone and corrupt its features.
    # In that case we freeze the backbone and train only the head for
    # `unfreeze_after` epochs before unfreezing everything.
    if checkpoint_loaded:
        unfreeze_all(model)
        unfreeze_epoch = 0          # already fine-tuned — never need to freeze
    else:
        freeze_backbone(model, model_name)
        unfreeze_epoch = max(1, args["unfreeze_after"])   # e.g. epoch 3

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────────
    criterion = LabelSmoothingCE(trial_cfg["label_smoothing"], class_weights)

    if trial_cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=trial_cfg["lr"], weight_decay=trial_cfg["weight_decay"])
    else:
        opt = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=trial_cfg["lr"], momentum=0.9,
            weight_decay=trial_cfg["weight_decay"], nesterov=True)

    scheduler = WarmupCosine(opt, warmup=2, total=args["hpo_epochs"])
    if _AMP_DEVICE:
        scaler = GradScaler(_AMP_DEVICE, enabled=torch.cuda.is_available())
    else:
        scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val        = 0.0
    best_state_dict = None   # FIX 5 — track weights of the best epoch
    no_improve      = 0

    for epoch in range(1, args["hpo_epochs"] + 1):

        # Unfreeze backbone after head-warmup phase (ImageNet-init only)
        if not checkpoint_loaded and epoch == unfreeze_epoch:
            unfreeze_all(model)
            # Re-create optimiser so the newly unfrozen params are included,
            # and use a lower LR to avoid disrupting the already-trained head.
            backbone_lr = trial_cfg["lr"] * 0.1
            if trial_cfg["optimizer"] == "adamw":
                opt = torch.optim.AdamW([
                    {"params": [p for n, p in model.named_parameters()
                                if not any(n.startswith(h)
                                           for h in get_head_prefixes(model_name))],
                     "lr": backbone_lr},
                    {"params": [p for n, p in model.named_parameters()
                                if any(n.startswith(h)
                                       for h in get_head_prefixes(model_name))],
                     "lr": trial_cfg["lr"]},
                ], weight_decay=trial_cfg["weight_decay"])
            else:
                opt = torch.optim.SGD([
                    {"params": [p for n, p in model.named_parameters()
                                if not any(n.startswith(h)
                                           for h in get_head_prefixes(model_name))],
                     "lr": backbone_lr},
                    {"params": [p for n, p in model.named_parameters()
                                if any(n.startswith(h)
                                       for h in get_head_prefixes(model_name))],
                     "lr": trial_cfg["lr"]},
                ], momentum=0.9, weight_decay=trial_cfg["weight_decay"], nesterov=True)
            # Remaining epochs for the scheduler after unfreeze
            scheduler = WarmupCosine(opt, warmup=1,
                                     total=args["hpo_epochs"] - unfreeze_epoch + 1)

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        running_loss  = 0.0
        running_steps = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Mixup (alpha=0 → no mixing)
            alpha = trial_cfg.get("mixup_alpha", 0.0)
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

            # FIX 4 — accumulate loss for logging
            running_loss  += loss.item()
            running_steps += 1

        mean_train_loss = running_loss / max(running_steps, 1)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        correct        = 0
        total          = 0
        # FIX 3 — per-class counters for per-class accuracy (must be on same device as preds)
        class_correct  = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
        class_total    = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
        all_preds      = []   # FIX 2b — for collapse detection

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with amp_autocast():
                    logits = model(images)
                preds = logits.float().argmax(1)
                correct += (preds == labels).sum().item()
                total   += images.size(0)
                # per-class
                for c in range(NUM_CLASSES):
                    mask = (labels == c)
                    class_correct[c] += (preds[mask] == c).sum()
                    class_total[c]   += mask.sum()
                all_preds.append(preds.cpu())

        val_acc = correct / total
        scheduler.step()

        # ── FIX 4: log train loss to ClearML ───────────────────────────────────
        if hpo_logger:
            hpo_logger.report_scalar(
                f"trial_{trial_num:02d}", "train_loss", mean_train_loss, epoch)

        # ── FIX 3: log per-class val accuracy to ClearML ──────────────────────
        if hpo_logger:
            hpo_logger.report_scalar(
                f"trial_{trial_num:02d}", "val_acc", val_acc, epoch)
            for c, cls_name in enumerate(CLASS_NAMES):
                cls_acc = (class_correct[c] / class_total[c].clamp(min=1)).item()
                hpo_logger.report_scalar(
                    f"trial_{trial_num:02d}_per_class", cls_name, cls_acc, epoch)

        # ── FIX 2b: majority-class collapse detector ──────────────────────────
        # If after epoch 2 more than 95% of val predictions are a single class,
        # the model has collapsed to a majority-class shortcut — prune immediately.
        if epoch >= 2:
            all_preds_cat = torch.cat(all_preds)
            pred_counts   = torch.bincount(all_preds_cat, minlength=NUM_CLASSES)
            top_class_frac = pred_counts.max().item() / len(all_preds_cat)
            if top_class_frac > 0.95:
                top_cls = pred_counts.argmax().item()
                print(f"    ⚠  Trial {trial_num}: collapse detected at epoch {epoch} — "
                      f"{100*top_class_frac:.1f}% of preds = '{CLASS_NAMES[top_cls]}'. Pruning.")
                raise optuna.TrialPruned()

        # ── FIX 2: per-epoch Optuna reporting (was after run_trial — pruner was blind)
        optuna_trial.report(val_acc, epoch)
        if optuna_trial.should_prune():
            raise optuna.TrialPruned()

        # ── FIX 5: save state dict when best val improves ─────────────────────
        if val_acc > best_val + 1e-4:
            best_val        = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve      = 0
        else:
            no_improve += 1
            if no_improve >= args["hpo_patience"]:
                break

    # Save best checkpoint for this trial to a temp file so hpo_one_model
    # can persist the overall best weights after the study finishes.
    if best_state_dict is not None:
        tmp_ckpt = out / f"_tmp_{model_name}_trial{trial_num:03d}.pth"
        torch.save(best_state_dict, tmp_ckpt)

    return best_val


# ─────────────────────────────────────────────────────────────────────────────
# HPO for one model
# ─────────────────────────────────────────────────────────────────────────────
def hpo_one_model(model_name: str) -> dict:
    """
    Run Optuna HPO for a single architecture.
    Returns dict with best_params and best_val_acc.
    """
    if not OPTUNA_AVAILABLE:
        print(f"  [SKIP] Optuna not installed — returning default params for {model_name}")
        return {"model_name": model_name, "best_val_acc": 0.0, "best_params": {}}

    print(f"\n{'='*60}")
    print(f"  HPO: {model_name.upper()}  ({args['hpo_trials']} trials × {args['hpo_epochs']} epochs)")
    print(f"{'='*60}")

    # Per-model ClearML task — Task.create() not Task.init()
    model_task = Task.create(
        project_name = "Banana Ripeness",
        task_name    = f"hpo/{model_name}",
        task_type    = Task.TaskTypes.optimizer,
    )
    model_task.add_tags([model_name, "hpo", "optuna", f"trials-{args['hpo_trials']}"])
    model_task.connect(args, name="HPO Config")
    model_task.mark_started()
    mlog = model_task.get_logger()

    def objective(trial: optuna.Trial) -> float:
        print(f"  Trial {trial.number:>3} starting ...  (model={model_name})", flush=True)

        cfg = {
            "lr":              trial.suggest_float("lr",              1e-5, 1e-2, log=True),
            "weight_decay":    trial.suggest_float("weight_decay",    1e-5, 1e-2, log=True),
            "dropout":         trial.suggest_float("dropout",         0.1,  0.5),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.05, 0.2),
            "mixup_alpha":     trial.suggest_float("mixup_alpha",     0.0,  0.6),
            "aug_strength":    trial.suggest_float("aug_strength",    0.4,  1.0),
            "batch_size":      trial.suggest_categorical("batch_size",  [32, 64]),
            "optimizer":       trial.suggest_categorical("optimizer",   ["adamw", "sgd"]),
        }

        # FIX 2 — pass trial into run_trial so per-epoch reporting works
        val_acc = run_trial(model_name, cfg, trial.number, mlog, trial)

        # Log trial summary to ClearML
        mlog.report_scalar("Trial Results", "val_acc", val_acc, trial.number)
        for k, v in cfg.items():
            if isinstance(v, float):
                mlog.report_scalar(f"Params/{k}", model_name, v, trial.number)

        print(f"  Trial {trial.number:>3}  val_acc={val_acc:.4f}  "
              f"lr={cfg['lr']:.2e}  aug={cfg['aug_strength']:.2f}  "
              f"bs={cfg['batch_size']}  opt={cfg['optimizer']}")

        return val_acc

    study = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=args["seed"]),
        pruner    = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )
    print(f"  Starting trial 0 — first download of pretrained weights may take 1-2 min ...")
    study.optimize(objective, n_trials=args["hpo_trials"], show_progress_bar=True)

    best_params  = study.best_params
    best_val_acc = study.best_value
    best_trial_n = study.best_trial.number

    # FIX 5 — Persist the best trial's checkpoint and clean up temp files.
    # run_trial saved a temp checkpoint whenever val improved; copy the best one.
    best_ckpt_src = out / f"_tmp_{model_name}_trial{best_trial_n:03d}.pth"
    best_ckpt_dst = out / f"{model_name}_hpo_best.pth"
    if best_ckpt_src.exists():
        import shutil as _shutil
        _shutil.copy2(best_ckpt_src, best_ckpt_dst)
        print(f"\n  ✓  Best checkpoint saved: {best_ckpt_dst}")
    # Clean up all temp checkpoints for this model
    for tmp in out.glob(f"_tmp_{model_name}_trial*.pth"):
        tmp.unlink(missing_ok=True)

    print(f"\n  ── Best result for {model_name} ──")
    print(f"  Best val_acc : {best_val_acc:.4f}  (trial {best_trial_n})")
    print(f"  Best params  :")
    for k, v in best_params.items():
        print(f"    {k:<20} = {v}")

    # ── ClearML: log best params ──────────────────────────────────────────────
    model_task.connect(best_params, name="Best Params")
    mlog.report_scalar("Best", "val_acc", best_val_acc, 0)

    # ── ClearML: optimisation history chart ──────────────────────────────────
    trial_vals = [t.value for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trial_vals, marker="o", color="#60a5fa", ms=5, lw=1.5)
    ax.axhline(best_val_acc, color="#f87171", ls="--", lw=1.5,
               label=f"Best={best_val_acc:.4f}")
    ax.set_title(f"HPO History — {model_name}")
    ax.set_xlabel("Trial"); ax.set_ylabel("Val Accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    mlog.report_matplotlib_figure("HPO", "Optimisation History", fig, 0)
    plt.close()

    # ── ClearML: param importance chart ──────────────────────────────────────
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(list(importances.keys()), list(importances.values()), color="#4ade80")
        ax.set_title(f"Hyperparameter Importance — {model_name}")
        ax.set_xlabel("Importance"); ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        mlog.report_matplotlib_figure("HPO", "Param Importance", fig, 0)
        plt.close()
    except Exception:
        pass

    # ── ClearML: Parallel Coordinates plot ───────────────────────────────────
    try:
        completed_trials = [t for t in study.trials
                            if t.value is not None and t.state.name == "COMPLETE"]
        if len(completed_trials) >= 2:
            param_names = list(completed_trials[0].params.keys())
            dimensions  = []

            for p in param_names:
                vals = [t.params.get(p) for t in completed_trials]
                if isinstance(vals[0], str):
                    # Categorical — encode as integers with tick labels
                    unique = sorted(set(vals))
                    encoded = [unique.index(v) for v in vals]
                    dimensions.append(dict(
                        label    = p,
                        values   = encoded,
                        tickvals = list(range(len(unique))),
                        ticktext = unique,
                    ))
                else:
                    dimensions.append(dict(label=p, values=vals))

            # Add val_acc as the final dimension (colour axis)
            accs = [t.value for t in completed_trials]
            dimensions.append(dict(label="val_acc", values=accs))

            # Build parallel coordinates figure using matplotlib
            # (avoids plotly/kaleido dependency)
            n_dims = len(dimensions)
            fig_pc, axes = plt.subplots(1, 1, figsize=(max(12, n_dims * 1.8), 5))
            fig_pc.patch.set_facecolor("#0d1117")
            axes.set_facecolor("#0d1117")

            # Normalise each dimension to [0,1] for plotting
            norm_data = []
            for dim in dimensions:
                v = np.array(dim["values"], dtype=float)
                mn, mx = v.min(), v.max()
                norm_data.append((v - mn) / (mx - mn + 1e-9))
            norm_data = np.array(norm_data)  # shape [n_dims, n_trials]

            cmap  = plt.cm.viridis
            accs_norm = norm_data[-1]   # last dim = val_acc

            xs = np.arange(n_dims)
            for i in range(len(completed_trials)):
                color = cmap(accs_norm[i])
                axes.plot(xs, norm_data[:, i], color=color, alpha=0.55, lw=1.2)

            axes.set_xticks(xs)
            axes.set_xticklabels([d["label"] for d in dimensions],
                                  rotation=25, ha="right", color="white", fontsize=9)
            axes.set_yticks([])
            axes.tick_params(colors="white")
            for sp in axes.spines.values():
                sp.set_edgecolor("#30363d")

            sm = plt.cm.ScalarMappable(cmap=cmap,
                                        norm=plt.Normalize(min(accs), max(accs)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes, pad=0.02)
            cbar.set_label("val_acc", color="white", fontsize=9)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

            axes.set_title(f"Parallel Coordinates — {model_name}",
                           color="white", fontsize=11, fontweight="bold")
            plt.tight_layout()

            pc_path = out / f"{model_name}_parallel_coords.png"
            fig_pc.savefig(str(pc_path), dpi=120, bbox_inches="tight",
                           facecolor=fig_pc.get_facecolor())
            plt.close(fig_pc)

            mlog.report_image(
                title      = "HPO",
                series     = "Parallel Coordinates",
                local_path = str(pc_path),
                iteration  = 0,
            )
            print(f"  ✓  Parallel Coordinates plot saved: {pc_path}")
        else:
            print(f"  ⚠  Not enough completed trials for Parallel Coordinates ({len(completed_trials)})")
    except Exception as e:
        print(f"  ⚠  Parallel Coordinates plot failed: {e}")

    model_task.close()

    return {
        "model_name":   model_name,
        "best_val_acc": best_val_acc,
        "best_params":  best_params,
        "checkpoint":   str(best_ckpt_dst) if best_ckpt_dst.exists() else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pre-warm: download all pretrained weights ONCE before HPO starts
# ─────────────────────────────────────────────────────────────────────────────
# On first call PyTorch downloads weights from CDN — can silently hang 5-10 min
# on SageMaker. Pre-warming forces the download now with visible progress so
# every HPO trial loads from ~/.cache/torch/hub/checkpoints/ immediately.
print("\n  Pre-warming pretrained weights (downloads once, cached after) ...")
_prewarm_map = {
    "efficientnet":   lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
    "efficientnetv2": lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT),
    "mobilenet":      lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT),
    "resnet":         lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
}
for _name in args["models_to_tune"]:
    _t0 = time.time()
    print(f"    Downloading {_name} weights ...", end=" ", flush=True)
    _prewarm_map[_name]()
    print(f"done ({time.time()-_t0:.1f}s)")
del _prewarm_map, _name, _t0
print("  All weights cached locally. HPO trials will start immediately.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Run HPO for all models
# ─────────────────────────────────────────────────────────────────────────────
all_hpo_results: list[dict] = []

for model_name in args["models_to_tune"]:
    result = hpo_one_model(model_name)
    all_hpo_results.append(result)

    # Save after each model so progress isn't lost if a later model crashes
    intermediate = out / "hpo_results.json"
    intermediate.write_text(json.dumps(all_hpo_results, indent=2))
    print(f"\n  Intermediate results saved: {intermediate}")


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
slog = stage_task.get_logger()
best_overall = max(all_hpo_results, key=lambda r: r["best_val_acc"])

sep = "─" * 60
print(f"\n{sep}")
print("  STAGE 4 — HPO RESULTS SUMMARY")
print(sep)
print(f"  {'Model':<16}  {'Best Val Acc':>12}  {'Best LR':>10}  {'Best Aug':>9}")
print(sep)
for r in all_hpo_results:
    bp   = r["best_params"]
    mark = "  ◀ BEST" if r is best_overall else ""
    print(f"  {r['model_name']:<16}  {r['best_val_acc']:>12.4f}  "
          f"  {bp.get('lr', 0):>9.2e}  {bp.get('aug_strength', 0):>9.3f}{mark}")
print(sep)

# Log all best val_accs to stage task
for r in all_hpo_results:
    slog.report_scalar("HPO Best Val Acc", r["model_name"], r["best_val_acc"], 0)

# ── ClearML: HPO Summary table ───────────────────────────────────────────────
try:
    headers = ["Model", "Best Val Acc", "LR", "Batch Size",
               "Dropout", "Optimizer", "Aug Strength"]
    table   = []
    for r in all_hpo_results:
        bp = r.get("best_params", {})
        table.append([
            r["model_name"],
            f"{r['best_val_acc']:.4f}",
            f"{bp.get('lr', 0):.2e}",
            str(bp.get("batch_size", "-")),
            f"{bp.get('dropout', 0):.3f}",
            bp.get("optimizer", "-"),
            f"{bp.get('aug_strength', 0):.3f}",
        ])

    # Plot as a matplotlib table
    fig_t, ax_t = plt.subplots(figsize=(14, max(3, len(all_hpo_results) * 1.2 + 1.5)))
    fig_t.patch.set_facecolor("#0d1117")
    ax_t.set_facecolor("#0d1117")
    ax_t.axis("off")

    col_widths = [0.14, 0.13, 0.10, 0.10, 0.10, 0.12, 0.13]
    tbl = ax_t.table(
        cellText    = table,
        colLabels   = headers,
        loc         = "center",
        cellLoc     = "center",
        colWidths   = col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # Style header
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#1F4E79")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Style rows — highlight best model
    best_idx = max(range(len(all_hpo_results)),
                   key=lambda i: all_hpo_results[i]["best_val_acc"])
    for i, row in enumerate(table):
        for j in range(len(headers)):
            if i == best_idx:
                tbl[i+1, j].set_facecolor("#375623")
                tbl[i+1, j].set_text_props(color="white", fontweight="bold")
            elif i % 2 == 0:
                tbl[i+1, j].set_facecolor("#161b22")
                tbl[i+1, j].set_text_props(color="#e6edf3")
            else:
                tbl[i+1, j].set_facecolor("#0d1117")
                tbl[i+1, j].set_text_props(color="#e6edf3")

    ax_t.set_title("HPO Summary — Best Parameters per Model",
                   color="white", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()

    summary_path = out / "hpo_summary_table.png"
    fig_t.savefig(str(summary_path), dpi=130, bbox_inches="tight",
                  facecolor=fig_t.get_facecolor())
    plt.close(fig_t)

    slog.report_image(
        title      = "HPO Summary",
        series     = "Summary Table",
        local_path = str(summary_path),
        iteration  = 0,
    )
    print(f"  ✓  HPO Summary table saved: {summary_path}")
except Exception as e:
    print(f"  ⚠  HPO Summary table failed: {e}")

# ── Save final results + artifacts ───────────────────────────────────────────
final_json = out / "hpo_results.json"
final_json.write_text(json.dumps(all_hpo_results, indent=2))

stage_task.upload_artifact("hpo_results",     str(final_json))
stage_task.upload_artifact("all_best_params", all_hpo_results)
stage_task.upload_artifact("best_model",      best_overall["model_name"])
stage_task.upload_artifact("data_path",       args["data_path"])

print(f"\n  Best overall: {best_overall['model_name'].upper()} "
      f"(val_acc={best_overall['best_val_acc']:.4f})")
print(f"  Results saved: {final_json}")
print(f"\n  Pass these to Stage 5:")
print(f"    model       = {best_overall['model_name']}")
for k, v in best_overall["best_params"].items():
    print(f"    {k:<20} = {v}")

stage_task.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIX 6 — Test-set evaluation for the best overall model
# ─────────────────────────────────────────────────────────────────────────────
# Val accuracy guided the HPO search so it is an optimistic estimate.
# The held-out test split (never seen during HPO) gives an unbiased number.
if test_ds is not None and best_overall.get("checkpoint"):
    print(f"\n{'─'*60}")
    print("  TEST SET EVALUATION — best model on held-out test split")
    print(f"{'─'*60}")

    best_model_name = best_overall["model_name"]
    best_bp         = best_overall["best_params"]

    test_model = build_model(best_model_name, best_bp.get("dropout", 0.2)).to(device)
    sd = torch.load(best_overall["checkpoint"], map_location=device, weights_only=True)
    test_model.load_state_dict(sd)
    test_model.eval()

    test_loader = make_loader(test_ds, shuffle=False, batch_size=64)
    correct      = 0
    total        = 0
    class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    class_total   = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with amp_autocast():
                logits = test_model(images)
            preds   = logits.float().argmax(1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)
            for c in range(NUM_CLASSES):
                mask = (labels == c)
                class_correct[c] += (preds[mask] == c).sum()
                class_total[c]   += mask.sum()

    test_acc = correct / total
    print(f"\n  Model       : {best_model_name.upper()}")
    print(f"  Val acc     : {best_overall['best_val_acc']:.4f}  (HPO optimised)")
    print(f"  Test acc    : {test_acc:.4f}  (unbiased estimate)")
    print()
    print(f"  Per-class test accuracy:")
    for c, cls_name in enumerate(CLASS_NAMES):
        cls_acc = (class_correct[c] / class_total[c].clamp(min=1)).item()
        bar     = "█" * int(cls_acc * 20)
        print(f"    {cls_name:<12}  {cls_acc:.4f}  {bar}")

    # Log to ClearML
    slog.report_scalar("Test Acc", best_model_name, test_acc, 0)
    for c, cls_name in enumerate(CLASS_NAMES):
        cls_acc = (class_correct[c] / class_total[c].clamp(min=1)).item()
        slog.report_scalar("Test Acc Per Class", cls_name, cls_acc, 0)

    # Add test_acc to the JSON results
    best_overall["test_acc"] = test_acc
    final_json.write_text(json.dumps(all_hpo_results, indent=2))
    print(f"\n  Test results appended to: {final_json}")
else:
    if test_ds is None:
        print("\n  [skip] No test split found — skipping test evaluation.")
    else:
        print("\n  [skip] No checkpoint saved — cannot run test evaluation.")

print("\nStage 4 COMPLETE ✓")
"# CI/CD trigger test - $(date)" 
#test 
#test 
# test ci/cd 
