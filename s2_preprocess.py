"""
s2_preprocess.py  —  Pipeline Stage 2: Verify & Preprocess Dataset
===================================================================
What this file does:
  1. Retrieves the versioned dataset from ClearML (uploaded by Stage 1)
  2. Verifies all 4 classes exist in all 3 splits
  3. Logs class distribution chart and per-split counts to ClearML
  4. Saves the verified local data path as artifact for Stages 3–5

Run this ONCE to register it as a ClearML Task:
  python s2_preprocess.py

Requirements:
  pip install clearml matplotlib
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clearml import Dataset, Task

# ── Step 1: Create ClearML Task ──────────────────────────────────────────────
task = Task.init(
    project_name = "Banana Ripeness",
    task_name    = "Pipeline step 2 process dataset",
    task_type    = Task.TaskTypes.data_processing,
)
task.add_tags(["preprocessing", "pipeline"])

# ── Step 2: Parameters ───────────────────────────────────────────────────────
args = {
    "dataset_task_id": "",              # filled by pipeline: "${stage_data.id}"
    "dataset_name":    "banana-ripeness",
}
task.connect(args, name="General")

print("=" * 55)
print("STAGE 2: Verifying dataset")
print("=" * 55)

CLASS_NAMES  = ["overripe", "ripe", "rotten", "unripe"]
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Step 3: Retrieve dataset from ClearML ─────────────────────────────────────
# The pipeline passes ${stage_data.id} which is the *task* ID of Stage 1,
# not a ClearML Dataset ID. Stage 1 saves the actual Dataset ID as an artifact
# named "dataset_id". We must fetch that artifact value first, then call
# Dataset.get() with the real Dataset ID.
if args["dataset_task_id"]:
    print(f"\n  Resolving dataset ID from Stage 1 task: {args['dataset_task_id']}")
    try:
        stage1_task = Task.get_task(task_id=args["dataset_task_id"])
        dataset_id  = stage1_task.artifacts["dataset_id"].get()
        print(f"  Dataset ID resolved: {dataset_id}")
        dataset = Dataset.get(dataset_id=dataset_id)
    except Exception as e:
        print(f"  Could not resolve dataset from task artifact ({e})")
        print(f"  Falling back to lookup by name: {args['dataset_name']}")
        dataset = Dataset.get(
            dataset_name    = args["dataset_name"],
            dataset_project = "Banana Ripeness",
        )
else:
    print(f"\n  Loading latest dataset by name: {args['dataset_name']}")
    dataset = Dataset.get(
        dataset_name    = args["dataset_name"],
        dataset_project = "Banana Ripeness",
    )

data_path = Path(dataset.get_local_copy())
print(f"  Dataset local path: {data_path}")

# ── Step 4: Count images per split per class ──────────────────────────────────
print("\n  Counting images …\n")
logger = task.get_logger()
stats: dict[str, dict[str, int]] = {}
total = 0

for split in ["train", "val", "test"]:
    split_path  = data_path / split
    split_stats: dict[str, int] = {}

    for cls in CLASS_NAMES:
        cls_path = split_path / cls
        n = (sum(1 for f in cls_path.iterdir()
                 if f.suffix.lower() in IMAGE_EXTS)
             if cls_path.exists() else 0)
        split_stats[cls] = n
        total            += n
        logger.report_scalar(f"Images / {split}", cls, n, iteration=0)

    stats[split] = split_stats
    split_total  = sum(split_stats.values())
    print(f"  {split.upper()} split  ({split_total:,} images)")
    for cls, n in split_stats.items():
        bar = "█" * min(40, n // 50)
        print(f"    {cls:<12}  {n:>5}  {bar}")
    print()

print(f"  Total: {total:,} images across all splits and classes")

# ── Step 5: Plot class distribution chart ─────────────────────────────────────
x      = np.arange(len(CLASS_NAMES))
width  = 0.25
colors = ["#60a5fa", "#4ade80", "#f87171"]
fig, ax = plt.subplots(figsize=(10, 5))

for i, (split, color) in enumerate(zip(["train", "val", "test"], colors)):
    counts = [stats[split][cls] for cls in CLASS_NAMES]
    ax.bar(x + i * width, counts, width, label=split.capitalize(), color=color, alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylabel("Image count")
ax.set_title("Dataset class distribution by split", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

logger.report_matplotlib_figure(
    title="Dataset", series="Class Distribution", figure=fig, iteration=0
)
plt.close()

# ── Step 6: Verify completeness ───────────────────────────────────────────────
print("\n  Verification:")
all_ok = True
for split in ["train", "val", "test"]:
    for cls in CLASS_NAMES:
        n = stats[split].get(cls, 0)
        if n == 0:
            print(f"  ✗  MISSING: {split}/{cls}")
            all_ok = False

if all_ok:
    print("  ✓  All splits and classes present")
else:
    raise RuntimeError(
        "Dataset verification failed — see MISSING entries above.\n"
        "Re-run Stage 1 to re-download and re-split the dataset."
    )

# ── Step 7: Save artifacts for Stages 3–5 ────────────────────────────────────
task.upload_artifact("data_path",     str(data_path))
task.upload_artifact("dataset_stats", json.dumps(stats))
task.upload_artifact("total_images",  total)
print(f"\n  Data path artifact saved for downstream stages: {data_path}")

task.close()
print("\nStage 2 COMPLETE ✓")
