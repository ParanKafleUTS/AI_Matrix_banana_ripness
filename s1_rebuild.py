"""
s1_rebuild.py  —  Rebuild pipeline dataset from clean source
=============================================================
Uses DATA/Banana_Ripeness_Classification as the single authoritative
source of truth, discarding the mislabelled data/  directory entirely.

Steps:
    1.  Pool all images from B (train + valid + test splits)
    2.  Deduplicate by MD5 hash (removes exact duplicates)
    3.  Stratified random split  →  train 70% / val 20% / test 10%
    4.  Hard-copy files into data/  (old data/ backed up first)
    5.  Verify final counts and class balance
    6.  Save a manifest CSV of every file + its assigned split

Run:
    python s1_rebuild.py

Output:
    data/
        train/overripe/  train/ripe/  train/rotten/  train/unripe/
        val/  ...
        test/ ...
    data_backup_<timestamp>/   ← your old data/ moved here, not deleted
    results_rebuild/
        report.txt
        distribution.png
        manifest.csv
"""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_DIR  = Path("DATA/Banana_Ripeness_Classification")
DEST_DIR    = Path("data")
OUT_DIR     = Path("results_rebuild")
EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED         = 42

random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

lines: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)

def fast_hash(path: Path, chunk: int = 16_384) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(chunk))
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Validate source
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 62)
log("  STAGE 1 REBUILD — from clean source")
log("=" * 62)
log(f"  Source : {SOURCE_DIR.resolve()}")
log(f"  Dest   : {DEST_DIR.resolve()}")
log()

if not SOURCE_DIR.exists():
    log(f"  ERROR: source not found: {SOURCE_DIR.resolve()}")
    exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Pool all images from source, grouped by class
# ─────────────────────────────────────────────────────────────────────────────
log("─" * 62)
log("  STEP 1 — Pooling images from source")
log("─" * 62)

# Collect every image path, keyed by class
# Handles both split→class and class-only layouts
raw_pool: dict[str, list[Path]] = defaultdict(list)
skip_dirs = {".ipynb_checkpoints", "__pycache__", ".git"}

for item in sorted(SOURCE_DIR.iterdir()):
    if not item.is_dir() or item.name in skip_dirs:
        continue
    # Check if this is a split folder (train/valid/test) or a class folder
    sub_items = [s for s in item.iterdir() if s.is_dir()]
    is_split  = any(
        any(f.suffix.lower() in EXTENSIONS for f in s.iterdir() if f.is_file())
        for s in sub_items[:3]
        if s.is_dir()
    )
    if is_split:
        # split → class → files
        for cls_dir in sorted(item.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name in skip_dirs:
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in EXTENSIONS:
                    raw_pool[cls_dir.name].append(f)
    else:
        # class → files directly
        for f in sorted(item.iterdir()):
            if f.suffix.lower() in EXTENSIONS:
                raw_pool[item.name].append(f)

all_classes = sorted(raw_pool)
log(f"  Classes found : {all_classes}")
log()

for cls in all_classes:
    log(f"    {cls:<14}  {len(raw_pool[cls]):>6,} images collected")
log(f"    {'TOTAL':<14}  {sum(len(v) for v in raw_pool.values()):>6,} images")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Deduplicate by hash
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 62)
log("  STEP 2 — Deduplicating by MD5 hash")
log("─" * 62)

deduped: dict[str, list[Path]] = {}
total_dupes = 0

for cls in all_classes:
    seen: set[str] = set()
    unique: list[Path] = []
    for p in raw_pool[cls]:
        try:
            h = fast_hash(p)
        except Exception:
            log(f"    WARNING: could not hash {p.name} — skipped")
            continue
        if h not in seen:
            seen.add(h)
            unique.append(p)
        else:
            total_dupes += 1
    deduped[cls] = unique
    log(f"    {cls:<14}  {len(raw_pool[cls]):>6,}  →  {len(unique):>6,} unique"
        f"  (removed {len(raw_pool[cls]) - len(unique):,} dupes)")

log(f"\n    Total duplicates removed: {total_dupes:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Verify images are readable before splitting
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 62)
log("  STEP 3 — Verifying image integrity")
log("─" * 62)

clean: dict[str, list[Path]] = {}
total_corrupt = 0

for cls in all_classes:
    good: list[Path] = []
    bad:  list[str]  = []
    for p in deduped[cls]:
        try:
            with Image.open(p) as img:
                img.verify()
            good.append(p)
        except Exception as e:
            bad.append(f"{p.name}: {e}")
            total_corrupt += 1
    clean[cls] = good
    status = f"  ({len(bad)} corrupt skipped)" if bad else "  ✓"
    log(f"    {cls:<14}  {len(good):>6,} clean images{status}")

if total_corrupt:
    log(f"\n    WARNING: {total_corrupt} corrupt images removed from pipeline.")
else:
    log(f"\n    ✓  All images are readable.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Stratified split  train / val / test
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 62)
log("  STEP 4 — Stratified split  "
    f"(train={SPLIT_RATIOS['train']:.0%} / "
    f"val={SPLIT_RATIOS['val']:.0%} / "
    f"test={SPLIT_RATIOS['test']:.0%})")
log("─" * 62)

# {split: {class: [paths]}}
splits: dict[str, dict[str, list[Path]]] = {s: {} for s in SPLIT_RATIOS}

for cls in all_classes:
    paths = clean[cls].copy()
    random.shuffle(paths)
    n     = len(paths)
    n_val  = max(1, round(n * SPLIT_RATIOS["val"]))
    n_test = max(1, round(n * SPLIT_RATIOS["test"]))
    n_train = n - n_val - n_test

    splits["train"][cls] = paths[:n_train]
    splits["val"][cls]   = paths[n_train:n_train + n_val]
    splits["test"][cls]  = paths[n_train + n_val:]

    log(f"    {cls:<14}  train={len(splits['train'][cls]):>5,}  "
        f"val={len(splits['val'][cls]):>4,}  test={len(splits['test'][cls]):>4,}")

log()
for split in SPLIT_RATIOS:
    total = sum(len(splits[split][c]) for c in all_classes)
    log(f"    {split.upper():<8}  total={total:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Back up old data/ and write new splits
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 62)
log("  STEP 5 — Writing new dataset")
log("─" * 62)

# Back up existing data/ directory
if DEST_DIR.exists():
    ts          = time.strftime("%Y%m%d_%H%M%S")
    backup_path = DEST_DIR.parent / f"data_backup_{ts}"
    log(f"  Backing up existing data/ → {backup_path.name} ...")
    shutil.move(str(DEST_DIR), str(backup_path))
    log(f"  ✓  Backup complete: {backup_path.resolve()}")
else:
    log("  No existing data/ directory — creating fresh.")

DEST_DIR.mkdir(parents=True, exist_ok=True)

# Copy files
manifest_rows: list[dict] = []
total_copied = 0

for split in SPLIT_RATIOS:
    for cls in all_classes:
        dest_cls = DEST_DIR / split / cls
        dest_cls.mkdir(parents=True, exist_ok=True)
        for src in splits[split][cls]:
            dest = dest_cls / src.name
            # If dest filename already exists (from another split source), append hash
            if dest.exists():
                h    = fast_hash(src)[:8]
                dest = dest_cls / f"{src.stem}_{h}{src.suffix}"
            shutil.copy2(src, dest)
            total_copied += 1
            manifest_rows.append({
                "split": split, "class": cls,
                "filename": dest.name, "source": str(src)
            })

log(f"\n  ✓  {total_copied:,} images copied to {DEST_DIR.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Verify final counts
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 62)
log("  STEP 6 — Final verification")
log("─" * 62)

final_counts: dict[str, dict[str, int]] = {}
for split in SPLIT_RATIOS:
    final_counts[split] = {}
    for cls in all_classes:
        dest_cls = DEST_DIR / split / cls
        final_counts[split][cls] = len([
            f for f in dest_cls.iterdir() if f.suffix.lower() in EXTENSIONS
        ]) if dest_cls.exists() else 0

col_w = max(len(c) for c in all_classes) + 2
header = f"  {'Class':<{col_w}}" + "".join(
    f"  {s.upper():>8}  {'%':>6}" for s in SPLIT_RATIOS)
log(header)
log("  " + "-" * (len(header) - 2))

for cls in all_classes:
    row = f"  {cls:<{col_w}}"
    for split in SPLIT_RATIOS:
        n   = final_counts[split].get(cls, 0)
        tot = sum(final_counts[split].values())
        pct = 100 * n / tot if tot > 0 else 0
        row += f"  {n:>8,}  {pct:>5.1f}%"
    log(row)

log("  " + "-" * (len(header) - 2))
totals_row = f"  {'TOTAL':<{col_w}}"
for split in SPLIT_RATIOS:
    tot = sum(final_counts[split].values())
    totals_row += f"  {tot:>8,}  {'100.0%':>6}"
log(totals_row)

# Class balance check
log()
log("  Class balance check (each split should mirror overall distribution):")
any_imbalance = False
for cls in all_classes:
    pcts = []
    for split in SPLIT_RATIOS:
        tot = sum(final_counts[split].values())
        pct = 100 * final_counts[split].get(cls, 0) / tot if tot > 0 else 0
        pcts.append(pct)
    spread = max(pcts) - min(pcts)
    flag   = "  ⚠  spread > 3pp" if spread > 3.0 else "  ✓"
    if spread > 3.0:
        any_imbalance = True
    log(f"    {cls:<14}  "
        + "  ".join(f"{p:.1f}%" for p in pcts)
        + f"  (spread={spread:.1f}pp){flag}")
if not any_imbalance:
    log("  ✓  All classes are evenly distributed across splits.")


# ─────────────────────────────────────────────────────────────────────────────
# Distribution chart
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = ["#60a5fa", "#4ade80", "#f87171", "#fbbf24"]

fig, axes = plt.subplots(1, len(SPLIT_RATIOS), figsize=(5 * len(SPLIT_RATIOS), 5))
for ax, split in zip(axes, SPLIT_RATIOS):
    vals  = [final_counts[split].get(cls, 0) for cls in all_classes]
    total = sum(vals)
    bars  = ax.bar(all_classes, vals,
                   color=PALETTE[:len(all_classes)], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        pct = 100 * v / total if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    ax.set_title(f"{split.upper()}  (n={total:,})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class"); ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Rebuilt Dataset — Class Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
chart = OUT_DIR / "distribution.png"
fig.savefig(chart, dpi=150, bbox_inches="tight")
plt.close()
log(f"\n  Distribution chart saved: {chart}")


# ─────────────────────────────────────────────────────────────────────────────
# Manifest CSV
# ─────────────────────────────────────────────────────────────────────────────
manifest_path = OUT_DIR / "manifest.csv"
with open(manifest_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["split", "class", "filename", "source"])
    writer.writeheader()
    writer.writerows(manifest_rows)
log(f"  Manifest CSV saved: {manifest_path}  ({len(manifest_rows):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 62)
log("  REBUILD COMPLETE")
log("=" * 62)

grand_total = sum(
    final_counts[s].get(c, 0)
    for s in SPLIT_RATIOS for c in all_classes
)
log(f"  Total images in new data/  : {grand_total:,}")
for split in SPLIT_RATIOS:
    tot = sum(final_counts[split].values())
    log(f"    {split:<8} : {tot:,}")

log()
log("  Old dataset safely backed up — nothing was deleted.")
log("  Next step: re-run s4_hpo.py with the clean dataset.")
log("=" * 62)

# Save report
(OUT_DIR / "report.txt").write_text("\n".join(lines))
print(f"\n  Report saved: {OUT_DIR / 'report.txt'}")
