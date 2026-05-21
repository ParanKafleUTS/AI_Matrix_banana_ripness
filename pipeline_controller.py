"""
pipeline_controller.py  —  ClearML 5-Stage Pipeline Controller
===============================================================
Full pipeline flow:
    stage_data ──► stage_process ──► stage_train ──┐
                                                     ├──► stage_final
                                  └──► stage_hpo  ──┘

  Stage 1  (stage_data)    : Download & version dataset from Kaggle
  Stage 2  (stage_process) : Verify dataset & log class distribution
  Stage 3  (stage_train)   : Baseline training — all 4 models, 50 epochs
  Stage 4  (stage_hpo)     : Optuna HPO — all 4 models, 15 trials × 8 epochs
  Stage 5  (stage_final)   : Full training with best HPO params, 50 epochs

NOTE: Stages 3 and 4 run in parallel (both depend on Stage 2).
      Stage 5 waits for both Stage 3 and Stage 4 to complete.

IMPORTANT — Dataset rebuild
    The original Stage 1 script (s1_dataset.py) produces a mislabelled dataset.
    Run the clean rebuild ONCE manually BEFORE launching this pipeline:
        python s1_rebuild.py
    This replaces data/ with a clean split from DATA/Banana_Ripeness_Classification.
    The pipeline then reads from data/ directly — stage_data is skipped if data/
    already exists with all four class folders present.

IMPORTANT — Register each stage task ONCE before running this controller:
    python s2_preprocess.py
    python s3_train_model.py
    python s4_hpo.py
    python s5_final_model.py

Then run this controller:
    python pipeline_controller.py

Alternatively, since all stages are self-contained, you can run them manually
in order without using this controller at all:
    python s1_rebuild.py          # clean dataset (run once)
    python s2_preprocess.py
    python s3_train_model.py      # can run in parallel with s4
    python s4_hpo.py              # can run in parallel with s3
    python s5_final_model.py      # run after both s3 and s4 finish

Usage:
    python pipeline_controller.py                         # local, all defaults
    python pipeline_controller.py --remote                # submit to agent queue
    python pipeline_controller.py --epochs 60             # longer final training
    python pipeline_controller.py --hpo_trials 20         # more HPO trials
"""

from __future__ import annotations

import argparse
from clearml.automation import PipelineController


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────


def build_pipeline(args: argparse.Namespace) -> PipelineController:
    """Construct and return the 5-stage pipeline (does not launch it)."""

    pipe = PipelineController(
        name              = "Banana Ripeness Pipeline",
        project           = "Banana Ripeness",
        version           = "4.0",
        add_pipeline_tags = True,
    )

    # ── Stage 1: Download & version dataset ───────────────────────────────────
    # NOTE: s1_rebuild.py does not have a ClearML Task, so it cannot be a
    # pipeline step. If you are starting from a raw Kaggle download (data/ does
    # not yet exist), run  python s1_rebuild.py  manually first.
    # If data/ already exists and is clean, this step is effectively a no-op
    # and stage_process will verify it.
    pipe.add_step(
        name               = "stage_data",
        base_task_project  = "Banana Ripeness",
        base_task_name     = "Pipeline step 1 dataset",
        parameter_override = {
            "General/val_split":      args.val_split,
            "General/test_split":     args.test_split,
            "General/out_dir":        "data",
            "General/force_download": False,
        },
    )
    print("  [+] stage_data")

    # ── Stage 2: Verify & log class distribution ──────────────────────────────
    pipe.add_step(
        name               = "stage_process",
        parents            = ["stage_data"],
        base_task_project  = "Banana Ripeness",
        base_task_name     = "Pipeline step 2 process dataset",
        parameter_override = {
            # FIX: ${stage_data.id} resolves to the ClearML *task* ID, but
            # Dataset.get(dataset_id=...) in s2 expects the ClearML *dataset* ID.
            # Stage 1 saves the dataset ID as an artifact named "dataset_id" —
            # use ${stage_data.artifacts.dataset_id} to get that value instead.
            "General/dataset_task_id": "${stage_data.artifacts.dataset_id}",
        },
    )
    print("  [+] stage_process  (waits for stage_data)")

    # ── Stage 3: Baseline training — all 4 models ─────────────────────────────
    # Runs in PARALLEL with Stage 4 (both depend only on Stage 2).
    # Parameter names match args{} in s3_train_model.py exactly.
    pipe.add_step(
        name               = "stage_train",
        parents            = ["stage_process"],
        base_task_project  = "Banana Ripeness",
        base_task_name     = "Pipeline step 3 train model",
        parameter_override = {
            "General/models_to_train": str(["efficientnet", "efficientnetv2",
                                            "mobilenet", "resnet"]),
            "General/data_path":       "data",
            "General/output_dir":      "results_baseline",
            "General/epochs":          50,
            "General/batch_size":      args.batch_size,
            "General/lr":              3e-4,
            "General/aug_strength":    0.5,
            "General/mixup_alpha":     0.0,
            "General/unfreeze_after":  5,
            "General/num_workers":     2,
        },
    )
    print("  [+] stage_train    (waits for stage_process, runs parallel to stage_hpo)")

    # ── Stage 4: Optuna HPO — all 4 models ────────────────────────────────────
    # Runs in PARALLEL with Stage 3.
    # Parameter names match args{} in s4_hpo.py exactly.
    #
    # Key fixes vs previous version:
    #   unfreeze_after: 0  →  3   (0 caused backbone corruption on ImageNet init;
    #                              backbone must be frozen for the first few epochs
    #                              when no warm-start checkpoint is loaded)
    #   num_workers:    0  →  2   (matches s4 default; fork context is safe on Linux)
    #   Added hpo_subset and seed  (were missing — controller now explicitly sets them)
    pipe.add_step(
        name               = "stage_hpo",
        parents            = ["stage_process"],
        base_task_project  = "Banana Ripeness",
        base_task_name     = "Pipeline step 4 hpo",
        parameter_override = {
            # Which models to tune — must match "models_to_tune" key in s4 args{}
            "General/models_to_tune":  str(["efficientnet", "efficientnetv2",
                                            "mobilenet", "resnet"]),
            "General/data_path":       "data",
            "General/output_dir":      "results_hpo",
            "General/hpo_trials":      args.hpo_trials,
            "General/hpo_epochs":      8,
            "General/hpo_patience":    4,

            # Fraction of train set used per trial (0.5 = faster trials)
            "General/hpo_subset":      0.5,

            # Warm-start: empty string = use ImageNet init (safe default)
            "General/warm_start_dir":  "",

            # FIX: was 0 — caused random head to corrupt pretrained backbone.
            # Must be ≥ 1 when starting from ImageNet init (no warm-start ckpt).
            "General/unfreeze_after":  3,

            # FIX: was 0 — matches s4 default; fork context safe on SageMaker Linux
            "General/num_workers":     2,

            "General/seed":            42,
        },
    )
    print("  [+] stage_hpo      (waits for stage_process, runs parallel to stage_train)")

    # ── Stage 5: Final model — best HPO params ────────────────────────────────
    # Waits for BOTH Stage 3 and Stage 4 to finish.
    # s5_final_model.py auto-loads the best model + params from
    # results_hpo/hpo_results.json, so most params here are overrides /
    # safety defaults. Parameter names match args{} in s5_final_model.py exactly.
    #
    # Key fixes vs previous version:
    #   REMOVED lr_finetune   — key does not exist in s5 args{}
    #   REMOVED sampler_mode  — key does not exist in s5 args{}
    #   REMOVED warmup_epochs — key does not exist in s5 args{}
    #   ADDED   weight_decay, dropout, label_smoothing, optimizer, tta_passes
    #           (were missing — s5 exposes all HPO search-space params)
    pipe.add_step(
        name               = "stage_final",
        parents            = ["stage_train", "stage_hpo"],
        base_task_project  = "Banana Ripeness",
        base_task_name     = "Pipeline step 5 final model",
        parameter_override = {
            # s5 auto-loads the best model from results_hpo/hpo_results.json.
            # "model" here is a fallback default if the JSON isn't found.
            "General/model":           "mobilenet",

            "General/data_path":       "data",
            "General/output_dir":      "results_final",
            "General/epochs":          args.epochs,
            "General/unfreeze_after":  5,
            "General/patience":        10,

            # ── Hyperparameters ───────────────────────────────────────────────
            # s5 auto-loads these from HPO results too; values below are
            # fallback defaults that match mobilenet's typical HPO output.
            # Pipeline overrides are only applied when the JSON isn't found.
            "General/batch_size":      args.batch_size,
            "General/lr":              1.33e-4,
            "General/weight_decay":    7.1e-3,   # typical HPO output for mobilenet
            "General/dropout":         0.39,
            "General/label_smoothing": 0.14,
            "General/mixup_alpha":     0.3,
            "General/aug_strength":    0.494,
            "General/optimizer":       "adamw",

            # TTA: number of augmented passes per test image (1 = disabled)
            "General/tta_passes":      5,

            "General/num_workers":     2,
            "General/seed":            42,
        },
    )
    print("  [+] stage_final    (waits for stage_train + stage_hpo)")

    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Banana Ripeness — ClearML 5-Stage Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages
---------------
  1  stage_data     Download & version Kaggle dataset  (skip if data/ already clean)
  2  stage_process  Verify splits + log class distribution
  3  stage_train    Baseline — all 4 models, 50 epochs
  4  stage_hpo      Optuna HPO — all 4 models, 15 trials × 8 epochs
  5  stage_final    Full training with best HPO params, 50 epochs + TTA

Stages 3 and 4 run in parallel.
Stage 5 waits for both.

Prerequisites
-------------
  1. Run  python s1_rebuild.py  ONCE to build a clean dataset from
     DATA/Banana_Ripeness_Classification (fixes 1402 label conflicts).
  2. Register each stage task:
       python s2_preprocess.py
       python s3_train_model.py
       python s4_hpo.py
       python s5_final_model.py

Examples
--------
  python pipeline_controller.py
  python pipeline_controller.py --epochs 60
  python pipeline_controller.py --hpo_trials 20
  python pipeline_controller.py --remote
        """,
    )
    p.add_argument("--epochs",       type=int,   default=50,
                   help="Full training epochs in Stage 5 (default: 50)")
    p.add_argument("--batch_size",   type=int,   default=32,
                   help="Batch size across all stages (default: 32)")
    p.add_argument("--hpo_trials",   type=int,   default=15,
                   help="Optuna trials per model in Stage 4 (default: 15)")
    p.add_argument("--val_split",    type=float, default=0.20,
                   help="Validation fraction for stage_data (default: 0.20)")
    p.add_argument("--test_split",   type=float, default=0.10,
                   help="Test fraction for stage_data (default: 0.10)")
    p.add_argument("--remote",       action="store_true",
                   help="Submit to clearml-agent queue instead of running locally")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 62)
    print("  Banana Ripeness — ClearML 5-Stage Pipeline")
    print("=" * 62)
    print(f"  Epochs (S5)  : {args.epochs}")
    print(f"  HPO trials   : {args.hpo_trials} per model  "
          f"({args.hpo_trials * 4} total)")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Mode         : {'remote agent' if args.remote else 'local'}")
    print("=" * 62)
    print()
    print("  PREREQUISITE: run  python s1_rebuild.py  before this controller")
    print("  if data/ does not yet exist or contains the original dataset.")
    print()
    print("  Alternatively, run stages manually without this controller:")
    print("    python s1_rebuild.py")
    print("    python s2_preprocess.py")
    print("    python s3_train_model.py  # parallel with s4")
    print("    python s4_hpo.py          # parallel with s3")
    print("    python s5_final_model.py  # after s3 + s4")
    print()
    print("Adding pipeline steps …")

    pipe = build_pipeline(args)

    print()
    print("Pipeline structure:")
    print("  stage_data")
    print("       └─► stage_process")
    print("                ├─► stage_train ──────────────────┐")
    print("                └─► stage_hpo  ──────────────────►┴─► stage_final")
    print()

    if args.remote:
        pipe.start(queue="default")
        print("  Pipeline submitted to remote agent queue.")
        print("  Start agent with:")
        print("    clearml-agent daemon --queue default --foreground")
    else:
        pipe.start_locally(run_pipeline_steps_locally=True)
        print("  Pipeline completed locally.")

    print()
    print("  View results: https://app.clear.ml")
    print("  Navigate to:  Banana Ripeness → Pipelines tab")


if __name__ == "__main__":
    main()
