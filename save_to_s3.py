"""
save_to_s3.py – Upload trained model artefacts to an S3 bucket.

Usage:
    python save_to_s3.py [--bucket BUCKET] [--prefix PREFIX] [--models-dir DIR]

Environment variables (alternative to CLI flags):
    S3_BUCKET   – target bucket name
    S3_PREFIX   – key prefix inside the bucket (default: banana-ripeness-models)
"""

import argparse
import os
from pathlib import Path

import boto3

DEFAULT_MODELS_DIR = "saved_models"
DEFAULT_PREFIX = "banana-ripeness-models"


def upload_models(bucket: str, prefix: str, models_dir: str) -> None:
    s3 = boto3.client("s3")
    models_path = Path(models_dir)

    if not models_path.exists():
        raise FileNotFoundError(f"Models directory '{models_dir}' does not exist. Run train.py first.")

    pth_files = sorted(models_path.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in '{models_dir}'.")

    print(f"Uploading {len(pth_files)} model(s) to s3://{bucket}/{prefix}/")
    for local_path in pth_files:
        s3_key = f"{prefix}/{local_path.name}"
        print(f"  Uploading {local_path} → s3://{bucket}/{s3_key} ...", end=" ")
        s3.upload_file(str(local_path), bucket, s3_key)
        print("done")

    print("\nAll models uploaded successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model weights to S3")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET"),
        required=not os.environ.get("S3_BUCKET"),
        help="S3 bucket name (or set S3_BUCKET env var)",
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("S3_PREFIX", DEFAULT_PREFIX),
        help="S3 key prefix (default: %(default)s)",
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help="Local directory containing .pth files (default: %(default)s)",
    )
    args = parser.parse_args()

    upload_models(bucket=args.bucket, prefix=args.prefix, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
