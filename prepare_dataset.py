"""
prepare_dataset.py – Download and organise the banana ripeness dataset.

The script can source the dataset from:
  1. An S3 path  (default, via --s3-uri or S3_DATASET_URI env var)
  2. A local zip file (via --local-zip)

After extraction it expects (or produces) the layout:
    <output-dir>/
      train/  <class>/  *.jpg
      valid/  <class>/  *.jpg
      test/   <class>/  *.jpg

Usage:
    # from S3
    python prepare_dataset.py --s3-uri s3://my-bucket/banana-ripeness-dataset-original

    # from a local zip
    python prepare_dataset.py --local-zip files.zip
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import boto3

DEFAULT_OUTPUT_DIR = "Dataset/banana-ripeness-dataset-original"
EXPECTED_SPLITS = {"train", "valid", "test"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_from_s3(s3_uri: str, dest_dir: Path) -> None:
    """Download all objects under *s3_uri* into *dest_dir*."""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    print(f"Listing objects at {s3_uri} ...")
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix):].lstrip("/")
            if not relative:
                continue
            local_file = dest_dir / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Downloading s3://{bucket}/{key} → {local_file}")
            s3.download_file(bucket, key, str(local_file))
            total += 1

    print(f"Downloaded {total} file(s).")


def extract_zip(zip_path: str, dest_dir: Path) -> None:
    print(f"Extracting {zip_path} → {dest_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(dest_dir))
    print("Extraction complete.")


def validate_dataset(output_dir: Path) -> None:
    missing = []
    for split in EXPECTED_SPLITS:
        split_path = output_dir / split
        if not split_path.is_dir():
            missing.append(str(split_path))

    if missing:
        print(
            f"\nWarning: the following expected split directories were not found:\n"
            + "\n".join(f"  {p}" for p in missing)
        )
    else:
        class_names = sorted(p.name for p in (output_dir / "train").iterdir() if p.is_dir())
        counts = {
            split: sum(1 for _ in (output_dir / split).rglob("*") if _.is_file())
            for split in EXPECTED_SPLITS
        }
        print(f"\nDataset ready at: {output_dir}")
        print(f"  Classes : {class_names}")
        for split in sorted(EXPECTED_SPLITS):
            print(f"  {split:6s}: {counts[split]} images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download / prepare the banana ripeness dataset")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--s3-uri",
        default=os.environ.get("S3_DATASET_URI"),
        help="S3 URI of the dataset folder (or set S3_DATASET_URI env var)",
    )
    source.add_argument(
        "--local-zip",
        help="Path to a local zip archive containing the dataset",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Where to place the prepared dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and re-download even if output directory already exists",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if output_dir.exists() and not args.force:
        print(f"Output directory '{output_dir}' already exists. Use --force to re-download.")
        validate_dataset(output_dir)
        return

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.local_zip:
        extract_zip(args.local_zip, output_dir)
    elif args.s3_uri:
        download_from_s3(args.s3_uri, output_dir)
    else:
        parser.error("Provide either --s3-uri / S3_DATASET_URI or --local-zip.")

    validate_dataset(output_dir)


if __name__ == "__main__":
    main()
