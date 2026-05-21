"""
hf_upload.py — uploads app files to Hugging Face Space
Called by GitHub Actions deploy workflow.
Reads HF_TOKEN, HF_USERNAME, HF_SPACE_NAME from environment.
"""
import os
import sys
from huggingface_hub import HfApi

token      = os.environ.get("HF_TOKEN")
username   = os.environ.get("HF_USERNAME")
space_name = os.environ.get("HF_SPACE_NAME")

if not all([token, username, space_name]):
    print("ERROR: HF_TOKEN, HF_USERNAME, HF_SPACE_NAME must all be set")
    sys.exit(1)

repo_id = f"{username}/{space_name}"
api     = HfApi(token=token)

files = [
    ("main.py",               "main.py"),
    ("requirements-prod.txt", "requirements-prod.txt"),
    ("Dockerfile",            "Dockerfile"),
    ("README.md",             "README.md"),
    ("static/index.html",     "static/index.html"),
]

for local, remote in files:
    if os.path.exists(local):
        print(f"Uploading {local} ...")
        api.upload_file(
            path_or_fileobj = local,
            path_in_repo    = remote,
            repo_id         = repo_id,
            repo_type       = "space",
        )
        print(f"  OK: {remote}")
    else:
        print(f"  SKIP: {local} not found")

print("Upload complete. HF Space will rebuild automatically.")
