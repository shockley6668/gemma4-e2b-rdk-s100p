#!/usr/bin/env python3
"""Upload Gemma4 E2B deploy package to OSS (resumable multipart).

Usage:
    OSS_AK=xxx OSS_SK=xxx python upload_oss_deploy.py [--gemma-root ../../]

Requires: pip install oss2
"""
import argparse
import os
import sys
from pathlib import Path

import oss2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENDPOINT = "https://oss-cn-beijing.aliyuncs.com"
BUCKET_NAME = "gemma-model"


def progress(consumed, total):
    if total and (consumed == total or int(consumed * 100 / total) % 5 == 0):
        print(f"\r  {consumed * 100 / total:.1f}% ({consumed / 1e9:.2f}/{total / 1e9:.2f} GB)", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Upload deploy package to OSS")
    parser.add_argument("--gemma-root", default=str(REPO_ROOT), help="Project root with gemma4_e2b_deploy/")
    args = parser.parse_args()

    root = Path(args.gemma_root)
    deploy_dir = root / "gemma4_e2b_deploy"

    ak = os.environ.get("OSS_AK")
    sk = os.environ.get("OSS_SK")
    if not ak or not sk:
        print("Set OSS_AK and OSS_SK environment variables", file=sys.stderr)
        sys.exit(1)

    auth = oss2.Auth(ak, sk)
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

    uploads = [
        (str(root / "gemma4_e2b_deploy.tar.gz"), "gemma4_e2b/gemma4_e2b_deploy.tar.gz"),
        (str(root / "gemma4_e2b_deploy.sha256"), "gemma4_e2b/gemma4_e2b_deploy.sha256"),
        (str(deploy_dir / "README_OSS.txt"), "gemma4_e2b/README_OSS.txt"),
        (str(deploy_dir / "model" / "gemma4-e2b_vit_ptq.hbm"), "gemma4_e2b/gemma4-e2b_vit_ptq.hbm"),
    ]

    for local, key in uploads:
        if not os.path.isfile(local):
            print(f"SKIP missing: {local}")
            continue
        size = os.path.getsize(local)
        print(f"upload {local} ({size / 1e9:.2f} GB) -> oss://{BUCKET_NAME}/{key}")
        oss2.resumable_upload(bucket, key, local, multipart_threshold=100 * 1024 * 1024, progress_callback=progress)
        print()
        print(f"  OK oss://{BUCKET_NAME}/{key}")

    print("\nObjects under gemma4_e2b/:")
    for obj in oss2.ObjectIterator(bucket, prefix="gemma4_e2b/"):
        print(f"  {obj.key}  ({obj.size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
