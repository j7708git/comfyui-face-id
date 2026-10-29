#!/usr/bin/env python
"""Download the buffalo_l ONNX models used by ComfyUI FaceID.

Usage:
    python custom_nodes/comfyui-face-id/tools/download_models.py
    python custom_nodes/comfyui-face-id/tools/download_models.py --dir D:\\models\\buffalo_l

The archive comes from the official InsightFace release page. The models are
licensed CC BY-NC 4.0 (non-commercial), see the README.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
NEEDED = ("det_10g.onnx", "w600k_r50.onnx")


def default_target() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir))
    if os.path.isdir(os.path.join(root, "custom_nodes")):
        return os.path.join(root, "models", "insightface", "models", "buffalo_l")
    return os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l")


def _progress(block: int, size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(100, int(block * size * 100 / total))
    sys.stdout.write("\r  downloading... {:3d}%".format(done))
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download buffalo_l models for ComfyUI FaceID")
    parser.add_argument("--dir", default="", help="target folder (default: <ComfyUI>/models/insightface/models/buffalo_l)")
    parser.add_argument("--force", action="store_true", help="download again even if the files already exist")
    args = parser.parse_args()

    target = args.dir or default_target()

    if not args.force and all(os.path.isfile(os.path.join(target, name)) for name in NEEDED):
        print("Already present in: {}\nNothing to do (use --force to re-download).".format(target))
        return 0

    os.makedirs(target, exist_ok=True)
    print("Target folder: {}".format(target))
    print("Source: {}".format(URL))

    tmp_dir = tempfile.mkdtemp(prefix="faceid-")
    archive = os.path.join(tmp_dir, "buffalo_l.zip")

    try:
        urllib.request.urlretrieve(URL, archive, reporthook=_progress)
        print()

        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(tmp_dir)

        # The archive contains a top-level buffalo_l/ folder.
        source = os.path.join(tmp_dir, "buffalo_l")
        if not os.path.isdir(source):
            source = tmp_dir

        for name in os.listdir(source):
            src = os.path.join(source, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(target, name))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    missing = [name for name in NEEDED if not os.path.isfile(os.path.join(target, name))]
    if missing:
        print("Download finished but these files are missing: {}".format(", ".join(missing)))
        return 1

    print("Done. Restart ComfyUI, then look for the FaceID nodes.")
    print("Note: the InsightFace model weights are CC BY-NC 4.0 (non-commercial use).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
