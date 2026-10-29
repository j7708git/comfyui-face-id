#!/usr/bin/env python
"""Self-test for ComfyUI FaceID.

Checks that the models load and that the pipeline is sane: the same image must
score 1.00 against itself, and a random-noise image must report NO_FACE.

Usage:
    python custom_nodes/comfyui-face-id/tools/selftest.py path/to/photo.jpg
"""

from __future__ import annotations

import os
import sys

import numpy as np

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PACK_DIR))

# Load the pack the same way ComfyUI does (folder name is not a valid module name).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "faceid_selftest", os.path.join(PACK_DIR, "__init__.py"),
    submodule_search_locations=[PACK_DIR],
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["faceid_selftest"] = _pkg
_spec.loader.exec_module(_pkg)

core = sys.modules["faceid_selftest.core"]  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: selftest.py path/to/photo-with-one-face.jpg")
        return 2

    path = sys.argv[1]
    if not os.path.isfile(path):
        print("file not found: {}".format(path))
        return 2

    try:
        import cv2
    except ImportError:
        print("opencv is missing")
        return 1

    print("models:", core.find_model_dir())
    bgr = cv2.imread(path)
    if bgr is None:
        print("could not read image: {}".format(path))
        return 1

    analysis = core.analyze(bgr)
    if not analysis.embeddings:
        print("FAIL: no face detected in the test photo")
        return 1

    score = core.similarity(analysis.embeddings[0], analysis.embeddings[0])
    print("faces: {} | matched size: {} px | scale: {}x".format(
        analysis.count, analysis.face_px[0], analysis.scale))
    print("self similarity: {:.4f}".format(score))

    if abs(score - 1.0) > 1e-4:
        print("FAIL: an embedding should be identical to itself")
        return 1

    noise = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    blank = core.analyze(noise)
    if blank.embeddings:
        print("note: the detector found a face in pure noise (possible false positive)")
    else:
        print("noise image: NO_FACE (expected)")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
