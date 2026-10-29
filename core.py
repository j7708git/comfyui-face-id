"""Face identity scoring core.

Pipeline
--------
1. SCRFD (``det_10g.onnx``) detects faces plus 5 landmarks.
2. The 5 landmarks are aligned to the standard ArcFace 112x112 template.
3. ArcFace (``w600k_r50.onnx``) turns the aligned crop into a 512-d embedding.
4. Two embeddings are compared with cosine similarity.

Only ``onnxruntime`` is used.  The ``insightface`` Python package is *not*
imported, so this works even when that package is broken (for example after a
NumPy ABI change).

Everything in this module works on plain NumPy ``uint8`` BGR images, which
keeps it testable outside of ComfyUI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # opencv is shipped with ComfyUI
    import cv2
except ImportError as exc:  # pragma: no cover - depends on the host install
    raise ImportError(
        "face-id: OpenCV is required. ComfyUI normally ships it; otherwise run "
        "`pip install opencv-python-headless` in the ComfyUI Python environment."
    ) from exc


# Standard ArcFace 5-point template (left eye, right eye, nose, mouth corners).
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

DET_FILE = "det_10g.onnx"
REC_FILE = "w600k_r50.onnx"
STRIDES = (8, 16, 32)
ANCHORS_PER_CELL = 2
DET_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


class ModelNotFound(RuntimeError):
    """Raised when the buffalo_l ONNX files are missing."""


@dataclass
class Face:
    bbox: np.ndarray  # x1, y1, x2, y2 in *working image* pixels
    score: float
    kps: np.ndarray  # 5 x 2 landmarks in *working image* pixels


@dataclass
class Analysis:
    faces: List[Face]
    embeddings: List[np.ndarray]
    face_px: List[int]  # per face, width in *original image* pixels
    scale: float  # 1.0 = original image, 2.0 = analysed on a 2x upscale

    @property
    def count(self) -> int:
        return len(self.faces)

    @property
    def largest_px(self) -> int:
        return max(self.face_px) if self.face_px else 0


# ---------------------------------------------------------------- model paths

def _comfy_root() -> Optional[str]:
    """Guess the ComfyUI root from this file (``<root>/custom_nodes/<pack>``)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    if os.path.isdir(os.path.join(root, "custom_nodes")):
        return root
    return None


def candidate_model_dirs() -> List[str]:
    """Places where a buffalo_l folder is usually found, most specific first."""
    dirs: List[str] = []

    try:  # ComfyUI's own model registry
        import folder_paths  # type: ignore

        base = folder_paths.models_dir
        dirs += [
            os.path.join(base, "insightface", "models", "buffalo_l"),
            os.path.join(base, "insightface", "buffalo_l"),
            os.path.join(base, "insightface", "models"),
        ]
    except Exception:
        pass

    root = _comfy_root()
    if root:
        base = os.path.join(root, "models")
        dirs += [
            os.path.join(base, "insightface", "models", "buffalo_l"),
            os.path.join(base, "insightface", "buffalo_l"),
            os.path.join(base, "insightface", "models"),
        ]

    dirs.append(os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l"))
    dirs.append(os.path.join(os.path.expanduser("~"), ".insightface", "models"))
    return dirs


def find_model_dir() -> str:
    """Return the folder containing both ONNX files, or raise ModelNotFound."""
    for d in candidate_model_dirs():
        if os.path.isfile(os.path.join(d, DET_FILE)) and os.path.isfile(os.path.join(d, REC_FILE)):
            return d
    raise ModelNotFound(
        "face-id: could not find {} and {}.\n"
        "Put the buffalo_l ONNX files in one of these folders:\n  {}\n"
        "Quickest fix: run `python custom_nodes/comfyui-face-id/tools/download_models.py`."
        .format(DET_FILE, REC_FILE, "\n  ".join(candidate_model_dirs()))
    )


# ------------------------------------------------------------------ sessions

_SESSIONS: Dict[str, "object"] = {}


def _providers() -> List[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "face-id: onnxruntime is required. Run "
            "`pip install onnxruntime` (or onnxruntime-gpu) in the ComfyUI "
            "Python environment."
        ) from exc

    available = ort.get_available_providers()
    for pref in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if pref in available:
            yield pref  # type: ignore[misc]


def _session(filename: str):
    cached = _SESSIONS.get(filename)
    if cached is not None:
        return cached

    import onnxruntime as ort

    # The SCRFD graph has data-dependent output shapes; ORT logs a warning on
    # every resolution change. Keep the console readable for ComfyUI users.
    ort.set_default_logger_severity(3)

    path = os.path.join(find_model_dir(), filename)
    sess = ort.InferenceSession(path, providers=list(_providers()))
    _SESSIONS[filename] = sess
    return sess


# ----------------------------------------------------------------- detection

def _anchor_centers(grid_w: int, grid_h: int, stride: int) -> np.ndarray:
    xs = np.arange(grid_w) * stride
    ys = np.arange(grid_h) * stride
    cx, cy = np.meshgrid(xs, ys)
    centers = np.stack([cx, cy], axis=-1).reshape(-1, 2).astype(np.float32)
    return np.repeat(centers, ANCHORS_PER_CELL, axis=0)


def detect(bgr: np.ndarray, det_size: int = 1280, threshold: float = DET_THRESHOLD) -> List[Face]:
    """Run SCRFD on a BGR uint8 image and return faces, largest first."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("face-id: expected a BGR image with 3 channels")

    size = int(np.ceil(det_size / 32.0) * 32)
    h, w = bgr.shape[:2]
    scale = size / float(max(h, w))
    resized = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized

    blob = canvas.astype(np.float32).transpose(2, 0, 1)[None]
    blob = (blob - 127.5) / 128.0

    sess = _session(DET_FILE)
    outputs = sess.run(None, {sess.get_inputs()[0].name: blob})
    # Output order is scores(stride 8,16,32), boxes, keypoints.
    scores_all, boxes_all, kps_all = outputs[0:3], outputs[3:6], outputs[6:9]

    boxes: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    kpss: List[np.ndarray] = []

    for index, stride in enumerate(STRIDES):
        cell = int(np.ceil(size / stride))
        score = scores_all[index].reshape(-1)
        delta = boxes_all[index] * stride
        kps_delta = kps_all[index] * stride
        centers = _anchor_centers(cell, cell, stride)

        count = min(len(score), len(centers))
        score, delta, kps_delta, centers = (
            score[:count],
            delta[:count],
            kps_delta[:count],
            centers[:count],
        )
        keep = score >= threshold
        if not keep.any():
            continue

        centers, delta, kps_delta, score = centers[keep], delta[keep], kps_delta[keep], score[keep]
        boxes.append(
            np.stack(
                [
                    centers[:, 0] - delta[:, 0],
                    centers[:, 1] - delta[:, 1],
                    centers[:, 0] + delta[:, 2],
                    centers[:, 1] + delta[:, 3],
                ],
                axis=1,
            )
        )
        scores.append(score)
        kpss.append(kps_delta.reshape(-1, 5, 2) + centers[:, None, :])

    if not boxes:
        return []

    all_boxes = np.concatenate(boxes)
    all_scores = np.concatenate(scores)
    all_kps = np.concatenate(kpss)

    wh = np.stack(
        [all_boxes[:, 0], all_boxes[:, 1], all_boxes[:, 2] - all_boxes[:, 0], all_boxes[:, 3] - all_boxes[:, 1]],
        axis=1,
    )
    keep_idx = cv2.dnn.NMSBoxes(wh.tolist(), all_scores.tolist(), threshold, NMS_THRESHOLD)
    if len(keep_idx) == 0:
        return []

    faces: List[Face] = []
    for idx in np.array(keep_idx).reshape(-1):
        faces.append(
            Face(
                bbox=all_boxes[idx] / scale,
                score=float(all_scores[idx]),
                kps=all_kps[idx] / scale,
            )
        )

    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces


# ---------------------------------------------------------------- embeddings

def embed(bgr: np.ndarray, face: Face) -> np.ndarray:
    """Return an L2-normalised 512-d ArcFace embedding for one face."""
    matrix, _ = cv2.estimateAffinePartial2D(
        face.kps.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS
    )
    if matrix is None:
        matrix, _ = cv2.estimateAffinePartial2D(
            face.kps.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LINEAR
        )
    if matrix is None:
        raise RuntimeError("face-id: could not align the detected face")

    aligned = cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0)
    blob = aligned[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None]
    blob = (blob - 127.5) / 127.5

    sess = _session(REC_FILE)
    vector = sess.run(None, {sess.get_inputs()[0].name: blob})[0].reshape(-1)

    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def analyze(
    bgr: np.ndarray,
    det_size: int = 1280,
    face_index: int = -1,
    min_face_px: int = 80,
) -> Analysis:
    """Detect and embed every face in an image.

    ``face_index`` = -1 keeps all faces (so callers can take the best match,
    which is what you want for group shots); ``>= 0`` keeps a single face from
    the detector output, largest first.

    Small faces (typical for full-body shots) are re-analysed on a 2x upscale so
    the 112x112 crop is built from more real pixels instead of being upscaled
    after the fact.
    """
    best: Optional[Analysis] = None

    for scale in (1.0, 2.0):
        if scale == 1.0:
            work = bgr
        else:
            work = cv2.resize(bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        faces = detect(work, det_size)
        if not faces:
            continue

        if face_index >= 0:
            faces = [faces[min(face_index, len(faces) - 1)]]

        face_px = [
            int(min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) / scale)
            for face in faces
        ]
        best = Analysis(
            faces=faces,
            embeddings=[embed(work, face) for face in faces],
            face_px=face_px,
            scale=scale,
        )
        if best.largest_px >= min_face_px:
            return best
        # Too small to trust: keep it as a fallback and try the 2x pass.

    return best if best is not None else Analysis([], [], [], 1.0)


# ---------------------------------------------------------------- comparison

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised embeddings (1.0 = identical)."""
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def average_embeddings(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of several normalised embeddings, re-normalised to unit length."""
    if not vectors:
        raise ValueError("face-id: no embeddings to average")
    mean = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else mean


def verdict_for(score: float, threshold: float) -> str:
    return "PASS" if score >= threshold else "DRIFT"


def best_match(ref: np.ndarray, analysis: Analysis) -> Tuple[float, int]:
    """Highest similarity between the reference and any face in the image.

    Taking the best face instead of only the largest one is what makes group
    shots and small-in-frame characters work: the question is "is this person
    somewhere in this image", not "is the biggest head the right one".
    """
    if not analysis.embeddings:
        return 0.0, -1
    scores = [similarity(ref, embedding) for embedding in analysis.embeddings]
    index = int(np.argmax(scores))
    return float(scores[index]), index
