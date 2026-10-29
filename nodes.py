"""ComfyUI nodes for face identity scoring."""

from __future__ import annotations

import csv
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch

from . import core

try:  # available inside ComfyUI
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - only outside ComfyUI
    folder_paths = None


CATEGORY = "FaceID"

PASS_COLOR = (96, 200, 96)
FAIL_COLOR = (72, 72, 230)
TEXT_COLOR = (255, 255, 255)


# ------------------------------------------------------------- tensor helpers

def _to_bgr(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.ascontiguousarray(array[:, :, ::-1])


def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1]).astype(np.float32) / 255.0
    return torch.from_numpy(rgb)


def _draw(
    bgr: np.ndarray,
    analysis: core.Analysis,
    score: float,
    verdict: str,
    match_index: int,
) -> np.ndarray:
    out = bgr.copy()
    height, width = out.shape[:2]
    thickness = max(2, int(round(min(height, width) / 400)))

    if not analysis.faces:
        label = "NO FACE"
        color = FAIL_COLOR
        x1 = y1 = 8
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, thickness)
        cv2.rectangle(
            out,
            (x1 - 4, y1 - 4),
            (x1 + text_size[0] + 6, y1 + text_size[1] + 10),
            color,
            -1,
        )
        cv2.putText(out, label, (x1, y1 + text_size[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, thickness)
        return out

    scale = analysis.scale
    for index, face in enumerate(analysis.faces):
        if index == match_index:
            continue
        box = face.bbox / scale
        cv2.rectangle(
            out,
            (max(0, int(box[0])), max(0, int(box[1]))),
            (min(width - 1, int(box[2])), min(height - 1, int(box[3]))),
            (150, 150, 150),
            max(1, thickness // 2),
        )

    chosen = analysis.faces[max(0, match_index)]
    box = chosen.bbox / scale
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)

    color = PASS_COLOR if verdict == "PASS" else FAIL_COLOR
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    label = "{} {:.2f}".format(verdict, score)
    font_scale = max(0.5, min(1.4, min(height, width) / 900.0))
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_w, text_h = text_size
    ty = y1 - 6 if y1 - text_h - 10 > 0 else y2 + text_h + 10
    cv2.rectangle(out, (x1, ty - text_h - 4), (x1 + text_w + 8, ty + 4), color, -1)
    cv2.putText(out, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, thickness)

    return out


# -------------------------------------------------------------------- nodes

class FaceIDReference:
    """Average the identity embedding of one or more reference photos."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Batch of reference photos of the same person"}),
                "det_size": (
                    "INT",
                    {"default": 1280, "min": 640, "max": 2048, "step": 64,
                     "tooltip": "Detector input resolution. Raise it when faces are small."},
                ),
                "face_index": (
                    "INT",
                    {"default": -1, "min": -1, "max": 16, "step": 1,
                     "tooltip": "-1 = largest face in the photo"},
                ),
                "min_face_px": (
                    "INT",
                    {"default": 80, "min": 0, "max": 1024, "step": 8,
                     "tooltip": "Faces smaller than this are re-analysed on a 2x upscale"},
                ),
            }
        }

    RETURN_TYPES = ("FACE_REF", "INT", "STRING")
    RETURN_NAMES = ("ref", "used", "info")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, image, det_size, face_index, min_face_px):
        total = int(image.shape[0])
        vectors: List[np.ndarray] = []
        # A reference photo should hold one person, so keep a single face here
        # even when the user leaves face_index at -1.
        wanted = face_index if face_index >= 0 else 0

        for index in range(total):
            analysis = core.analyze(_to_bgr(image[index]), det_size, wanted, min_face_px)
            if not analysis.embeddings:
                continue
            vectors.append(analysis.embeddings[0])

        if not vectors:
            raise ValueError(
                "face-id: no face found in the reference image(s). "
                "Try a larger det_size, a tighter crop, or a different photo."
            )

        ref = {
            "embedding": core.average_embeddings(vectors),
            "vectors": vectors,
            "count": len(vectors),
        }

        info = ["face-id: averaged {} of {} reference image(s)".format(len(vectors), total)]
        if len(vectors) > 1:
            pairs = [
                core.similarity(vectors[i], vectors[j])
                for i in range(len(vectors))
                for j in range(i + 1, len(vectors))
            ]
            low, high = min(pairs), max(pairs)
            info.append("reference self-similarity: min {:.2f} / max {:.2f}".format(low, high))
            info.append("suggested threshold: {:.2f} (0.7 x the lowest pair, lower it if good images fail)".format(
                max(0.20, low * 0.7)
            ))

        return (ref, len(vectors), " | ".join(info))


class FaceIDScore:
    """Compare one image against a FACE_REF and return a score plus a verdict."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref": ("FACE_REF",),
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Cosine similarity needed for PASS. Calibrate it with your own reference set."},
                ),
                "det_size": ("INT", {"default": 1280, "min": 640, "max": 2048, "step": 64}),
                "face_index": ("INT", {"default": -1, "min": -1, "max": 16, "step": 1}),
                "min_face_px": ("INT", {"default": 80, "min": 0, "max": 1024, "step": 8}),
                "draw": ("BOOLEAN", {"default": True, "tooltip": "Draw the box and score on the output image"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "score", "distance", "verdict", "is_same", "info")
    FUNCTION = "score"
    CATEGORY = CATEGORY

    def score(self, ref, image, threshold, det_size, face_index, min_face_px, draw):
        bgr = _to_bgr(image[0])
        analysis = core.analyze(bgr, det_size, face_index, min_face_px)
        score, match = core.best_match(ref["embedding"], analysis)

        if match < 0:
            score = 0.0
            verdict = "NO_FACE"
            info = "face-id: no face detected - check the photo or raise det_size"
        else:
            verdict = core.verdict_for(score, threshold)
            info = "face-id: score {:.3f} | matched face {}px (of {} face(s)) | analysed at {}x".format(
                score, analysis.face_px[match], analysis.count, analysis.scale
            )

        out = _draw(bgr, analysis, score, verdict, match) if draw else bgr
        is_same = verdict == "PASS"

        return (
            _to_tensor(out)[None],
            float(score),
            float(1.0 - score),
            verdict,
            is_same,
            info,
        )


class FaceIDBatchScore:
    """Score a whole batch and (optionally) write a CSV report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref": ("FACE_REF",),
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "det_size": ("INT", {"default": 1280, "min": 640, "max": 2048, "step": 64}),
                "face_index": ("INT", {"default": -1, "min": -1, "max": 16, "step": 1}),
                "min_face_px": ("INT", {"default": 80, "min": 0, "max": 1024, "step": 8}),
                "save_csv": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("images", "report", "mean_score", "min_score", "fails")
    FUNCTION = "score_batch"
    CATEGORY = CATEGORY

    def score_batch(self, ref, images, threshold, det_size, face_index, min_face_px, save_csv):
        rows: List[Tuple[int, float, int, str]] = []
        annotated = []

        for index in range(int(images.shape[0])):
            bgr = _to_bgr(images[index])
            analysis = core.analyze(bgr, det_size, face_index, min_face_px)
            score, match = core.best_match(ref["embedding"], analysis)

            if match < 0:
                score = 0.0
                verdict = "NO_FACE"
                face_px = 0
            else:
                verdict = core.verdict_for(score, threshold)
                face_px = analysis.face_px[match]

            rows.append((index, score, face_px, verdict))
            annotated.append(_to_tensor(_draw(bgr, analysis, score, verdict, match))[None])

        scored = [r[1] for r in rows if r[3] != "NO_FACE"]
        mean_score = float(np.mean(scored)) if scored else 0.0
        min_score = float(np.min(scored)) if scored else 0.0
        fails = sum(1 for r in rows if r[3] != "PASS")

        lines = ["#    score   face  verdict"]
        for index, score, face_px, verdict in rows:
            lines.append("{:>4}  {:>6.3f}  {:>3}px  {}".format(index, score, face_px, verdict))
        lines.append("mean {:.3f} | min {:.3f} | threshold {:.2f} | {} of {} not PASS".format(
            mean_score, min_score, threshold, fails, len(rows)
        ))
        report = "\n".join(lines)

        if save_csv and folder_paths is not None:
            out_dir = os.path.join(folder_paths.get_output_directory(), "faceid")
            os.makedirs(out_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(out_dir, "scores_{}.csv".format(stamp))
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "score", "face_px", "verdict", "threshold"])
                for index, score, face_px, verdict in rows:
                    writer.writerow([index, "{:.4f}".format(score), face_px, verdict, threshold])
            report += "\ncsv: " + path

        return (torch.cat(annotated, dim=0), report, mean_score, min_score, fails)


NODE_CLASS_MAPPINGS = {
    "FaceIDReference": FaceIDReference,
    "FaceIDScore": FaceIDScore,
    "FaceIDBatchScore": FaceIDBatchScore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceIDReference": "FaceID Reference",
    "FaceIDScore": "FaceID Score",
    "FaceIDBatchScore": "FaceID Batch Score",
}
