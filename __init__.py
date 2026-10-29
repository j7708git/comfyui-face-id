"""ComfyUI FaceID - offline face identity scoring.

SCRFD face detection + ArcFace embeddings through onnxruntime.
No cloud calls, no ``insightface`` package required.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "0.1.0"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
