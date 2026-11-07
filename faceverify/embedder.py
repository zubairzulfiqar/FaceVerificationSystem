"""Face detection + embedding.

The API depends only on the small `Embedder` interface, so the heavy
`face_recognition`/dlib stack can be swapped out (another model, or a fake
one in tests) without touching the routes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class DetectedFace:
    box: tuple[int, int, int, int]   # top, right, bottom, left (face_recognition convention)
    embedding: np.ndarray            # (128,) float64


class Embedder(Protocol):
    name: str

    def detect_and_embed(self, rgb: np.ndarray) -> list[DetectedFace]: ...


class FaceRecognitionEmbedder:
    """dlib ResNet face embeddings via the face_recognition library (128-d)."""

    name = "dlib-resnet-128"

    def __init__(self, model: str = "hog", upsample: int = 1):
        import face_recognition  # imported here: heavy, and not needed in tests

        self._fr = face_recognition
        self.model = model
        self.upsample = upsample

    def detect_and_embed(self, rgb: np.ndarray) -> list[DetectedFace]:
        boxes = self._fr.face_locations(rgb, number_of_times_to_upsample=self.upsample, model=self.model)
        if not boxes:
            return []
        encodings = self._fr.face_encodings(rgb, boxes)
        return [DetectedFace(tuple(int(v) for v in b), np.asarray(e, dtype=np.float64))
                for b, e in zip(boxes, encodings)]


def crop(rgb: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.15) -> np.ndarray:
    """Face crop with a small margin, clipped to the image."""
    top, right, bottom, left = box
    h, w = bottom - top, right - left
    t = max(0, int(top - margin * h))
    b = min(rgb.shape[0], int(bottom + margin * h))
    lft = max(0, int(left - margin * w))
    r = min(rgb.shape[1], int(right + margin * w))
    return rgb[t:b, lft:r]
