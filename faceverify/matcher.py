"""Matching probe embeddings against enrolled ones.

Two operations that are easy to mix up:

* Verification (1:1) - "is this person who they claim to be?" Compare the
  probe with ONE enrolled template.
* Identification (1:N) - "who is this?" Compare with EVERY template and take
  the closest. With N people there are N chances for an impostor to fall
  under the threshold by luck, so the false-match risk grows with the gallery.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def distances(gallery: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Euclidean distance from the probe to every row (same as face_recognition.face_distance)."""
    if len(gallery) == 0:
        return np.zeros(0)
    return np.linalg.norm(gallery - probe[None, :], axis=1)


@dataclass
class Decision:
    match: bool
    index: int | None
    distance: float | None


def identify(gallery: np.ndarray, probe: np.ndarray, threshold: float) -> Decision:
    d = distances(gallery, probe)
    if d.size == 0:
        return Decision(False, None, None)
    i = int(np.argmin(d))
    return Decision(bool(d[i] <= threshold), i, float(d[i]))


def verify(template: np.ndarray, probe: np.ndarray, threshold: float) -> Decision:
    d = float(np.linalg.norm(template - probe))
    return Decision(d <= threshold, 0, d)
