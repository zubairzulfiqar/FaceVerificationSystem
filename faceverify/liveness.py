"""Presentation-attack detection (anti-spoofing) baseline.

A face matcher alone accepts a printed photo or a phone screen showing the
enrolled person. Presentation-attack detection (PAD) tries to tell a live
face from such a replay.

This module implements a classic, explainable baseline: Local Binary
Pattern (LBP) texture histograms + an SVM (Määttä, Hadid & Pietikäinen,
2011). Printed and screen-displayed faces lose fine skin texture and gain
print dots, moiré and blur, which shows up in LBP statistics. Modern
systems use CNNs trained on large spoof datasets; this baseline is the
reference point they are compared with.

Metrics follow ISO/IEC 30107-3:
* APCER - attacks wrongly accepted as live (security failure)
* BPCER - live users wrongly rejected as attacks (usability failure)
* ACER  - their mean
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.transform import resize

LBP_CONFIGS = ((8, 1), (8, 2), (16, 2))  # (points, radius): fine and coarser texture
FACE_SIZE = 64


def lbp_features(face_rgb: np.ndarray) -> np.ndarray:
    """Concatenated, normalised uniform-LBP histograms of a face crop."""
    gray = rgb2gray(face_rgb) if face_rgb.ndim == 3 else face_rgb.astype(float)
    gray = resize(gray, (FACE_SIZE, FACE_SIZE), anti_aliasing=True)
    gray = (gray * 255).astype(np.uint8)
    feats = []
    for p, r in LBP_CONFIGS:
        codes = local_binary_pattern(gray, p, r, method="uniform")  # values 0..p+1
        hist, _ = np.histogram(codes, bins=np.arange(p + 3))
        feats.append(hist / max(hist.sum(), 1))
    return np.concatenate(feats).astype(np.float32)


def pad_metrics(spoof_scores: np.ndarray, is_attack: np.ndarray, threshold: float) -> dict:
    """APCER / BPCER / ACER at a threshold on the spoof probability."""
    spoof_scores, is_attack = np.asarray(spoof_scores), np.asarray(is_attack, bool)
    pred_attack = spoof_scores > threshold
    apcer = float((~pred_attack[is_attack]).mean()) if is_attack.any() else float("nan")
    bpcer = float(pred_attack[~is_attack].mean()) if (~is_attack).any() else float("nan")
    return {"threshold": threshold, "apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2}


class LivenessModel:
    def __init__(self, pipeline=None):
        if pipeline is None:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            # The SVM gives a margin, not a probability; Platt scaling ("sigmoid")
            # fitted with cross-validation turns it into a calibrated spoof probability.
            svm = SVC(kernel="rbf", C=10, gamma="scale")
            pipeline = make_pipeline(StandardScaler(),
                                     CalibratedClassifierCV(svm, method="sigmoid", cv=3, ensemble=False))
        self.pipeline = pipeline

    def fit(self, faces: list[np.ndarray], is_attack: list[bool]) -> "LivenessModel":
        self.pipeline.fit(np.stack([lbp_features(f) for f in faces]), np.asarray(is_attack, int))
        return self

    def spoof_probability(self, face_rgb: np.ndarray) -> float:
        return float(self.pipeline.predict_proba(lbp_features(face_rgb)[None])[0, 1])

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "LivenessModel":
        import joblib

        return cls(joblib.load(path))  # joblib uses pickle: only load model files you trained yourself


def load_folder(root: str | Path) -> tuple[list[np.ndarray], list[bool]]:
    """root/live/*.jpg and root/attack/*.jpg (face crops) -> images, labels."""
    from PIL import Image

    faces, labels = [], []
    for sub, attack in (("live", False), ("attack", True)):
        for p in sorted((Path(root) / sub).glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                faces.append(np.asarray(Image.open(p).convert("RGB")))
                labels.append(attack)
    if not faces:
        raise FileNotFoundError(f"Expected images in {root}/live and {root}/attack")
    return faces, labels
