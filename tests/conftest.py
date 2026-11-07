import io

import numpy as np
import pytest
from PIL import Image

from faceverify.api import create_app
from faceverify.config import Settings
from faceverify.embedder import DetectedFace
from faceverify.store import SQLiteStore


class FakeEmbedder:
    """Stands in for dlib. The 'identity' of an image is its colour:
    same colour -> same embedding. A black image has no face; an image more
    than twice as wide as tall contains two faces."""

    name = "fake"

    def detect_and_embed(self, rgb):
        h, w, _ = rgb.shape
        if rgb.max() == 0:
            return []
        colour = rgb.reshape(-1, 3).mean(0) / 255.0
        rng = np.random.default_rng(int(colour.sum() * 1000))
        emb = rng.normal(0, 0.09, 128)                    # different colours -> distance ~1.4
        face = DetectedFace((0, w, h, 0), emb)
        return [face, face] if w > 2 * h else [face]


def png(colour=(200, 120, 90), size=(64, 64)) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(str(tmp_path / "faces.db"))


@pytest.fixture
def client(store):
    app = create_app(Settings(db_backend="sqlite", match_tolerance=0.6, max_upload_mb=1),
                     embedder=FakeEmbedder(), store=store)
    app.testing = True
    return app.test_client()
