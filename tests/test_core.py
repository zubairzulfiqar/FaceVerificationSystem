import pickle
import sqlite3

import numpy as np
import pytest

from faceverify.fairness import (PairScores, audit, eer, fmr, fnmr, make_pairs, model_card,
                                 synthetic_embeddings, threshold_for_fmr)
from faceverify.liveness import LivenessModel, lbp_features, pad_metrics
from faceverify.matcher import identify, verify
from faceverify.monitoring import Histogram
from faceverify.store import decode_embedding, encode_embedding


# ------------------------------------------------------------- storage ---

def test_embedding_roundtrip_and_legacy_pickle():
    e = np.random.default_rng(0).normal(size=128)
    assert np.array_equal(decode_embedding(encode_embedding(e)), e)
    assert len(encode_embedding(e)) == 1024
    assert np.allclose(decode_embedding(pickle.dumps(e)), e)      # rows from the old version


def test_malicious_pickle_is_refused():
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))
    with pytest.raises(pickle.UnpicklingError):
        decode_embedding(pickle.dumps(Evil()))


def test_wrong_dimension_rejected():
    with pytest.raises(ValueError):
        encode_embedding(np.zeros(64))


def test_migrate_converts_legacy_rows(store):
    e = np.ones(128) * 0.1
    store.add("new", e)
    with sqlite3.connect(store.path) as c:
        c.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", ("old", pickle.dumps(e)))
    assert store.migrate() == 1 and store.migrate() == 0
    _, names, gal = store.gallery()
    assert names == ["new", "old"] and np.allclose(gal[1], e)


# ------------------------------------------------------------- matching ---

def test_identify_and_verify():
    gallery = np.stack([np.zeros(128), np.ones(128)])
    probe = np.full(128, 0.01)
    d = identify(gallery, probe, 0.6)
    assert d.match and d.index == 0 and d.distance == pytest.approx(0.01 * np.sqrt(128))
    assert not identify(gallery, np.full(128, 0.5), 0.6).match
    assert identify(np.zeros((0, 128)), probe, 0.6).index is None
    assert not verify(np.ones(128), probe, 0.6).match


# ------------------------------------------------------------ fairness ---

def test_rates_on_hand_made_scores():
    s = PairScores(genuine=np.array([0.3, 0.4, 0.7]), impostor=np.array([0.5, 0.8, 0.9, 1.0]))
    assert fnmr(s, 0.6) == pytest.approx(1 / 3)
    assert fmr(s, 0.6) == pytest.approx(1 / 4)
    e, t = eer(s)
    assert 0 <= e <= 0.34 and 0.4 <= t <= 0.7


def test_threshold_for_fmr():
    imp = np.linspace(0.5, 1.5, 1000)
    t = threshold_for_fmr(imp, 0.01)
    assert (imp <= t).mean() <= 0.01


def test_pairs_stay_inside_groups():
    emb, ids, groups = synthetic_embeddings(n_ids=10, imgs_per_id=3)
    scores = make_pairs(emb, ids, groups)
    assert set(scores) == {"group_A", "group_B", "group_C"}
    assert all(s.genuine.size == 10 * 3 for s in scores.values())   # C(3,2)=3 per identity


def test_audit_detects_built_in_disparity():
    emb, ids, groups = synthetic_embeddings(n_ids=60)
    res = audit(make_pairs(emb, ids, groups), threshold=0.6)
    by = {r["group"]: r for r in res["groups"]}
    assert by["group_C"]["fnmr"] > by["group_A"]["fnmr"]          # C was generated noisier
    assert res["disparity"]["fnmr_ratio"] > 2
    card = model_card(res, "m", "d")
    assert "| group_C |" in card and "Intended use" in card


# ------------------------------------------------------------ liveness ---

def _live(rng):      # fine random skin-like texture
    return (rng.normal(150, 25, (80, 80, 3))).clip(0, 255).astype(np.uint8)


def _attack(rng):    # print: blurred, with a regular halftone dot pattern
    from scipy import ndimage
    yy, xx = np.mgrid[:80, :80]
    base = ndimage.gaussian_filter(rng.normal(150, 25, (80, 80)), 2)
    img = base + 30 * (np.sin(xx * 1.6) * np.sin(yy * 1.6) > 0)
    return np.repeat(img.clip(0, 255).astype(np.uint8)[..., None], 3, axis=2)


def test_lbp_features_shape_and_normalised():
    f = lbp_features(_live(np.random.default_rng(0)))
    assert f.shape == (10 + 10 + 18,) and np.isclose(f[:10].sum(), 1)


def test_liveness_pipeline_on_synthetic_textures(tmp_path):
    """Checks the plumbing (features -> SVM -> save/load -> metrics), not real-world accuracy."""
    rng = np.random.default_rng(0)
    faces = [_live(rng) for _ in range(20)] + [_attack(rng) for _ in range(20)]
    labels = [False] * 20 + [True] * 20
    model = LivenessModel().fit(faces, labels)
    model.save(tmp_path / "m.joblib")
    loaded = LivenessModel.load(tmp_path / "m.joblib")
    test = [_live(rng) for _ in range(10)] + [_attack(rng) for _ in range(10)]
    scores = np.array([loaded.spoof_probability(f) for f in test])
    m = pad_metrics(scores, np.array([False] * 10 + [True] * 10), 0.5)
    assert m["acer"] <= 0.1


def test_pad_metrics_definitions():
    m = pad_metrics(np.array([0.9, 0.2, 0.8, 0.1]), np.array([True, True, False, False]), 0.5)
    assert m["apcer"] == 0.5 and m["bpcer"] == 0.5 and m["acer"] == 0.5


# ----------------------------------------------------------- monitoring ---

def test_histogram_buckets_are_cumulative():
    h = Histogram("x", "help", [0.5, 1.0])
    for v in (0.2, 0.5, 0.7, 3.0):
        h.observe(v)
    text = "\n".join(h.render())
    assert 'x_bucket{le="0.5"} 2' in text and 'x_bucket{le="1.0"} 3' in text
    assert 'x_bucket{le="+Inf"} 4' in text and "x_count 4" in text
