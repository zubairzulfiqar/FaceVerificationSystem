"""Flask application factory.

`create_app()` builds the app from its parts (settings, embedder, store,
optional liveness model). Production wires in the real ones; tests pass
fakes, so every route is tested without dlib or a database server.
"""
from __future__ import annotations

import io
import json
import logging
import time
import uuid

import numpy as np
from flask import Flask, g, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from . import __version__
from .config import Settings
from .embedder import crop
from .matcher import identify, verify
from .monitoring import Metrics
from .store import make_store

log = logging.getLogger("faceverify")


class BadRequest(Exception):
    def __init__(self, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.status, self.extra = status, extra


def _read_image(field: str = "image") -> np.ndarray:
    if field not in request.files:
        raise BadRequest(f"Missing {field}")
    try:
        img = Image.open(io.BytesIO(request.files[field].read()))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise BadRequest("File is not a readable image")
    return np.asarray(img.convert("RGB"))


def create_app(settings: Settings | None = None, embedder=None, store=None, liveness=None) -> Flask:
    settings = settings or Settings.from_env()
    if embedder is None:
        from .embedder import FaceRecognitionEmbedder
        embedder = FaceRecognitionEmbedder()
    store = store or make_store(settings)
    if liveness is None and settings.liveness_model:
        from .liveness import LivenessModel
        liveness = LivenessModel.load(settings.liveness_model)

    app = Flask(__name__, template_folder="../home")
    app.config["MAX_CONTENT_LENGTH"] = int(settings.max_upload_mb * 1024 * 1024)
    metrics = Metrics()
    app.extensions["faceverify"] = {"settings": settings, "store": store, "metrics": metrics}

    # ------------------------------------------------------ request logging --
    @app.before_request
    def _start():
        g.t0, g.rid = time.perf_counter(), request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])

    @app.after_request
    def _finish(resp):
        dt = time.perf_counter() - g.t0
        endpoint = request.url_rule.rule if request.url_rule else "unmatched"
        metrics.requests.inc(endpoint=endpoint, status=resp.status_code)
        metrics.latency.observe(dt, endpoint=endpoint)
        resp.headers["X-Request-ID"] = g.rid
        log.info(json.dumps({"rid": g.rid, "method": request.method, "path": endpoint,
                             "status": resp.status_code, "ms": round(dt * 1000, 1)}))
        return resp

    @app.errorhandler(BadRequest)
    def _bad(e: BadRequest):
        return jsonify({"error": str(e), **e.extra}), e.status

    @app.errorhandler(RequestEntityTooLarge)
    def _too_big(_):
        return jsonify({"error": f"Image larger than {settings.max_upload_mb:g} MB"}), 413

    @app.errorhandler(Exception)
    def _crash(e):
        if isinstance(e, HTTPException):  # normal 404 / 405 etc.: keep their status
            return jsonify({"error": e.description}), e.code
        # Log the details server-side; never send internals (SQL, paths) to the client.
        log.exception("unhandled error rid=%s", getattr(g, "rid", "?"))
        return jsonify({"error": "Internal server error", "request_id": getattr(g, "rid", None)}), 500

    # --------------------------------------------------------------- helpers --
    def single_face(rgb: np.ndarray, purpose: str):
        faces = embedder.detect_and_embed(rgb)
        if not faces:
            metrics.decisions.inc(result="no_face")
            raise BadRequest("No face detected")
        if len(faces) > 1:
            metrics.decisions.inc(result="multiple_faces")
            raise BadRequest(f"Multiple faces detected. Please upload an image with a single face for {purpose}.")
        face = faces[0]
        if liveness is not None:
            p = liveness.spoof_probability(crop(rgb, face.box))
            if p > settings.liveness_threshold:
                metrics.decisions.inc(result="spoof")
                raise BadRequest("Presentation attack suspected (photo or screen)", 403,
                                 spoof_probability=round(p, 3))
        return face

    # ---------------------------------------------------------------- routes --
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        db_ok = store.ping()
        return jsonify({"status": "ok" if db_ok else "degraded", "database": db_ok,
                        "model": embedder.name, "liveness": liveness is not None,
                        "version": __version__}), (200 if db_ok else 503)

    @app.get("/metrics")
    def prometheus():
        return metrics.render(), 200, {"Content-Type": "text/plain; version=0.0.4"}

    @app.post("/enroll")
    def enroll():
        name = (request.form.get("name") or "").strip()
        if not name:
            raise BadRequest("Missing name or image")
        if len(name) > 255:
            raise BadRequest("Name is longer than 255 characters")
        face = single_face(_read_image(), "enrolment")
        user_id = store.add(name, face.embedding)
        return jsonify({"message": f"User {name} enrolled successfully", "id": user_id}), 201

    @app.post("/verify")
    def verify_route():
        """1:N identification by default; 1:1 verification when `user_id` is given."""
        rgb = _read_image()
        claimed = request.form.get("user_id")
        face = single_face(rgb, "verification")
        thr = settings.match_tolerance

        if claimed:
            if not claimed.isdigit():
                raise BadRequest("user_id must be an integer")
            row = store.get(int(claimed))
            if row is None:
                raise BadRequest("User not found", 404)
            name, template = row
            d = verify(template, face.embedding, thr)
            mode = "1:1"
        else:
            ids, names, gallery = store.gallery()
            if not ids:
                return jsonify({"message": "No users in database", "match": False}), 404
            d = identify(gallery, face.embedding, thr)
            name, mode = names[d.index], "1:N"

        metrics.distance.observe(d.distance)
        metrics.decisions.inc(result="match" if d.match else "no_match")
        if d.match:
            return jsonify({"message": "User verified", "name": name, "match": True,
                            "distance": round(d.distance, 4), "threshold": thr, "mode": mode}), 200
        return jsonify({"message": "User not found", "match": False,
                        "closest_distance": round(d.distance, 4), "threshold": thr, "mode": mode}), 404

    @app.get("/users")
    def list_users():
        return jsonify(store.list_users()), 200

    @app.delete("/users/<int:user_id>")
    def delete_user(user_id: int):
        if not store.delete(user_id):
            raise BadRequest("User not found", 404)
        return jsonify({"message": f"User {user_id} deleted"}), 200

    return app
