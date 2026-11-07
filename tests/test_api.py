import io

from conftest import png


def enroll(client, name, colour):
    return client.post("/enroll", data={"name": name, "image": (png(colour), "a.png")},
                       content_type="multipart/form-data")


def verify(client, colour, **extra):
    return client.post("/verify", data={"image": (png(colour), "b.png"), **extra},
                       content_type="multipart/form-data")


def test_enroll_list_verify_delete(client):
    r = enroll(client, "Alice", (200, 120, 90))
    assert r.status_code == 201 and r.json["id"] == 1
    enroll(client, "Bob", (40, 160, 220))
    assert [u["name"] for u in client.get("/users").json] == ["Alice", "Bob"]

    r = verify(client, (200, 120, 90))
    assert r.status_code == 200 and r.json["name"] == "Alice" and r.json["mode"] == "1:N"
    assert r.json["distance"] == 0.0

    r = verify(client, (10, 250, 10))
    assert r.status_code == 404 and r.json["match"] is False and r.json["closest_distance"] > 0.6

    assert client.delete("/users/1").status_code == 200
    assert client.delete("/users/1").status_code == 404
    assert [u["name"] for u in client.get("/users").json] == ["Bob"]


def test_one_to_one_verification(client):
    enroll(client, "Alice", (200, 120, 90))
    enroll(client, "Bob", (40, 160, 220))
    ok = verify(client, (40, 160, 220), user_id="2")
    assert ok.status_code == 200 and ok.json["mode"] == "1:1" and ok.json["name"] == "Bob"
    wrong_claim = verify(client, (40, 160, 220), user_id="1")   # Bob claiming to be Alice
    assert wrong_claim.status_code == 404
    assert verify(client, (1, 1, 1), user_id="99").status_code == 404
    assert verify(client, (1, 1, 1), user_id="x").status_code == 400


def test_input_validation(client):
    assert client.post("/enroll", data={"name": "A"}, content_type="multipart/form-data").status_code == 400
    assert enroll(client, "", (1, 2, 3)).status_code == 400
    r = client.post("/enroll", data={"name": "A", "image": (io.BytesIO(b"not an image"), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "not a readable image" in r.json["error"]
    assert enroll(client, "Nobody", (0, 0, 0)).json["error"] == "No face detected"
    two = client.post("/enroll", data={"name": "Two", "image": (png(size=(200, 64)), "t.png")},
                      content_type="multipart/form-data")
    assert two.status_code == 400 and "Multiple faces" in two.json["error"]
    big = client.post("/enroll", data={"name": "Big", "image": (io.BytesIO(b"0" * 2_000_000), "b.png")},
                      content_type="multipart/form-data")
    assert big.status_code == 413


def test_empty_gallery_and_unknown_routes(client):
    assert verify(client, (5, 5, 5)).status_code == 404
    assert client.get("/nope").status_code == 404           # not turned into a 500
    assert client.get("/enroll").status_code == 405


def test_health_metrics_and_request_id(client):
    assert b"Face Verification System" in client.get("/").data   # web page still served
    h = client.get("/health")
    assert h.status_code == 200 and h.json["database"] is True and h.json["model"] == "fake"
    assert len(h.headers["X-Request-ID"]) == 12
    enroll(client, "Alice", (200, 120, 90))
    verify(client, (200, 120, 90))
    verify(client, (0, 0, 0))
    text = client.get("/metrics").data.decode()
    assert 'faceverify_decisions_total{result="match"} 1.0' in text
    assert 'faceverify_decisions_total{result="no_face"} 1.0' in text
    assert 'faceverify_best_distance_bucket{le="0.2"} 1' in text
    assert 'faceverify_requests_total{endpoint="/enroll",status="201"} 1.0' in text


def test_internal_errors_do_not_leak(client, store, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("SELECT secret FROM users -- password=hunter2")
    monkeypatch.setattr(store, "list_users", boom)
    r = client.get("/users")
    assert r.status_code == 500 and "secret" not in r.get_data(as_text=True) and r.json["request_id"]


class AlwaysSpoof:
    def spoof_probability(self, face):
        return 0.97


def test_liveness_rejects_attacks(store):
    from conftest import FakeEmbedder
    from faceverify.api import create_app
    from faceverify.config import Settings

    app = create_app(Settings(db_backend="sqlite"), embedder=FakeEmbedder(), store=store, liveness=AlwaysSpoof())
    r = enroll(app.test_client(), "Mallory", (200, 120, 90))
    assert r.status_code == 403 and r.json["spoof_probability"] == 0.97
