# Face Verification System

A small face enrolment and verification service. A user is enrolled from one photo, and a new photo is later checked against everyone enrolled. Built with Flask, the `face_recognition` library (dlib ResNet, 128-d embeddings) and MySQL.

![Web interface](docs/screenshot.png)

## How it works

1. **Enrol**: detect exactly one face in the uploaded image, compute its 128-d embedding, and store the name and embedding in MySQL.
2. **Verify**: detect the face in the probe image, compute its embedding, and compute the Euclidean distance to every stored embedding.
3. **Decide**: the closest identity is accepted if its distance is below `MATCH_TOLERANCE` (default 0.6). The response includes the distance, so the threshold can be studied.

Enrolment rejects images with no face or more than one face, so the gallery holds one clean template per person.

## API

| Method | Route | Input | Returns |
|---|---|---|---|
| POST | `/enroll` | form: `name`, `image` | new user id |
| POST | `/verify` | form: `image` | match, name, distance |
| GET | `/users` | | list of enrolled users (no embeddings) |
| DELETE | `/users/<id>` | | confirmation |

The web page in `home/index.html` calls these routes.

## Run it

```bash
pip install -r requirements.txt   # face_recognition needs dlib and CMake
mysql -u root < schema.sql
export DB_USER=root DB_PASSWORD=yourpassword   # optional, defaults to root with no password
python app.py
```

Open http://127.0.0.1:5000.

Settings via environment variables: `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `MATCH_TOLERANCE`, `FLASK_DEBUG=1`.

## Limitations

- Identification is a linear scan over all embeddings, fine for small galleries only.
- No liveness or presentation-attack detection, so a printed photo can pass.
- The 0.6 threshold is the library default and was not tuned on a benchmark.
- Embeddings are stored with `pickle`; a fixed binary format would be safer.
