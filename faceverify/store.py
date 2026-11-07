"""Storage of enrolled users.

Two interchangeable stores share the same SQL: MySQL for deployment and
SQLite for local development, tests and the Docker demo.

Embeddings are now stored as raw little-endian float64 bytes (128 x 8 =
1024 bytes) instead of `pickle`. Unpickling data can execute code, so a
compromised database could have attacked the server. Rows written by the
previous version are still read, with a restricted unpickler that only
allows NumPy array reconstruction, and can be converted with `migrate()`.
"""
from __future__ import annotations

import io
import pickle
import sqlite3
from contextlib import closing

import numpy as np

EMBED_DIM = 128
_DTYPE = np.dtype("<f8")

SCHEMA = """CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY {autoinc},
    name VARCHAR(255) NOT NULL,
    embedding BLOB NOT NULL
)"""


def encode_embedding(e: np.ndarray) -> bytes:
    e = np.asarray(e, dtype=_DTYPE).ravel()
    if e.shape != (EMBED_DIM,):
        raise ValueError(f"expected a {EMBED_DIM}-d embedding, got shape {e.shape}")
    return e.tobytes()


class _NumpyOnlyUnpickler(pickle.Unpickler):
    ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"),
               ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")


def decode_embedding(blob: bytes) -> np.ndarray:
    blob = bytes(blob)
    if len(blob) == EMBED_DIM * _DTYPE.itemsize and not blob.startswith(b"\x80"):
        return np.frombuffer(blob, dtype=_DTYPE).copy()
    # Legacy row written with pickle by the first version of this app.
    arr = _NumpyOnlyUnpickler(io.BytesIO(blob)).load()
    return np.asarray(arr, dtype=_DTYPE).ravel()


class SQLStore:
    """Shared logic; subclasses provide `_connect` and the parameter marker."""

    marker = "?"

    def _connect(self):
        raise NotImplementedError

    def init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.cursor().execute(SCHEMA.format(autoinc=self._autoinc))
            conn.commit()

    def add(self, name: str, embedding: np.ndarray) -> int:
        with closing(self._connect()) as conn:
            cur = conn.cursor()
            cur.execute(f"INSERT INTO users (name, embedding) VALUES ({self.marker}, {self.marker})",
                        (name, encode_embedding(embedding)))
            conn.commit()
            return int(cur.lastrowid)

    def gallery(self) -> tuple[list[int], list[str], np.ndarray]:
        with closing(self._connect()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, embedding FROM users ORDER BY id")
            rows = cur.fetchall()
        if not rows:
            return [], [], np.zeros((0, EMBED_DIM))
        return ([int(r[0]) for r in rows], [r[1] for r in rows],
                np.stack([decode_embedding(r[2]) for r in rows]))

    def get(self, user_id: int) -> tuple[str, np.ndarray] | None:
        with closing(self._connect()) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT name, embedding FROM users WHERE id = {self.marker}", (user_id,))
            row = cur.fetchone()
        return None if row is None else (row[0], decode_embedding(row[1]))

    def list_users(self) -> list[dict]:
        with closing(self._connect()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name FROM users ORDER BY id")
            return [{"id": int(r[0]), "name": r[1]} for r in cur.fetchall()]

    def delete(self, user_id: int) -> bool:
        with closing(self._connect()) as conn:
            cur = conn.cursor()
            cur.execute(f"DELETE FROM users WHERE id = {self.marker}", (user_id,))
            conn.commit()
            return cur.rowcount > 0

    def ping(self) -> bool:
        try:
            with closing(self._connect()) as conn:
                conn.cursor().execute("SELECT 1")
            return True
        except Exception:
            return False

    def migrate(self) -> int:
        """Rewrite legacy pickled embeddings in the new binary format."""
        with closing(self._connect()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, embedding FROM users")
            legacy = [(r[0], r[1]) for r in cur.fetchall() if bytes(r[1]).startswith(b"\x80")]
            for uid, blob in legacy:
                cur.execute(f"UPDATE users SET embedding = {self.marker} WHERE id = {self.marker}",
                            (encode_embedding(decode_embedding(blob)), uid))
            conn.commit()
        return len(legacy)


class SQLiteStore(SQLStore):
    marker = "?"
    _autoinc = "AUTOINCREMENT"

    def __init__(self, path: str):
        self.path = path
        self.init_schema()

    def _connect(self):
        return sqlite3.connect(self.path)


class MySQLStore(SQLStore):
    marker = "%s"
    _autoinc = "AUTO_INCREMENT"

    def __init__(self, host: str, user: str, password: str, database: str):
        self.params = dict(host=host, user=user, password=password, database=database)

    def _connect(self):
        import mysql.connector

        return mysql.connector.connect(**self.params)


def make_store(settings) -> SQLStore:
    if settings.db_backend == "sqlite":
        return SQLiteStore(settings.sqlite_path)
    if settings.db_backend == "mysql":
        return MySQLStore(settings.db_host, settings.db_user, settings.db_password, settings.db_name)
    raise ValueError(f"DB_BACKEND must be mysql or sqlite, got {settings.db_backend!r}")
