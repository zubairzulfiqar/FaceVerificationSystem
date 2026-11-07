"""Settings, read once from environment variables (12-factor style)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    db_backend: str = "mysql"            # "mysql" or "sqlite"
    sqlite_path: str = "faces.db"
    db_host: str = "localhost"
    db_user: str = "root"
    db_password: str = ""
    db_name: str = "face_recognition_db"
    match_tolerance: float = 0.6         # max embedding distance accepted as the same person
    max_upload_mb: float = 8.0
    liveness_model: str = ""             # path to a trained anti-spoofing model; empty = disabled
    liveness_threshold: float = 0.5      # spoof probability above which a probe is rejected

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_backend=_env("DB_BACKEND", "mysql").lower(),
            sqlite_path=_env("SQLITE_PATH", "faces.db"),
            db_host=_env("DB_HOST", "localhost"),
            db_user=_env("DB_USER", "root"),
            db_password=_env("DB_PASSWORD", ""),
            db_name=_env("DB_NAME", "face_recognition_db"),
            match_tolerance=float(_env("MATCH_TOLERANCE", "0.6")),
            max_upload_mb=float(_env("MAX_UPLOAD_MB", "8")),
            liveness_model=_env("LIVENESS_MODEL", ""),
            liveness_threshold=float(_env("LIVENESS_THRESHOLD", "0.5")),
        )
