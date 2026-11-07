"""Experiment tracking: MLflow when installed, a local JSON log otherwise.

Every audit or training run records its parameters (model, threshold,
dataset), metrics (FMR, FNMR, EER per group, APCER/BPCER) and artifacts
(model card, plots), so results can be compared and reproduced later.

    pip install mlflow && mlflow ui     # browse runs at http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Run:
    def __init__(self, experiment: str, run_name: str, out_dir: str | Path = "runs"):
        self.experiment, self.run_name = experiment, run_name
        self.dir = Path(out_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{run_name}"
        self.params, self.metrics, self.artifacts = {}, {}, []
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            self._mlflow = None

    def __enter__(self) -> "Run":
        self.dir.mkdir(parents=True, exist_ok=True)
        if self._mlflow:
            self._mlflow.set_experiment(self.experiment)
            self._mlflow.start_run(run_name=self.run_name)
        return self

    def log_params(self, **params) -> None:
        self.params.update(params)
        if self._mlflow:
            self._mlflow.log_params(params)

    def log_metrics(self, **metrics) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v == v}  # drop NaN
        self.metrics.update(clean)
        if self._mlflow:
            self._mlflow.log_metrics(clean)

    def artifact_path(self, name: str) -> Path:
        p = self.dir / name
        self.artifacts.append(str(p))
        return p

    def __exit__(self, *exc) -> None:
        if self._mlflow:
            for a in self.artifacts:
                if Path(a).exists():
                    self._mlflow.log_artifact(a)
            self._mlflow.end_run()
        (self.dir / "run.json").write_text(json.dumps(
            {"experiment": self.experiment, "run": self.run_name, "params": self.params,
             "metrics": self.metrics, "artifacts": self.artifacts, "tracker": "mlflow" if self._mlflow else "json"},
            indent=2))
