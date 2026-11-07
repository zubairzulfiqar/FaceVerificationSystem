"""Prometheus metrics without extra dependencies.

`/metrics` exposes counters and histograms in the Prometheus text format, so
a Prometheus server (or Grafana Agent) can scrape the service.

Besides request counts and latency, the service records the distribution of
best-match distances. If that distribution drifts (new camera, lighting, a
different user population), accuracy changes silently; watching it is the
simplest form of model monitoring.
"""
from __future__ import annotations

import threading
from bisect import bisect_left
from collections import defaultdict


def _labels(lbl: tuple) -> str:
    return "{" + ",".join(f'{k}="{v}"' for k, v in lbl) + "}" if lbl else ""


class Counter:
    def __init__(self, name: str, help_: str):
        self.name, self.help, self.values, self._lock = name, help_, defaultdict(float), threading.Lock()

    def inc(self, amount: float = 1.0, **labels) -> None:
        with self._lock:
            self.values[tuple(sorted(labels.items()))] += amount

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        out += [f"{self.name}{_labels(k)} {v}" for k, v in sorted(self.values.items())]
        return out


class Histogram:
    def __init__(self, name: str, help_: str, buckets: list[float]):
        self.name, self.help, self.buckets = name, help_, sorted(buckets)
        self.counts: dict = defaultdict(lambda: [0] * (len(self.buckets) + 1))
        self.sums: dict = defaultdict(float)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self.counts[key][bisect_left(self.buckets, value)] += 1
            self.sums[key] += value

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key, counts in sorted(self.counts.items()):
            cum = 0
            for b, c in zip(self.buckets + [float("inf")], counts):
                cum += c
                le = "+Inf" if b == float("inf") else repr(b)
                out.append(f"{self.name}_bucket{_labels(key + (('le', le),))} {cum}")
            out.append(f"{self.name}_sum{_labels(key)} {self.sums[key]}")
            out.append(f"{self.name}_count{_labels(key)} {cum}")
        return out


class Metrics:
    def __init__(self):
        self.requests = Counter("faceverify_requests_total", "HTTP requests by endpoint and status")
        self.latency = Histogram("faceverify_request_seconds", "Request latency in seconds",
                                 [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5])
        self.decisions = Counter("faceverify_decisions_total",
                                 "Verification outcomes: match, no_match, no_face, multiple_faces, spoof")
        self.distance = Histogram("faceverify_best_distance", "Distance to the closest enrolled face",
                                  [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])

    def render(self) -> str:
        lines = []
        for m in (self.requests, self.latency, self.decisions, self.distance):
            lines += m.render()
        return "\n".join(lines) + "\n"
