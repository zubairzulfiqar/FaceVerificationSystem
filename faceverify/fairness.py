"""Demographic fairness audit of the face matcher.

Face recognition error rates are known to differ between demographic groups
(NIST FRVT Part 3, 2019). This module measures that for *this* system, with
the methodology NIST uses:

1. Embed every image of a labelled dataset (identity + group per image).
2. Within each group, form genuine pairs (two images of the same person) and
   impostor pairs (two different people from the same group: the realistic
   case, because impostors usually look broadly similar to the target).
3. At the operating threshold, per group:
   * FNMR - false non-match rate: genuine pairs rejected (users locked out)
   * FMR  - false match rate: impostor pairs accepted (security failure)
   * EER  - the rate where FMR = FNMR, a threshold-free summary
4. Disparity = worst group / best group, for FMR and FNMR. 1.0 is equal.

It also reports the threshold that gives a target FMR (e.g. 1 in 1,000)
overall, which is how a deployment threshold should be chosen, instead of a
library default.
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PairScores:
    genuine: np.ndarray    # distances
    impostor: np.ndarray


def make_pairs(emb: np.ndarray, ids: np.ndarray, groups: np.ndarray, max_impostors: int = 20000,
               seed: int = 0) -> dict[str, PairScores]:
    rng = np.random.default_rng(seed)
    out = {}
    for grp in sorted(set(groups)):
        idx = np.flatnonzero(groups == grp)
        gen, imp = [], []
        by_id: dict = {}
        for i in idx:
            by_id.setdefault(ids[i], []).append(i)
        for members in by_id.values():
            gen += [np.linalg.norm(emb[a] - emb[b]) for a, b in itertools.combinations(members, 2)]
        n_imp = min(max_impostors, len(idx) * (len(idx) - 1) // 2)
        tries = 0
        while len(imp) < n_imp and tries < n_imp * 20:
            a, b = rng.choice(idx, 2, replace=False)
            tries += 1
            if ids[a] != ids[b]:
                imp.append(np.linalg.norm(emb[a] - emb[b]))
        out[str(grp)] = PairScores(np.asarray(gen), np.asarray(imp))
    return out


def fnmr(s: PairScores, t: float) -> float:
    return float((s.genuine > t).mean()) if s.genuine.size else float("nan")


def fmr(s: PairScores, t: float) -> float:
    return float((s.impostor <= t).mean()) if s.impostor.size else float("nan")


def eer(s: PairScores) -> tuple[float, float]:
    """(EER, threshold where it occurs), scanning every observed distance."""
    ts = np.unique(np.concatenate([s.genuine, s.impostor]))
    if ts.size == 0:
        return float("nan"), float("nan")
    gen, imp = np.sort(s.genuine), np.sort(s.impostor)
    fnmr_t = 1 - np.searchsorted(gen, ts, side="right") / max(len(gen), 1)
    fmr_t = np.searchsorted(imp, ts, side="right") / max(len(imp), 1)
    i = int(np.argmin(np.abs(fnmr_t - fmr_t)))
    return float((fnmr_t[i] + fmr_t[i]) / 2), float(ts[i])


def threshold_for_fmr(impostor: np.ndarray, target: float) -> float:
    """Largest threshold whose FMR does not exceed `target`."""
    imp = np.sort(impostor)
    k = int(np.floor(target * len(imp)))      # number of impostors allowed through
    return float(imp[k] - 1e-9) if k < len(imp) else float(imp[-1])


def audit(scores: dict[str, PairScores], threshold: float, target_fmr: float = 1e-3) -> dict:
    all_imp = np.concatenate([s.impostor for s in scores.values()])
    all_gen = np.concatenate([s.genuine for s in scores.values()])
    t_target = threshold_for_fmr(all_imp, target_fmr)
    rows = []
    for grp, s in scores.items():
        e, te = eer(s)
        rows.append({"group": grp, "genuine_pairs": int(s.genuine.size), "impostor_pairs": int(s.impostor.size),
                     "fnmr": fnmr(s, threshold), "fmr": fmr(s, threshold), "eer": e, "eer_threshold": te,
                     "fnmr_at_target": fnmr(s, t_target), "fmr_at_target": fmr(s, t_target)})

    def ratio(key, count_key):
        # A group with zero observed errors gets the smallest rate it could have
        # measured (1 / number of pairs) instead of 0, so the ratio stays finite.
        vals = [max(r[key], 1.0 / max(r[count_key], 1)) for r in rows if not np.isnan(r[key])]
        return float(max(vals) / min(vals)) if vals else float("nan")

    overall = PairScores(all_gen, all_imp)
    return {
        "threshold": threshold, "target_fmr": target_fmr, "threshold_at_target_fmr": t_target,
        "overall": {"fnmr": fnmr(overall, threshold), "fmr": fmr(overall, threshold), "eer": eer(overall)[0]},
        "groups": rows,
        "disparity": {"fmr_ratio": ratio("fmr", "impostor_pairs"), "fnmr_ratio": ratio("fnmr", "genuine_pairs"),
                      "fmr_ratio_at_target": ratio("fmr_at_target", "impostor_pairs"),
                      "fnmr_ratio_at_target": ratio("fnmr_at_target", "genuine_pairs")},
    }


# ------------------------------------------------------------- reporting --

def plot_distributions(scores: dict[str, PairScores], threshold: float, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(scores), figsize=(3.4 * len(scores), 2.8), sharey=True, squeeze=False)
    bins = np.linspace(0, 1.4, 50)
    for ax, (grp, s) in zip(axes[0], scores.items()):
        ax.hist(s.genuine, bins, alpha=0.6, density=True, label="same person")
        ax.hist(s.impostor, bins, alpha=0.6, density=True, label="different people")
        ax.axvline(threshold, color="k", ls="--", lw=1)
        ax.set_title(grp, fontsize=9)
        ax.set_xlabel("embedding distance", fontsize=8)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def model_card(res: dict, model_name: str, dataset: str) -> str:
    pct = lambda x: "n/a" if x != x else f"{x:.2%}"  # noqa: E731
    lines = [
        "# Model card: face verification", "",
        f"- **Model:** {model_name}",
        f"- **Decision rule:** match if embedding distance <= {res['threshold']}",
        f"- **Evaluation data:** {dataset}", "",
        "## Error rates by group", "",
        f"At the deployed threshold {res['threshold']}:", "",
        "| Group | Genuine pairs | Impostor pairs | FNMR | FMR | EER |", "|---|---|---|---|---|---|",
    ]
    for r in res["groups"]:
        lines.append(f"| {r['group']} | {r['genuine_pairs']} | {r['impostor_pairs']} | {pct(r['fnmr'])} | "
                     f"{pct(r['fmr'])} | {pct(r['eer'])} |")
    o, d = res["overall"], res["disparity"]
    lines += [f"| **all** | | | {pct(o['fnmr'])} | {pct(o['fmr'])} | {pct(o['eer'])} |", "",
              f"Disparity (worst / best group): FMR x{d['fmr_ratio']:.1f}, FNMR x{d['fnmr_ratio']:.1f}.", "",
              f"## Threshold for FMR = {res['target_fmr']:g}", "",
              f"Threshold {res['threshold_at_target_fmr']:.3f} gives the target FMR overall. Per group at that threshold:", "",
              "| Group | FNMR | FMR |", "|---|---|---|"]
    lines += [f"| {r['group']} | {pct(r['fnmr_at_target'])} | {pct(r['fmr_at_target'])} |" for r in res["groups"]]
    lines += ["", f"Disparity at that threshold: FMR x{d['fmr_ratio_at_target']:.1f}, "
                  f"FNMR x{d['fnmr_ratio_at_target']:.1f}.", "",
              "## Intended use and limitations", "",
              "- Consent-based enrolment and verification in low-risk settings (demo, lab access).",
              "- Not for surveillance or identifying people without consent.",
              "- Under the EU AI Act, remote biometric identification is high-risk or prohibited depending on use; "
              "biometric verification of a person who claims an identity is treated differently. "
              "Check the obligations before any deployment.",
              "- Results depend on the evaluation data; groups with few pairs have wide uncertainty.",
              "- No presentation-attack detection unless a liveness model is configured.", ""]
    return "\n".join(lines)


def load_labels(csv_path: str | Path) -> list[dict]:
    """CSV with columns: path, identity, group."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = {"path", "identity", "group"} - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"labels CSV is missing columns: {sorted(missing)}")
    return rows


def synthetic_embeddings(n_ids: int = 60, imgs_per_id: int = 4, seed: int = 0,
                         spreads: dict[str, float] | None = None):
    """Fake 128-d embeddings for three groups with different intra-person
    spread and different between-person separation, to exercise the audit.
    Not a statement about any real population."""
    # Scales chosen so distances look like dlib's: same person ~0.35-0.55,
    # different people ~0.8-1.0, with a tail of poor-quality images.
    rng = np.random.default_rng(seed)
    spreads = spreads or {"group_A": 0.024, "group_B": 0.027, "group_C": 0.031}
    centre_scale = {"group_A": 0.055, "group_B": 0.050, "group_C": 0.044}
    emb, ids, groups = [], [], []
    for grp, sp in spreads.items():
        base = rng.normal(0, 0.02, 128)
        for i in range(n_ids):
            centre = base + rng.normal(0, centre_scale.get(grp, 0.05) * rng.lognormal(0, 0.15), 128)
            for _ in range(imgs_per_id):
                emb.append(centre + rng.normal(0, sp * rng.lognormal(0, 0.25), 128))
                ids.append(f"{grp}_{i}")
                groups.append(grp)
    return np.asarray(emb), np.asarray(ids), np.asarray(groups)
