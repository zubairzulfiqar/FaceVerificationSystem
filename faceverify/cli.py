"""Offline tools.

    python -m faceverify.cli audit --synthetic
    python -m faceverify.cli audit --labels data/labels.csv --threshold 0.6
    python -m faceverify.cli liveness-train data/pad_train --out liveness.joblib
    python -m faceverify.cli liveness-eval data/pad_test --model liveness.joblib
    python -m faceverify.cli migrate        # convert old pickled embeddings
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .fairness import (audit, load_labels, make_pairs, model_card, plot_distributions,
                       synthetic_embeddings)
from .tracking import Run


def _embed_labelled(labels_csv: str):
    from PIL import Image

    from .embedder import FaceRecognitionEmbedder

    emb_model = FaceRecognitionEmbedder()
    base = Path(labels_csv).parent
    emb, ids, groups, skipped = [], [], [], 0
    for row in load_labels(labels_csv):
        rgb = np.asarray(Image.open(base / row["path"]).convert("RGB"))
        faces = emb_model.detect_and_embed(rgb)
        if len(faces) != 1:
            skipped += 1
            continue
        emb.append(faces[0].embedding)
        ids.append(row["identity"])
        groups.append(row["group"])
    print(f"embedded {len(emb)} images, skipped {skipped} (no face or several faces)")
    return np.asarray(emb), np.asarray(ids), np.asarray(groups), emb_model.name


def cmd_audit(args) -> None:
    if args.synthetic:
        emb, ids, groups = synthetic_embeddings(n_ids=args.n_ids, seed=args.seed)
        model, dataset = "synthetic embeddings", "synthetic: 3 groups with different spreads (not real people)"
    else:
        if not args.labels:
            sys.exit("error: give --labels path/to/labels.csv or --synthetic")
        emb, ids, groups, model = _embed_labelled(args.labels)
        dataset = args.dataset_name or args.labels

    with Run("fairness-audit", "synthetic" if args.synthetic else Path(args.labels).stem, args.out) as run:
        scores = make_pairs(emb, ids, groups, seed=args.seed)
        res = audit(scores, args.threshold, args.target_fmr)
        run.log_params(model=model, dataset=dataset, threshold=args.threshold, target_fmr=args.target_fmr,
                       n_images=len(emb), n_identities=len(set(ids)))
        run.log_metrics(fnmr=res["overall"]["fnmr"], fmr=res["overall"]["fmr"], eer=res["overall"]["eer"],
                        threshold_at_target_fmr=res["threshold_at_target_fmr"],
                        **{f"{k}": v for k, v in res["disparity"].items()},
                        **{f"fnmr_{r['group']}": r["fnmr"] for r in res["groups"]},
                        **{f"fmr_{r['group']}": r["fmr"] for r in res["groups"]})
        card = model_card(res, model, dataset)
        run.artifact_path("MODEL_CARD.md").write_text(card)
        plot_distributions(scores, args.threshold, run.artifact_path("distances.png"))
    print(card)
    print(f"run saved to {run.dir}")


def cmd_liveness_train(args) -> None:
    from .liveness import LivenessModel, load_folder, pad_metrics

    faces, labels = load_folder(args.folder)
    with Run("liveness", "lbp-svm-train", args.runs) as run:
        model = LivenessModel().fit(faces, labels)
        model.save(args.out)
        scores = np.array([model.spoof_probability(f) for f in faces])
        m = pad_metrics(scores, np.array(labels), 0.5)
        run.log_params(features="LBP (8,1)(8,2)(16,2)", classifier="RBF SVM", n_train=len(faces))
        run.log_metrics(train_apcer=m["apcer"], train_bpcer=m["bpcer"], train_acer=m["acer"])
    print(f"saved {args.out}; training APCER {m['apcer']:.2%} BPCER {m['bpcer']:.2%} (optimistic: same data)")


def cmd_liveness_eval(args) -> None:
    from .liveness import LivenessModel, load_folder, pad_metrics

    faces, labels = load_folder(args.folder)
    model = LivenessModel.load(args.model)
    scores = np.array([model.spoof_probability(f) for f in faces])
    with Run("liveness", "lbp-svm-eval", args.runs) as run:
        m = pad_metrics(scores, np.array(labels), args.threshold)
        run.log_params(model=args.model, n_test=len(faces), threshold=args.threshold)
        run.log_metrics(apcer=m["apcer"], bpcer=m["bpcer"], acer=m["acer"])
    print(f"APCER {m['apcer']:.2%}  BPCER {m['bpcer']:.2%}  ACER {m['acer']:.2%}  (threshold {args.threshold})")


def cmd_migrate(args) -> None:
    from .config import Settings
    from .store import make_store

    n = make_store(Settings.from_env()).migrate()
    print(f"converted {n} legacy embeddings")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="faceverify")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="per-group FMR/FNMR fairness audit + model card")
    a.add_argument("--labels", help="CSV with columns path,identity,group (paths relative to the CSV)")
    a.add_argument("--dataset-name", default=None)
    a.add_argument("--synthetic", action="store_true", help="use synthetic embeddings (demo / CI)")
    a.add_argument("--n-ids", type=int, default=80, help="identities per group (synthetic)")
    a.add_argument("--threshold", type=float, default=0.6)
    a.add_argument("--target-fmr", type=float, default=1e-3)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="runs")
    a.set_defaults(func=cmd_audit)

    t = sub.add_parser("liveness-train", help="train the LBP+SVM anti-spoofing baseline")
    t.add_argument("folder", help="folder with live/ and attack/ face crops")
    t.add_argument("--out", default="liveness.joblib")
    t.add_argument("--runs", default="runs")
    t.set_defaults(func=cmd_liveness_train)

    e = sub.add_parser("liveness-eval", help="APCER / BPCER / ACER on a held-out folder")
    e.add_argument("folder")
    e.add_argument("--model", default="liveness.joblib")
    e.add_argument("--threshold", type=float, default=0.5)
    e.add_argument("--runs", default="runs")
    e.set_defaults(func=cmd_liveness_eval)

    m = sub.add_parser("migrate", help="convert legacy pickled embeddings in the database")
    m.set_defaults(func=cmd_migrate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
