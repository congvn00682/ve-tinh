#!/usr/bin/env python3
"""Aggregate cross-domain AID results over random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from satdomain.constants import CLASS_NAMES
from satdomain.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows = []
    class_rows = []

    for architecture in args.models:
        for seed in args.seeds:
            prediction_path = (
                output_root
                / "final"
                / architecture
                / f"seed_{seed}"
                / "aid_evaluation"
                / "predictions.csv"
            )
            predictions = pd.read_csv(prediction_path)
            metrics = classification_metrics(
                predictions["target"],
                predictions["prediction"],
                len(CLASS_NAMES),
                confidences=predictions["confidence"],
            )
            result_path = prediction_path.parent / "metrics.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            for index, label in enumerate(CLASS_NAMES):
                class_rows.append({
                    "architecture": architecture, "seed": seed, "class_name": label,
                    "precision": metrics["per_class_precision"][index],
                    "recall": metrics["per_class_recall"][index],
                    "f1": metrics["per_class_f1"][index],
                    "support": metrics["support"][index],
                })
            rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "checkpoint_sha256": result.get("checkpoint_sha256"),
                    "target_manifest_sha256": result.get("target_manifest_sha256"),
                    "cross_domain_accuracy": metrics["accuracy"],
                    "cross_domain_balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "cross_domain_macro_precision": metrics["macro_precision"],
                    "cross_domain_macro_recall": metrics["macro_recall"],
                    "cross_domain_macro_f1": metrics["macro_f1"],
                    "cross_domain_ece": metrics[
                        "expected_calibration_error_15bin"
                    ],
                }
            )

    runs = pd.DataFrame(rows)
    runs.to_csv(output_root / "cross_domain_runs.csv", index=False)
    metric_columns = [
        column for column in runs if column.startswith("cross_domain_")
    ]
    models = runs.groupby("architecture")[metric_columns].agg(["mean", "std"])
    models.columns = ["_".join(column) for column in models.columns]
    models.to_csv(output_root / "cross_domain_models.csv")
    per_class = pd.DataFrame(class_rows)
    per_class.to_csv(output_root / "per_class_runs.csv", index=False)
    class_summary = per_class.groupby(["architecture", "class_name"])[
        ["precision", "recall", "f1"]
    ].agg(["mean", "std"])
    class_summary.columns = ["_".join(column) for column in class_summary.columns]
    class_summary.to_csv(output_root / "per_class_models.csv")
    print(models.to_string())


if __name__ == "__main__":
    main()
