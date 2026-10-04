#!/usr/bin/env python3
"""Validate submitted CSVs and rebuild tables without torch or third-party packages.

This summarizes existing predictions, never trains or evaluates a model.
Raw CSVs and their import provenance are read-only. Only derived tables are
replaced. No clean-to-degraded drops are computed without paired-image evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from satdomain.constants import CLASS_NAMES

MODELS = ("small_cnn", "resnet18_scratch", "resnet18_pretrained", "deit_tiny_pretrained")
SEEDS = (13, 37, 73)
METRICS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Inconsistent {context}: {actual} != {expected}")


def grouped_summary(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        if len(group) != len(SEEDS) or {int(r["seed"]) for r in group} != set(SEEDS):
            raise ValueError(f"Expected exactly seeds {SEEDS} for {key}")
        summary = dict(zip(keys, key))
        summary["runs"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values)
        output.append(summary)
    return output


def validate_clean(root: Path) -> list[dict]:
    raw = root / "raw" / "clean"
    runs = read_csv(raw / "cross_domain_runs.csv")
    classes = read_csv(raw / "per_class_runs.csv")
    expected_runs = {(model, str(seed)) for model in MODELS for seed in SEEDS}
    if len(runs) != 12 or {(r["architecture"], r["seed"]) for r in runs} != expected_runs:
        raise ValueError("Clean results must contain exactly four models and three seeds")
    expected_classes = {(m, s, c) for m, s in expected_runs for c in CLASS_NAMES}
    if len(classes) != 108 or {(r["architecture"], r["seed"], r["class_name"]) for r in classes} != expected_classes:
        raise ValueError("Clean per-class rows are missing or duplicated")
    for run in runs:
        group = [c for c in classes if (c["architecture"], c["seed"]) == (run["architecture"], run["seed"])]
        for c in group:
            if int(c["support"]) != 20:
                raise ValueError("Expected 20 clean images per class")
            p, r = float(c["precision"]), float(c["recall"])
            if not all(math.isfinite(float(c[k])) and 0 <= float(c[k]) <= 1 for k in ("precision", "recall", "f1")):
                raise ValueError("Invalid clean per-class metric")
            require_close(r * 20, round(r * 20), "clean true-positive count")
            require_close(float(c["f1"]), 2 * p * r / (p + r) if p + r else 0, "clean class F1")
        for metric in ("precision", "recall", "f1"):
            require_close(float(run[f"cross_domain_macro_{metric}"]),
                          statistics.mean(float(c[metric]) for c in group), f"clean macro {metric}")
        require_close(float(run["cross_domain_accuracy"]),
                      sum(round(float(c["recall"]) * 20) for c in group) / 180, "clean accuracy")
        require_close(float(run["cross_domain_balanced_accuracy"]),
                      float(run["cross_domain_macro_recall"]), "clean balanced accuracy")
    metrics = tuple(k for k in runs[0] if k.startswith("cross_domain_"))
    models = read_csv(raw / "cross_domain_models.csv")
    class_models = read_csv(raw / "per_class_models.csv")
    for raw_rows, generated, keys, metric_names in (
        (models, grouped_summary(runs, ("architecture",), metrics), ("architecture",), metrics),
        (class_models, grouped_summary(classes, ("architecture", "class_name"), ("precision", "recall", "f1")),
         ("architecture", "class_name"), ("precision", "recall", "f1")),
    ):
        lookup = {tuple(r[k] for k in keys): r for r in raw_rows}
        if len(raw_rows) != len(generated) or len(lookup) != len(generated):
            raise ValueError("Invalid clean summary row count")
        for row in generated:
            match = lookup.get(tuple(row[k] for k in keys))
            if match is None:
                raise ValueError("Missing clean summary group")
            for metric in metric_names:
                for suffix in ("mean", "std"):
                    key = f"{metric}_{suffix}"
                    require_close(float(match[key]), row[key], f"clean summary {key}")
    return runs


def prediction_metrics(rows: list[dict]) -> tuple[dict, list[dict], list[list[int]]]:
    if len(rows) != 180 or len({(r["File_Name"], r["True_Label"]) for r in rows}) != 180:
        raise ValueError("Expected 180 distinct filename/label pairs per seed")
    if Counter(r["True_Label"] for r in rows) != Counter({c: 20 for c in CLASS_NAMES}):
        raise ValueError("Expected all nine classes, with 20 images each")
    counts = Counter()
    for row in rows:
        if row["Predicted_Label"] not in CLASS_NAMES:
            raise ValueError(f"Unknown predicted label: {row['Predicted_Label']}")
        confidence = float(row["Confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be finite and between 0 and 1")
        flag = row["Is_Correct"].strip().lower()
        if flag not in {"true", "false"} or (flag == "true") != (row["True_Label"] == row["Predicted_Label"]):
            raise ValueError(f"Is_Correct disagrees with labels for {row['File_Name']}")
        counts[row["True_Label"], row["Predicted_Label"]] += 1
    matrix = [[counts[truth, pred] for pred in CLASS_NAMES] for truth in CLASS_NAMES]
    per_class = []
    for index, label in enumerate(CLASS_NAMES):
        tp = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / 20
        per_class.append({"class_name": label, "precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                          "support": 20})
    correct = sum(matrix[i][i] for i in range(len(CLASS_NAMES)))
    metrics = {"samples": len(rows), "correct": correct, "errors": len(rows) - correct,
               "accuracy": correct / len(rows),
               "balanced_accuracy": statistics.mean(c["recall"] for c in per_class)}
    metrics.update({f"macro_{k}": statistics.mean(c[k] for c in per_class) for k in ("precision", "recall", "f1")})
    return metrics, per_class, matrix


def summarize(root: Path) -> dict:
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    expected_paths = {f"raw/clean/{name}.csv" for name in
                      ("cross_domain_models", "cross_domain_runs", "per_class_models", "per_class_runs")}
    expected_paths.update(f"raw/{condition}/{model}.csv" for condition in ("bright", "lowres") for model in MODELS)
    if {item["path"] for item in provenance["files"]} != expected_paths or len(provenance["files"]) != 12:
        raise ValueError("Import provenance must identify exactly the 12 expected raw CSVs")
    for item in provenance["files"]:
        digest = hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Raw CSV changed since import: {item['path']}")
    clean = validate_clean(root)
    runs, class_runs, matrices = [], [], []
    identities: dict[str, set[tuple[str, str]]] = {}
    for condition in ("bright", "lowres"):
        for model in MODELS:
            rows = read_csv(root / "raw" / condition / f"{model}.csv")
            if len(rows) != 540 or {int(r["Seed"]) for r in rows} != set(SEEDS):
                raise ValueError(f"Expected 540 predictions across three seeds: {condition}/{model}")
            for seed in SEEDS:
                group = [r for r in rows if int(r["Seed"]) == seed]
                metrics, per_class, matrix = prediction_metrics(group)
                names = {(r["File_Name"], r["True_Label"]) for r in group}
                if condition in identities and names != identities[condition]:
                    raise ValueError(f"Filename/label set differs across models or seeds: {condition}/{model}/{seed}")
                identities[condition] = names
                base = {"condition": condition, "architecture": model, "seed": seed}
                runs.append({**base, **metrics})
                class_runs.extend({**base, **c} for c in per_class)
                matrices.append((condition, model, seed, matrix))
    model_summary = grouped_summary(runs, ("condition", "architecture"), METRICS)
    class_summary = grouped_summary(class_runs, ("condition", "architecture", "class_name"), ("precision", "recall", "f1"))
    verification = {
        "raw_file_hashes_match_import": True,
        "clean_summary_matches_submitted_run_and_class_tables": True,
        "clean_ece_recomputed": False,
        "degraded_prediction_rows": sum(r["samples"] for r in runs),
        "degraded_runs": len(runs), "samples_per_run": 180,
        "samples_per_class_per_run": 20,
        "filename_label_sets_match_within_each_condition": True,
        "bright_lowres_filename_label_overlap": len(identities["bright"] & identities["lowres"]),
        "clean_target_manifest_hashes_as_reported": sorted({r["target_manifest_sha256"] for r in clean}),
        "clean_checkpoint_hash_count_as_reported": len({r["checkpoint_sha256"] for r in clean}),
        "image_content_verified": False, "checkpoint_files_verified": False,
        "paired_clean_degraded_comparison_verified": False,
        "drop_metrics_computed": False,
        "uncertainty": "Sample standard deviation over three runs, not a confidence interval.",
        "note": "Rows repeat image identities over training seeds. No image independence is inferred from row count."
    }
    # All inputs are validated before replacing any derived output.
    for name, data in (("condition_runs", runs), ("condition_models", model_summary),
                       ("per_class_condition_runs", class_runs), ("per_class_condition_models", class_summary)):
        write_csv(root / f"{name}.csv", data)
    for condition, model, seed, matrix in matrices:
        write_csv(root / "confusion_matrices" / condition / model / f"seed_{seed}.csv",
                  [{"true_label": label, **dict(zip(CLASS_NAMES, row))} for label, row in zip(CLASS_NAMES, matrix)])
    (root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "cuda_aid_9class")
    args = parser.parse_args()
    print(json.dumps(summarize(args.results_root), indent=2))


if __name__ == "__main__":
    main()
