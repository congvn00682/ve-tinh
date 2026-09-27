"""Export readable research tables/figures from saved results; no torch required."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_table(path: Path, columns: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def export_history(directory: Path, history: list[dict]) -> None:
    if not history:
        raise ValueError("Cannot plot an empty training history")
    columns = list(dict.fromkeys(key for row in history for key in row))
    write_table(directory / "history.csv", columns, ([row.get(k, "") for k in columns] for row in history))
    plt = plotting()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = [r["epoch"] for r in history]
    for ax, metric, title in zip(axes, ("loss", "accuracy", "macro_f1"), ("Loss", "Accuracy", "Macro F1")):
        for phase in ("train", "validation"):
            key = f"{phase}_{metric}"
            if all(key in r for r in history):
                ax.plot(epochs, [r[key] for r in history], label=phase)
        if metric == "macro_f1":
            for key, label in (("validation_robust_macro_f1", "validation degraded"),
                               ("validation_selection_score", "selection score")):
                if all(key in r for r in history) and "validation_robust_macro_f1" in history[0]:
                    ax.plot(epochs, [r[key] for r in history], label=label, linestyle="--")
        ax.set(title=title, xlabel="Epoch")
        if metric != "loss":
            ax.set_ylim(0, 1)
        ax.legend()
    fig.suptitle("Source development" if "validation_loss" in history[0] else "Full-source final fit (no validation)")
    fig.tight_layout()
    fig.savefig(directory / "training_curves.png", dpi=180)
    plt.close(fig)


def export_metrics(directory: Path, result: dict) -> None:
    labels = result["class_names"]
    metrics = result["metrics"]
    degradation = result.get("degradation", {})
    condition_label = (
        f" — {degradation['condition']} (synthetic level {degradation['severity']})"
        if degradation.get("condition", "clean") != "clean" else ""
    )
    matrix = metrics["confusion_matrix"]
    count = len(labels)
    if len(matrix) != count or any(len(row) != count for row in matrix):
        raise ValueError("Confusion matrix and class names do not match")
    fields = ("per_class_precision", "per_class_recall", "per_class_f1", "support")
    if any(len(metrics[key]) != count for key in fields):
        raise ValueError("Per-class arrays and class names do not match")
    write_table(directory / "per_class_metrics.csv", ["class_index", "class_name", "precision", "recall", "f1", "support"],
                ([i, label, *(metrics[key][i] for key in fields)] for i, label in enumerate(labels)))
    normalized = [[value / sum(row) if sum(row) else 0 for value in row] for row in matrix]
    plt = plotting()
    for name, values, fractional in (
        ("confusion_matrix_counts", matrix, False),
        ("confusion_matrix_normalized", normalized, True),
    ):
        write_table(directory / f"{name}.csv", ["true_label", *labels], ([label, *row] for label, row in zip(labels, values)))
        fig, ax = plt.subplots(figsize=(11, 9))
        maximum = 1 if fractional else max(1, max(max(row) for row in values))
        img = ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.1%}" if fractional else str(value), ha="center", va="center",
                        color="white" if value > maximum / 2 else "black", fontsize=8)
        ax.set_xticks(range(count), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(count), labels=labels)
        ax.set(xlabel="Predicted class", ylabel="True class",
               title=("AID confusion matrix — row-normalized" if fractional else "AID confusion matrix — counts") + condition_label)
        fig.colorbar(img, ax=ax)
        fig.tight_layout()
        fig.savefig(directory / f"{name}.png", dpi=180)
        if fractional:
            fig.savefig(directory / "confusion_matrix.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, metrics["per_class_f1"])
    for i, (f1, support) in enumerate(zip(metrics["per_class_f1"], metrics["support"])):
        ax.text(min(f1 + .01, 1.01), i, f"{f1:.1%} (n={support})", va="center", fontsize=9)
    ax.set(xlim=(0, 1.3), xlabel="F1 (0–1)", title="AID per-class F1" + condition_label)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(directory / "per_class_f1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    if not root.is_dir():
        parser.error(f"Directory not found: {root}")
    histories = sorted(root.rglob("history.json"))
    evaluations = sorted(root.rglob("metrics.json"))
    if not histories and not evaluations:
        parser.error("No history.json or metrics.json found. Aggregate CSVs cannot reconstruct these reports.")
    for path in histories:
        export_history(path.parent, json.loads(path.read_text(encoding="utf-8")))
        print(f"Training curves: {path.parent}")
    for path in evaluations:
        export_metrics(path.parent, json.loads(path.read_text(encoding="utf-8")))
        print(f"Evaluation reports: {path.parent}")
    print("Reports exported. Historical manifests, seeds or checkpoints were not reconstructed.")


if __name__ == "__main__":
    main()
