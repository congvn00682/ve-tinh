"""Evaluate a trusted checkpoint on a manifest without changing the model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .data import build_transforms, read_manifest
from .models import create_model
from .runtime import choose_device, write_json
from .train import make_loader, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--domain", default="target")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def save_predictions(path: Path, rows: list[dict], class_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "target",
                "target_label",
                "prediction",
                "prediction_label",
                "confidence",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "target_label": class_names[row["target"]],
                    "prediction_label": class_names[row["prediction"]],
                }
            )


def plot_confusion_matrix(path: Path, matrix: list[list[int]], labels: list[str]) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    row_sums = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Row-normalized confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    # Only load checkpoints produced by this project or another trusted source.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = list(checkpoint["class_names"])
    model = create_model(
        checkpoint["architecture"],
        len(class_names),
        load_pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    frame = read_manifest(args.manifest, domain=args.domain)
    _, eval_transform = build_transforms(int(checkpoint["image_size"]))
    loader = make_loader(
        frame,
        args.data_root,
        eval_transform,
        args.batch_size,
        args.workers,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    loss, metrics, rows = run_epoch(model, loader, criterion, device)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "domain": args.domain,
        "samples": len(frame),
        "loss": loss,
        "metrics": metrics,
        "class_names": class_names,
    }
    write_json(output_dir / "metrics.json", result)
    save_predictions(output_dir / "predictions.csv", rows, class_names)
    plot_confusion_matrix(
        output_dir / "confusion_matrix.png",
        metrics["confusion_matrix"],
        class_names,
    )
    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
