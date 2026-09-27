"""Evaluate a trusted checkpoint on a manifest without changing the model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from .data import build_transforms, read_manifest
from .constants import CLASS_NAMES
from .artifacts import complete_run, file_hash, snapshot_file, start_run, verify_manifest
from .reports import export_metrics
from .models import create_model
from .runtime import choose_device, write_json
from .train import make_loader, run_epoch
from .robustness import CONDITIONS, VERSION, EvaluationDegradation


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
    parser.add_argument("--condition", choices=("clean", *CONDITIONS), default="clean")
    parser.add_argument("--severity", type=int, default=0)
    parser.add_argument("--corruption-seed", type=int, default=2026)
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = choose_device(args.device)
    degradation = EvaluationDegradation(args.condition, args.severity, args.corruption_seed)

    # Only load checkpoints produced by this project or another trusted source.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = list(checkpoint["class_names"])
    if class_names != CLASS_NAMES:
        raise ValueError("Checkpoint class order differs from this project")
    verify_manifest(args.manifest, args.data_root, class_names, args.domain)
    metadata = start_run(output_dir, vars(args), str(device), "evaluation")
    metadata["checkpoint_sha256"] = file_hash(args.checkpoint)
    metadata["checkpoint"] = str(Path(args.checkpoint).resolve())
    metadata["training_seed"] = checkpoint.get("seed")
    metadata["training_run_id"] = checkpoint.get("provenance", {}).get("run_id")
    metadata["degradation"] = {
        "version": VERSION, "condition": args.condition, "severity": args.severity,
        "seed": args.corruption_seed, "identity": "image_sha256",
        "stage": "before_resize_and_center_crop", "synthetic_only": True,
    }
    metadata["target_manifest"] = snapshot_file(args.manifest, output_dir / "target_manifest.csv")
    metadata["cuda_version"] = torch.version.cuda
    metadata["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    write_json(output_dir / "run.json", metadata)
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
        degradation=degradation,
    )
    criterion = nn.CrossEntropyLoss()
    loss, metrics, rows = run_epoch(model, loader, criterion, device)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "run_id": metadata["run_id"],
        "training_run_id": metadata["training_run_id"],
        "architecture": checkpoint["architecture"],
        "seed": checkpoint.get("seed"),
        "epoch": checkpoint.get("epoch"),
        "target_manifest_sha256": metadata["target_manifest"]["sha256"],
        "device": str(device),
        "domain": args.domain,
        "samples": len(frame),
        "loss": loss,
        "metrics": metrics,
        "class_names": class_names,
        "degradation": metadata["degradation"],
        "augmentation": checkpoint.get("training_args", {}).get("augmentation", "baseline"),
    }
    write_json(output_dir / "metrics.json", result)
    save_predictions(output_dir / "predictions.csv", rows, class_names)
    export_metrics(output_dir, result)
    complete_run(output_dir, ["run.json", "metrics.json", "predictions.csv", "target_manifest.csv"])
    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
