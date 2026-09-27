"""Train source development/final models and preserve their research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .constants import CLASS_NAMES, CLASS_TO_INDEX, IMAGENET_MEAN, IMAGENET_STD
from .artifacts import complete_run, file_hash, snapshot_file, start_run, verify_manifest
from .reports import export_history
from .data import (
    ManifestDataset,
    build_transforms,
    read_manifest,
    source_train_validation_split,
)
from .metrics import classification_metrics
from .models import create_model
from .runtime import choose_device, seed_everything, write_json
from .robustness import CONDITIONS, PROFILES, EvaluationDegradation, profile_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--arch",
        required=True,
        choices=[
            "small_cnn",
            "resnet18_scratch",
            "resnet18_pretrained",
            "deit_tiny_pretrained",
        ],
    )
    parser.add_argument(
        "--mode",
        choices=["development", "final"],
        default="development",
    )
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--augmentation", choices=PROFILES, default="baseline")
    parser.add_argument("--development-summary", help="Source-only summary that selected final epochs")
    return parser.parse_args()


def make_loader(
    frame,
    root: str,
    transform,
    batch_size: int,
    workers: int,
    shuffle: bool,
    degradation: EvaluationDegradation | None = None,
) -> DataLoader:
    dataset = ManifestDataset(frame, root=root, transform=transform, degradation=degradation)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler=None,
    use_amp: bool = False,
) -> tuple[float, dict, list[dict]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    rows: list[dict] = []

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            probabilities = torch.softmax(logits.detach(), dim=1)
            confidence, predicted = probabilities.max(dim=1)
            total_loss += float(loss.item()) * labels.size(0)
            batch_targets = labels.detach().cpu().tolist()
            batch_predictions = predicted.cpu().tolist()
            batch_confidence = confidence.cpu().tolist()
            targets.extend(batch_targets)
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidence)
            rows.extend(
                {
                    "path": path,
                    "target": target,
                    "prediction": prediction,
                    "confidence": probability,
                }
                for path, target, prediction, probability in zip(
                    paths,
                    batch_targets,
                    batch_predictions,
                    batch_confidence,
                )
            )

    metrics = classification_metrics(
        targets,
        predictions,
        len(CLASS_NAMES),
        confidences=confidences,
    )
    return total_loss / len(loader.dataset), metrics, rows


def checkpoint_payload(model: nn.Module, args: argparse.Namespace, epoch: int, provenance: dict) -> dict:
    return {
        "state_dict": model.state_dict(),
        "architecture": args.arch,
        "class_names": CLASS_NAMES,
        "class_to_index": CLASS_TO_INDEX,
        "image_size": args.image_size,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "seed": args.seed,
        "epoch": epoch,
        "temperature": 1.0,
        "training_args": vars(args),
        "provenance": provenance,
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    seed_everything(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    verify_manifest(args.manifest, args.data_root, CLASS_NAMES, "source")

    frame = read_manifest(args.manifest, domain="source")
    train_transform, eval_transform = build_transforms(args.image_size, args.augmentation)

    if args.mode == "development":
        train_frame, validation_frame = source_train_validation_split(
            frame,
            validation_fold=args.validation_fold,
        )
    else:
        train_frame = frame
        validation_frame = None

    metadata = start_run(output_dir, vars(args), str(device), "training")
    metadata["amp_enabled"] = args.amp and device.type == "cuda"
    metadata["cuda_version"] = torch.version.cuda
    metadata["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    metadata["source_manifest"] = snapshot_file(args.manifest, output_dir / "source_manifest.csv")
    train_frame.to_csv(output_dir / "train_manifest.csv", index=False)
    splits = {"train_manifest.csv": file_hash(output_dir / "train_manifest.csv")}
    if validation_frame is not None:
        validation_frame.to_csv(output_dir / "validation_manifest.csv", index=False)
        splits["validation_manifest.csv"] = file_hash(output_dir / "validation_manifest.csv")
    metadata["split_sha256"] = splits
    if args.development_summary:
        selected = json.loads(Path(args.development_summary).read_text(encoding="utf-8"))
        if (args.mode != "final" or selected["architecture"] != args.arch
                or selected["seed"] != args.seed or selected["best_epoch"] != args.epochs
                or selected.get("augmentation", "baseline") != args.augmentation
                or selected["source_manifest_sha256"] != metadata["source_manifest"]["sha256"]):
            raise ValueError("Development summary does not match this final run")
        metadata["development_summary"] = snapshot_file(
            args.development_summary, output_dir / "development_summary.json")
    metadata["preprocessing"] = {
        "resize_short_side": 256, "image_size": args.image_size,
        "evaluation_crop": "center", "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
        "train_crop_scale": [0.8, 1.0], "horizontal_vertical_flip": True,
        "rotation_degrees": [0, 90, 180, 270],
        "augmentation": profile_metadata(args.augmentation),
    }
    metadata["selection_metric"] = (
        "mean_clean_and_four_degraded_source_macro_f1"
        if args.augmentation == "weather_robust" else "clean_source_macro_f1")
    if args.augmentation == "weather_robust":
        metadata["robust_validation"] = {"severity": 2, "seed": 7919, "conditions": list(CONDITIONS)}
    write_json(output_dir / "run.json", metadata)

    train_loader = make_loader(
        train_frame,
        args.data_root,
        train_transform,
        args.batch_size,
        args.workers,
        shuffle=True,
    )
    validation_loader = (
        make_loader(
            validation_frame,
            args.data_root,
            eval_transform,
            args.batch_size,
            args.workers,
            shuffle=False,
        )
        if validation_frame is not None
        else None
    )
    model = create_model(args.arch, len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = float("-inf")
    best_epoch = 0
    best_clean_validation_score = None
    epochs_without_improvement = 0
    history: list[dict] = []
    checkpoint_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        if validation_loader is not None:
            validation_loss, validation_metrics, _ = run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                use_amp=use_amp,
            )
            score = validation_metrics["macro_f1"]
            row.update(
                {
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_macro_f1": score,
                }
            )
            if args.augmentation == "weather_robust":
                condition_scores = []
                # Sequential loaders avoid multiplying persistent worker pools.
                # The same held-out source images and corruptions are used each epoch.
                for condition in CONDITIONS:
                    robust_loader = make_loader(
                        validation_frame, args.data_root, eval_transform,
                        args.batch_size, 0, shuffle=False,
                        degradation=EvaluationDegradation(condition, 2, seed=7919))
                    _, robust_metrics, _ = run_epoch(
                        model, robust_loader, criterion, device, use_amp=use_amp)
                    condition_scores.append(robust_metrics["macro_f1"])
                    row[f"validation_{condition}_macro_f1"] = robust_metrics["macro_f1"]
                row["validation_robust_macro_f1"] = sum(condition_scores) / len(condition_scores)
                score = (score + sum(condition_scores)) / (1 + len(condition_scores))
            row["validation_selection_score"] = score
        else:
            score = train_metrics["macro_f1"]

        history.append(row)
        # Preserve curves even if a later epoch fails or the process is stopped.
        write_json(output_dir / "history.json", history)
        if args.mode == "final":
            best_score = score
            best_epoch = epoch
            torch.save(checkpoint_payload(model, args, epoch, metadata), checkpoint_path)
        elif score > best_score:
            best_score = score
            best_clean_validation_score = row["validation_macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint_payload(model, args, epoch, metadata), checkpoint_path)
        elif args.mode == "development":
            epochs_without_improvement += 1

        scheduler.step()
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} score={score:.4f}",
            flush=True,
        )
        if validation_loader is not None and epochs_without_improvement >= args.patience:
            break

    write_json(output_dir / "history.json", history)
    export_history(output_dir, history)
    summary = {
        "architecture": args.arch,
        "run_id": metadata["run_id"],
        "checkpoint": "best.pt",
        "checkpoint_sha256": file_hash(checkpoint_path),
        "source_manifest_sha256": metadata["source_manifest"]["sha256"],
        "mode": args.mode,
        "seed": args.seed,
        "validation_fold": (
            args.validation_fold if args.mode == "development" else None
        ),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": (
            best_clean_validation_score if args.mode == "development" else None
        ),
        "augmentation": args.augmentation,
        "selection_metric": metadata["selection_metric"],
        "best_selection_score": best_score if args.mode == "development" else None,
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame) if validation_frame is not None else 0,
    }

    write_json(output_dir / "summary.json", summary)
    completed_files = ["best.pt", "summary.json", "run.json", "history.json", "source_manifest.csv", *splits]
    if args.development_summary:
        completed_files.append("development_summary.json")
    complete_run(output_dir, completed_files)
    print(f"checkpoint={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
