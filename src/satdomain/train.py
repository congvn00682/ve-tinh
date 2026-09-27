"""Train one source-CV fold or one final source-domain model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .constants import CLASS_NAMES, CLASS_TO_INDEX, IMAGENET_MEAN, IMAGENET_STD
from .data import (
    ManifestDataset,
    build_transforms,
    read_manifest,
    source_train_validation_split,
)
from .metrics import classification_metrics
from .models import create_model
from .runtime import choose_device, seed_everything, write_json


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
    return parser.parse_args()


def make_loader(
    frame,
    root: str,
    transform,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ManifestDataset(frame, root=root, transform=transform)
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


def plot_history(path: Path, history: list[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    if "validation_loss" in history[0]:
        axes[0].plot(
            epochs,
            [row["validation_loss"] for row in history],
            label="validation",
        )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_macro_f1"] for row in history], label="train")
    if "validation_macro_f1" in history[0]:
        axes[1].plot(
            epochs,
            [row["validation_macro_f1"] for row in history],
            label="validation",
        )
    axes[1].set_title("Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def checkpoint_payload(model: nn.Module, args: argparse.Namespace, epoch: int) -> dict:
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
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_manifest(args.manifest, domain="source")
    train_transform, eval_transform = build_transforms(args.image_size)

    if args.mode == "development":
        train_frame, validation_frame = source_train_validation_split(
            frame,
            validation_fold=args.validation_fold,
        )
    else:
        train_frame = frame
        validation_frame = None

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
        else:
            score = train_metrics["macro_f1"]

        history.append(row)
        if args.mode == "final":
            best_score = score
            best_epoch = epoch
            torch.save(checkpoint_payload(model, args, epoch), checkpoint_path)
        elif score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint_payload(model, args, epoch), checkpoint_path)
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
    plot_history(output_dir / "training_curves.png", history)
    summary = {
        "architecture": args.arch,
        "mode": args.mode,
        "seed": args.seed,
        "validation_fold": (
            args.validation_fold if args.mode == "development" else None
        ),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": (
            best_score if args.mode == "development" else None
        ),
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame) if validation_frame is not None else 0,
    }

    write_json(output_dir / "summary.json", summary)
    print(f"checkpoint={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
