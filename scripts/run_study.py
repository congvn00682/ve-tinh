#!/usr/bin/env python3
"""Develop on source only, fit on all source data, then test once on AID."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from satdomain.artifacts import environment_info, file_hash, is_complete, snapshot_file, write_json
from satdomain.robustness import PROFILES

DEFAULT_MODELS = [
    "small_cnn",
    "resnet18_scratch",
    "resnet18_pretrained",
    "deit_tiny_pretrained",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--augmentation", choices=PROFILES, default="baseline")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def training_base(args: argparse.Namespace, architecture: str, seed: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "satdomain.train",
        "--manifest",
        args.source_manifest,
        "--data-root",
        args.source_root,
        "--arch",
        architecture,
        "--seed",
        str(seed),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--device",
        args.device,
        "--augmentation",
        args.augmentation,
    ]
    if args.amp:
        command.append("--amp")
    return command


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    # Freeze configuration and manifests before training or --skip-existing.
    lock_path = output_root / "study.json"
    signature = {k: v for k, v in vars(args).items() if k not in {"skip_existing", "output_root"}}
    signature["source_manifest_sha256"] = file_hash(args.source_manifest)
    signature["target_manifest_sha256"] = file_hash(args.target_manifest)
    signature["code_sha256"] = environment_info()["code_sha256"]
    if lock_path.exists():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        if previous != signature:
            raise ValueError("Study configuration/manifests changed. Use a new --output-root.")
    else:
        if (output_root / "development").exists() or (output_root / "final").exists():
            raise ValueError("Existing study has no provenance lock. Use a new --output-root; keep old results.")
        output_root.mkdir(parents=True, exist_ok=True)
        snapshot_file(args.source_manifest, output_root / "manifests" / "source.csv")
        snapshot_file(args.target_manifest, output_root / "manifests" / "aid_test.csv")
        generation = Path(args.source_manifest).parent / "summary.json"
        if generation.exists():
            snapshot_file(generation, output_root / "manifests" / "generation_summary.json")
        write_json(lock_path, signature)

    # Phase 1: source-only development. Target images are not opened.
    for architecture in args.models:
        for seed_index, seed in enumerate(args.seeds):
            validation_fold = seed_index % args.folds
            destination = output_root / "development" / architecture / f"seed_{seed}"
            if args.skip_existing and is_complete(destination):
                continue
            command = training_base(args, architecture, seed)
            command.extend(
                [
                    "--mode",
                    "development",
                    "--validation-fold",
                    str(validation_fold),
                    "--epochs",
                    str(args.max_epochs),
                    "--patience",
                    str(args.patience),
                    "--output-dir",
                    str(destination),
                ]
            )
            run(command)

    # Phase 2: refit with every source image using source-selected epoch counts.
    for architecture in args.models:
        for seed in args.seeds:
            development_dir = (
                output_root / "development" / architecture / f"seed_{seed}"
            )
            summary = json.loads(
                (development_dir / "summary.json").read_text(encoding="utf-8")
            )
            final_epochs = max(1, int(summary["best_epoch"]))
            destination = output_root / "final" / architecture / f"seed_{seed}"
            if args.skip_existing and is_complete(destination):
                continue
            command = training_base(args, architecture, seed)
            command.extend(
                [
                    "--mode",
                    "final",
                    "--epochs",
                    str(final_epochs),
                    "--development-summary",
                    str(development_dir / "summary.json"),
                    "--output-dir",
                    str(destination),
                ]
            )
            run(command)

    # Phase 3: all training is complete; test final models on AID only.
    for architecture in args.models:
        for seed in args.seeds:
            final_dir = output_root / "final" / architecture / f"seed_{seed}"
            evaluation_dir = final_dir / "aid_evaluation"
            if args.skip_existing and is_complete(evaluation_dir):
                continue
            run(
                [
                    sys.executable,
                    "-m",
                    "satdomain.evaluate",
                    "--checkpoint",
                    str(final_dir / "best.pt"),
                    "--manifest",
                    args.target_manifest,
                    "--data-root",
                    args.target_root,
                    "--domain",
                    "target",
                    "--output-dir",
                    str(evaluation_dir),
                    "--batch-size",
                    str(args.batch_size),
                    "--workers",
                    str(args.workers),
                    "--device",
                    args.device,
                ]
            )


if __name__ == "__main__":
    main()
