#!/usr/bin/env python3
"""Evaluate a fixed checkpoint on clean AID and four synthetic degradations.

This is a finite research evaluation, not a hardware load/stress benchmark.
Run it on the Windows experiment machine, never train on these target views.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from satdomain.artifacts import complete_run, file_hash, snapshot_file, start_run, write_json
from satdomain.reports import plotting, write_table
from satdomain.robustness import CONDITIONS, VERSION


def summarize(results: list[dict]) -> list[dict]:
    clean = next(result for result in results if result["degradation"]["condition"] == "clean")
    rows = []
    for result in results:
        for key in ("checkpoint_sha256", "target_manifest_sha256", "samples"):
            if result[key] != clean[key]:
                raise ValueError(f"Cannot compare robustness results with different {key}")
        degradation = result["degradation"]
        metrics = result["metrics"]
        rows.append({
            "architecture": result["architecture"], "training_seed": result["seed"],
            "augmentation": result["augmentation"], "condition": degradation["condition"],
            "severity": degradation["severity"], "corruption_seed": degradation["seed"],
            "samples": result["samples"], "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy_drop_pp": 100 * (clean["metrics"]["accuracy"] - metrics["accuracy"]),
            "macro_f1_drop_pp": 100 * (clean["metrics"]["macro_f1"] - metrics["macro_f1"]),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "target_manifest_sha256": result["target_manifest_sha256"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--corruption-seed", type=int, default=2026)
    args = parser.parse_args()
    output = Path(args.output_dir)
    checkpoint_hash = file_hash(args.checkpoint)
    manifest_hash = file_hash(args.manifest)
    metadata = start_run(output, vars(args), args.device, "synthetic_robustness_evaluation")
    metadata.update({"degradation_version": VERSION, "checkpoint_sha256": checkpoint_hash,
                     "target_manifest": snapshot_file(args.manifest, output / "target_manifest.csv")})
    write_json(output / "run.json", metadata)
    cases = [("clean", 0), *((condition, level) for condition in CONDITIONS for level in (1, 2, 3))]
    results = []
    for condition, severity in cases:
        destination = output / f"{condition}_{severity}"
        command = [sys.executable, "-m", "satdomain.evaluate",
                   "--checkpoint", args.checkpoint, "--manifest", args.manifest,
                   "--data-root", args.data_root, "--output-dir", str(destination),
                   "--device", args.device, "--workers", str(args.workers),
                   "--batch-size", str(args.batch_size), "--condition", condition,
                   "--severity", str(severity), "--corruption-seed", str(args.corruption_seed)]
        print(f"Evaluate {condition}, severity={severity}", flush=True)
        subprocess.run(command, check=True)
        result = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
        if result["checkpoint_sha256"] != checkpoint_hash or result["target_manifest_sha256"] != manifest_hash:
            raise ValueError("Checkpoint or manifest changed during evaluation")
        results.append(result)
    rows = summarize(results)
    write_table(output / "robustness.csv", list(rows[0]), ([row[key] for key in rows[0]] for row in rows))
    degraded = rows[1:]
    write_json(output / "summary.json", {
        "synthetic_only": True, "independent_images": rows[0]["samples"],
        "clean": rows[0], "degraded_mean_macro_f1": sum(r["macro_f1"] for r in degraded) / len(degraded),
        "worst_case": min(degraded, key=lambda r: r["macro_f1"]),
        "note": "Degraded views are paired transformations, not additional independent samples."
    })
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ("accuracy", "macro_f1")):
        for condition in CONDITIONS:
            selected = [r for r in rows if r["condition"] == condition]
            ax.plot([0, 1, 2, 3], [rows[0][metric], *(r[metric] for r in selected)], marker="o", label=condition)
        ax.set(xlabel="Synthetic severity (0 = clean)", ylabel=metric, ylim=(0, 1), xticks=[0, 1, 2, 3])
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "robustness_curves.png", dpi=180)
    plt.close(fig)
    complete_run(output, ["run.json", "target_manifest.csv", "robustness.csv", "summary.json", "robustness_curves.png",
                          *(f"{condition}_{severity}/metrics.json" for condition, severity in cases)])
    print(f"Saved: {output / 'robustness.csv'}", flush=True)


if __name__ == "__main__":
    main()
