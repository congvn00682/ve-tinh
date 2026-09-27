#!/usr/bin/env python3
"""Audit source/AID folders and create reproducible CSV manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


CLASS_NAMES = [
    "airport",
    "baseball_diamond",
    "beach",
    "bridge",
    "church",
    "commercial_area",
    "dense_residential",
    "desert",
    "forest",
]

ALIASES = {
    "airport": {"airport"},
    "baseball_diamond": {"baseballdiamond", "baseballfield"},
    "beach": {"beach"},
    "bridge": {"bridge"},
    "church": {"church"},
    "commercial_area": {"commercialarea", "commercial"},
    "dense_residential": {"denseresidential", "denseresidentialarea"},
    "desert": {"desert"},
    "forest": {"forest"},
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--aid-root")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--expected-source-per-class", type=int)
    parser.add_argument("--min-source-per-class", type=int, default=5)
    parser.add_argument("--expected-target-per-class", type=int)
    parser.add_argument("--min-target-per-class", type=int, default=1)
    return parser.parse_args()


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_class_directory(root: Path, canonical: str) -> Path:
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and normalized(child.name) in ALIASES[canonical]
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one directory for {canonical!r} in {root}, found {candidates}"
        )
    return candidates[0]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_domain(
    root: Path,
    domain: str,
    expected_per_class: int | None,
    min_per_class: int,
    folds: int | None,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    seen_hashes: dict[str, str] = {}
    for class_index, canonical in enumerate(CLASS_NAMES):
        directory = find_class_directory(root, canonical)
        images = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if expected_per_class is not None and len(images) != expected_per_class:
            raise ValueError(
                f"{domain}/{canonical}: expected {expected_per_class} images, "
                f"found {len(images)}"
            )
        if len(images) < min_per_class:
            raise ValueError(
                f"{domain}/{canonical}: requires at least {min_per_class} images, "
                f"found {len(images)}"
            )

        shuffled = images.copy()
        random.Random(f"{seed}:{domain}:{canonical}").shuffle(shuffled)
        fold_by_path = (
            {path: index % folds for index, path in enumerate(shuffled)}
            if folds is not None
            else {}
        )
        for path in images:
            digest = file_hash(path)
            if digest in seen_hashes:
                raise ValueError(
                    f"Duplicate content: {path} and {seen_hashes[digest]}"
                )
            seen_hashes[digest] = str(path)
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "label": canonical,
                    "class_index": class_index,
                    "fold": fold_by_path.get(path, -1),
                    "domain": domain,
                    "sha256": digest,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], root: Path) -> dict:
    return {
        "root": str(root.resolve()),
        "images": len(rows),
        "classes": dict(sorted(Counter(row["label"] for row in rows).items())),
        "folds": dict(sorted(Counter(row["fold"] for row in rows).items())),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    source_root = Path(args.source_root)
    source_rows = collect_domain(
        source_root,
        domain="source",
        expected_per_class=args.expected_source_per_class,
        min_per_class=max(args.min_source_per_class, args.folds),
        folds=args.folds,
        seed=args.seed,
    )
    write_csv(output_dir / "source.csv", source_rows)
    summary = {"source": summarize(source_rows, source_root)}

    if args.aid_root:
        aid_root = Path(args.aid_root)
        target_rows = collect_domain(
            aid_root,
            domain="target",
            expected_per_class=args.expected_target_per_class,
            min_per_class=args.min_target_per_class,
            folds=None,
            seed=args.seed,
        )
        source_hashes = {row["sha256"]: row["path"] for row in source_rows}
        overlap = [
            (source_hashes[row["sha256"]], row["path"])
            for row in target_rows
            if row["sha256"] in source_hashes
        ]
        if overlap:
            raise ValueError(f"Source/target duplicate content detected: {overlap[:5]}")
        write_csv(output_dir / "aid_test.csv", target_rows)
        summary["target"] = summarize(target_rows, aid_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
