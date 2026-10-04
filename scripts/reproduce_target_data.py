#!/usr/bin/env python3
"""Verify/package/restore the supplied AID test snapshots without ML dependencies.

Restoration preserves existing image bytes, including PNG images named .jpg.
It does not infer the historical brightness/blur formula or recreate checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from satdomain.artifacts import file_hash, verify_manifest
from satdomain.constants import CLASS_NAMES
from reproduce_source_data import missing_paths

CONDITIONS = ("clean", "bright", "lowres")


def load_bundle(metadata_path: Path) -> tuple[dict, dict[str, list[dict]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata["class_names"] != CLASS_NAMES or metadata["domain"] != "target"
            or set(metadata["conditions"]) != set(CONDITIONS)):
        raise ValueError("Target metadata has unexpected classes, domain or conditions")
    manifests = {}
    for condition in CONDITIONS:
        info = metadata["conditions"][condition]
        if Path(info["manifest"]).name != info["manifest"] or info["root_subdirectory"] != condition:
            raise ValueError("Unsafe manifest or condition directory")
        manifest = metadata_path.parent / info["manifest"]
        if file_hash(manifest) != info["manifest_sha256"]:
            raise ValueError(f"Frozen target manifest changed: {condition}")
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != info["images"] or Counter(r["label"] for r in rows) != Counter(info["class_counts"]):
            raise ValueError(f"Target image/class counts differ: {condition}")
        if set(info["class_counts"]) != set(CLASS_NAMES):
            raise ValueError("Target snapshot must include all nine classes")
        for row in rows:
            path = PurePosixPath(row["path"])
            if path.is_absolute() or ".." in path.parts or "\\" in row["path"] or len(path.parts) != 2:
                raise ValueError(f"Unsafe image path: {row['path']}")
            index = int(row["class_index"])
            if index < 0 or index >= len(CLASS_NAMES) or CLASS_NAMES[index] != row["label"]:
                raise ValueError("Target class index differs")
            if row["domain"] != "target" or int(row["fold"]) != -1:
                raise ValueError("Target data must have domain target and fold -1")
        if len({r["path"] for r in rows}) != len(rows) or len({r["sha256"] for r in rows}) != len(rows):
            raise ValueError("Duplicate target paths or image content")
        manifests[condition] = rows
    if sum(len(rows) for rows in manifests.values()) != metadata["total_images"]:
        raise ValueError("Total target snapshot count differs")
    return metadata, manifests


def verify_images(root: Path, metadata_path: Path, conditions=CONDITIONS) -> int:
    metadata, _ = load_bundle(metadata_path)
    for condition in conditions:
        manifest = metadata_path.parent / metadata["conditions"][condition]["manifest"]
        verify_manifest(manifest, root / condition, CLASS_NAMES, "target")
    return sum(metadata["conditions"][c]["images"] for c in conditions)


def package(root: Path, metadata_path: Path, archive_dir: Path, condition: str) -> str:
    metadata, manifests = load_bundle(metadata_path)
    verify_images(root, metadata_path, (condition,))
    info = metadata["conditions"][condition]
    archive = archive_dir / f"aid_{condition}_v1.zip"
    checksum = Path(str(archive) + ".sha256")
    if archive.exists() or checksum.exists():
        raise ValueError(f"Archive/checksum exists; preserving it: {archive}")
    prefix = f"aid_{condition}_v1/"
    notice = ("AID test subset supplied by the project user; image bytes are unchanged.\n"
              f"Condition: {condition}; processing description: {info.get('processing_user_reported', 'not recorded')}\n"
              f"Official dataset URL: {metadata['official_url']}\n"
              "Citation: Xia et al. AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification. IEEE TGRS, 2017.\n"
              "Original processing code/formula and historical download revision are not recorded.\n"
              "These snapshots are not verified paired views of the same underlying original images.\n")
    entries = {prefix + "target.csv": (metadata_path.parent / info["manifest"]).read_bytes(),
               prefix + "dataset.json": metadata_path.read_bytes(),
               prefix + "NOTICE.txt": notice.encode("utf-8")}
    entries.update({prefix + "images/" + r["path"]: (root / condition / r["path"]).read_bytes()
                    for r in manifests[condition]})
    archive_dir.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as output:
        try:
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as bundle:
                for name, content in sorted(entries.items()):
                    info_zip = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info_zip.create_system = 3
                    info_zip.external_attr = 0o100644 << 16
                    bundle.writestr(info_zip, content)
        except BaseException:
            output.close()
            archive.unlink()
            raise
    digest = file_hash(archive)
    with checksum.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {archive.name}\n")
    return digest


def verify_archive(metadata_path: Path, archive_dir: Path, condition: str) -> list[dict]:
    metadata, manifests = load_bundle(metadata_path)
    info = metadata["conditions"][condition]
    rows = manifests[condition]
    archive = archive_dir / f"aid_{condition}_v1.zip"
    expected_hash = Path(str(archive) + ".sha256").read_text(encoding="utf-8").split()[0]
    if file_hash(archive) != expected_hash:
        raise ValueError(f"Archive checksum differs: {condition}")
    prefix = f"aid_{condition}_v1/"
    expected = {prefix + "images/" + r["path"] for r in rows}
    expected.update(prefix + name for name in ("target.csv", "dataset.json", "NOTICE.txt"))
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(expected) or set(bundle.namelist()) != expected:
            raise ValueError("Archive contains missing, duplicate or unexpected paths")
        if (bundle.read(prefix + "target.csv") != (metadata_path.parent / info["manifest"]).read_bytes()
                or bundle.read(prefix + "dataset.json") != metadata_path.read_bytes()):
            raise ValueError("Archive manifest/metadata differs")
        for row in rows:
            if hashlib.sha256(bundle.read(prefix + "images/" + row["path"])).hexdigest() != row["sha256"]:
                raise ValueError(f"Archive image hash differs: {row['path']}")
    return rows


def restore(root: Path, metadata_path: Path, archive_dir: Path, conditions=CONDITIONS) -> int:
    pending = {}
    # Validate every requested archive and destination before writing any image.
    for condition in conditions:
        rows = verify_archive(metadata_path, archive_dir, condition)
        pending[condition] = missing_paths(root / condition, rows)
    for condition, rows in pending.items():
        prefix = f"aid_{condition}_v1/images/"
        with zipfile.ZipFile(archive_dir / f"aid_{condition}_v1.zip") as bundle:
            for row in rows:
                path = root / condition / row["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    handle.write(bundle.read(prefix + row["path"]))
    return sum(len(rows) for rows in pending.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "package", "verify-archive", "restore"))
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "aid_subsets")
    parser.add_argument("--metadata", type=Path, default=PROJECT / "data" / "provenance" / "aid_subsets_v1.json")
    parser.add_argument("--archive-dir", type=Path, default=PROJECT / "data" / "releases")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    args = parser.parse_args()
    conditions = CONDITIONS if args.condition == "all" else (args.condition,)
    if args.command == "verify":
        print(f"Verified {verify_images(args.root, args.metadata, conditions)} target images.")
    elif args.command == "package":
        for condition in conditions:
            print(condition, package(args.root, args.metadata, args.archive_dir, condition))
    elif args.command == "verify-archive":
        count = sum(len(verify_archive(args.metadata, args.archive_dir, c)) for c in conditions)
        print(f"Verified archive checksums and {count} target images.")
    else:
        count = restore(args.root, args.metadata, args.archive_dir, conditions)
        verify_images(args.root, args.metadata, conditions)
        print(f"Restored {count} target images; all requested images verified.")


if __name__ == "__main__":
    main()
