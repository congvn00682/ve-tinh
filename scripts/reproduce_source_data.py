#!/usr/bin/env python3
"""Verify, download, package or restore the frozen source subset (stdlib only).

Image bytes are checked against the frozen manifest; existing mismatching files
are never overwritten. Dataset revision is pinned, not inferred from `main`.
This covers source only; supplied AID snapshots use reproduce_target_data.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from satdomain.artifacts import file_hash, verify_manifest
from satdomain.constants import CLASS_NAMES

ARCHIVE_PREFIX = "source_subset_v1/"


def load_snapshot(manifest: Path, metadata_path: Path) -> tuple[list[dict], dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if file_hash(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Frozen source manifest differs from snapshot metadata")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != metadata["images"] or metadata["class_names"] != CLASS_NAMES:
        raise ValueError("Source snapshot count or class order differs")
    seen_paths, seen_hashes = set(), set()
    for row in rows:
        path = PurePosixPath(row["path"])
        if (path.is_absolute() or ".." in path.parts or "\\" in row["path"]
                or len(path.parts) != 2 or path.parts[0] != row["label"]):
            raise ValueError(f"Unsafe or invalid image path: {row['path']}")
        index = int(row["class_index"])
        if index < 0 or index >= len(CLASS_NAMES) or CLASS_NAMES[index] != row["label"]:
            raise ValueError("Invalid class mapping in snapshot")
        if row["domain"] != "source" or not 0 <= int(row["fold"]) < metadata["folds"]:
            raise ValueError("Invalid source domain/fold")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid image SHA-256")
        if row["path"] in seen_paths or row["sha256"] in seen_hashes:
            raise ValueError("Duplicate path or image content in snapshot")
        seen_paths.add(row["path"])
        seen_hashes.add(row["sha256"])
    if {row["label"] for row in rows} != set(CLASS_NAMES):
        raise ValueError("Source snapshot must include all nine classes")
    for label, count in metadata["class_counts"].items():
        if sum(row["label"] == label for row in rows) != count:
            raise ValueError("Snapshot class count differs")
    return rows, metadata


def missing_paths(root: Path, rows: list[dict]) -> list[dict]:
    root = root.resolve()
    missing = []
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Destination escapes data root: {path}")
        if path.exists():
            if not path.is_file() or file_hash(path) != row["sha256"]:
                raise ValueError(f"Existing file differs; preserving it: {path}")
        else:
            missing.append(row)
    return missing


def download_source(root: Path, rows: list[dict], metadata: dict, workers: int = 2) -> int:
    if not 1 <= workers <= 4:
        raise ValueError("Download workers must be 1..4")
    if not re.fullmatch(r"[0-9a-f]{40}", metadata["mirror_revision"]):
        raise ValueError("A pinned 40-character dataset revision is required")
    pending = missing_paths(root, rows)

    def download(row):
        remote_path = metadata["upstream_path_prefix"] + row["path"]
        url = (f"https://huggingface.co/datasets/{metadata['mirror_id']}/resolve/"
               f"{metadata['mirror_revision']}/{urllib.parse.quote(remote_path, safe='/')}")
        request = urllib.request.Request(url, headers={"User-Agent": "satdomain-source-reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Downloaded image SHA-256 differs: {row['path']}")
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
        return row["path"]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for name in executor.map(download, pending):
            print(f"Downloaded and verified: {name}", flush=True)
    return len(pending)


def package_source(root: Path, manifest: Path, metadata_path: Path, archive: Path) -> str:
    rows, metadata = load_snapshot(manifest, metadata_path)
    verify_manifest(manifest, root, CLASS_NAMES, "source")
    checksum = Path(str(archive) + ".sha256")
    if archive.exists() or checksum.exists():
        raise ValueError("Archive/checksum exists; choose a new output path")
    notice = (f"Source subset of NWPU-RESISC45; {metadata['images']} retained images in the repository snapshot.\n"
              f"Official dataset page: {metadata['official_url']}\n"
              f"Dataset license reported there: {metadata['license_as_reported_by_official_page']}\n"
              f"License terms: {metadata['license_url']}\n"
              f"Citation: {metadata['citation']}\n"
              "Image bytes are unchanged; subset selection and fold assignments are listed in source.csv.\n"
              "This snapshot does not establish the historical Windows run's source manifest.\n")
    entries = {ARCHIVE_PREFIX + "source.csv": manifest.read_bytes(),
               ARCHIVE_PREFIX + "dataset.json": metadata_path.read_bytes(),
               ARCHIVE_PREFIX + "NOTICE.txt": notice.encode("utf-8")}
    entries.update({ARCHIVE_PREFIX + "images/" + r["path"]: (root / r["path"]).read_bytes() for r in rows})
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as output:
        try:
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as bundle:
                for name, content in sorted(entries.items()):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    bundle.writestr(info, content)
        except BaseException:
            output.close()
            archive.unlink()
            raise
    digest = file_hash(archive)
    with checksum.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {archive.name}\n")
    return digest


def verify_archive(archive: Path, manifest: Path, metadata_path: Path) -> list[dict]:
    rows, _ = load_snapshot(manifest, metadata_path)
    expected_digest = Path(str(archive) + ".sha256").read_text(encoding="utf-8").split()[0]
    if file_hash(archive) != expected_digest:
        raise ValueError("Archive SHA-256 differs from companion checksum")
    expected = {ARCHIVE_PREFIX + "images/" + row["path"] for row in rows}
    expected.update(ARCHIVE_PREFIX + name for name in ("source.csv", "dataset.json", "NOTICE.txt"))
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(expected) or set(bundle.namelist()) != expected:
            raise ValueError("Archive has missing, duplicate or unexpected paths")
        if (bundle.read(ARCHIVE_PREFIX + "source.csv") != manifest.read_bytes()
                or bundle.read(ARCHIVE_PREFIX + "dataset.json") != metadata_path.read_bytes()):
            raise ValueError("Archive manifest/metadata differs from frozen snapshot")
        for row in rows:
            content = bundle.read(ARCHIVE_PREFIX + "images/" + row["path"])
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError(f"Archive image differs: {row['path']}")
    return rows


def restore_source(root: Path, archive: Path, manifest: Path, metadata_path: Path) -> int:
    rows = verify_archive(archive, manifest, metadata_path)
    pending = missing_paths(root, rows)
    with zipfile.ZipFile(archive) as bundle:
        for row in pending:
            path = root / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(bundle.read(ARCHIVE_PREFIX + "images/" + row["path"]))
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "download", "package", "verify-archive", "restore"))
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "source_subset")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "provenance" / "source_subset_v1.csv")
    parser.add_argument("--metadata", type=Path, default=PROJECT / "data" / "provenance" / "source_subset_v1.json")
    parser.add_argument("--archive", type=Path, default=PROJECT / "data" / "releases" / "source_subset_v1.zip")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    rows, metadata = load_snapshot(args.manifest, args.metadata)
    if args.command == "verify":
        verify_manifest(args.manifest, args.root, CLASS_NAMES, "source")
        print(f"Verified {len(rows)} source images against frozen manifest.")
    elif args.command == "download":
        count = download_source(args.root, rows, metadata, args.workers)
        verify_manifest(args.manifest, args.root, CLASS_NAMES, "source")
        print(f"Downloaded {count}; verified {len(rows)} source images.")
    elif args.command == "package":
        print(f"Archive SHA-256: {package_source(args.root, args.manifest, args.metadata, args.archive)}")
    elif args.command == "verify-archive":
        print(f"Verified archive and all {len(verify_archive(args.archive, args.manifest, args.metadata))} images.")
    else:
        count = restore_source(args.root, args.archive, args.manifest, args.metadata)
        verify_manifest(args.manifest, args.root, CLASS_NAMES, "source")
        print(f"Restored {count}; verified {len(rows)} source images.")


if __name__ == "__main__":
    main()
