"""Portable run provenance and integrity checks (standard library only)."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def environment_info() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "matplotlib", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    project = Path(__file__).resolve().parents[2]
    git = {}
    for key, arguments in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
        try:
            git[key] = subprocess.check_output(
                ["git", "-C", str(project), *arguments], stderr=subprocess.DEVNULL,
                text=True, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            git[key] = None
    sources = sorted((project / "src" / "satdomain").glob("*.py"))
    sources += sorted((project / "scripts").glob("*.py"))
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "git": git,
        "code_sha256": {p.relative_to(project).as_posix(): file_hash(p) for p in sources},
    }


def start_run(output_dir: Path, args: dict, device: str, kind: str) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}. Use a new run directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1, "run_id": str(uuid4()), "kind": kind,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": args, "device": device, "environment": environment_info(),
    }
    write_json(output_dir / "run.json", metadata)
    return metadata


def snapshot_file(source: str | Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {"file": destination.name, "sha256": file_hash(destination)}


def verify_manifest(path: str | Path, root: str | Path, classes: list[str], domain: str) -> None:
    """Check actual image bytes, canonical labels, domains and duplicates before use."""
    root = Path(root).resolve()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    for row in rows:
        index = int(row["class_index"])
        if index < 0 or index >= len(classes) or classes[index] != row["label"]:
            raise ValueError(f"Class mapping mismatch: {row['path']}")
        if row["domain"] != domain:
            raise ValueError(f"Unexpected domain: {row['domain']}")
        image = (root / row["path"]).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"Image is outside data root: {row['path']}")
        digest = file_hash(image)
        if digest != row["sha256"]:
            raise ValueError(f"Image changed since manifest creation: {row['path']}")
        if digest in seen:
            raise ValueError(f"Duplicate image content: {row['path']}")
        seen.add(digest)
    if {r["label"] for r in rows} != set(classes):
        raise ValueError("Manifest must contain every class")


def complete_run(directory: Path, filenames: list[str]) -> None:
    write_json(directory / "complete.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: file_hash(directory / name) for name in filenames},
    })


def is_complete(directory: Path) -> bool:
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    return bool(record["files"]) and all(
        (directory / name).is_file() and file_hash(directory / name) == digest
        for name, digest in record["files"].items()
    )
