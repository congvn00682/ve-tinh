"""Check frozen source reconstruction and preservation using byte fixtures."""

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

spec = importlib.util.spec_from_file_location(
    "reproduce_source_data", Path(__file__).resolve().parents[1] / "scripts" / "reproduce_source_data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class DataReproductionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.root = self.folder / "source"
        self.rows = []
        for index, label in enumerate(data.CLASS_NAMES):
            path = self.root / label / "sample.jpg"
            path.parent.mkdir(parents=True)
            path.write_bytes(f"fixture:{label}".encode())
            self.rows.append({"path": f"{label}/sample.jpg", "label": label,
                              "class_index": index, "fold": 0, "domain": "source",
                              "sha256": data.file_hash(path)})
        self.manifest = self.folder / "source.csv"
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.metadata = self.folder / "dataset.json"
        self.metadata.write_text(json.dumps({
            "manifest_sha256": data.file_hash(self.manifest), "images": 9,
            "class_names": data.CLASS_NAMES, "class_counts": {c: 1 for c in data.CLASS_NAMES},
            "folds": 5, "mirror_id": "fixture/dataset", "mirror_revision": "f" * 40,
            "upstream_path_prefix": "data/", "official_url": "https://example.com",
            "license_as_reported_by_official_page": "fixture", "license_url": "https://example.com/license",
            "citation": "fixture",
        }))
        self.archive = self.folder / "subset.zip"

    def package(self):
        return data.package_source(self.root, self.manifest, self.metadata, self.archive)

    def test_archive_roundtrip_preserves_every_image_and_manifest(self):
        self.package()
        destination = self.folder / "restored"
        self.assertEqual(data.restore_source(destination, self.archive, self.manifest, self.metadata), 9)
        for row in self.rows:
            self.assertEqual((self.root / row["path"]).read_bytes(), (destination / row["path"]).read_bytes())
        self.assertEqual(data.restore_source(destination, self.archive, self.manifest, self.metadata), 0)

    def test_archive_is_deterministic_and_existing_archive_is_preserved(self):
        digest = self.package()
        other = self.folder / "another.zip"
        self.assertEqual(digest, data.package_source(self.root, self.manifest, self.metadata, other))
        original = self.archive.read_bytes()
        with self.assertRaisesRegex(ValueError, "exists"):
            self.package()
        self.assertEqual(self.archive.read_bytes(), original)

    def test_changed_source_is_rejected_before_packaging(self):
        (self.root / self.rows[0]["path"]).write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.package()
        self.assertFalse(self.archive.exists())

    def test_conflicting_destination_is_preserved_before_restore(self):
        self.package()
        destination = self.folder / "restored"
        conflict = destination / self.rows[0]["path"]
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"existing user file")
        with self.assertRaisesRegex(ValueError, "preserving"):
            data.restore_source(destination, self.archive, self.manifest, self.metadata)
        self.assertEqual(conflict.read_bytes(), b"existing user file")
        self.assertEqual(len(list(destination.rglob("*.jpg"))), 1)

    def test_archive_tampering_is_rejected(self):
        self.package()
        with self.archive.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            data.verify_archive(self.archive, self.manifest, self.metadata)

    def test_unsafe_manifest_path_is_rejected_even_with_matching_manifest_hash(self):
        self.rows[0]["path"] = "../outside.jpg"
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        metadata = json.loads(self.metadata.read_text())
        metadata["manifest_sha256"] = data.file_hash(self.manifest)
        self.metadata.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            data.load_snapshot(self.manifest, self.metadata)

    def test_downloaded_bytes_must_match_pinned_manifest(self):
        _, metadata = data.load_snapshot(self.manifest, self.metadata)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"wrong bytes"
        destination = self.folder / "downloaded"
        with patch.object(data.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "Downloaded image SHA-256 differs"):
                data.download_source(destination, self.rows[:1], metadata, workers=1)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
