"""Preserve supplied target bytes, manifests and existing user files."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "target_data", Path(__file__).resolve().parents[1] / "scripts" / "reproduce_target_data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class TargetReproductionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.root = self.folder / "images"
        self.metadata = self.folder / "dataset.json"
        self.archive_dir = self.folder / "archives"
        conditions = {}
        for condition in data.CONDITIONS:
            rows = []
            for index, label in enumerate(data.CLASS_NAMES):
                path = self.root / condition / label / "test.jpg"
                path.parent.mkdir(parents=True)
                # A byte fixture with a PNG signature and .jpg suffix; no decoding is claimed.
                path.write_bytes(b"\x89PNG\r\n\x1a\n" + f"{condition}:{label}".encode())
                rows.append({"path": f"{label}/test.jpg", "label": label, "class_index": index,
                             "fold": -1, "domain": "target", "sha256": data.file_hash(path)})
            manifest = self.folder / f"{condition}.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            conditions[condition] = {"manifest": manifest.name, "manifest_sha256": data.file_hash(manifest),
                                     "root_subdirectory": condition, "images": 9,
                                     "class_counts": {c: 1 for c in data.CLASS_NAMES}}
        self.metadata.write_text(json.dumps({"class_names": data.CLASS_NAMES, "domain": "target",
                                             "conditions": conditions, "total_images": 27,
                                             "official_url": "https://example.com"}))

    def package_all(self):
        for condition in data.CONDITIONS:
            data.package(self.root, self.metadata, self.archive_dir, condition)

    def test_restore_preserves_all_bytes_despite_filename_extension(self):
        self.package_all()
        destination = self.folder / "restored"
        self.assertEqual(data.restore(destination, self.metadata, self.archive_dir), 27)
        self.assertEqual(data.verify_images(destination, self.metadata), 27)
        for original in self.root.rglob("*.jpg"):
            self.assertEqual(original.read_bytes(), (destination / original.relative_to(self.root)).read_bytes())
        self.assertEqual(data.restore(destination, self.metadata, self.archive_dir), 0)

    def test_archive_is_deterministic_and_existing_archive_is_preserved(self):
        digest = data.package(self.root, self.metadata, self.archive_dir, "bright")
        other = self.folder / "other"
        self.assertEqual(digest, data.package(self.root, self.metadata, other, "bright"))
        with self.assertRaisesRegex(ValueError, "preserving"):
            data.package(self.root, self.metadata, self.archive_dir, "bright")

    def test_bad_archive_blocks_all_restoration_before_writing(self):
        self.package_all()
        archive = self.archive_dir / "aid_lowres_v1.zip"
        with archive.open("ab") as handle:
            handle.write(b"tampered")
        destination = self.folder / "restored"
        with self.assertRaisesRegex(ValueError, "checksum"):
            data.restore(destination, self.metadata, self.archive_dir)
        self.assertFalse(destination.exists())

    def test_restore_does_not_overwrite_conflicting_user_image(self):
        self.package_all()
        destination = self.folder / "restored"
        conflict = destination / "clean" / data.CLASS_NAMES[0] / "test.jpg"
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"user content")
        with self.assertRaisesRegex(ValueError, "preserving"):
            data.restore(destination, self.metadata, self.archive_dir)
        self.assertEqual(conflict.read_bytes(), b"user content")
        self.assertEqual(len(list(destination.rglob("*.jpg"))), 1)

    def test_target_train_fold_is_rejected(self):
        metadata = json.loads(self.metadata.read_text())
        manifest = self.folder / metadata["conditions"]["clean"]["manifest"]
        with manifest.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["fold"] = "0"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata["conditions"]["clean"]["manifest_sha256"] = data.file_hash(manifest)
        self.metadata.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "fold -1"):
            data.load_bundle(self.metadata)

    def test_real_clean_snapshot_matches_every_reported_run_hash(self):
        metadata_path = data.PROJECT / "data" / "provenance" / "aid_subsets_v1.json"
        metadata, manifests = data.load_bundle(metadata_path)
        self.assertEqual(sum(len(rows) for rows in manifests.values()), 540)
        with (data.PROJECT / "results" / "cuda_aid_9class" / "raw" / "clean" / "cross_domain_runs.csv").open(newline="") as handle:
            hashes = {r["target_manifest_sha256"] for r in csv.DictReader(handle)}
        self.assertEqual(hashes, {metadata["conditions"]["clean"]["manifest_sha256"]})


if __name__ == "__main__":
    unittest.main()
