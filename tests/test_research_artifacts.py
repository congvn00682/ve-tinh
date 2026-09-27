"""Integrity and reporting regression tests; use only the standard library."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from satdomain.artifacts import complete_run, file_hash, is_complete, snapshot_file, start_run, verify_manifest
from satdomain.reports import export_history, export_metrics


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_manifest_verifies_bytes_and_labels(self):
        image = self.root / "sample.jpg"
        image.write_bytes(b"fixture image bytes")
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "class_index", "label", "domain", "sha256"])
            writer.writerow([image.name, 0, "forest", "target", file_hash(image)])
        verify_manifest(manifest, self.root, ["forest"], "target")
        with self.assertRaisesRegex(ValueError, "Class mapping"):
            verify_manifest(manifest, self.root, ["desert"], "target")
        image.write_bytes(b"changed image")
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_manifest(manifest, self.root, ["forest"], "target")

    def test_snapshot_and_completed_checkpoint_integrity(self):
        original = self.root / "manifest.csv"
        original.write_text("original manifest", encoding="utf-8")
        run = self.root / "run"
        run.mkdir()
        reference = snapshot_file(original, run / "source_manifest.csv")
        original.write_text("later manifest", encoding="utf-8")
        self.assertNotEqual(reference["sha256"], file_hash(original))
        checkpoint = run / "best.pt"
        checkpoint.write_bytes(b"dummy weights")
        self.assertFalse(is_complete(run))
        complete_run(run, ["best.pt", "source_manifest.csv"])
        self.assertTrue(is_complete(run))
        checkpoint.write_bytes(b"different weights")
        self.assertFalse(is_complete(run))

    def test_new_run_preserves_existing_output(self):
        with patch("satdomain.artifacts.environment_info", return_value={}):
            info = start_run(self.root / "run", {"seed": 13}, "cpu", "training")
            self.assertEqual(info["arguments"]["seed"], 13)
            self.assertTrue(info["run_id"])
            with self.assertRaisesRegex(ValueError, "not empty"):
                start_run(self.root / "run", {"seed": 37}, "cpu", "training")

    def test_report_tables_preserve_class_order_counts_and_f1(self):
        result = {"class_names": ["water", "forest"], "metrics": {
            "confusion_matrix": [[2, 1], [0, 0]],
            "per_class_precision": [1.0, 0.0], "per_class_recall": [2/3, 0.0],
            "per_class_f1": [.8, 0.0], "support": [3, 0],
        }}
        fake_plot = MagicMock()
        fake_plot.subplots.return_value = (MagicMock(), MagicMock())
        with patch("satdomain.reports.plotting", return_value=fake_plot):
            export_metrics(self.root, result)
        with (self.root / "per_class_metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["class_name"] for r in rows], ["water", "forest"])
        self.assertEqual(float(rows[0]["f1"]), .8)
        self.assertEqual(int(rows[0]["support"]), 3)
        with (self.root / "confusion_matrix_normalized.csv").open() as handle:
            normalized = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(normalized[0]["water"]), 2/3)
        self.assertEqual(float(normalized[1]["forest"]), 0.0)
        self.assertEqual(fake_plot.close.call_count, 3)

    def test_curves_keep_validation_and_final_fit_distinct(self):
        history = [{"epoch": 1, "train_loss": .5, "train_accuracy": .8, "train_macro_f1": .7,
                    "validation_loss": .6, "validation_accuracy": .6, "validation_macro_f1": .5}]
        fake_plot = MagicMock()
        axes = [MagicMock() for _ in range(3)]
        fake_plot.subplots.return_value = (MagicMock(), axes)
        with patch("satdomain.reports.plotting", return_value=fake_plot):
            export_history(self.root, history)
            self.assertEqual(axes[1].plot.call_count, 2)
            final_history = [{k: v for k, v in history[0].items() if not k.startswith("validation")}]
            export_history(self.root, final_history)
            self.assertEqual(axes[1].plot.call_count, 3)
        with (self.root / "history.csv").open() as handle:
            row = next(csv.DictReader(handle))
        self.assertNotIn("validation_loss", row)


if __name__ == "__main__":
    unittest.main()
