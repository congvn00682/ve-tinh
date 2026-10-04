"""Validate archived experiment tables; no ML dependencies or model execution."""

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "submitted_results", PROJECT / "scripts" / "summarize_submitted_results.py")
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)


def fixture_rows():
    return [{"File_Name": f"{label}_{index}.jpg", "True_Label": label,
             "Predicted_Label": label, "Confidence": "0.95", "Is_Correct": "TRUE"}
            for label in results.CLASS_NAMES for index in range(20)]


class SubmittedResultsTests(unittest.TestCase):
    def copy_bundle(self, directory):
        source = PROJECT / "results" / "cuda_aid_9class"
        root = Path(directory) / "bundle"
        shutil.copytree(source / "raw", root / "raw")
        shutil.copyfile(source / "provenance.json", root / "provenance.json")
        return root

    def test_metrics_and_confusion_orientation_from_known_errors(self):
        rows = fixture_rows()
        rows[0]["Predicted_Label"] = results.CLASS_NAMES[1]
        rows[20]["Predicted_Label"] = results.CLASS_NAMES[0]
        rows[0]["Is_Correct"] = rows[20]["Is_Correct"] = "FALSE"
        metric, classes, matrix = results.prediction_metrics(rows)
        self.assertEqual(metric["correct"], 178)
        self.assertAlmostEqual(metric["accuracy"], 178 / 180)
        self.assertAlmostEqual(metric["macro_f1"], (7 + 2 * .95) / 9)
        self.assertEqual(matrix[0][1], 1)
        self.assertEqual(matrix[1][0], 1)
        self.assertEqual(classes[0]["support"], 20)
        self.assertAlmostEqual(classes[0]["f1"], .95)

    def test_missing_predictions_for_a_class_produce_zero_f1(self):
        rows = fixture_rows()
        for row in rows:
            if row["True_Label"] == "dense_residential":
                row["Predicted_Label"] = "commercial_area"
                row["Is_Correct"] = "FALSE"
        _, classes, _ = results.prediction_metrics(rows)
        residential = next(c for c in classes if c["class_name"] == "dense_residential")
        self.assertEqual((residential["precision"], residential["recall"], residential["f1"]), (0, 0, 0))

    def test_inconsistent_correctness_flag_is_rejected(self):
        rows = fixture_rows()
        rows[0]["Is_Correct"] = "FALSE"
        with self.assertRaisesRegex(ValueError, "Is_Correct"):
            results.prediction_metrics(rows)

    def test_duplicate_image_or_class_imbalance_is_rejected(self):
        rows = fixture_rows()
        rows[0] = rows[1].copy()
        with self.assertRaisesRegex(ValueError, "distinct"):
            results.prediction_metrics(rows)
        rows = fixture_rows()
        rows[0]["True_Label"] = "forest"
        with self.assertRaisesRegex(ValueError, "20 images"):
            results.prediction_metrics(rows)

    def test_nonfinite_confidence_is_rejected(self):
        rows = fixture_rows()
        rows[0]["Confidence"] = "nan"
        with self.assertRaisesRegex(ValueError, "finite"):
            results.prediction_metrics(rows)

    def test_summary_uses_sample_sd_and_distinct_seeds(self):
        rows = [{"model": "test", "seed": seed, "score": score}
                for seed, score in zip(results.SEEDS, (.5, .75, 1.0))]
        summary = results.grouped_summary(rows, ("model",), ("score",))[0]
        self.assertEqual(summary["score_mean"], .75)
        self.assertEqual(summary["score_std"], .25)
        rows[0]["seed"] = 37
        with self.assertRaisesRegex(ValueError, "seeds"):
            results.grouped_summary(rows, ("model",), ("score",))

    def test_archived_inputs_rebuild_expected_tables_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_bundle(directory)
            files = [root / "provenance.json", *sorted((root / "raw").rglob("*.csv"))]
            before = {p: p.read_bytes() for p in files}
            verification = results.summarize(root)
            self.assertEqual(before, {p: p.read_bytes() for p in files})
            self.assertEqual(verification["degraded_runs"], 24)
            self.assertFalse(verification["drop_metrics_computed"])
            self.assertFalse(verification["paired_clean_degraded_comparison_verified"])
            summaries = results.read_csv(root / "condition_models.csv")
            deit = next(r for r in summaries if (r["condition"], r["architecture"]) == ("bright", "deit_tiny_pretrained"))
            self.assertAlmostEqual(float(deit["accuracy_mean"]), 471 / 540)
            classes = results.read_csv(root / "per_class_condition_runs.csv")
            self.assertEqual(len(classes), 216)
            self.assertEqual(len(list((root / "confusion_matrices").rglob("*.csv"))), 24)
            self.assertNotIn("accuracy_drop_pp", results.read_csv(root / "condition_runs.csv")[0])

    def test_raw_csv_tampering_is_rejected_before_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_bundle(directory)
            raw = root / "raw" / "bright" / "small_cnn.csv"
            raw.write_bytes(raw.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed since import"):
                results.summarize(root)
            self.assertFalse((root / "condition_runs.csv").exists())

    def test_different_image_lists_across_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_bundle(directory)
            raw = root / "raw" / "bright" / "resnet18_scratch.csv"
            rows = results.read_csv(raw)
            rows[0]["File_Name"] = "different_airport.jpg"
            results.write_csv(raw, rows)
            path = root / "provenance.json"
            provenance = json.loads(path.read_text())
            for record in provenance["files"]:
                if record["path"] == raw.relative_to(root).as_posix():
                    record["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
            path.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, "Filename/label set differs"):
                results.summarize(root)
            self.assertFalse((root / "condition_runs.csv").exists())


if __name__ == "__main__":
    unittest.main()
