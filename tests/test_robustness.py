"""Check reproducible paired degradations and scientific summary integrity."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from satdomain.robustness import CONDITIONS, EvaluationDegradation, degrade, profile_metadata

spec = importlib.util.spec_from_file_location(
    "evaluate_robustness", Path(__file__).resolve().parents[1] / "scripts/evaluate_robustness.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.image = Image.fromarray(rng.integers(30, 200, size=(48, 64, 3), dtype=np.uint8))

    def test_clean_is_exact_and_original_is_never_mutated(self):
        original = self.image.tobytes()
        for condition in CONDITIONS:
            self.assertEqual(degrade(self.image, condition, 0, 1).tobytes(), original)
            for severity in (1, 2, 3):
                output = degrade(self.image, condition, severity, 1)
                self.assertEqual(output.size, self.image.size)
                self.assertEqual(output.mode, "RGB")
                self.assertNotEqual(output.tobytes(), original)
        self.assertEqual(self.image.tobytes(), original)

    def test_evaluation_is_independent_of_iteration_order_and_global_rng(self):
        transform = EvaluationDegradation("cloud_haze", 2, 2026)
        before = transform(self.image, "sha-image-a").tobytes()
        transform(self.image, "sha-image-b")
        np.random.seed(999)
        self.assertEqual(before, transform(self.image, "sha-image-a").tobytes())
        self.assertNotEqual(before, transform(self.image, "sha-image-b").tobytes())
        self.assertNotEqual(before, EvaluationDegradation("cloud_haze", 2, 2027)(self.image, "sha-image-a").tobytes())

    def test_increasing_haze_and_noise_have_increasing_effect_on_gray_image(self):
        image = Image.new("RGB", (100, 100), (80, 80, 80))
        for condition in ("cloud_haze", "sensor_noise"):
            changes = [np.mean(np.abs(np.asarray(degrade(image, condition, level, 19), dtype=float) - 80))
                       for level in (1, 2, 3)]
            self.assertTrue(changes[0] < changes[1] < changes[2])

    def test_bad_conditions_and_severities_fail(self):
        for condition, severity in (("storm", 1), ("clean", 2), ("cloud_haze", 0), ("resolution", 4)):
            with self.assertRaises(ValueError):
                EvaluationDegradation(condition, severity)
        with self.assertRaises(ValueError):
            profile_metadata("unknown")

    def test_summary_uses_percentage_points_without_multiplying_sample_count(self):
        clean = {"checkpoint_sha256": "a", "target_manifest_sha256": "b", "samples": 180,
                 "architecture": "resnet18_pretrained", "seed": 13, "augmentation": "baseline",
                 "degradation": {"condition": "clean", "severity": 0, "seed": 2026},
                 "metrics": {"accuracy": .90, "macro_f1": .88, "balanced_accuracy": .90}}
        degraded = {**clean, "degradation": {"condition": "cloud_haze", "severity": 1, "seed": 2026},
                    "metrics": {"accuracy": .75, "macro_f1": .70, "balanced_accuracy": .75}}
        rows = suite.summarize([clean, degraded])
        self.assertAlmostEqual(rows[1]["accuracy_drop_pp"], 15)
        self.assertAlmostEqual(rows[1]["macro_f1_drop_pp"], 18)
        self.assertEqual(rows[1]["samples"], 180)
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            suite.summarize([clean, {**degraded, "checkpoint_sha256": "changed"}])
        with self.assertRaisesRegex(ValueError, "target_manifest_sha256"):
            suite.summarize([clean, {**degraded, "target_manifest_sha256": "changed"}])


if __name__ == "__main__":
    unittest.main()
