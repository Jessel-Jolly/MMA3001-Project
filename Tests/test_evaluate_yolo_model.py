"""
Unit tests for evaluate_yolo_model.py module.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

from evaluate_yolo_model import (
    load_ground_truth_classes,
    evaluate_image_level_accuracy,
)


class MockBox:
    """Mock YOLO bounding box object for testing."""
    def __init__(self, cls_id: int):
        self.cls = [cls_id]


class MockResult:
    """Mock YOLO prediction result object."""
    def __init__(self, path: str, box_classes):
        self.path = path
        self.boxes = [MockBox(cid) for cid in box_classes]


class MockYOLOModel:
    """Mock YOLO model providing .predict() for unit tests."""
    def __init__(self, predictions_dict):
        self.predictions_dict = predictions_dict

    def predict(self, source, conf=0.25, verbose=False):
        results = []
        for img_path, boxes in self.predictions_dict.items():
            results.append(MockResult(img_path, boxes))
        return results


class TestEvaluateYOLOModel(unittest.TestCase):
    """Test suite for YOLO packaging evaluation."""

    def setUp(self):
        """Set up mock test images and labels."""
        self.temp_dir = tempfile.mkdtemp(prefix="test_eval_yolo_")
        self.img_dir = os.path.join(self.temp_dir, "images")
        self.lbl_dir = os.path.join(self.temp_dir, "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_ground_truth_classes(self):
        """Validates class parsing for empty (undamaged) and defect label files."""
        empty_lbl = Path(self.lbl_dir) / "clean.txt"
        with open(empty_lbl, "w") as f:
            f.write("")
        self.assertEqual(load_ground_truth_classes(empty_lbl), {-1})

        defect_lbl = Path(self.lbl_dir) / "defect.txt"
        with open(defect_lbl, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
            f.write("3 0.6 0.6 0.1 0.1\n")
        self.assertEqual(load_ground_truth_classes(defect_lbl), {0, 3})

        missing_lbl = Path(self.lbl_dir) / "missing.txt"
        self.assertEqual(load_ground_truth_classes(missing_lbl), {-1})

    def test_evaluate_image_level_accuracy(self):
        """
        Tests binary and multiclass accuracy calculation on a 4-sample mock dataset:
        1. Clean image correctly predicted as Clean (TN)
        2. Clean image falsely predicted as Loose Meat / Class 0 (FP)
        3. Defect image (Class 1) correctly predicted as Class 1 (TP)
        4. Defect image (Class 2) missed / predicted as Clean (FN)
        """
        # Create 4 dummy image files
        names = ["sample_tn.jpg", "sample_fp.jpg", "sample_tp.jpg", "sample_fn.jpg"]
        for name in names:
            p = os.path.join(self.img_dir, name)
            Image.new("RGB", (64, 64), color="white").save(p)

        # Create 4 corresponding label files
        # 1. TN: Clean
        with open(os.path.join(self.lbl_dir, "sample_tn.txt"), "w") as f:
            f.write("")
        # 2. FP: Clean
        with open(os.path.join(self.lbl_dir, "sample_fp.txt"), "w") as f:
            f.write("")
        # 3. TP: Class 1
        with open(os.path.join(self.lbl_dir, "sample_tp.txt"), "w") as f:
            f.write("1 0.5 0.5 0.3 0.3\n")
        # 4. FN: Class 2
        with open(os.path.join(self.lbl_dir, "sample_fn.txt"), "w") as f:
            f.write("2 0.4 0.4 0.2 0.2\n")

        # Mock predictions
        predictions = {
            os.path.join(self.img_dir, "sample_tn.jpg"): [],      # Clean -> TN
            os.path.join(self.img_dir, "sample_fp.jpg"): [0],     # Loose Meat -> FP
            os.path.join(self.img_dir, "sample_tp.jpg"): [1],     # Class 1 -> TP
            os.path.join(self.img_dir, "sample_fn.jpg"): [],      # Clean -> FN
        }
        mock_model = MockYOLOModel(predictions)

        results = evaluate_image_level_accuracy(
            model=mock_model,
            test_img_dir=self.img_dir,
            test_lbl_dir=self.lbl_dir,
            verbose=False
        )

        self.assertIn("binary_metrics", results)
        b_metrics = results["binary_metrics"]

        # Expected: 2 correct out of 4 (TN and TP) = 50%
        self.assertEqual(b_metrics["total_images"] if "total_images" in b_metrics else results["total_images"], 4)
        self.assertAlmostEqual(b_metrics["accuracy"], 50.0)
        self.assertEqual(b_metrics["true_positives"], 1)
        self.assertEqual(b_metrics["false_positives"], 1)
        self.assertEqual(b_metrics["true_negatives"], 1)
        self.assertEqual(b_metrics["false_negatives"], 1)

        # Precision = 1 / (1 + 1) = 50%
        self.assertAlmostEqual(b_metrics["precision"], 50.0)
        # Recall = 1 / (1 + 1) = 50%
        self.assertAlmostEqual(b_metrics["recall"], 50.0)
        # F1 = 50%
        self.assertAlmostEqual(b_metrics["f1"], 50.0)

        # Multiclass check: sample_tn matched, sample_tp matched -> 2/4 = 50%
        mc_metrics = results["multiclass_metrics"]
        self.assertAlmostEqual(mc_metrics["primary_accuracy"], 50.0)


if __name__ == "__main__":
    unittest.main()
