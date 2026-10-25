"""
Unit tests for the YOLO defect inference script (inference.py).
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inference
from inference import (
    load_yolo_model,
    predict_single_image,
    predict_images_batch,
    format_cli_summary,
)


class TestYOLOInference(unittest.TestCase):
    """Test suite for YOLOv11 defect detection inference module."""

    @classmethod
    def setUpClass(cls):
        """Creates a temporary workspace with dummy test images."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_yolo_infer_")
        cls.img_dir = os.path.join(cls.temp_dir, "images")
        os.makedirs(cls.img_dir, exist_ok=True)

        # Create two sample dummy images
        cls.sample_img1 = os.path.join(cls.img_dir, "pack_01.jpg")
        img1 = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        img1.save(cls.sample_img1)

        cls.sample_img2 = os.path.join(cls.img_dir, "pack_02.png")
        img2 = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        img2.save(cls.sample_img2)

        # Non-image file to test extension filtering
        with open(os.path.join(cls.img_dir, "ignore.txt"), "w") as f:
            f.write("text content")

    @classmethod
    def tearDownClass(cls):
        """Cleans up temporary directory after test execution."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_load_yolo_model_missing_file_raises_error(self):
        """Tests that loading a non-existent weights file raises FileNotFoundError."""
        with patch.object(inference, "YOLO", MagicMock()):
            fake_weights = os.path.join(self.temp_dir, "non_existent.pt")
            with self.assertRaises(FileNotFoundError):
                load_yolo_model(fake_weights)

    def test_predict_single_image_missing_image_raises_error(self):
        """Tests that predicting on a non-existent image raises FileNotFoundError."""
        mock_model = MagicMock()
        fake_img = os.path.join(self.temp_dir, "ghost.jpg")
        with self.assertRaises(FileNotFoundError):
            predict_single_image(mock_model, fake_img)

    def test_predict_single_image_damaged_detection(self):
        """Tests single image prediction when defect bounding boxes are detected."""
        mock_box = MagicMock()
        mock_box.cls = [0]
        mock_box.conf = [0.915]
        mock_box.xyxy = [[12.5, 20.0, 55.0, 80.5]]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.plot.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        annotated_out = os.path.join(self.temp_dir, "annotated_pack_01.jpg")
        pred = predict_single_image(
            model=mock_model,
            image_path=self.sample_img1,
            conf_thresh=0.25,
            save_annotated_path=annotated_out
        )

        self.assertTrue(pred["is_damaged"])
        self.assertEqual(pred["classification_label"], "Damaged Packaging")
        self.assertEqual(pred["num_defects"], 1)
        self.assertEqual(len(pred["detections"]), 1)
        self.assertEqual(pred["detections"][0]["class_name"], "Loose Meat")
        self.assertEqual(pred["detections"][0]["confidence"], 0.915)
        self.assertTrue(os.path.exists(annotated_out))

    def test_predict_single_image_undamaged_detection(self):
        """Tests single image prediction when no bounding boxes are detected (intact)."""
        mock_result = MagicMock()
        mock_result.boxes = []

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        pred = predict_single_image(
            model=mock_model,
            image_path=self.sample_img2,
            conf_thresh=0.25
        )

        self.assertFalse(pred["is_damaged"])
        self.assertEqual(pred["classification_label"], "Undamaged Packaging")
        self.assertEqual(pred["num_defects"], 0)
        self.assertEqual(len(pred["detections"]), 0)

    def test_predict_images_batch_directory_and_json_export(self):
        """Tests batch directory processing and structured JSON export."""
        mock_result = MagicMock()
        mock_result.boxes = []

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        json_output_path = os.path.join(self.temp_dir, "batch_output.json")
        annotated_dir = os.path.join(self.temp_dir, "annotated_batch")

        results = predict_images_batch(
            model=mock_model,
            target_path=self.img_dir,
            conf_thresh=0.25,
            save_dir=annotated_dir,
            output_json=json_output_path
        )

        # Exactly 2 image files should be processed, ignoring ignore.txt
        self.assertEqual(len(results), 2)
        self.assertTrue(os.path.exists(json_output_path))

        with open(json_output_path, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)

    def test_predict_images_batch_empty_folder(self):
        """Tests batch processing on an empty folder without errors."""
        empty_dir = os.path.join(self.temp_dir, "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)

        mock_model = MagicMock()
        results = predict_images_batch(mock_model, empty_dir)
        self.assertEqual(len(results), 0)

    def test_format_cli_summary(self):
        """Tests terminal summary formatting for damaged and undamaged records."""
        damaged_res = {
            "image_path": "/path/to/pork_chop.jpg",
            "is_damaged": True,
            "num_defects": 1,
            "detections": [{"class_name": "Wrinkle", "confidence": 0.88}],
        }
        undamaged_res = {
            "image_path": "/path/to/pork_clean.jpg",
            "is_damaged": False,
            "num_defects": 0,
            "detections": [],
        }

        damaged_str = format_cli_summary(damaged_res)
        undamaged_str = format_cli_summary(undamaged_res)

        self.assertIn("[DAMAGED]", damaged_str)
        self.assertIn("Wrinkle (88.0%)", damaged_str)
        self.assertIn("[OK/UNDAMAGED]", undamaged_str)
        self.assertIn("Packaging intact", undamaged_str)


if __name__ == "__main__":
    unittest.main()
