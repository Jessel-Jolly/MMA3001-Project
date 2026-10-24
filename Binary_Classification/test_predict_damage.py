"""
Unit tests for the packaging damage inference script (predict_damage.py).

MMA3001 Project Brief Alignment:
- Validates executable computational workflow and inference pipeline.
- Tests edge cases (missing model file, missing image file, unsupported extensions).
- Confirms output types, probability boundaries [0.0, 1.0], and batch processing consistency.
- Compatible with both standard unittest and pytest.
"""

import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image
import joblib

# Ensure current directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_mlp_pipeline
from predict_damage import (
    load_binary_model,
    predict_image,
    predict_directory,
    format_prediction_result,
)


class TestPredictDamage(unittest.TestCase):
    """
    Test suite for packaging damage model inference.
    """

    @classmethod
    def setUpClass(cls):
        """
        Creates a temporary directory, a dummy fitted model, and sample dummy images.
        """
        cls.temp_dir = tempfile.mkdtemp(prefix="test_predict_damage_")

        # Create dummy training data & train a miniature pipeline
        X_dummy = np.random.rand(10, 16 * 16 * 3).astype(np.float32)
        y_dummy = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        cls.pipeline = build_mlp_pipeline(
            hidden_layer_sizes=(8,),
            alpha=0.01,
            max_iter=30,
            random_state=42,
            early_stopping=False
        )
        cls.pipeline.fit(X_dummy, y_dummy)

        # Save dummy model to disk
        cls.model_path = os.path.join(cls.temp_dir, "binary_model.joblib")
        joblib.dump(cls.pipeline, cls.model_path)

        # Create a sample query image
        cls.sample_image_path = os.path.join(cls.temp_dir, "sample_pack.jpg")
        img_arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        img.save(cls.sample_image_path, format="JPEG")

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up temporary directory and files.
        """
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_load_binary_model_success(self):
        """Tests that a saved model is successfully loaded from disk."""
        loaded = load_binary_model(self.model_path)
        self.assertIsNotNone(loaded)
        self.assertTrue(hasattr(loaded, "predict"))
        self.assertTrue(hasattr(loaded, "predict_proba"))

    def test_load_binary_model_missing_file(self):
        """Tests that loading a non-existent model file raises FileNotFoundError."""
        non_existent = os.path.join(self.temp_dir, "ghost_model.joblib")
        with self.assertRaises(FileNotFoundError):
            load_binary_model(non_existent)

    def test_predict_single_image_with_pipeline(self):
        """Tests single image prediction passing the pipeline instance directly."""
        result = predict_image(
            self.sample_image_path,
            model=self.pipeline,
            image_size=(16, 16)
        )

        self.assertIn("image_path", result)
        self.assertIn("is_damaged", result)
        self.assertIn("classification_label", result)
        self.assertIn("damage_probability", result)
        self.assertIn("undamaged_probability", result)
        self.assertIn("confidence", result)

        self.assertIsInstance(result["is_damaged"], bool)
        self.assertIn(result["classification_label"], ["Damaged Packaging", "Undamaged Packaging"])
        self.assertGreaterEqual(result["damage_probability"], 0.0)
        self.assertLessEqual(result["damage_probability"], 1.0)
        self.assertGreaterEqual(result["undamaged_probability"], 0.0)
        self.assertLessEqual(result["undamaged_probability"], 1.0)
        self.assertGreaterEqual(result["confidence"], 0.5)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_predict_single_image_with_model_path_str(self):
        """Tests single image prediction resolving model from file path string."""
        result = predict_image(
            self.sample_image_path,
            model=self.model_path,
            image_size=(16, 16)
        )
        self.assertIsInstance(result["is_damaged"], bool)
        self.assertEqual(os.path.abspath(self.sample_image_path), result["image_path"])

    def test_predict_missing_image_raises_error(self):
        """Tests that predicting on a non-existent image file raises FileNotFoundError."""
        fake_path = os.path.join(self.temp_dir, "does_not_exist.jpg")
        with self.assertRaises(FileNotFoundError):
            predict_image(fake_path, model=self.pipeline, image_size=(16, 16))

    def test_predict_directory_batch(self):
        """Tests batch processing on a directory of mixed images and non-image files."""
        batch_dir = os.path.join(self.temp_dir, "batch_test")
        os.makedirs(batch_dir, exist_ok=True)

        # Create 2 JPEGs, 1 PNG, and 1 non-image text file
        img1 = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
        img1.save(os.path.join(batch_dir, "pack_1.jpg"))

        img2 = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
        img2.save(os.path.join(batch_dir, "pack_2.jpeg"))

        img3 = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
        img3.save(os.path.join(batch_dir, "pack_3.png"))

        with open(os.path.join(batch_dir, "notes.txt"), "w") as f:
            f.write("Some notes about meat packaging")

        results = predict_directory(batch_dir, model=self.pipeline, image_size=(16, 16))

        # Only 3 image files should be processed
        self.assertEqual(len(results), 3)
        for res in results:
            self.assertIn("is_damaged", res)
            self.assertIsInstance(res["is_damaged"], bool)
            self.assertFalse(res["image_path"].endswith(".txt"))

    def test_format_prediction_result(self):
        """Tests the formatted CLI text block generation."""
        mock_res = {
            "image_path": "/path/to/pork_chop_01.jpg",
            "is_damaged": True,
            "classification_label": "Damaged Packaging",
            "damage_probability": 0.9421,
            "undamaged_probability": 0.0579,
            "confidence": 0.9421,
        }
        formatted = format_prediction_result(mock_res)
        self.assertIn("pork_chop_01.jpg", formatted)
        self.assertIn("[DAMAGED]", formatted)
        self.assertIn("94.2%", formatted)


if __name__ == "__main__":
    unittest.main()
