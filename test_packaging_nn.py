"""
Unit test suite for packaging damage classification system.

This module provides unit tests for:
- YOLO label parsing (empty vs non-empty annotation files)
- Image feature extraction & normalization
- Dataset split loading
- Scikit-learn MLP pipeline configuration
- Training, evaluation, and single-image inference

MMA3001 Project Brief Alignment:
- Compatible with pytest and standard unittest.
- Validates code correctness, robustness, and handling of edge cases.
"""

import os
import tempfile
import unittest
import numpy as np
from PIL import Image

from dataset import parse_label_file, extract_image_features, load_dataset_split
from model import build_mlp_pipeline, train_and_evaluate, predict_packaging_damage


class TestPackagingDamageNN(unittest.TestCase):
    """Unit tests for the packaging damage classification dataset and MLP model pipeline."""

    def setUp(self):
        """Creates a temporary dataset folder with sample undamaged and damaged image/label pairs."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = self.tmp_dir.name

        self.train_img_dir = os.path.join(self.dataset_dir, "train", "images")
        self.train_lbl_dir = os.path.join(self.dataset_dir, "train", "labels")
        os.makedirs(self.train_img_dir, exist_ok=True)
        os.makedirs(self.train_lbl_dir, exist_ok=True)

        # 1. Create Undamaged Sample (empty txt file)
        self.img1_path = os.path.join(self.train_img_dir, "sample_undamaged.jpg")
        self.lbl1_path = os.path.join(self.train_lbl_dir, "sample_undamaged.txt")
        img1 = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 200)
        img1.save(self.img1_path)
        with open(self.lbl1_path, "w", encoding="utf-8") as f:
            f.write("")  # Empty txt = Undamaged (0)

        # 2. Create Damaged Sample (non-empty txt file with YOLO bounding box)
        self.img2_path = os.path.join(self.train_img_dir, "sample_damaged.jpg")
        self.lbl2_path = os.path.join(self.train_lbl_dir, "sample_damaged.txt")
        img2 = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 50)
        img2.save(self.img2_path)
        with open(self.lbl2_path, "w", encoding="utf-8") as f:
            f.write("1 0.5 0.5 0.2 0.3\n")  # Non-empty txt = Damaged (1)

    def tearDown(self):
        """Cleans up the temporary directory after tests complete."""
        self.tmp_dir.cleanup()

    def test_parse_label_file_empty_and_nonempty(self):
        """Test label file parsing logic for empty (undamaged) and non-empty (damaged) txt files."""
        self.assertEqual(parse_label_file(self.lbl1_path), 0)
        self.assertEqual(parse_label_file(self.lbl2_path), 1)

        missing_lbl = os.path.join(self.train_lbl_dir, "non_existent.txt")
        self.assertEqual(parse_label_file(missing_lbl), 0)

    def test_extract_image_features(self):
        """Test image feature extraction shape and normalization bounds [0.0, 1.0]."""
        image_size = (16, 16)
        features = extract_image_features(self.img1_path, image_size=image_size)

        expected_dim = 16 * 16 * 3
        self.assertEqual(features.shape, (expected_dim,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(features >= 0.0) and np.all(features <= 1.0))

    def test_load_dataset_split(self):
        """Test dataset loading split across images and labels."""
        X, y = load_dataset_split(self.dataset_dir, split="train", image_size=(16, 16))

        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)
        self.assertEqual(set(y), {0, 1})

    def test_build_mlp_pipeline(self):
        """Test creation and configuration of the Scikit-Learn MLP pipeline."""
        pipeline = build_mlp_pipeline(hidden_layer_sizes=(10, 5), alpha=0.01)

        self.assertIn("standardscaler", pipeline.named_steps)
        self.assertIn("mlpclassifier", pipeline.named_steps)

        mlp_step = pipeline.named_steps["mlpclassifier"]
        self.assertEqual(mlp_step.hidden_layer_sizes, (10, 5))
        self.assertEqual(mlp_step.alpha, 0.01)
        self.assertTrue(mlp_step.early_stopping)

    def test_train_and_evaluate(self):
        """Test pipeline training and validation metrics evaluation."""
        np.random.seed(42)
        X_train = np.random.randn(40, 48).astype(np.float32)
        y_train = np.random.randint(0, 2, size=40)
        X_val = np.random.randn(10, 48).astype(np.float32)
        y_val = np.random.randint(0, 2, size=10)

        pipeline = build_mlp_pipeline(hidden_layer_sizes=(10,), max_iter=50, random_state=42)
        metrics = train_and_evaluate(pipeline, X_train, y_train, X_val, y_val)

        required_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"}
        self.assertTrue(required_keys.issubset(metrics.keys()))
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)

    def test_predict_packaging_damage(self):
        """Test single-image damage classification inference."""
        X_train = np.random.rand(20, 16 * 16 * 3).astype(np.float32)
        y_train = np.array([0, 1] * 10)

        pipeline = build_mlp_pipeline(hidden_layer_sizes=(10,), max_iter=50)
        pipeline.fit(X_train, y_train)

        result = predict_packaging_damage(pipeline, self.img2_path, image_size=(16, 16))

        self.assertIn("is_damaged", result)
        self.assertIn("classification_label", result)
        self.assertIn("damage_probability", result)
        self.assertIsInstance(result["is_damaged"], bool)
        self.assertIn(result["classification_label"], ["Damaged Packaging", "Undamaged Packaging"])
        self.assertTrue(0.0 <= result["damage_probability"] <= 1.0)

    def test_missing_file_exceptions(self):
        """Test error handling when passing non-existent image path."""
        with self.assertRaises(FileNotFoundError):
            extract_image_features("non_existent_image.jpg")


if __name__ == "__main__":
    unittest.main()
