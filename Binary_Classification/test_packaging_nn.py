"""
Unit tests for packaging neural network dataset, model pipeline, and evaluation workflow.

MMA3001 Project Brief Alignment:
- Validates code correctness and reproducible software engineering standards.
- Tests label parsing, feature extraction, scaling pipeline, side-by-side evaluation,
  model persistence via joblib, and output results folder creation.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

# Robust import of dataset module
try:
    from dataset import parse_label_file, extract_image_features, load_dataset_split
except ImportError:
    import importlib.util
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    _ds_path = os.path.join(_cur_dir, "dataset.py")
    if os.path.exists(_ds_path):
        _spec = importlib.util.spec_from_file_location("dataset", _ds_path)
        _ds_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ds_mod)
        parse_label_file = _ds_mod.parse_label_file
        extract_image_features = _ds_mod.extract_image_features
        load_dataset_split = _ds_mod.load_dataset_split
    else:
        raise ImportError("Could not find dataset.py module.")

# Robust import of model module
try:
    from model import (
        build_mlp_pipeline,
        evaluate_model,
        train_and_evaluate,
        save_model,
        load_model,
        predict_packaging_damage,
    )
except ImportError:
    import importlib.util
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = ["model-v2.py", "model_v2.py", "model.py"]
    _m_mod = None
    for cand in _candidates:
        cand_path = os.path.join(_cur_dir, cand)
        if os.path.exists(cand_path):
            _spec = importlib.util.spec_from_file_location("model", cand_path)
            _m_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_m_mod)
            break
    if _m_mod is not None:
        build_mlp_pipeline = _m_mod.build_mlp_pipeline
        evaluate_model = _m_mod.evaluate_model
        train_and_evaluate = _m_mod.train_and_evaluate
        save_model = _m_mod.save_model
        load_model = _m_mod.load_model
        predict_packaging_damage = _m_mod.predict_packaging_damage
    else:
        raise ImportError("Could not import model components.")


class TestPackagingNeuralNetwork(unittest.TestCase):
    """Test suite for dataset parsing and Scikit-Learn MLP classification pipeline."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_parse_label_file_empty_undamaged(self):
        """Empty label text file must be parsed as class 0 (Undamaged)."""
        empty_label = os.path.join(self.test_dir, "empty.txt")
        with open(empty_label, "w") as f:
            f.write("")
        self.assertEqual(parse_label_file(empty_label), 0)

    def test_parse_label_file_non_empty_damaged(self):
        """Non-empty label text file with bounding boxes must be parsed as class 1 (Damaged)."""
        damaged_label = os.path.join(self.test_dir, "damaged.txt")
        with open(damaged_label, "w") as f:
            f.write("0 0.512 0.498 0.120 0.250\n")
        self.assertEqual(parse_label_file(damaged_label), 1)

    def test_extract_image_features_shape_and_bounds(self):
        """Image feature vector must match target flattened dimension and fall within [0.0, 1.0]."""
        img_path = os.path.join(self.test_dir, "sample.jpg")
        img = Image.new("RGB", (100, 100), color=(128, 64, 32))
        img.save(img_path)

        features = extract_image_features(img_path, image_size=(16, 16))
        self.assertEqual(features.shape, (16 * 16 * 3,))
        self.assertTrue(np.all(features >= 0.0) and np.all(features <= 1.0))

    def test_pipeline_architecture(self):
        """Pipeline should contain StandardScaler and MLPClassifier with specified parameters."""
        pipeline = build_mlp_pipeline(hidden_layer_sizes=(20, 10), alpha=0.05)
        self.assertIn("standardscaler", pipeline.named_steps)
        self.assertIn("mlpclassifier", pipeline.named_steps)
        mlp = pipeline.named_steps["mlpclassifier"]
        self.assertEqual(mlp.hidden_layer_sizes, (20, 10))
        self.assertEqual(mlp.alpha, 0.05)

    def test_train_and_evaluate_with_test_split(self):
        """Pipeline must evaluate both validation and withheld test splits properly."""
        np.random.seed(42)
        X_train = np.random.rand(40, 12).astype(np.float32)
        y_train = np.array([0, 1] * 20)

        X_val = np.random.rand(10, 12).astype(np.float32)
        y_val = np.array([0, 1] * 5)

        X_test = np.random.rand(10, 12).astype(np.float32)
        y_test = np.array([0, 1] * 5)

        pipeline = build_mlp_pipeline(hidden_layer_sizes=(10,), max_iter=30, random_state=42)
        results = train_and_evaluate(pipeline, X_train, y_train, X_val, y_val, X_test, y_test)

        self.assertIn("validation", results)
        self.assertIn("test", results)
        for split_key in ["validation", "test"]:
            self.assertIn("accuracy", results[split_key])
            self.assertIn("precision", results[split_key])
            self.assertIn("recall", results[split_key])
            self.assertIn("f1_score", results[split_key])
            self.assertIn("confusion_matrix", results[split_key])

    def test_save_and_load_model(self):
        """Verifies model saving to Binary_Classification/results folder and reloading with joblib."""
        np.random.seed(42)
        X_train = np.random.rand(30, 8).astype(np.float32)
        y_train = np.array([0, 1] * 15)

        pipeline = build_mlp_pipeline(hidden_layer_sizes=(8,), max_iter=20, random_state=42)
        pipeline.fit(X_train, y_train)

        save_path = os.path.join(self.test_dir, "Binary_Classification", "results", "binary_model.joblib")
        saved_file = save_model(pipeline, save_path)
        self.assertTrue(os.path.exists(saved_file))

        loaded_pipeline = load_model(saved_file)
        preds_orig = pipeline.predict(X_train[:5])
        preds_loaded = loaded_pipeline.predict(X_train[:5])
        np.testing.assert_array_equal(preds_orig, preds_loaded)


if __name__ == "__main__":
    unittest.main()
