"""
Unit test suite for packaging damage inference engine (v2).

Validates:
- Automatic image size inference directly from the trained model pipeline.
- Single image inference without requiring manual image_size inputs.
- Input file handling for both direct image files and text/CSV files listing image paths.
- Directory batch inference without requiring image_size.
- Unified predict() dispatcher across images, list files, and directories.
- Edge case handling (missing files, unsupported extensions).

Compatible with both pytest and python -m unittest.
"""

import os
import sys
import shutil
import tempfile
import unittest
import importlib.util
import numpy as np
from PIL import Image
import joblib

# Ensure current scratch directory is on path
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _cur_dir)

from model import build_mlp_pipeline

# Robust dynamic import to handle hyphenated file names like predict_damage-v2.py
_pred_v2_path = os.path.join(_cur_dir, "predict_damage-v2.py")
if not os.path.exists(_pred_v2_path):
    _pred_v2_path = os.path.join(_cur_dir, "predict_damage.py")

_spec = importlib.util.spec_from_file_location("predict_damage_v2", _pred_v2_path)
_pred_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pred_mod)

load_binary_model = _pred_mod.load_binary_model
infer_image_size_from_model = _pred_mod.infer_image_size_from_model
predict_image = _pred_mod.predict_image
predict_file = _pred_mod.predict_file
predict_directory = _pred_mod.predict_directory
predict = _pred_mod.predict
format_prediction_result = _pred_mod.format_prediction_result


class TestPredictDamageV2(unittest.TestCase):
    """
    Test suite for v2 prediction module.
    """

    @classmethod
    def setUpClass(cls):
        """
        Creates a temporary working directory, a trained miniature model, and sample test files.
        """
        cls.temp_dir = tempfile.mkdtemp(prefix="test_predict_damage_v2_")

        # 16x16 RGB image -> 768 features
        cls.test_side = 16
        cls.n_features = cls.test_side * cls.test_side * 3

        X_dummy = np.random.rand(10, cls.n_features).astype(np.float32)
        y_dummy = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        cls.pipeline = build_mlp_pipeline(
            hidden_layer_sizes=(8,),
            alpha=0.01,
            max_iter=30,
            random_state=42,
            early_stopping=False
        )
        cls.pipeline.fit(X_dummy, y_dummy)

        # Save dummy model
        cls.model_path = os.path.join(cls.temp_dir, "binary_model.joblib")
        joblib.dump(cls.pipeline, cls.model_path)

        # Create sample test images
        cls.img1_path = os.path.join(cls.temp_dir, "pack_01.jpg")
        img1 = Image.fromarray((np.random.rand(48, 48, 3) * 255).astype(np.uint8))
        img1.save(cls.img1_path, format="JPEG")

        cls.img2_path = os.path.join(cls.temp_dir, "pack_02.png")
        img2 = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        img2.save(cls.img2_path, format="PNG")

        # Create a text file listing images
        cls.list_file_path = os.path.join(cls.temp_dir, "image_manifest.txt")
        with open(cls.list_file_path, "w", encoding="utf-8") as f:
            f.write(f"# Manifest list\n{cls.img1_path}\n{cls.img2_path}\n")

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up temporary directory.
        """
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_infer_image_size_from_model(self):
        """Tests that spatial dimensions (16, 16) are correctly inferred from model features."""
        inferred_size = infer_image_size_from_model(self.pipeline)
        self.assertEqual(inferred_size, (16, 16))

        # Also test with model path string
        inferred_from_str = infer_image_size_from_model(self.model_path)
        self.assertEqual(inferred_from_str, (16, 16))

    def test_load_binary_model(self):
        """Tests model loading and missing file exception."""
        loaded = load_binary_model(self.model_path)
        self.assertIsNotNone(loaded)
        self.assertTrue(hasattr(loaded, "predict"))

        with self.assertRaises(FileNotFoundError):
            load_binary_model(os.path.join(self.temp_dir, "non_existent.joblib"))

    def test_predict_image_without_image_size(self):
        """Tests that predict_image runs successfully without providing image_size."""
        result = predict_image(self.img1_path, model=self.pipeline)

        self.assertIn("image_path", result)
        self.assertIn("is_damaged", result)
        self.assertIn("classification_label", result)
        self.assertIn("damage_probability", result)
        self.assertIn("undamaged_probability", result)
        self.assertIn("confidence", result)
        self.assertIn("image_size_used", result)

        self.assertEqual(result["image_size_used"], [16, 16])
        self.assertIsInstance(result["is_damaged"], bool)
        self.assertIn(result["classification_label"], ["Damaged Packaging", "Undamaged Packaging"])
        self.assertGreaterEqual(result["damage_probability"], 0.0)
        self.assertLessEqual(result["damage_probability"], 1.0)
        self.assertGreaterEqual(result["confidence"], 0.5)

    def test_predict_file_with_image(self):
        """Tests predict_file when given a direct image file without image_size."""
        result = predict_file(self.img1_path, model=self.model_path)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["image_path"], os.path.abspath(self.img1_path))
        self.assertEqual(result["image_size_used"], [16, 16])

    def test_predict_file_with_text_list(self):
        """Tests predict_file when given a .txt file containing image paths."""
        results = predict_file(self.list_file_path, model=self.pipeline)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("is_damaged", r)
            self.assertEqual(r["image_size_used"], [16, 16])

    def test_predict_directory_without_image_size(self):
        """Tests predict_directory without passing image_size."""
        results = predict_directory(self.temp_dir, model=self.pipeline)
        # Should process img1 and img2, ignoring .txt and .joblib files
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("is_damaged", r)
            self.assertEqual(r["image_size_used"], [16, 16])

    def test_predict_unified_dispatcher(self):
        """Tests the unified predict() entry point on image, text list, and directory."""
        # 1. Image
        res_img = predict(self.img1_path, model=self.pipeline)
        self.assertIsInstance(res_img, dict)

        # 2. Text list file
        res_list = predict(self.list_file_path, model=self.pipeline)
        self.assertIsInstance(res_list, list)
        self.assertEqual(len(res_list), 2)

        # 3. Directory
        res_dir = predict(self.temp_dir, model=self.pipeline)
        self.assertIsInstance(res_dir, list)
        self.assertEqual(len(res_dir), 2)

    def test_missing_target_raises_error(self):
        """Tests that passing a non-existent target raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            predict(os.path.join(self.temp_dir, "ghost_image.jpg"), model=self.pipeline)


if __name__ == "__main__":
    unittest.main()
