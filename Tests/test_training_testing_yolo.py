"""
Unit tests for YOLO training and inferencing modules (training.py, inference.py).
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from training import train_yolo_model
from testing import test_yolo_model


class TestYOLOTrainingTesting(unittest.TestCase):
    """Test suite for YOLO train and test function interfaces."""

    def setUp(self):
        """Set up temporary directory and mock config files."""
        self.temp_dir = tempfile.mkdtemp(prefix="test_yolo_train_test_")
        self.yaml_path = os.path.join(self.temp_dir, "data.yaml")
        with open(self.yaml_path, "w") as f:
            f.write("train: train/images\nval: valid/images\ntest: test/images\n")

        self.weights_path = os.path.join(self.temp_dir, "best.pt")
        with open(self.weights_path, "w") as f:
            f.write("dummy weights content")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_train_missing_yaml_raises_error(self):
        """train_yolo_model should raise FileNotFoundError if data.yaml is missing."""
        missing_yaml = os.path.join(self.temp_dir, "non_existent.yaml")
        with self.assertRaises(FileNotFoundError):
            train_yolo_model(data_yaml=missing_yaml)

    def test_test_missing_weights_raises_error(self):
        """test_yolo_model should raise FileNotFoundError if weights do not exist."""
        missing_weights = os.path.join(self.temp_dir, "ghost.pt")
        with self.assertRaises(FileNotFoundError):
            test_yolo_model(weights_path=missing_weights, data_yaml=self.yaml_path)

    def test_train_yolo_model_mocked_execution(self):
        """train_yolo_model should instantiate YOLO and call .train() with correct arguments."""
        mock_ultralytics = MagicMock()
        mock_model_instance = MagicMock()
        mock_results = MagicMock()
        mock_model_instance.train.return_value = mock_results
        mock_ultralytics.YOLO.return_value = mock_model_instance

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            model, results = train_yolo_model(
                data_yaml=self.yaml_path,
                model_name="yolo11n.pt",
                epochs=5,
                imgsz=320,
                batch=8,
                project_name="unit_test_run"
            )

            mock_ultralytics.YOLO.assert_called_once_with("yolo11n.pt")
            mock_model_instance.train.assert_called_once()
            call_kwargs = mock_model_instance.train.call_args[1]
            self.assertEqual(call_kwargs["epochs"], 5)
            self.assertEqual(call_kwargs["imgsz"], 320)
            self.assertEqual(call_kwargs["batch"], 8)
            self.assertEqual(call_kwargs["name"], "unit_test_run")
            self.assertEqual(results, mock_results)

    def test_test_yolo_model_mocked_execution(self):
        """test_yolo_model should instantiate YOLO and call .val() with correct arguments."""
        mock_ultralytics = MagicMock()
        mock_model_instance = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.box.ap50 = [0.85, 0.90]
        mock_metrics.save_dir = "runs/detect/val_test"
        mock_model_instance.val.return_value = mock_metrics
        mock_ultralytics.YOLO.return_value = mock_model_instance

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            metrics = test_yolo_model(
                weights_path=self.weights_path,
                data_yaml=self.yaml_path,
                split="test"
            )

            mock_ultralytics.YOLO.assert_called_once_with(str(self.weights_path))
            mock_model_instance.val.assert_called_once_with(data=str(self.yaml_path), split="test")
            self.assertEqual(metrics, mock_metrics)


if __name__ == "__main__":
    unittest.main()
