"""
Unit tests for visualisation.py module.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

from visualisation import (
    get_all_image_pairs,
    parse_yolo_labels,
    plot_sample_images,
)


class TestVisualisation(unittest.TestCase):
    """Test suite for YOLO bounding box parsing and visualization."""

    def setUp(self):
        """Create mock image and label files."""
        self.temp_dir = tempfile.mkdtemp(prefix="test_vis_")
        self.img_dir = os.path.join(self.temp_dir, "train", "images")
        self.lbl_dir = os.path.join(self.temp_dir, "train", "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

        # Create dummy image (100x200)
        self.img_path = os.path.join(self.img_dir, "test_sample.jpg")
        img_arr = (np.random.rand(200, 100, 3) * 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        img.save(self.img_path)

        # Create dummy label
        self.lbl_path = os.path.join(self.lbl_dir, "test_sample.txt")
        # Center x=0.5, y=0.5, w=0.4, h=0.2 on 100x200 image
        # x_min = 50 - 20 = 30, y_min = 100 - 20 = 80, w = 40, h = 40
        with open(self.lbl_path, "w") as f:
            f.write("1 0.5 0.5 0.4 0.2\n")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_all_image_pairs(self):
        """Pairs should correctly associate images with matching label txt files."""
        pairs = get_all_image_pairs(dataset_path=self.temp_dir, split="train")
        self.assertEqual(len(pairs), 1)
        img_p, lbl_p, split = pairs[0]
        self.assertEqual(img_p.name, "test_sample.jpg")
        self.assertEqual(lbl_p.name, "test_sample.txt")
        self.assertEqual(split, "train")

    def test_parse_yolo_labels_valid(self):
        """Normalized YOLO format should correctly de-normalize to pixel coordinates."""
        boxes = parse_yolo_labels(self.lbl_path, img_width=100, img_height=200)
        self.assertEqual(len(boxes), 1)
        box = boxes[0]
        self.assertEqual(box["class_id"], 1)

        x_min, y_min, w, h = box["bbox"]
        self.assertAlmostEqual(x_min, 30.0)
        self.assertAlmostEqual(y_min, 80.0)
        self.assertAlmostEqual(w, 40.0)
        self.assertAlmostEqual(h, 40.0)

    def test_parse_yolo_labels_empty_or_missing(self):
        """Missing or empty files should return empty box lists."""
        missing_file = Path(self.temp_dir) / "missing.txt"
        self.assertEqual(parse_yolo_labels(missing_file, 100, 100), [])

        empty_file = Path(self.temp_dir) / "empty.txt"
        with open(empty_file, "w") as f:
            f.write("")
        self.assertEqual(parse_yolo_labels(empty_file, 100, 100), [])

    def test_plot_sample_images_headless(self):
        """plot_sample_images should execute cleanly with show=False and save figure."""
        save_file = os.path.join(self.temp_dir, "output_plot.png")
        result = plot_sample_images(
            n=1,
            split="train",
            dataset_path=self.temp_dir,
            save_path=save_file,
            show=False
        )
        self.assertIsNotNone(result)
        fig, axes = result
        self.assertTrue(os.path.exists(save_file))
        self.assertGreater(os.path.getsize(save_file), 0)


if __name__ == "__main__":
    unittest.main()
