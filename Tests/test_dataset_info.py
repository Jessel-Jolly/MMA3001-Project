"""
Unit tests for dataset_info.py module.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from dataset_info import get_instances, summarize_dataset


class TestDatasetInfo(unittest.TestCase):
    """Test suite for YOLO dataset label instance counting."""

    def setUp(self):
        """Create temporary mock dataset structure."""
        self.temp_dir = tempfile.mkdtemp(prefix="test_dataset_info_")
        self.labels_dir = os.path.join(self.temp_dir, "train", "labels")
        os.makedirs(self.labels_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_empty_labels_counted_as_undamaged(self):
        """Empty label text files should be counted as undamaged (last index)."""
        clean_file = os.path.join(self.labels_dir, "clean_01.txt")
        with open(clean_file, "w") as f:
            f.write("")  # empty file = undamaged

        result = get_instances(
            numbers=[0, 1, 2, 3, 4],
            file_split="train",
            dataset_dir=self.temp_dir,
            verbose=False
        )

        self.assertEqual(result["total_images"], 1)
        # Class instances 0..4 should be 0, undamaged (index -1) should be 1
        self.assertEqual(result["instance_counts"][-1], 1)
        self.assertEqual(result["image_counts"][-1], 1)
        self.assertTrue(all(c == 0 for c in result["instance_counts"][:-1]))

    def test_single_and_multiple_defect_instances(self):
        """Valid defect annotations should update corresponding class indices."""
        # File 1 has two instances of class 0 (Loose Meat) and one of class 2 (Twisted Meat)
        lbl_file = os.path.join(self.labels_dir, "defect_01.txt")
        with open(lbl_file, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
            f.write("0 0.7 0.7 0.1 0.1\n")
            f.write("2 0.3 0.3 0.15 0.15\n")

        result = get_instances(
            numbers=[0, 1, 2, 3, 4],
            file_split="train",
            dataset_dir=self.temp_dir,
            verbose=False
        )

        counts = result["instance_counts"]
        images = result["image_counts"]

        # Class 0: 2 instances in 1 image
        self.assertEqual(counts[0], 2)
        self.assertEqual(images[0], 1)

        # Class 1: 0 instances
        self.assertEqual(counts[1], 0)

        # Class 2: 1 instance in 1 image
        self.assertEqual(counts[2], 1)
        self.assertEqual(images[2], 1)

        # Undamaged: 0
        self.assertEqual(counts[-1], 0)

    def test_summarize_dataset_splits(self):
        """summarize_dataset should iterate across all requested splits."""
        valid_lbl = os.path.join(self.temp_dir, "valid", "labels")
        os.makedirs(valid_lbl, exist_ok=True)
        with open(os.path.join(valid_lbl, "v1.txt"), "w") as f:
            f.write("1 0.4 0.4 0.2 0.2\n")

        summaries = summarize_dataset(
            dataset_dir=self.temp_dir,
            splits=["train", "valid"],
            numbers=[0, 1],
            verbose=False
        )

        self.assertIn("train", summaries)
        self.assertIn("valid", summaries)
        self.assertEqual(summaries["valid"]["instance_counts"][1], 1)


if __name__ == "__main__":
    unittest.main()
