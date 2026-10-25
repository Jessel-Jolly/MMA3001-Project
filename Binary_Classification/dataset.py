"""
Dataset loading and feature extraction module for binary packaging defect classification.

This module processes image files and YOLO-formatted label text files to prepare feature
matrices (X) and binary targets (y) for machine learning classifiers.
"""

import os
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image


def parse_label_file(label_path: str) -> int:
    """
    Parses a YOLO format label file to determine packaging damage status.

    According to the dataset schema:
    - An empty text file indicates NO packaging damage (Label: 0).
    - A non-empty text file containing bounding boxes/classifications indicates damage (Label: 1).

    Parameters
    ----------
    label_path : str
        Path to the label text file (.txt).

    Returns
    -------
    int
        0 if packaging is undamaged (empty file or file missing), 1 if damaged (contains annotations).

    Raises
    ------
    FileNotFoundError
        If label_path does not exist and strict checking is required.
    """
    if not os.path.exists(label_path):
        # Treat missing annotation file as undamaged (0)
        return 0

    if os.path.getsize(label_path) == 0:
        return 0

    with open(label_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return 0
        return 1


def extract_image_features(image_path: str, image_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """
    Extracts normalized, flattened pixel feature vectors from a packaging RGB image.

    Parameters
    ----------
    image_path : str
        Path to the target image file (.jpg / .jpeg).
    image_size : Tuple[int, int], default=(32, 32)
        Target spatial resolution (width, height) to downsample images to.

    Returns
    -------
    np.ndarray
        1D float32 numpy array of flattened normalized RGB pixel values in range [0.0, 1.0].
        Dimension is (width * height * 3,).

    Raises
    ------
    FileNotFoundError
        If image_path does not exist.
    ValueError
        If the image file cannot be read or is corrupted.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_resized = img.resize(image_size, Image.Resampling.BILINEAR)
            img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
            return img_array.flatten()
    except Exception as e:
        raise ValueError(f"Failed to process image file {image_path}: {str(e)}") from e


def load_dataset_split(
    dataset_dir: str,
    split: str = "train",
    image_size: Tuple[int, int] = (32, 32)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads images and labels from a split folder ('train', 'valid', or 'test') and constructs
    the feature matrix X and target vector y.

    Directory structure expected:
    - {dataset_dir}/{split}/images/*.jpg
    - {dataset_dir}/{split}/labels/*.txt

    Parameters
    ----------
    dataset_dir : str
        Base path to the YOLO dataset folder.
    split : str, default='train'
        Data split name ('train', 'valid', or 'test').
    image_size : Tuple[int, int], default=(32, 32)
        Resized spatial dimensions for image feature extraction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X : 2D numpy array of shape (n_samples, n_features) containing image pixel features.
        y : 1D numpy array of shape (n_samples,) containing binary classification targets (0 or 1).

    Raises
    ------
    FileNotFoundError
        If the specified split or images directory does not exist.
    """
    split_dir = os.path.join(dataset_dir, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    image_files.sort()

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        try:
            features = extract_image_features(img_path, image_size=image_size)
            label = parse_label_file(label_path)
            X_list.append(features)
            y_list.append(label)
        except Exception as err:
            print(f"Skipping file {img_name} due to error: {err}")

    if not X_list:
        return np.empty((0, image_size[0] * image_size[1] * 3), dtype=np.float32), np.empty((0,), dtype=int)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)
