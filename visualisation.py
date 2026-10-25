"""
Visualisation utility for YOLO packaging defect annotations and bounding boxes.
"""

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

# 1. Base dataset folder relative to project repository
DATASET_DIR_DEFAULT = Path("Pork Rasher Error-Packaging-.v4i.yolov11")
SPLITS_DEFAULT = ["train", "valid", "test"]

# 2. Defect class names
CLASS_NAMES_DEFAULT = {
    0: "Loose Meat",
    1: "Packaging Error",
    2: "Twisted Meat",
    3: "Unsealed",
    4: "Wrinkle",
}

# Color palette for bounding box labels
CLASS_COLORS_DEFAULT = ["#FF3838", "#FF9D9A", "#3070B3", "#FFB800", "#9D4EDD"]


def get_all_image_pairs(
    dataset_path: Union[str, Path] = DATASET_DIR_DEFAULT,
    split: Optional[str] = None
) -> List[Tuple[Path, Path, str]]:
    """
    Finds all images and pairs them with their corresponding label text files.

    Parameters
    ----------
    dataset_path : Union[str, Path], default=DATASET_DIR_DEFAULT
        Base directory of the YOLO-formatted dataset.
    split : Optional[str], default=None
        If specified ('train', 'valid', or 'test'), only searches that directory.

    Returns
    -------
    List[Tuple[Path, Path, str]]
        List of tuples: (image_path, label_path, split_name).
    """
    path_obj = Path(dataset_path)
    valid_splits = [split] if split else SPLITS_DEFAULT
    pairs = []

    for s in valid_splits:
        img_dir = path_obj / s / "images"
        lbl_dir = path_obj / s / "labels"

        if not img_dir.exists():
            continue

        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in sorted(img_dir.glob(ext)):
                # Matched label file in labels/ folder with same stem name
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                pairs.append((img_path, lbl_path, s))

    return pairs


def parse_yolo_labels(
    label_path: Union[str, Path],
    img_width: int,
    img_height: int
) -> List[Dict[str, Union[int, Tuple[float, float, float, float]]]]:
    """
    Reads YOLO normalized bounding box format (class x_center y_center width height)
    and converts them to pixel coordinates (x_min, y_min, box_w, box_h).

    Parameters
    ----------
    label_path : Union[str, Path]
        Path to the YOLO .txt label file.
    img_width : int
        Pixel width of the target image.
    img_height : int
        Pixel height of the target image.

    Returns
    -------
    List[Dict[str, Union[int, Tuple[float, float, float, float]]]]
        List of dictionaries with 'class_id' and 'bbox': (x_min, y_min, width, height).
        Returns an empty list if file is missing, empty, or unreadable.
    """
    boxes = []
    lbl_obj = Path(label_path)
    if not lbl_obj.exists():
        return boxes

    with open(lbl_obj, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 5:
            try:
                class_id = int(parts[0])
                # De-normalize coordinates
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height

                # Convert center-based to top-left corner (x_min, y_min)
                x_min = x_center - (w / 2.0)
                y_min = y_center - (h / 2.0)

                boxes.append({"class_id": class_id, "bbox": (x_min, y_min, w, h)})
            except (ValueError, IndexError):
                continue

    return boxes


def plot_sample_images(
    n: int = 6,
    split: Optional[str] = None,
    dataset_path: Union[str, Path] = DATASET_DIR_DEFAULT,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    seed: Optional[int] = None
) -> Optional[Tuple[plt.Figure, Any]]:
    """
    Selects N images, parses bounding boxes, and plots them in a grid.

    Parameters
    ----------
    n : int, default=6
        Number of images to sample and visualize.
    split : Optional[str], default=None
        Dataset split to filter by ('train', 'valid', 'test', or None for all).
    dataset_path : Union[str, Path], default=DATASET_DIR_DEFAULT
        Path to the root YOLO dataset directory.
    save_path : Optional[Union[str, Path]], default=None
        Path to save the generated figure.
    show : bool, default=True
        Whether to call plt.show() (set False in automated scripts or tests).
    seed : Optional[int], default=None
        Random seed for reproducible sample selection.

    Returns
    -------
    Optional[Tuple[plt.Figure, Any]]
        Tuple of (fig, axes) if samples are found, or None if no images found.
    """
    path_obj = Path(dataset_path)
    pairs = get_all_image_pairs(path_obj, split=split)

    if not pairs:
        print(f"No images found under: {path_obj.resolve()}")
        return None

    if seed is not None:
        random.seed(seed)

    # Select N random images (or total available if less than N)
    num_samples = min(n, len(pairs))
    sampled_pairs = random.sample(pairs, num_samples)

    # Calculate grid layout dimensions
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_samples == 1:
        axes_array = [axes]
    else:
        axes_array = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (img_path, lbl_path, split_name) in enumerate(sampled_pairs):
        ax = axes_array[idx]
        img = Image.open(img_path)
        img_w, img_h = img.size

        ax.imshow(img)

        # Extract annotations if present
        boxes = parse_yolo_labels(lbl_path, img_w, img_h)

        if boxes:
            for box in boxes:
                cid = box["class_id"]
                x_min, y_min, w, h = box["bbox"]

                color = CLASS_COLORS_DEFAULT[cid % len(CLASS_COLORS_DEFAULT)]
                class_label = CLASS_NAMES_DEFAULT.get(cid, f"Class {cid}")

                # Draw bounding box rectangle
                rect = patches.Rectangle(
                    (x_min, y_min),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Draw text tag above box
                ax.text(
                    x_min,
                    max(0, y_min - 5),
                    class_label,
                    color="white",
                    fontsize=8,
                    weight="bold",
                    bbox=dict(
                        boxstyle="square,pad=0.2",
                        facecolor=color,
                        alpha=0.85,
                        edgecolor="none",
                    ),
                )
            title_text = f"[{split_name}] {img_path.name}\nStatus: {len(boxes)} defect(s)"
        else:
            title_text = f"[{split_name}] {img_path.name}\nStatus: Undamaged"

        ax.set_title(title_text, fontsize=9)
        ax.axis("off")

    # Hide extra unused subplots in the grid
    for j in range(num_samples, len(axes_array)):
        axes_array[j].axis("off")

    plt.tight_layout()

    if save_path:
        save_obj = Path(save_path)
        save_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_obj, dpi=300, bbox_inches="tight")
        print(f"Annotated plot saved to {save_obj.resolve()}")

    if show:
        plt.show()

    return fig, axes


def main():
    """CLI entrypoint for dataset visualization."""
    parser = argparse.ArgumentParser(description="Visualize annotated YOLO images with bounding boxes.")
    parser.add_argument("--n", type=int, default=6, help="Number of sample images to visualize.")
    parser.add_argument("--split", type=str, default=None, help="Filter by split ('train', 'valid', 'test').")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(DATASET_DIR_DEFAULT),
        help="Path to YOLO dataset directory."
    )
    parser.add_argument("--save", type=str, default=None, help="Optional path to save plot image.")
    args = parser.parse_args()

    plot_sample_images(n=args.n, split=args.split, dataset_path=args.dataset_path, save_path=args.save)


if __name__ == "__main__":
    main()
