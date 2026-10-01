import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 1. Base dataset folder relative to your python script
DATASET_DIR = Path("Pork Rasher Error-Packaging-.v4i.yolov11")
SPLITS = ["train", "valid", "test"]

# 2. Optional: Map class IDs to human-readable names
CLASS_NAMES = { 0: "Loose Meat",
                1: "Packaging Error",
                2: "Twisted Meat",
                3: "Unsealed",
                4: "Wrinkle", }

# Color palette for bounding box labels
CLASS_COLORS = ["#FF3838", "#FF9D9A", "#3070B3", "#FFB800", "#9D4EDD"]


def get_all_image_pairs(dataset_path=DATASET_DIR, split=None):
    """
    Finds all images and pairs them with their corresponding label text files.
    If split is specified ('train', 'valid', or 'test'), only searches that directory.
    """
    valid_splits = [split] if split else SPLITS
    pairs = []

    for s in valid_splits:
        img_dir = dataset_path / s / "images"
        lbl_dir = dataset_path / s / "labels"

        if not img_dir.exists():
            continue

        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in img_dir.glob(ext):
                # Matched label file in labels/ folder with same stem name
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                pairs.append((img_path, lbl_path, s))

    return pairs


def parse_yolo_labels(label_path, img_width, img_height):
    """
    Reads YOLO normalized bounding box format (class x_center y_center width height)
    and converts them to pixel coordinates (x_min, y_min, box_w, box_h).
    Returns an empty list if file is missing or empty (undamaged).
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 5:
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

    return boxes


def plot_sample_images(n=6, split=None, dataset_path=DATASET_DIR, save_path=None):
    """
    Selects N images, parses bounding boxes, and plots them in a grid.
    """
    pairs = get_all_image_pairs(dataset_path, split=split)

    if not pairs:
        print(f"No images found under: {dataset_path.resolve()}")
        return

    # Select N random images (or total available if less than N)
    num_samples = min(n, len(pairs))
    sampled_pairs = random.sample(pairs, num_samples)

    # Calculate grid layout dimensions
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (img_path, lbl_path, split_name) in enumerate(sampled_pairs):
        ax = axes[idx]
        img = Image.open(img_path)
        img_w, img_h = img.size

        ax.imshow(img)

        # Extract annotations if present
        boxes = parse_yolo_labels(lbl_path, img_w, img_h)

        if boxes:
            for box in boxes:
                cid = box["class_id"]
                x_min, y_min, w, h = box["bbox"]

                color = CLASS_COLORS[cid % len(CLASS_COLORS)]
                class_label = CLASS_NAMES.get(cid, f"Class {cid}")

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
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Annotated plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Change N to whatever sample size you'd like to visualize
    # You can also pass split="train" or split="valid" to filter by folder
    plot_sample_images(n=6)
