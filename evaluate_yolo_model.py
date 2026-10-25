"""
Comprehensive evaluation pipeline for YOLO packaging defect detection model.
Mainly used for training purposes.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# 1. Default configuration paths
DATA_YAML_DEFAULT = Path("Pork Rasher Error-Packaging-.v4i.yolov11/data.yaml")
MODEL_WEIGHTS_DEFAULT = Path("runs/detect/pork_rasher_detection/weights/best.pt")
TEST_IMG_DIR_DEFAULT = Path("Pork Rasher Error-Packaging-.v4i.yolov11/test/images")
TEST_LBL_DIR_DEFAULT = Path("Pork Rasher Error-Packaging-.v4i.yolov11/test/labels")

# Standard 0-indexed defect mapping (matches data.yaml)
CLASS_NAMES_DEFAULT = {
    0: "Loose Meat",
    1: "Packaging Error",
    2: "Twisted Meat",
    3: "Unsealed",
    4: "Wrinkle",
}

# Full 6-category mapping including undamaged status
ALL_CATEGORIES_DEFAULT = {
    -1: "Undamaged",
    0: "Loose Meat",
    1: "Packaging Error",
    2: "Twisted Meat",
    3: "Unsealed",
    4: "Wrinkle",
}


def load_ground_truth_classes(label_path: Union[str, Path]) -> Set[int]:
    """
    Reads ground truth classes from a YOLO label file.

    Parameters
    ----------
    label_path : Union[str, Path]
        Path to the label .txt file.

    Returns
    -------
    Set[int]
        Set of detected class IDs, or {-1} if file is empty or missing (Undamaged).
    """
    lbl_obj = Path(label_path)
    if not lbl_obj.exists():
        return {-1}

    classes = set()
    with open(lbl_obj, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        classes.add(int(parts[0]))
                    except ValueError:
                        continue

    return classes if len(classes) > 0 else {-1}


def evaluate_image_level_accuracy(
    model: Any,
    test_img_dir: Union[str, Path] = TEST_IMG_DIR_DEFAULT,
    test_lbl_dir: Union[str, Path] = TEST_LBL_DIR_DEFAULT,
    conf_thresh: float = 0.25,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Performs custom image-level evaluation across a test set of images and labels.

    Calculates:
      1. Binary Image Accuracy (Damaged vs. Undamaged) with Precision, Recall, F1, 2x2 CM
      2. Multi-Class Image Classification Accuracy across 6 categories:
         [Undamaged, Loose Meat, Packaging Error, Twisted Meat, Unsealed, Wrinkle]

    Parameters
    ----------
    model : Any
        Fitted ultralytics YOLO model instance or mock.
    test_img_dir : Union[str, Path], default=TEST_IMG_DIR_DEFAULT
        Directory containing test images.
    test_lbl_dir : Union[str, Path], default=TEST_LBL_DIR_DEFAULT
        Directory containing corresponding test label files.
    conf_thresh : float, default=0.25
        Confidence threshold for considering a predicted bounding box detection.
    verbose : bool, default=True
        Whether to print formatted evaluation tables.

    Returns
    -------
    Dict[str, Any]
        Structured dictionary containing binary and multi-class performance metrics.
    """
    img_dir_obj = Path(test_img_dir)
    lbl_dir_obj = Path(test_lbl_dir)

    img_paths = sorted(
        list(img_dir_obj.glob("*.jpg")) +
        list(img_dir_obj.glob("*.jpeg")) +
        list(img_dir_obj.glob("*.png"))
    )

    if not img_paths:
        if verbose:
            print(f"[Warning] No test images found in: {img_dir_obj.resolve()}")
        return {}

    if verbose:
        print(f"\nRunning image-level prediction on {len(img_paths)} test images...")

    # Run predictions on test images
    results = model.predict(source=str(img_dir_obj), conf=conf_thresh, verbose=False)

    # Map results by image stem name
    predictions_map = {}
    for res in results:
        img_stem = Path(res.path).stem
        pred_classes = set()
        if hasattr(res, "boxes") and len(res.boxes) > 0:
            for box in res.boxes:
                cls_attr = box.cls[0]
                # Support PyTorch Tensor or numpy float
                if hasattr(cls_attr, "cpu"):
                    cid = int(cls_attr.cpu().numpy())
                else:
                    cid = int(cls_attr)
                pred_classes.add(cid)
        if not pred_classes:
            pred_classes = {-1}  # Undamaged
        predictions_map[img_stem] = pred_classes

    # Track metrics
    gt_binary = []    # 0 = Undamaged, 1 = Damaged
    pred_binary = []  # 0 = Undamaged, 1 = Damaged

    gt_primary = []   # Primary category ID (-1 to 4)
    pred_primary = [] # Primary category ID (-1 to 4)

    exact_matches = 0

    for img_path in img_paths:
        stem = img_path.stem
        lbl_path = lbl_dir_obj / f"{stem}.txt"

        gt_cls = load_ground_truth_classes(lbl_path)
        pred_cls = predictions_map.get(stem, {-1})

        # --- Binary Ground Truth & Prediction ---
        is_gt_damaged = 0 if gt_cls == {-1} else 1
        is_pred_damaged = 0 if pred_cls == {-1} else 1

        gt_binary.append(is_gt_damaged)
        pred_binary.append(is_pred_damaged)

        # --- Multi-Class Primary Category Ground Truth & Prediction ---
        gt_p = -1 if gt_cls == {-1} else sorted(list(gt_cls - {-1}))[0]
        pred_p = -1 if pred_cls == {-1} else sorted(list(pred_cls - {-1}))[0]

        gt_primary.append(gt_p)
        pred_primary.append(pred_p)

        if gt_cls == pred_cls:
            exact_matches += 1

    # 1. BINARY IMAGE-LEVEL ACCURACY (Damaged vs Undamaged)
    gt_bin_arr = np.array(gt_binary)
    pred_bin_arr = np.array(pred_binary)

    correct_binary = int(np.sum(gt_bin_arr == pred_bin_arr))
    total_images = len(gt_bin_arr)
    binary_acc = (correct_binary / total_images) * 100.0 if total_images > 0 else 0.0

    tp = int(np.sum((gt_bin_arr == 1) & (pred_bin_arr == 1)))
    fp = int(np.sum((gt_bin_arr == 0) & (pred_bin_arr == 1)))
    tn = int(np.sum((gt_bin_arr == 0) & (pred_bin_arr == 0)))
    fn = int(np.sum((gt_bin_arr == 1) & (pred_bin_arr == 0)))

    precision_bin = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall_bin = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1_bin = (2 * precision_bin * recall_bin) / (precision_bin + recall_bin) if (precision_bin + recall_bin) > 0 else 0.0

    if verbose:
        print("\n" + "=" * 55)
        print("   1. BINARY IMAGE-LEVEL ACCURACY (Damaged vs. Undamaged) ")
        print("=" * 55)
        print(f"Overall Image Classification Accuracy: {binary_acc:.2f}% ({correct_binary}/{total_images})")
        print(f"Precision (Avoiding False Alarms):    {precision_bin:.2f}%")
        print(f"Recall    (Catching All Defective):    {recall_bin:.2f}%")
        print(f"F1-Score:                            {f1_bin:.2f}%")
        print("\n2x2 Confusion Matrix (Image Status):")
        print("                  Pred Undamaged    Pred Damaged")
        print(f"Actual Undamaged       {tn:<14} {fp:<12}")
        print(f"Actual Damaged         {fn:<14} {tp:<12}")

    # 2. MULTI-CLASS CATEGORY ACCURACY (6 Categories)
    gt_prim_arr = np.array(gt_primary)
    pred_prim_arr = np.array(pred_primary)

    primary_acc = (float(np.sum(gt_prim_arr == pred_prim_arr)) / total_images) * 100.0 if total_images > 0 else 0.0
    exact_match_acc = (float(exact_matches) / total_images) * 100.0 if total_images > 0 else 0.0

    if verbose:
        print("\n" + "=" * 55)
        print("   2. MULTI-CLASS IMAGE CLASSIFICATION ACCURACY ")
        print("=" * 55)
        print(f"Primary Category Classification Accuracy: {primary_acc:.2f}%")
        print(f"Exact Multi-Label Set Match Accuracy:    {exact_match_acc:.2f}%")

    cat_metrics = []
    cat_ids = [-1, 0, 1, 2, 3, 4]

    for cid in cat_ids:
        cname = ALL_CATEGORIES_DEFAULT[cid]
        actual_mask = (gt_prim_arr == cid)
        total_actual = int(np.sum(actual_mask))

        if total_actual > 0:
            correct_cat = int(np.sum((gt_prim_arr == cid) & (pred_prim_arr == cid)))
            cat_acc = (correct_cat / total_actual) * 100.0
            acc_str = f"{cat_acc:.1f}% ({correct_cat}/{total_actual})"
        else:
            cat_acc = 0.0
            correct_cat = 0
            acc_str = "N/A (0 images)"

        cat_metrics.append({
            "Category ID": "Clean" if cid == -1 else cid,
            "Packaging Status / Defect": cname,
            "Total Images": total_actual,
            "Correct Images": correct_cat,
            "Accuracy": acc_str,
            "Accuracy_pct": cat_acc,
        })

    if verbose:
        df_cat = pd.DataFrame(cat_metrics)
        print("\nPer-Category Image Classification Breakdown:")
        print(df_cat[["Category ID", "Packaging Status / Defect", "Total Images", "Accuracy"]].to_string(index=False))

    cm_6x6 = np.zeros((6, 6), dtype=int)
    for g, p in zip(gt_prim_arr, pred_prim_arr):
        row_idx = cat_ids.index(g)
        col_idx = cat_ids.index(p)
        cm_6x6[row_idx, col_idx] += 1

    if verbose:
        headers = ["Undam.", "Loose", "PkgErr", "Twist", "Unseal", "Wrinkle"]
        df_cm = pd.DataFrame(cm_6x6, index=[f"Act: {h}" for h in headers], columns=[f"Pred: {h}" for h in headers])
        print("\n6x6 Image-Level Category Confusion Matrix:")
        print(df_cm.to_string())

    return {
        "total_images": total_images,
        "binary_metrics": {
            "accuracy": binary_acc,
            "precision": precision_bin,
            "recall": recall_bin,
            "f1": f1_bin,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "confusion_matrix": [[tn, fp], [fn, tp]],
        },
        "multiclass_metrics": {
            "primary_accuracy": primary_acc,
            "exact_match_accuracy": exact_match_acc,
            "category_metrics": cat_metrics,
            "confusion_matrix_6x6": cm_6x6.tolist(),
        },
    }


def evaluate_test_set(
    data_yaml: Union[str, Path] = DATA_YAML_DEFAULT,
    weights_path: Union[str, Path] = MODEL_WEIGHTS_DEFAULT,
    test_img_dir: Union[str, Path] = TEST_IMG_DIR_DEFAULT,
    test_lbl_dir: Union[str, Path] = TEST_LBL_DIR_DEFAULT,
    conf_thresh: float = 0.25
) -> Optional[Dict[str, Any]]:
    """
    Main evaluation pipeline running both standard object detection mAP
    and custom image-level classification accuracy.

    Parameters
    ----------
    data_yaml : Union[str, Path], default=DATA_YAML_DEFAULT
        Path to data.yaml dataset definition.
    weights_path : Union[str, Path], default=MODEL_WEIGHTS_DEFAULT
        Path to trained best.pt model weights.
    test_img_dir : Union[str, Path], default=TEST_IMG_DIR_DEFAULT
        Path to test images folder.
    test_lbl_dir : Union[str, Path], default=TEST_LBL_DIR_DEFAULT
        Path to test labels folder.
    conf_thresh : float, default=0.25
        Confidence threshold for YOLO predictions.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing both standard detection metrics and image-level metrics.
    """
    weights_obj = Path(weights_path)
    if not weights_obj.exists():
        print(f"Weights file not found at: {weights_obj.resolve()}")
        print("Please train your model first or update MODEL_WEIGHTS path.")
        return None

    try:
        from ultralytics import YOLO
    except ImportError as err:
        raise ImportError(
            "ultralytics is required for YOLO evaluation. Install via 'pip install ultralytics'."
        ) from err

    # Load trained model
    model = YOLO(str(weights_obj))

    # --- Standard YOLO Bounding Box Evaluation ---
    print("\nRunning standard YOLO evaluation (mAP, Bounding Box IoU)...")
    detection_metrics = model.val(data=str(data_yaml), split="test")

    print("\n" + "=" * 55)
    print("      STANDARD OBJECT DETECTION PERFORMANCE      ")
    print("=" * 55)
    if hasattr(detection_metrics, "box"):
        print(f"mAP@50-95 (Overall Detection): {detection_metrics.box.map:.4f}")
        print(f"mAP@50    (IoU 0.50 Benchmark): {detection_metrics.box.map50:.4f}")
        print(f"Precision (BBox Precision):    {detection_metrics.box.mp:.4f}")
        print(f"Recall    (BBox Recall):       {detection_metrics.box.mr:.4f}")

    # --- Image-Level Classification Accuracy Evaluation ---
    image_metrics = evaluate_image_level_accuracy(
        model=model,
        test_img_dir=test_img_dir,
        test_lbl_dir=test_lbl_dir,
        conf_thresh=conf_thresh,
        verbose=True
    )

    return {
        "detection_metrics": detection_metrics,
        "image_level_metrics": image_metrics,
    }


def main():
    """CLI entrypoint for full YOLO test set evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate YOLO packaging model on test set.")
    parser.add_argument("--data", type=str, default=str(DATA_YAML_DEFAULT), help="Path to data.yaml.")
    parser.add_argument("--weights", type=str, default=str(MODEL_WEIGHTS_DEFAULT), help="Path to best.pt.")
    parser.add_argument("--img_dir", type=str, default=str(TEST_IMG_DIR_DEFAULT), help="Test images dir.")
    parser.add_argument("--lbl_dir", type=str, default=str(TEST_LBL_DIR_DEFAULT), help="Test labels dir.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    args = parser.parse_args()

    evaluate_test_set(
        data_yaml=args.data,
        weights_path=args.weights,
        test_img_dir=args.img_dir,
        test_lbl_dir=args.lbl_dir,
        conf_thresh=args.conf
    )


if __name__ == "__main__":
    main()
