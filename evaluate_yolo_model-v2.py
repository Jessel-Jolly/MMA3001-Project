import os
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO

# 1. Configuration paths
DATA_YAML = Path("Pork Rasher Error-Packaging-.v4i.yolov11/data.yaml")
MODEL_WEIGHTS = Path("runs/detect/pork_rasher_detection/weights/best.pt")
TEST_IMG_DIR = Path("Pork Rasher Error-Packaging-.v4i.yolov11/test/images")
TEST_LBL_DIR = Path("Pork Rasher Error-Packaging-.v4i.yolov11/test/labels")

# Standard 0-indexed defect mapping (matches data.yaml)
CLASS_NAMES = {
    0: "Loose Meat",
    1: "Packaging Error",
    2: "Twisted Meat",
    3: "Unsealed",
    4: "Wrinkle",
}

# Full 6-category mapping including undamaged status
ALL_CATEGORIES = {
    -1: "Undamaged",
    0: "Loose Meat",
    1: "Packaging Error",
    2: "Twisted Meat",
    3: "Unsealed",
    4: "Wrinkle",
}


def load_ground_truth_classes(label_path):
    """
    Reads ground truth classes from a YOLO label file.
    Returns a set of class IDs, or {-1} if file is empty/missing (Undamaged).
    """
    if not label_path.exists():
        return {-1}

    classes = set()
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    classes.add(int(parts[0]))

    return classes if len(classes) > 0 else {-1}


def evaluate_image_level_accuracy(model, test_img_dir, test_lbl_dir, conf_thresh=0.25):
    """
    Performs custom image-level evaluation across the entire test set.
    Calculates:
      1. Binary Image Accuracy (Damaged vs. Undamaged)
      2. Multi-Class Image Classification Accuracy across 6 categories:
         [Undamaged, Loose Meat, Packaging Error, Twisted Meat, Unsealed, Wrinkle]
    """
    img_paths = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.jpeg")) + list(test_img_dir.glob("*.png"))
    if not img_paths:
        print(f"[Warning] No test images found in: {test_img_dir}")
        return

    print(f"\nRunning image-level prediction on {len(img_paths)} test images...")
    
    # Run predictions on test images
    results = model.predict(source=str(test_img_dir), conf=conf_thresh, verbose=False)
    
    # Map results by image stem name
    predictions_map = {}
    for res in results:
        img_stem = Path(res.path).stem
        pred_classes = set()
        if len(res.boxes) > 0:
            for box in res.boxes:
                cid = int(box.cls[0].cpu().numpy())
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
        lbl_path = test_lbl_dir / f"{stem}.txt"

        gt_cls = load_ground_truth_classes(lbl_path)
        pred_cls = predictions_map.get(stem, {-1})

        # --- Binary Ground Truth & Prediction ---
        is_gt_damaged = 0 if gt_cls == {-1} else 1
        is_pred_damaged = 0 if pred_cls == {-1} else 1

        gt_binary.append(is_gt_damaged)
        pred_binary.append(is_pred_damaged)

        # --- Multi-Class Primary Category Ground Truth & Prediction ---
        # Select primary class (first defect class or -1 for Undamaged)
        gt_p = -1 if gt_cls == {-1} else sorted(list(gt_cls - {-1}))[0]
        pred_p = -1 if pred_cls == {-1} else sorted(list(pred_cls - {-1}))[0]

        gt_primary.append(gt_p)
        pred_primary.append(pred_p)

        if gt_cls == pred_cls:
            exact_matches += 1

    # ==========================================
    # 1. BINARY IMAGE-LEVEL ACCURACY (Damaged vs Undamaged)
    # ==========================================
    gt_bin_arr = np.array(gt_binary)
    pred_bin_arr = np.array(pred_binary)

    correct_binary = np.sum(gt_bin_arr == pred_bin_arr)
    total_images = len(gt_bin_arr)
    binary_acc = (correct_binary / total_images) * 100.0

    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((gt_bin_arr == 1) & (pred_bin_arr == 1))
    fp = np.sum((gt_bin_arr == 0) & (pred_bin_arr == 1))
    tn = np.sum((gt_bin_arr == 0) & (pred_bin_arr == 0))
    fn = np.sum((gt_bin_arr == 1) & (pred_bin_arr == 0))

    precision_bin = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall_bin = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1_bin = (2 * precision_bin * recall_bin) / (precision_bin + recall_bin) if (precision_bin + recall_bin) > 0 else 0.0

    print("\n" + "=" * 55)
    print("   1. BINARY IMAGE-LEVEL ACCURACY (Damaged vs. Undamaged) ")
    print("=" * 55)
    print(f"Overall Image Classification Accuracy: {binary_acc:.2f}% ({correct_binary}/{total_images})")
    print(f"Precision (Avoiding False Alarms):    {precision_bin:.2f}%")
    print(f"Recall    (Catching All Defective):    {recall_bin:.2f}%")
    print(f"F1-Score:                            {f1_bin:.2f}%")
    print("\n2x2 Confusion Matrix (Image Status):")
    print(f"                  Pred Undamaged    Pred Damaged")
    print(f"Actual Undamaged       {tn:<14} {fp:<12}")
    print(f"Actual Damaged         {fn:<14} {tp:<12}")

    # ==========================================
    # 2. MULTI-CLASS CATEGORY ACCURACY (6 Categories)
    # ==========================================
    gt_prim_arr = np.array(gt_primary)
    pred_prim_arr = np.array(pred_primary)

    primary_acc = (np.sum(gt_prim_arr == pred_prim_arr) / total_images) * 100.0
    exact_match_acc = (exact_matches / total_images) * 100.0

    print("\n" + "=" * 55)
    print("   2. MULTI-CLASS IMAGE CLASSIFICATION ACCURACY ")
    print("=" * 55)
    print(f"Primary Category Classification Accuracy: {primary_acc:.2f}%")
    print(f"Exact Multi-Label Set Match Accuracy:    {exact_match_acc:.2f}%")

    # Category Breakdown
    cat_metrics = []
    cat_ids = [-1, 0, 1, 2, 3, 4]

    for cid in cat_ids:
        cname = ALL_CATEGORIES[cid]
        actual_mask = (gt_prim_arr == cid)
        total_actual = np.sum(actual_mask)

        if total_actual > 0:
            correct_cat = np.sum((gt_prim_arr == cid) & (pred_prim_arr == cid))
            cat_acc = (correct_cat / total_actual) * 100.0
            acc_str = f"{cat_acc:.1f}% ({correct_cat}/{total_actual})"
        else:
            acc_str = "N/A (0 images)"

        cat_metrics.append({
            "Category ID": "Clean" if cid == -1 else cid,
            "Packaging Status / Defect": cname,
            "Total Images": total_actual,
            "Accuracy": acc_str
        })

    df_cat = pd.DataFrame(cat_metrics)
    print("\nPer-Category Image Classification Breakdown:")
    print(df_cat.to_string(index=False))

    # 6x6 Confusion Matrix
    print("\n6x6 Image-Level Category Confusion Matrix:")
    cm_6x6 = np.zeros((6, 6), dtype=int)
    for g, p in zip(gt_prim_arr, pred_prim_arr):
        row_idx = cat_ids.index(g)
        col_idx = cat_ids.index(p)
        cm_6x6[row_idx, col_idx] += 1

    headers = ["Undam.", "Loose", "PkgErr", "Twist", "Unseal", "Wrinkle"]
    df_cm = pd.DataFrame(cm_6x6, index=[f"Act: {h}" for h in headers], columns=[f"Pred: {h}" for h in headers])
    print(df_cm.to_string())


def evaluate_test_set(data_yaml=DATA_YAML, weights_path=MODEL_WEIGHTS):
    """
    Main evaluation pipeline running both standard object detection mAP
    and custom image-level classification accuracy.
    """
    if not weights_path.exists():
        print(f"Weights file not found at: {weights_path.resolve()}")
        print("Please train your model first or update MODEL_WEIGHTS path.")
        return

    # Load trained model
    model = YOLO(weights_path)

    # --- Standard YOLO Bounding Box Evaluation ---
    print("\nRunning standard YOLO evaluation (mAP, Bounding Box IoU)...")
    metrics = model.val(data=str(data_yaml), split="test")

    print("\n" + "=" * 55)
    print("      STANDARD OBJECT DETECTION PERFORMANCE      ")
    print("=" * 55)
    print(f"mAP@50-95 (Overall Detection): {metrics.box.map:.4f}")
    print(f"mAP@50    (IoU 0.50 Benchmark): {metrics.box.map50:.4f}")
    print(f"Precision (BBox Precision):    {metrics.box.mp:.4f}")
    print(f"Recall    (BBox Recall):       {metrics.box.mr:.4f}")

    # --- Image-Level Classification Accuracy Evaluation ---
    evaluate_image_level_accuracy(
        model=model,
        test_img_dir=TEST_IMG_DIR,
        test_lbl_dir=TEST_LBL_DIR,
        conf_thresh=0.25
    )


if __name__ == "__main__":
    evaluate_test_set()
