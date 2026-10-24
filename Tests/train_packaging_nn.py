"""
Main execution script for training, evaluating, and exporting the packaging damage
binary neural network classifier.

MMA3001 Project Brief Alignment:
- Demonstrates executable computational workflow.
- Integrates feature scaling, L2 regularisation, early stopping, and model persistence.
- Evaluates performance across both external Validation and withheld Test splits.
- Automatically exports the trained model and metrics to a designated results subfolder.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any
import numpy as np

# Robust import of dataset module
try:
    from dataset import load_dataset_split
except ImportError:
    import importlib.util
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    _ds_path = os.path.join(_cur_dir, "dataset.py")
    if os.path.exists(_ds_path):
        _spec = importlib.util.spec_from_file_location("dataset", _ds_path)
        _ds_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ds_mod)
        load_dataset_split = _ds_mod.load_dataset_split
    else:
        raise ImportError("Could not find dataset.py module.")

# Robust import of model module (supports model.py or model-v2.py)
try:
    from model import (
        build_mlp_pipeline,
        train_and_evaluate,
        save_model,
        evaluate_model
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
        train_and_evaluate = _m_mod.train_and_evaluate
        save_model = _m_mod.save_model
        evaluate_model = _m_mod.evaluate_model
    else:
        raise ImportError("Could not import build_mlp_pipeline, train_and_evaluate, or save_model.")


def plot_confusion_matrices(val_cm: list, test_cm: list, output_path: str):
    """
    Plots and exports side-by-side confusion matrix heatmaps for validation and test splits.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        splits = [("Validation Set", np.array(val_cm), axes[0]),
                  ("Test Set (Withheld)", np.array(test_cm), axes[1])]

        labels = ["Undamaged (0)", "Damaged (1)"]

        for title, cm, ax in splits:
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            tick_marks = np.arange(len(labels))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(labels)

            ax.set_ylabel("True Ground Truth Label")
            ax.set_xlabel("Predicted Model Label")

            # Annotate numbers
            thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight="bold"
                    )

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[Notice] Could not generate confusion matrix plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train, Evaluate, and Save Scikit-Learn MLP for Packaging Damage Binary Classification.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="Pork Rasher Error-Packaging-.v4i.yolov11",
        help="Path to YOLO format dataset root folder."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Binary_Classification/results",
        help="Target folder where the model and evaluation outputs will be stored."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="binary_model",
        help="Filename prefix for the saved model (e.g. 'binary_model')."
    )
    parser.add_argument("--image_size", type=int, default=32, help="Target image size (width & height in pixels).")
    parser.add_argument("--alpha", type=float, default=0.01, help="L2 regularization penalty parameter.")
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[30, 15], help="Neurons per hidden layer.")
    args = parser.parse_args()

    img_size_tuple = (args.image_size, args.image_size)

    print("=" * 75)
    print("MMA3001 Packaging Quality Inspection - Binary Classification Pipeline")
    print("=" * 75)
    print(f"Dataset Directory : {args.dataset_dir}")
    print(f"Output Directory  : {args.output_dir}")
    print(f"Model Save Name   : {args.model_name}.joblib")
    print(f"Feature Dimension : {args.image_size}x{args.image_size}x3 = {args.image_size * args.image_size * 3} features")
    print(f"MLP Architecture  : Hidden Layers = {tuple(args.hidden_layers)}, L2 Penalty (alpha) = {args.alpha}")
    print("-" * 75)

    if not os.path.exists(args.dataset_dir):
        print(f"\n[Warning] Dataset directory '{args.dataset_dir}' not found on local filesystem.")
        print("Please place your dataset folder in the project repository root and re-run.")
        return

    # 1. Load Train, Validation, and Test splits
    print("\n[1/4] Loading dataset splits (train, valid, test)...")
    X_train, y_train = load_dataset_split(args.dataset_dir, split="train", image_size=img_size_tuple)
    X_val, y_val = load_dataset_split(args.dataset_dir, split="valid", image_size=img_size_tuple)
    X_test, y_test = load_dataset_split(args.dataset_dir, split="test", image_size=img_size_tuple)

    print(f"  - Train Set : {len(X_train)} samples (Damaged: {sum(y_train)}, Undamaged: {len(y_train) - sum(y_train)})")
    print(f"  - Valid Set : {len(X_val)} samples (Damaged: {sum(y_val)}, Undamaged: {len(y_val) - sum(y_val)})")
    print(f"  - Test Set  : {len(X_test)} samples (Damaged: {sum(y_test)}, Undamaged: {len(y_test) - sum(y_test)})")

    if len(X_train) == 0:
        print("[Error] No training samples found. Please verify your dataset directory structure.")
        return

    # 2. Build Pipeline & Train Model
    print("\n[2/4] Training Scikit-Learn MLP Classifier Pipeline with StandardScaler...")
    pipeline = build_mlp_pipeline(
        hidden_layer_sizes=tuple(args.hidden_layers),
        alpha=args.alpha,
        random_state=42,
        early_stopping=True
    )

    results = train_and_evaluate(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )

    val_res = results.get("validation", {})
    test_res = results.get("test", {})

    # 3. Print Side-by-Side Performance Comparison
    print("\n[3/4] Performance Evaluation Results:")
    print("-" * 75)
    print(f"{'Metric':<20} | {'Validation Set':<20} | {'Test Set (Withheld)':<20}")
    print("-" * 75)
    print(f"{'Accuracy':<20} | {val_res.get('accuracy', 0.0):<20.4f} | {test_res.get('accuracy', 0.0):<20.4f}")
    print(f"{'Precision':<20} | {val_res.get('precision', 0.0):<20.4f} | {test_res.get('precision', 0.0):<20.4f}")
    print(f"{'Recall':<20} | {val_res.get('recall', 0.0):<20.4f} | {test_res.get('recall', 0.0):<20.4f}")
    print(f"{'F1-Score':<20} | {val_res.get('f1_score', 0.0):<20.4f} | {test_res.get('f1_score', 0.0):<20.4f}")
    print(f"{'ROC-AUC':<20} | {val_res.get('roc_auc', 0.0):<20.4f} | {test_res.get('roc_auc', 0.0):<20.4f}")
    print("-" * 75)
    print(f"Validation Confusion Matrix : {val_res.get('confusion_matrix')}")
    print(f"Test Set Confusion Matrix   : {test_res.get('confusion_matrix')}")
    print("-" * 75)

    # 4. Save Model and Outputs
    print(f"\n[4/4] Saving model and results to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)

    # A. Save trained model with joblib
    model_filename = f"{args.model_name}.joblib"
    model_save_path = os.path.join(args.output_dir, model_filename)
    saved_abs_path = save_model(pipeline, model_save_path)
    print(f"  [+] Saved trained model : {model_save_path}")

    # B. Save JSON metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    export_payload = {
        "model_name": args.model_name,
        "image_size": list(img_size_tuple),
        "hyperparameters": {
            "hidden_layer_sizes": list(args.hidden_layers),
            "alpha": args.alpha,
            "early_stopping": True,
        },
        "sample_counts": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test)
        },
        "validation_metrics": val_res,
        "test_metrics": test_res
    }
    with open(metrics_path, "w") as f:
        json.dump(export_payload, f, indent=4)
    print(f"  [+] Saved metrics JSON  : {metrics_path}")

    # C. Save formatted text summary
    summary_path = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("MMA3001 PACKAGING QUALITY INSPECTION - BINARY NN REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Model Name       : {args.model_name}.joblib\n")
        f.write(f"Dataset          : {args.dataset_dir}\n")
        f.write(f"Hidden Layers    : {tuple(args.hidden_layers)}\n")
        f.write(f"Alpha (L2)       : {args.alpha}\n")
        f.write(f"Feature Dimension: {args.image_size}x{args.image_size}x3\n\n")
        f.write("-" * 65 + "\n")
        f.write("SIDE-BY-SIDE PERFORMANCE SUMMARY\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Metric':<20} | {'Validation Set':<18} | {'Test Set':<18}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Accuracy':<20} | {val_res.get('accuracy', 0.0):<18.4f} | {test_res.get('accuracy', 0.0):<18.4f}\n")
        f.write(f"{'Precision':<20} | {val_res.get('precision', 0.0):<18.4f} | {test_res.get('precision', 0.0):<18.4f}\n")
        f.write(f"{'Recall':<20} | {val_res.get('recall', 0.0):<18.4f} | {test_res.get('recall', 0.0):<18.4f}\n")
        f.write(f"{'F1-Score':<20} | {val_res.get('f1_score', 0.0):<18.4f} | {test_res.get('f1_score', 0.0):<18.4f}\n")
        f.write(f"{'ROC-AUC':<20} | {val_res.get('roc_auc', 0.0):<18.4f} | {test_res.get('roc_auc', 0.0):<18.4f}\n\n")
        f.write("-" * 65 + "\n")
        f.write("VALIDATION CLASSIFICATION REPORT:\n")
        f.write(val_res.get("classification_report", "N/A") + "\n\n")
        f.write("-" * 65 + "\n")
        f.write("TEST CLASSIFICATION REPORT:\n")
        f.write(test_res.get("classification_report", "N/A") + "\n")
    print(f"  [+] Saved report summary: {summary_path}")

    # D. Save Confusion Matrix Plot
    if "confusion_matrix" in val_res and "confusion_matrix" in test_res:
        cm_plot_path = os.path.join(args.output_dir, "confusion_matrices.png")
        plot_confusion_matrices(val_res["confusion_matrix"], test_res["confusion_matrix"], cm_plot_path)
        print(f"  [+] Saved confusion plot: {cm_plot_path}")

    print("\n[Done] All training, validation, testing, and file exports completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
