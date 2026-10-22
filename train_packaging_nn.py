"""
Main execution script for training and evaluating the packaging damage binary neural network classifier.

MMA3001 Project Brief Alignment:
- Demonstrates executable computational workflow.
- Integrates feature scaling, L2 regularisation, early stopping, and external validation set evaluation.
- Outputs quantitative performance metrics (Precision, Recall, F1, Accuracy, ROC-AUC) as required for project report.
"""

import os
import argparse
from dataset import load_dataset_split
from model import build_mlp_pipeline, train_and_evaluate, predict_packaging_damage


def main():
    parser = argparse.ArgumentParser(description="Train Scikit-Learn MLP for Packaging Damage Binary Classification.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="Pork Rasher Error-Packaging-.v4i.yolov11",
        help="Path to YOLO format dataset root folder."
    )
    parser.add_argument("--image_size", type=int, default=32, help="Target image size (width & height in pixels).")
    parser.add_argument("--alpha", type=float, default=0.01, help="L2 regularization penalty parameter.")
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[30, 15], help="Neurons per hidden layer.")
    args = parser.parse_args()

    img_size_tuple = (args.image_size, args.image_size)

    print("=" * 70)
    print("MMA3001 Packaging Quality Inspection - Binary Classification Pipeline")
    print("=" * 70)
    print(f"Dataset Directory : {args.dataset_dir}")
    print(f"Feature Dimension : {args.image_size}x{args.image_size}x3 = {args.image_size * args.image_size * 3} features")
    print(f"MLP Architecture  : Hidden Layers = {tuple(args.hidden_layers)}, L2 Penalty (alpha) = {args.alpha}")
    print("-" * 70)

    if not os.path.exists(args.dataset_dir):
        print(f"\n[Warning] Dataset directory '{args.dataset_dir}' not found on local filesystem.")
        print("Please place your dataset folder in the project repository root and re-run.")
        return

    # 1. Load Train, Validation, and Test splits
    print("\n[1/3] Loading dataset splits...")
    X_train, y_train = load_dataset_split(args.dataset_dir, split="train", image_size=img_size_tuple)
    X_val, y_val = load_dataset_split(args.dataset_dir, split="valid", image_size=img_size_tuple)
    X_test, y_test = load_dataset_split(args.dataset_dir, split="test", image_size=img_size_tuple)

    print(f"  - Train Set : {len(X_train)} samples (Damaged: {sum(y_train)}, Undamaged: {len(y_train) - sum(y_train)})")
    print(f"  - Valid Set : {len(X_val)} samples (Damaged: {sum(y_val)}, Undamaged: {len(y_val) - sum(y_val)})")
    print(f"  - Test Set  : {len(X_test)} samples (Damaged: {sum(y_test)}, Undamaged: {len(y_test) - sum(y_test)})")

    if len(X_train) == 0:
        print("[Error] No training samples loaded. Check dataset directory path.")
        return

    # 2. Build and Train Pipeline
    print("\n[2/3] Training Scikit-Learn MLP Classifier Pipeline with StandardScaler...")
    pipeline = build_mlp_pipeline(
        hidden_layer_sizes=tuple(args.hidden_layers),
        alpha=args.alpha,
        random_state=42,
        early_stopping=True
    )

    val_metrics = train_and_evaluate(pipeline, X_train, y_train, X_val, y_val)

    # 3. Validation Performance Summary
    print("\n[3/3] Validation Results:")
    print("-" * 40)
    print(f"  Accuracy  : {val_metrics['accuracy']:.4f}")
    print(f"  Precision : {val_metrics['precision']:.4f}")
    print(f"  Recall    : {val_metrics['recall']:.4f}")
    print(f"  F1-Score  : {val_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {val_metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix : {val_metrics['confusion_matrix']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
