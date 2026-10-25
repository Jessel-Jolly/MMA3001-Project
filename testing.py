"""
YOLOv11 model testing and validation script for packaging defect detection.
"""

import argparse
from pathlib import Path
from typing import Any, Union


def test_yolo_model(
    weights_path: Union[str, Path] = "runs/detect/pork_rasher_detection/weights/best.pt",
    data_yaml: Union[str, Path] = "Pork Rasher Error-Packaging-.v4i.yolov11/data.yaml",
    split: str = "test",
    **kwargs: Any
) -> Any:
    """
    Evaluates a trained YOLO model checkpoint on a specific dataset split.

    Parameters
    ----------
    weights_path : Union[str, Path], default="runs/detect/pork_rasher_detection/weights/best.pt"
        Path to the trained model weights (.pt).
    data_yaml : Union[str, Path], default="Pork Rasher Error-Packaging-.v4i.yolov11/data.yaml"
        Path to the dataset's data.yaml configuration file.
    split : str, default="test"
        Dataset split to evaluate ('test', 'val', or 'train').
    **kwargs : Any
        Additional keyword arguments passed to model.val().

    Returns
    -------
    Any
        Ultralytics metrics object containing detection performance (box.ap50, save_dir, etc.).

    Raises
    ------
    FileNotFoundError
        If weights_path or data_yaml does not exist.
    ImportError
        If ultralytics is not installed.
    """
    weights = Path(weights_path)
    yaml_path = Path(data_yaml)

    if not weights.exists():
        raise FileNotFoundError(f"Trained model weights not found at: {weights.resolve()}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found at: {yaml_path.resolve()}")

    try:
        from ultralytics import YOLO
    except ImportError as err:
        raise ImportError(
            "ultralytics is required for YOLO evaluation. Install via 'pip install ultralytics'."
        ) from err

    # Load trained model
    model = YOLO(str(weights))

    print("=" * 60)
    print(f"Evaluating YOLO Model on '{split}' Split")
    print(f"Weights     : {weights}")
    print(f"Dataset YAML: {yaml_path}")
    print("=" * 60)

    # Run evaluation on specified split
    metrics = model.val(data=str(yaml_path), split=split, **kwargs)

    # Access and display per-class mAP and confusion matrix
    if hasattr(metrics, "box") and hasattr(metrics.box, "ap50"):
        print("\nPer-class mAP50:", metrics.box.ap50)
    if hasattr(metrics, "save_dir"):
        print("Confusion Matrix and validation outputs saved to:", metrics.save_dir)

    return metrics


def main():
    """CLI entrypoint for YOLO model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv11 model on test data.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/pork_rasher_detection/weights/best.pt",
        help="Path to trained weights file."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="Pork Rasher Error-Packaging-.v4i.yolov11/data.yaml",
        help="Path to data.yaml file."
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on.")
    args = parser.parse_args()

    test_yolo_model(weights_path=args.weights, data_yaml=args.data, split=args.split)


if __name__ == "__main__":
    main()
