"""
YOLOv11 model training pipeline for packaging defect detection.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union


def train_yolo_model(
    data_yaml: Union[str, Path] = "Pork Rasher Error-Packaging-.v4i.yolov11/data.yaml",
    model_name: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project_name: str = "pork_rasher_detection",
    device: Optional[Union[str, int]] = None,
    **kwargs: Any
) -> Tuple[Any, Any]:
    """
    Initializes and trains a YOLOv11 model on the packaging defect dataset.

    Parameters
    ----------
    data_yaml : Union[str, Path], default="Pork Rasher Error-Packaging-.v4i.yolov11/data.yaml"
        Path to the dataset's data.yaml configuration file.
    model_name : str, default="yolo11n.pt"
        Base pretrained YOLO weights or model architecture configuration.
    epochs : int, default=50
        Number of training epochs.
    imgsz : int, default=640
        Target square image dimension (pixels) for training.
    batch : int, default=16
        Batch size per training iteration.
    project_name : str, default="pork_rasher_detection"
        Experiment run name for output checkpoints and logs.
    device : Optional[Union[str, int]], default=None
        Computation device (e.g. 'cpu', 0, 'cuda:0').
    **kwargs : Any
        Additional keyword arguments passed to ultralytics YOLO.train().

    Returns
    -------
    Tuple[Any, Any]
        Tuple of (model, results) where model is the YOLO instance and results
        contains the training metrics and output directory information.

    Raises
    ------
    FileNotFoundError
        If data_yaml does not exist on disk.
    ImportError
        If the 'ultralytics' package is not installed.
    """
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found at: {yaml_path.resolve()}")

    try:
        from ultralytics import YOLO
    except ImportError as err:
        raise ImportError(
            "ultralytics is required for YOLO training. Install via 'pip install ultralytics'."
        ) from err

    # Load pre-trained YOLO model
    model = YOLO(model_name)

    # Compile training parameters
    train_args = {
        "data": str(yaml_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "name": project_name,
        **kwargs,
    }
    if device is not None:
        train_args["device"] = device

    print("=" * 60)
    print("Starting YOLO Model Training")
    print(f"Model       : {model_name}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Epochs      : {epochs} | Batch: {batch} | ImgSz: {imgsz}")
    print(f"Run Name    : {project_name}")
    print("=" * 60)

    # Train the model
    results = model.train(**train_args)
    return model, results


def main():
    """CLI entrypoint for YOLO model training."""
    parser = argparse.ArgumentParser(description="Train YOLOv11 model for packaging error detection.")
    parser.add_argument(
        "--data",
        type=str,
        default="Pork Rasher Error-Packaging-.v4i.yolov11/data.yaml",
        help="Path to data.yaml dataset definition."
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model weights.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--name", type=str, default="pork_rasher_detection", help="Experiment name.")
    args = parser.parse_args()

    train_yolo_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project_name=args.name,
    )


if __name__ == "__main__":
    main()
