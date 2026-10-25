"""
Inference module for YOLOv11 packaging defect detection.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Default class label mapping matching dataset annotations
DEFAULT_CLASS_NAMES = {
    0: "Loose Meat",
    1: "Packaging Error",
    2: "Twisted Meat",
    3: "Unsealed",
    4: "Wrinkle",
}

DEFAULT_WEIGHTS = Path("runs/detect/pork_rasher_detection/weights/best.pt")


def load_yolo_model(weights_path: Union[str, Path] = DEFAULT_WEIGHTS) -> Any:
    """
    Loads a trained YOLO model from a weight file (.pt).

    Parameters
    ----------
    weights_path : Union[str, Path], default=DEFAULT_WEIGHTS
        Path to the YOLO model checkpoint weights file.

    Returns
    -------
    YOLO
        An instantiated Ultralytics YOLO model object.

    Raises
    ------
    ImportError
        If the ultralytics package is not installed.
    FileNotFoundError
        If the specified weights file does not exist on disk.
    """
    if YOLO is None:
        raise ImportError(
            "The 'ultralytics' library is required for YOLO inference. "
            "Please install it via 'pip install ultralytics'."
        )

    w_path = Path(weights_path)
    if not w_path.exists():
        raise FileNotFoundError(
            f"YOLO model weights not found at: '{w_path.resolve()}'. "
            "Please provide a valid checkpoint path or train the model first."
        )

    return YOLO(str(w_path))


def predict_single_image(
    model: Any,
    image_path: Union[str, Path],
    conf_thresh: float = 0.25,
    save_annotated_path: Optional[Union[str, Path]] = None,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Runs YOLO defect detection on an individual packaging image.

    Parameters
    ----------
    model : Any
        Instantiated YOLO model instance or path to weights file.
    image_path : Union[str, Path]
        Path to the target packaging image (e.g. JPEG, PNG).
    conf_thresh : float, default=0.25
        Minimum confidence threshold for predicted bounding boxes.
    save_annotated_path : Optional[Union[str, Path]], default=None
        Optional destination path to save the visualized image with bounding boxes.
    class_names : Optional[Dict[int, str]], default=None
        Dictionary mapping class indices to human-readable names.

    Returns
    -------
    Dict[str, Any]
        Dictionary detailing the prediction outcome:
        - 'image_path' (str): Absolute file path to the analyzed image.
        - 'is_damaged' (bool): True if at least one packaging defect was detected.
        - 'classification_label' (str): "Damaged Packaging" or "Undamaged Packaging".
        - 'num_defects' (int): Total number of defect boxes detected.
        - 'detections' (List[Dict[str, Any]]): List of defect bounding box records,
          each containing 'class_id', 'class_name', 'confidence', and 'bbox_xyxy'.
        - 'annotated_saved_to' (Optional[str]): Path where annotated image was saved, if requested.

    Raises
    ------
    FileNotFoundError
        If the input image file does not exist.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: '{img_path.resolve()}'")

    if isinstance(model, (str, Path)):
        model = load_yolo_model(model)

    c_names = class_names if class_names is not None else DEFAULT_CLASS_NAMES

    # Run Ultralytics YOLO inference
    results = model.predict(source=str(img_path), conf=conf_thresh, verbose=False)
    result = results[0]

    detections: List[Dict[str, Any]] = []
    if len(result.boxes) > 0:
        for box in result.boxes:
            cid = int(box.cls[0].cpu().numpy()) if hasattr(box.cls[0], "cpu") else int(box.cls[0])
            conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], "cpu") else float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist() if hasattr(box.xyxy[0], "cpu") else [float(x) for x in box.xyxy[0]]

            name = c_names.get(cid, getattr(model, "names", {}).get(cid, f"Class {cid}"))

            detections.append({
                "class_id": cid,
                "class_name": name,
                "confidence": round(conf, 4),
                "bbox_xyxy": [round(coord, 2) for coord in xyxy],
            })

    num_defects = len(detections)
    is_damaged = num_defects > 0
    status_label = "Damaged Packaging" if is_damaged else "Undamaged Packaging"

    saved_dest = None
    if save_annotated_path:
        out_path = Path(save_annotated_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # result.plot() returns a numpy array in BGR format
        annotated_arr = result.plot()
        # Convert BGR array to RGB for PIL Image
        pil_img = Image.fromarray(annotated_arr[..., ::-1])
        pil_img.save(out_path)
        saved_dest = str(out_path.resolve())

    return {
        "image_path": str(img_path.resolve()),
        "is_damaged": is_damaged,
        "classification_label": status_label,
        "num_defects": num_defects,
        "detections": detections,
        "annotated_saved_to": saved_dest,
    }


def predict_images_batch(
    model: Any,
    target_path: Union[str, Path],
    conf_thresh: float = 0.25,
    save_dir: Optional[Union[str, Path]] = None,
    output_json: Optional[Union[str, Path]] = None,
    recursive: bool = False,
    class_names: Optional[Dict[int, str]] = None
) -> List[Dict[str, Any]]:
    """
    Runs YOLO defect detection across a directory of images or a single file.

    Parameters
    ----------
    model : Any
        YOLO model instance or path to checkpoint weights.
    target_path : Union[str, Path]
        Path to an image file or a directory containing images.
    conf_thresh : float, default=0.25
        Minimum detection confidence threshold.
    save_dir : Optional[Union[str, Path]], default=None
        Optional directory to save plotted images with detected bounding boxes.
    output_json : Optional[Union[str, Path]], default=None
        Optional path to save all prediction results in structured JSON format.
    recursive : bool, default=False
        Whether to search subdirectories recursively when target_path is a folder.
    class_names : Optional[Dict[int, str]], default=None
        Mapping of class IDs to human-readable names.

    Returns
    -------
    List[Dict[str, Any]]
        List of per-image prediction outcome dictionaries.

    Raises
    ------
    FileNotFoundError
        If target_path does not exist.
    """
    path_obj = Path(target_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Target path does not exist: '{path_obj.resolve()}'")

    if isinstance(model, (str, Path)):
        model = load_yolo_model(model)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths: List[Path] = []

    if path_obj.is_file():
        image_paths.append(path_obj)
    elif path_obj.is_dir():
        pattern = "**/*" if recursive else "*"
        for f in path_obj.glob(pattern):
            if f.is_file() and f.suffix.lower() in valid_exts:
                image_paths.append(f)
        image_paths.sort()

    if not image_paths:
        print(f"[Warning] No supported image files found under: '{path_obj.resolve()}'")
        return []

    save_dir_obj = Path(save_dir) if save_dir else None
    if save_dir_obj:
        save_dir_obj.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for img_p in image_paths:
        save_annot = None
        if save_dir_obj:
            save_annot = save_dir_obj / f"annotated_{img_p.name}"

        try:
            pred = predict_single_image(
                model=model,
                image_path=img_p,
                conf_thresh=conf_thresh,
                save_annotated_path=save_annot,
                class_names=class_names
            )
            results.append(pred)
        except Exception as err:
            results.append({
                "image_path": str(img_p.resolve()),
                "is_damaged": None,
                "classification_label": "Error",
                "num_defects": 0,
                "detections": [],
                "error": str(err),
            })

    if output_json:
        out_json_path = Path(output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[+] Prediction records exported to JSON: {out_json_path.resolve()}")

    return results


def format_cli_summary(res: Dict[str, Any]) -> str:
    """
    Formats a single prediction dictionary into a clean CLI string.

    Parameters
    ----------
    res : Dict[str, Any]
        Prediction outcome dictionary.

    Returns
    -------
    str
        Human-readable summary line.
    """
    fname = Path(res["image_path"]).name
    status = "[DAMAGED]" if res.get("is_damaged") else "[OK/UNDAMAGED]"
    num = res.get("num_defects", 0)

    if num > 0:
        det_summary = ", ".join([f"{d['class_name']} ({d['confidence']*100:.1f}%)" for d in res["detections"]])
        return f"{fname:<35} | {status:<14} | Defects: {num} | Found: {det_summary}"
    else:
        return f"{fname:<35} | {status:<14} | Defects: 0 | Packaging intact"


def main():
    """CLI entrypoint for YOLO packaging defect inference."""
    parser = argparse.ArgumentParser(
        description="Run YOLO packaging defect detection on an image or directory of images."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an input image file or a directory containing images."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to YOLO .pt weights file (default: {DEFAULT_WEIGHTS})."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save visualized images with bounding boxes drawn."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to export inference results as a JSON file."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search image directory recursively."
    )

    args = parser.parse_args()

    print("=" * 75)
    print("MMA3001 Packaging Inspection - YOLOv11 Defect Inference Engine")
    print("=" * 75)
    print(f"Target Input : {args.input_path}")
    print(f"Weights Path : {args.weights}")
    print(f"Confidence   : {args.conf}")
    print("-" * 75)

    if not os.path.exists(args.weights):
        print(f"[Error] Weights file not found at: '{args.weights}'")
        print("Please train your model using 'training.py' or provide a valid weights checkpoint path.")
        return

    try:
        model = load_yolo_model(args.weights)
    except Exception as e:
        print(f"[Error] Failed to load YOLO model: {e}")
        return

    predictions = predict_images_batch(
        model=model,
        target_path=args.input_path,
        conf_thresh=args.conf,
        save_dir=args.save_dir,
        output_json=args.output_json,
        recursive=args.recursive
    )

    print("\nInference Results:")
    print("-" * 75)
    damaged_count = sum(1 for p in predictions if p.get("is_damaged") is True)
    total_count = len(predictions)

    for p in predictions:
        print(format_cli_summary(p))

    print("-" * 75)
    print(f"Summary: {total_count} image(s) processed | {damaged_count} damaged | {total_count - damaged_count} undamaged")
    print("=" * 75)


if __name__ == "__main__":
    main()
