"""
Packaging defect classification inference engine (v2).

This module provides standalone inference capabilities for the packaging damage
binary classification neural network. It can evaluate any individual image, a file
(an image file or a text/CSV file listing image paths), or an entire directory of images.
"""

import os
import sys
import math
import argparse
import json
from typing import Dict, Any, Tuple, List, Union, Optional
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# Dynamic / fallback import for dataset feature extraction
try:
    from dataset import extract_image_features
except ImportError:
    import importlib.util
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    _ds_path = os.path.join(_cur_dir, "dataset.py")
    if os.path.exists(_ds_path):
        _spec = importlib.util.spec_from_file_location("dataset", _ds_path)
        _ds_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ds_mod)
        extract_image_features = _ds_mod.extract_image_features
    else:
        raise ImportError("Could not locate dataset.py for image feature extraction.")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TEXT_LIST_EXTENSIONS = {".txt", ".csv"}


def load_binary_model(model_path: str = "Binary_Classification/results/binary_model.joblib") -> Pipeline:
    """
    Loads a pre-trained Scikit-Learn binary classification pipeline from disk.

    Parameters
    ----------
    model_path : str, default="Binary_Classification/results/binary_model.joblib"
        Path to the saved .joblib model file.

    Returns
    -------
    Pipeline
        Loaded Scikit-Learn Pipeline containing StandardScaler and MLPClassifier.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model file not found at: '{model_path}'. "
            "Please train the model first using train_packaging_nn-v2.py or verify the path."
        )
    return joblib.load(model_path)


def infer_image_size_from_model(model: Union[str, Pipeline]) -> Tuple[int, int]:
    """
    Automatically infers the required (height, width) image resolution from the model.

    The model expects a flattened RGB vector of dimension (H * W * 3).
    For square image inputs (standard in YOLO / classification pipelines),
    H = W = sqrt(n_features_in_ / 3).

    Parameters
    ----------
    model : Union[str, Pipeline]
        Loaded Scikit-Learn Pipeline or file path to the saved model.

    Returns
    -------
    Tuple[int, int]
        Inferred (height, width) spatial dimensions (e.g. (32, 32)).

    Raises
    ------
    ValueError
        If the input dimension cannot be converted into valid square RGB image dimensions.
    """
    if isinstance(model, str):
        pipeline = load_binary_model(model)
    else:
        pipeline = model

    # Retrieve input feature dimension from pipeline or inner estimators
    n_features = getattr(pipeline, "n_features_in_", None)
    if n_features is None and hasattr(pipeline, "named_steps"):
        if "standardscaler" in pipeline.named_steps:
            n_features = getattr(pipeline.named_steps["standardscaler"], "n_features_in_", None)
        elif "mlpclassifier" in pipeline.named_steps:
            n_features = getattr(pipeline.named_steps["mlpclassifier"], "n_features_in_", None)

    if n_features is None:
        # Fallback to default if model has not recorded n_features_in_
        return (32, 32)

    # 3 color channels (RGB)
    if n_features % 3 != 0:
        raise ValueError(
            f"Model feature count ({n_features}) is not divisible by 3 (RGB channels)."
        )

    pixels_per_channel = n_features // 3
    side_len = int(math.isqrt(pixels_per_channel))

    if side_len * side_len != pixels_per_channel:
        raise ValueError(
            f"Model feature count ({n_features}) does not correspond to a square image "
            f"(pixels per channel = {pixels_per_channel} is not a square number)."
        )

    return (side_len, side_len)


def predict_image(
    image_path: str,
    model: Union[str, Pipeline] = "Binary_Classification/results/binary_model.joblib",
    image_size: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Predicts whether an individual packaging image exhibits packaging damage.

    Parameters
    ----------
    image_path : str
        Path to the query image file (e.g. JPEG or PNG).
    model : Union[str, Pipeline], default="Binary_Classification/results/binary_model.joblib"
        Either a pre-loaded Scikit-Learn Pipeline instance or the file path to a saved .joblib model.
    image_size : Optional[Tuple[int, int]], default=None
        Spatial resolution. If None, it is automatically inferred from the model pipeline.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing prediction results:
        - 'image_path' (str): Absolute path to the analyzed image.
        - 'is_damaged' (bool): True if classified as damaged (class 1), False if undamaged (class 0).
        - 'classification_label' (str): "Damaged Packaging" or "Undamaged Packaging".
        - 'damage_probability' (float): Probability score of damage in range [0.0, 1.0].
        - 'undamaged_probability' (float): Probability score of undamaged packaging in [0.0, 1.0].
        - 'confidence' (float): Confidence score of the chosen class [0.5, 1.0].
        - 'image_size_used' (Tuple[int, int]): The spatial dimensions used during feature extraction.

    Raises
    ------
    FileNotFoundError
        If the input image file cannot be found.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: '{image_path}'")

    # Resolve pipeline instance
    if isinstance(model, str):
        pipeline = load_binary_model(model)
    elif hasattr(model, "predict"):
        pipeline = model
    else:
        raise TypeError(f"Expected model to be a file path str or fitted Pipeline, received: {type(model)}")

    # Auto-infer image size from model if not provided
    if image_size is None:
        resolved_image_size = infer_image_size_from_model(pipeline)
    else:
        resolved_image_size = image_size

    # Extract normalised flattened feature vector
    features = extract_image_features(image_path, image_size=resolved_image_size)
    features_2d = features.reshape(1, -1)

    # Predict class and probabilities
    prediction = int(pipeline.predict(features_2d)[0])

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features_2d)[0]
        prob_undamaged = float(probabilities[0])
        prob_damaged = float(probabilities[1])
    else:
        prob_damaged = float(prediction)
        prob_undamaged = 1.0 - prob_damaged

    is_damaged = bool(prediction == 1)
    label_str = "Damaged Packaging" if is_damaged else "Undamaged Packaging"
    confidence = prob_damaged if is_damaged else prob_undamaged

    return {
        "image_path": os.path.abspath(image_path),
        "is_damaged": is_damaged,
        "classification_label": label_str,
        "damage_probability": round(prob_damaged, 4),
        "undamaged_probability": round(prob_undamaged, 4),
        "confidence": round(confidence, 4),
        "image_size_used": list(resolved_image_size),
    }


def predict_file(
    file_path: str,
    model: Union[str, Pipeline] = "Binary_Classification/results/binary_model.joblib",
    image_size: Optional[Tuple[int, int]] = None
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Performs inference on a given file.

    The file can be:
    1. A single image file (.jpg, .jpeg, .png, .bmp, .webp, etc.):
       Returns a single prediction dictionary.
    2. A text or CSV file (.txt, .csv) containing paths to images (one per line):
       Returns a list of prediction dictionaries for all valid image paths in the file.

    Parameters
    ----------
    file_path : str
        Path to the target input file.
    model : Union[str, Pipeline], default="Binary_Classification/results/binary_model.joblib"
        Fitted Pipeline or file path to saved model.
    image_size : Optional[Tuple[int, int]], default=None
        Spatial resolution. If None, it is automatically inferred from the model.

    Returns
    -------
    Union[Dict[str, Any], List[Dict[str, Any]]]
        Single prediction dict (for an image file) or list of dicts (for a text list file).

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file format is unrecognized.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: '{file_path}'")

    ext = os.path.splitext(file_path)[1].lower()

    # Pre-load model once
    pipeline = load_binary_model(model) if isinstance(model, str) else model
    resolved_size = image_size if image_size is not None else infer_image_size_from_model(pipeline)

    # Case 1: Direct image file
    if ext in IMAGE_EXTENSIONS:
        return predict_image(file_path, model=pipeline, image_size=resolved_size)

    # Case 2: Text / CSV file listing image paths
    if ext in TEXT_LIST_EXTENSIONS:
        image_paths: List[str] = []
        base_dir = os.path.dirname(os.path.abspath(file_path))

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                # Support CSV rows: take the first column
                candidate = stripped.split(",")[0].strip().strip('"').strip("'")
                # Handle relative paths relative to the text file location or current working directory
                if not os.path.isabs(candidate):
                    rel_candidate = os.path.join(base_dir, candidate)
                    if os.path.exists(rel_candidate):
                        candidate = rel_candidate
                image_paths.append(candidate)

        results: List[Dict[str, Any]] = []
        for img_p in image_paths:
            try:
                pred = predict_image(img_p, model=pipeline, image_size=resolved_size)
                results.append(pred)
            except Exception as err:
                results.append({
                    "image_path": os.path.abspath(img_p) if os.path.exists(img_p) else img_p,
                    "error": str(err),
                    "is_damaged": None,
                    "classification_label": "Error",
                    "damage_probability": 0.0,
                    "undamaged_probability": 0.0,
                    "confidence": 0.0,
                    "image_size_used": list(resolved_size),
                })
        return results

    # Unrecognized extension - attempt image opening as a fallback
    try:
        return predict_image(file_path, model=pipeline, image_size=resolved_size)
    except Exception:
        raise ValueError(
            f"Unsupported file format '{ext}' for file '{file_path}'. "
            f"Supported image formats: {sorted(list(IMAGE_EXTENSIONS))}; "
            f"Supported list formats: {sorted(list(TEXT_LIST_EXTENSIONS))}."
        )


def predict_directory(
    directory_path: str,
    model: Union[str, Pipeline] = "Binary_Classification/results/binary_model.joblib",
    image_size: Optional[Tuple[int, int]] = None,
    recursive: bool = False
) -> List[Dict[str, Any]]:
    """
    Processes all images within a directory and returns prediction records.

    Parameters
    ----------
    directory_path : str
        Path to the folder containing image files.
    model : Union[str, Pipeline], default="Binary_Classification/results/binary_model.joblib"
        Fitted Pipeline or file path to saved model.
    image_size : Optional[Tuple[int, int]], default=None
        Spatial resolution. Automatically inferred from model if None.
    recursive : bool, default=False
        Whether to search subdirectories recursively.

    Returns
    -------
    List[Dict[str, Any]]
        List of prediction outcome dictionaries for all valid images found.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    """
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: '{directory_path}'")

    pipeline = load_binary_model(model) if isinstance(model, str) else model
    resolved_size = image_size if image_size is not None else infer_image_size_from_model(pipeline)

    image_paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(directory_path):
            for file in sorted(files):
                if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(root, file))
    else:
        for file in sorted(os.listdir(directory_path)):
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(directory_path, file))

    results: List[Dict[str, Any]] = []
    for img_path in image_paths:
        try:
            pred = predict_image(img_path, model=pipeline, image_size=resolved_size)
            results.append(pred)
        except Exception as err:
            results.append({
                "image_path": os.path.abspath(img_path),
                "error": str(err),
                "is_damaged": None,
                "classification_label": "Error",
                "damage_probability": 0.0,
                "undamaged_probability": 0.0,
                "confidence": 0.0,
                "image_size_used": list(resolved_size),
            })

    return results


def predict(
    target: str,
    model: Union[str, Pipeline] = "Binary_Classification/results/binary_model.joblib",
    image_size: Optional[Tuple[int, int]] = None
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Unified high-level prediction function.

    Automatically handles:
    - An image file path
    - A text/CSV list file path
    - A folder / directory path

    Parameters
    ----------
    target : str
        Path to image, list file, or directory.
    model : Union[str, Pipeline]
        Model instance or model file path.
    image_size : Optional[Tuple[int, int]], default=None
        Image resolution. Automatically inferred from model if None.

    Returns
    -------
    Union[Dict[str, Any], List[Dict[str, Any]]]
        Inference results.
    """
    if not os.path.exists(target):
        raise FileNotFoundError(f"Target path does not exist: '{target}'")

    if os.path.isdir(target):
        return predict_directory(target, model=model, image_size=image_size)
    return predict_file(target, model=model, image_size=image_size)


def format_prediction_result(result: Dict[str, Any]) -> str:
    """
    Formats an individual inference dictionary into a readable CLI summary block.

    Parameters
    ----------
    result : Dict[str, Any]
        Dictionary output from predict_image.

    Returns
    -------
    str
        Formatted string summary.
    """
    filename = os.path.basename(result["image_path"])
    if result.get("error"):
        return f"File: {filename:<35} | Error: {result['error']}"

    status = "[DAMAGED]" if result["is_damaged"] else "[OK/UNDAMAGED]"
    return (
        f"File: {filename:<35} | Status: {status:<14} | "
        f"Confidence: {result['confidence']*100:5.1f}% | "
        f"Damage Prob: {result['damage_probability']:.4f}"
    )


def main():
    """
    Command-line interface execution for packaging defect classification inference.
    """
    parser = argparse.ArgumentParser(
        description="Run packaging damage neural network inference on any given image, file, or directory."
    )
    # Target can be supplied as a positional argument or via --input / --file / --image / --dir
    parser.add_argument(
        "input_target",
        nargs="?",
        default=None,
        help="Path to an image file, a text/CSV list file of image paths, or a directory of images."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        dest="file_flag",
        default=None,
        help="Path to an input file (single image or text/CSV list of image paths)."
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        dest="image_flag",
        default=None,
        help="Path to a single query packaging image (JPEG/PNG)."
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        dest="dir_flag",
        default=None,
        help="Path to a directory of images to process in batch."
    )
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        default="Binary_Classification/results/binary_model.joblib",
        help="Path to the saved .joblib model file (default: Binary_Classification/results/binary_model.joblib)."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Optional manual override for image size (default: None, auto-inferred from model)."
    )
    parser.add_argument(
        "--output_json", "-o",
        type=str,
        default=None,
        help="Optional path to save prediction results as a JSON file."
    )

    args = parser.parse_args()

    # Determine input target from arguments
    target = args.input_target or args.file_flag or args.image_flag or args.dir_flag

    if target is None:
        parser.print_help()
        print("\n[Error] Please provide an input image, file, or directory.")
        sys.exit(1)

    print("=" * 80)
    print("MMA3001 Packaging Defect Quality Inspection - Inference Engine (v2)")
    print("=" * 80)
    print(f"Target Input : {target}")
    print(f"Model Path   : {args.model_path}")

    # Check model path
    if not os.path.exists(args.model_path):
        print(f"\n[Error] Trained model file not found at: '{args.model_path}'.")
        print("Please train the model first using train_packaging_nn-v2.py or verify the path.")
        sys.exit(1)

    # Load model and display auto-detected image size
    pipeline = load_binary_model(args.model_path)
    if args.image_size is not None:
        size_tuple = (args.image_size, args.image_size)
        print(f"Image Size   : {args.image_size}x{args.image_size} (Manual override)")
    else:
        size_tuple = infer_image_size_from_model(pipeline)
        print(f"Image Size   : {size_tuple[0]}x{size_tuple[1]} (Auto-inferred from model)")
    print("-" * 80)

    try:
        results = predict(target, model=pipeline, image_size=size_tuple)
    except Exception as exc:
        print(f"\n[Error during inference] {exc}")
        sys.exit(1)

    if isinstance(results, dict):
        # Single image result
        print("\nInference Result:")
        print("-" * 50)
        print(f"  Target Image        : {results['image_path']}")
        print(f"  Classification      : {results['classification_label']}")
        status_label = "DAMAGED (Defect detected)" if results['is_damaged'] else "UNDAMAGED (Quality passed)"
        print(f"  Damage Status       : {status_label}")
        print(f"  Damage Probability  : {results['damage_probability']*100:.2f}%")
        print(f"  Classification Conf : {results['confidence']*100:.2f}%")
        print(f"  Resolution Used     : {results['image_size_used'][0]}x{results['image_size_used'][1]}")
        print("=" * 80)
        export_payload = results
    else:
        # Multiple results (file list or directory)
        total = len(results)
        damaged_count = sum(1 for r in results if r.get("is_damaged") is True)
        undamaged_count = sum(1 for r in results if r.get("is_damaged") is False)
        error_count = sum(1 for r in results if r.get("error") is not None)

        print(f"\nProcessed {total} items:")
        print("-" * 80)
        for r in results:
            print(format_prediction_result(r))
        print("-" * 80)
        print(f"Summary: Damaged: {damaged_count} | Undamaged: {undamaged_count} | Errors: {error_count}")
        print("=" * 80)
        export_payload = {
            "total_samples": total,
            "damaged_count": damaged_count,
            "undamaged_count": undamaged_count,
            "error_count": error_count,
            "results": results
        }

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=4)
        print(f"[+] Prediction records exported to: {args.output_json}")


if __name__ == "__main__":
    main()
