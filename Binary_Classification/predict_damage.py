"""
Inference script to evaluate packaging quality and damage classification on new images.

This module provides standalone inference capabilities for the packaging damage
binary classification neural network. It can evaluate a single packaging image
or batch-process entire directories of images, producing confidence scores and damage classifications.

MMA3001 Project Brief Alignment:
- Implements an executable prediction workflow for demonstration and deployment.
- Incorporates robust exception handling, path validation, and Google-style docstrings.
- Supports both programmatic API usage and command-line interface (CLI) execution.
"""

import os
import argparse
import json
from typing import Dict, Any, Tuple, List, Union
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


def predict_image(
    image_path: str,
    model: Union[str, Pipeline] = "Binary_Classification/results/binary_model.joblib",
    image_size: Tuple[int, int] = (32, 32)
) -> Dict[str, Any]:
    """
    Predicts whether an individual packaging image exhibits damage.

    Parameters
    ----------
    image_path : str
        Path to the query image file (e.g. JPEG or PNG).
    model : Union[str, Pipeline], default="Binary_Classification/results/binary_model.joblib"
        Either a pre-loaded Scikit-Learn Pipeline instance or the file path to a saved .joblib model.
    image_size : Tuple[int, int], default=(32, 32)
        Spatial resolution matching the input dimensions used during training.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing prediction results:
        - 'image_path' (str): Absolute path to the analyzed image.
        - 'is_damaged' (bool): True if classified as damaged (class 1), False if undamaged (class 0).
        - 'classification_label' (str): "Damaged Packaging" or "Undamaged Packaging".
        - 'damage_probability' (float): Probability score of damage in range [0.0, 1.0].
        - 'undamaged_probability' (float): Probability score of undamaged packaging.
        - 'confidence' (float): Confidence score of the chosen class [0.5, 1.0].

    Raises
    ------
    FileNotFoundError
        If the input image file cannot be found.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: '{image_path}'")

    # Resolve model instance
    if isinstance(model, str):
        pipeline = load_binary_model(model)
    elif hasattr(model, "predict"):
        pipeline = model
    else:
        raise TypeError(f"Expected model to be a file path str or fitted Pipeline, received: {type(model)}")

    # Extract normalised flattened feature vector
    features = extract_image_features(image_path, image_size=image_size)
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
    }


def predict_directory(
    directory_path: str,
    model: Union[str, Pipeline] = "Binary_Classification/results/binary_model.joblib",
    image_size: Tuple[int, int] = (32, 32),
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
    image_size : Tuple[int, int], default=(32, 32)
        Input spatial resolution.
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

    # Load model once for batch processing efficiency
    if isinstance(model, str):
        pipeline = load_binary_model(model)
    else:
        pipeline = model

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths: List[str] = []

    if recursive:
        for root, _, files in os.walk(directory_path):
            for file in sorted(files):
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_paths.append(os.path.join(root, file))
    else:
        for file in sorted(os.listdir(directory_path)):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(directory_path, file))

    results: List[Dict[str, Any]] = []
    for img_path in image_paths:
        try:
            pred = predict_image(img_path, model=pipeline, image_size=image_size)
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
            })

    return results


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
        description="Run packaging damage neural network inference on an image or directory of images."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=str,
        help="Path to a single query packaging image (JPEG/PNG)."
    )
    group.add_argument(
        "--dir",
        type=str,
        help="Path to a directory of images to process in batch."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Binary_Classification/results/binary_model.joblib",
        help="Path to the saved .joblib model file (default: Binary_Classification/results/binary_model.joblib)."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="Target image size matching training dimensions (default: 32)."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when --dir is provided."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to export inference results as a JSON file."
    )

    args = parser.parse_args()
    img_size_tuple = (args.image_size, args.image_size)

    print("=" * 80)
    print("MMA3001 Packaging Defect Quality Inspection - Inference Engine")
    print("=" * 80)
    print(f"Model Path : {args.model_path}")
    print(f"Image Size : {args.image_size}x{args.image_size}")
    print("-" * 80)

    try:
        pipeline = load_binary_model(args.model_path)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    if args.image:
        try:
            res = predict_image(args.image, model=pipeline, image_size=img_size_tuple)
            print("\nInference Result:")
            print("-" * 50)
            print(f"  Target Image        : {res['image_path']}")
            print(f"  Classification      : {res['classification_label']}")
            print(f"  Damage Status       : {'DAMAGED (Defect detected)' if res['is_damaged'] else 'UNDAMAGED (Quality passed)'}")
            print(f"  Damage Probability  : {res['damage_probability'] * 100:.2f}%")
            print(f"  Classification Conf : {res['confidence'] * 100:.2f}%")
            print("=" * 80)

            if args.output_json:
                with open(args.output_json, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=4)
                print(f"[+] Exported prediction result to: {args.output_json}")

        except Exception as err:
            print(f"[Error during inference] {err}")

    elif args.dir:
        try:
            results = predict_directory(
                args.dir,
                model=pipeline,
                image_size=img_size_tuple,
                recursive=args.recursive
            )

            print(f"\nBatch Inference Results ({len(results)} images found):")
            print("-" * 80)
            damaged_count = 0
            undamaged_count = 0
            error_count = 0

            for r in results:
                if r.get("error"):
                    error_count += 1
                    print(f"File: {os.path.basename(r['image_path']):<35} | Error: {r['error']}")
                else:
                    if r["is_damaged"]:
                        damaged_count += 1
                    else:
                        undamaged_count += 1
                    print(format_prediction_result(r))

            print("-" * 80)
            print(f"Summary: Total: {len(results)} | Damaged: {damaged_count} | Undamaged: {undamaged_count} | Errors: {error_count}")
            print("=" * 80)

            if args.output_json:
                with open(args.output_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)
                print(f"[+] Exported batch results to: {args.output_json}")

        except Exception as err:
            print(f"[Error processing directory] {err}")


if __name__ == "__main__":
    main()
