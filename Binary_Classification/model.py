"""
Neural Network model pipeline for binary packaging defect classification.

This module builds, trains, evaluates, saves, and loads a Scikit-Learn
Multilayer Perceptron (MLP) pipeline for detecting packaging damage in frozen meat products.

MMA3001 Project Brief & Week 05 Notes Alignment:
- Feature scaling via StandardScaler in a Scikit-learn Pipeline to condition neural network optimization.
- L2 Regularisation (alpha) to control model capacity and weight magnitude.
- Early stopping to monitor internal validation loss and prevent overfitting.
- Model persistence via joblib for reproducible deployment and verification.
- Comprehensive model evaluation across validation and test sets (Precision, Recall, F1, ROC-AUC).
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import joblib
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Robust import of extract_image_features from dataset
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
        raise ImportError("Could not find dataset.py module.")


def build_mlp_pipeline(
    hidden_layer_sizes: Tuple[int, ...] = (30, 15),
    alpha: float = 0.01,
    learning_rate_init: float = 0.001,
    max_iter: int = 500,
    random_state: int = 42,
    early_stopping: bool = True
) -> Pipeline:
    """
    Constructs a Scikit-Learn Pipeline integrating StandardScaler and MLPClassifier.

    Parameters
    ----------
    hidden_layer_sizes : Tuple[int, ...], default=(30, 15)
        Number of neurons in each hidden layer.
    alpha : float, default=0.01
        L2 regularization penalty parameter.
    learning_rate_init : float, default=0.001
        Initial learning rate for Adam optimizer.
    max_iter : int, default=500
        Maximum number of optimization epochs.
    random_state : int, default=42
        Seed for weight initialization reproducibility.
    early_stopping : bool, default=True
        Whether to use early stopping to terminate training when internal validation score stops improving.

    Returns
    -------
    Pipeline
        A scikit-learn Pipeline with 'standardscaler' and 'mlpclassifier' steps.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=random_state,
    )

    return make_pipeline(StandardScaler(), mlp)


def evaluate_model(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates a fitted classification pipeline on feature matrix X against true targets y.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted Scikit-Learn classification pipeline.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Binary target ground truth vector (0 or 1).

    Returns
    -------
    Dict[str, Any]
        Dictionary of performance metrics:
        - 'accuracy' (float)
        - 'precision' (float)
        - 'recall' (float)
        - 'f1_score' (float)
        - 'roc_auc' (float)
        - 'confusion_matrix' (list of lists)
        - 'classification_report' (str)
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Cannot evaluate model on empty features X or labels y.")

    y_pred = pipeline.predict(X)

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X)[:, 1]
        try:
            roc_auc = float(roc_auc_score(y, y_proba))
        except ValueError:
            # When only a single class is present in the split
            roc_auc = 0.5
    else:
        roc_auc = 0.5

    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, target_names=["Undamaged", "Damaged"], zero_division=0)

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Trains the MLP pipeline on training data and computes classification metrics
    on validation data and optionally on withheld test data.

    Parameters
    ----------
    pipeline : Pipeline
        The Scikit-Learn classification pipeline.
    X_train : np.ndarray
        Training feature matrix of shape (n_train_samples, n_features).
    y_train : np.ndarray
        Training binary target labels (0 or 1).
    X_val : np.ndarray
        Validation feature matrix of shape (n_val_samples, n_features).
    y_val : np.ndarray
        Validation binary target labels (0 or 1).
    X_test : Optional[np.ndarray], default=None
        Withheld test feature matrix of shape (n_test_samples, n_features).
    y_test : Optional[np.ndarray], default=None
        Withheld test binary target labels (0 or 1).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'validation' metrics, and 'test' metrics if test data is provided.
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training dataset X_train and y_train cannot be empty.")

    # Guard against early_stopping failure when training size is very small
    mlp_step = pipeline.named_steps.get("mlpclassifier")
    if mlp_step is not None and mlp_step.early_stopping and len(X_train) < 20:
        mlp_step.early_stopping = False

    pipeline.fit(X_train, y_train)

    val_metrics = evaluate_model(pipeline, X_val, y_val)
    results = {"validation": val_metrics}

    if X_test is not None and y_test is not None and len(X_test) > 0:
        test_metrics = evaluate_model(pipeline, X_test, y_test)
        results["test"] = test_metrics

    return results


def save_model(pipeline: Pipeline, filepath: str) -> str:
    """
    Saves the fitted Scikit-Learn pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted classification pipeline.
    filepath : str
        Target file path (e.g. 'Binary_Classification/results/binary_model.joblib').

    Returns
    -------
    str
        Absolute file path where the model was saved.
    """
    target_dir = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(target_dir, exist_ok=True)
    joblib.dump(pipeline, filepath)
    return os.path.abspath(filepath)


def load_model(filepath: str) -> Pipeline:
    """
    Loads a saved Scikit-Learn pipeline from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved .joblib model file.

    Returns
    -------
    Pipeline
        Loaded Scikit-Learn Pipeline.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Saved model file not found at: {filepath}")
    return joblib.load(filepath)


def predict_packaging_damage(
    pipeline: Pipeline,
    image_path: str,
    image_size: Tuple[int, int] = (32, 32)
) -> Dict[str, Any]:
    """
    Performs inference on a single packaging image to classify if packaging damage is present.

    Parameters
    ----------
    pipeline : Pipeline
        Trained classification pipeline.
    image_path : str
        Path to the query packaging JPEG image.
    image_size : Tuple[int, int], default=(32, 32)
        Spatial resolution matching the training pipeline input format.

    Returns
    -------
    Dict[str, Any]
        Inference results containing:
        - 'is_damaged' (bool): True if damaged (class 1), False if undamaged (class 0).
        - 'classification_label' (str): "Damaged Packaging" or "Undamaged Packaging".
        - 'damage_probability' (float): Predicted probability of packaging damage.
    """
    features = extract_image_features(image_path, image_size=image_size)
    features_2d = features.reshape(1, -1)

    prediction = int(pipeline.predict(features_2d)[0])
    proba = float(pipeline.predict_proba(features_2d)[0][1])

    is_damaged = bool(prediction == 1)
    label_str = "Damaged Packaging" if is_damaged else "Undamaged Packaging"

    return {
        "is_damaged": is_damaged,
        "classification_label": label_str,
        "damage_probability": proba,
    }
