"""
Neural Network model pipeline for binary packaging defect classification.

This module builds, trains, and evaluates a Scikit-Learn Multilayer Perceptron (MLP) pipeline
for detecting packaging damage in frozen meat products.

MMA3001 Project Brief & Week 05 Notes Alignment:
- Feature scaling via StandardScaler in a Scikit-learn Pipeline to condition neural network optimization.
- L2 Regularisation (alpha) to control model capacity and weight magnitude.
- Early stopping to monitor internal validation loss and prevent overfitting.
- Comprehensive model evaluation using Precision, Recall, F1-score, Accuracy, and ROC-AUC metrics.
"""

from typing import Dict, Any, Tuple
import numpy as np
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
)

from dataset import extract_image_features


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


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Dict[str, Any]:
    """
    Trains the MLP pipeline on training data and computes classification metrics on validation data.

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

    Returns
    -------
    Dict[str, Any]
        Dictionary containing validation performance metrics:
        'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', and 'confusion_matrix'.

    Raises
    ------
    ValueError
        If training data is empty or inputs have mismatched lengths.
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training dataset X_train and y_train cannot be empty.")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    # Calculate probabilities if available
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        try:
            roc_auc = float(roc_auc_score(y_val, y_proba))
        except ValueError:
            # Handle single-class validation set case
            roc_auc = 0.5
    else:
        roc_auc = 0.5

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
    }

    return metrics


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
