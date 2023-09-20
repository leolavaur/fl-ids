"""Metrics for evaluating model performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from eiffel.utils.typing import MetricsDict, NDArray


def mean_absolute_error(x_orig: pd.DataFrame, x_pred: pd.DataFrame) -> np.ndarray:
    """Mean absolute error.

    Parameters
    ----------
    x_orig : pd.DataFrame
        True labels.
    x_pred : pd.DataFrame
        Predicted labels.

    Returns
    -------
    ndarray[float]
        Mean absolute error.
    """
    return np.mean(np.abs(x_orig - x_pred), axis=1)


def mean_squared_error(x_orig: pd.DataFrame, x_pred: pd.DataFrame) -> np.ndarray:
    """Mean squared error.

    Parameters
    ----------
    x_orig : pd.DataFrame
        True labels.
    x_pred : pd.DataFrame
        Predicted labels.

    Returns
    -------
    ndarray[float]
        Mean squared error.
    """
    return np.mean((x_orig - x_pred) ** 2, axis=1)


def root_mean_squared_error(x_orig: pd.DataFrame, x_pred: pd.DataFrame) -> np.ndarray:
    """Root mean squared error.

    Parameters
    ----------
    x_orig : pd.DataFrame
        True labels.
    x_pred : pd.DataFrame
        Predicted labels.

    Returns
    -------
    ndarray[float]
        Root mean squared error.
    """
    return np.sqrt(np.mean((x_orig - x_pred) ** 2, axis=1))


def metrics_from_preds(y_true: NDArray, y_pred: NDArray) -> MetricsDict:
    """Evaluate the predictions of a model.

    Parameters
    ----------
    y_true : NDArray
        True labels.
    y_pred : NDArray
        Predicted labels.

    Returns
    -------
    Dict[str, float]
        Dictionary with the evaluation metrics (accuracy, precision, recall, f1, mcc,
        missrate, fallout).
    """
    conf = confusion_matrix(y_true, y_pred)
    tn = conf[0][0]
    fp = conf[0][1]
    fn = conf[1][0]
    tp = conf[1][1]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "missrate": fn / (fn + tp) if (fn + tp) != 0 else 0,
        "fallout": fp / (fp + tn) if (fp + tn) != 0 else 0,
    }
