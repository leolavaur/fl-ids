"""Metrics for evaluating model performance."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from flwr.server.history import History as FlwrHistory
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from eiffel.utils.typing import EiffelCID, MetricsDict, NDArray


class History:
    """History of a model training session.

    The History object contains the training and evaluation metrics of a model training
    session.
    """

    def __init__(
        self,
        fit: dict[EiffelCID, dict[int, MetricsDict]] = {},
        distributed: dict[EiffelCID, dict[int, MetricsDict]] = {},
        centralized: dict[int, MetricsDict] = {},
    ) -> None:
        """Initialize a History object.

        Parameters
        ----------
        fit : dict[EiffelCID, dict[int, MetricsDict]], optional
            Training metrics of the clients, by client ID and round number.
        distributed : dict[EiffelCID, dict[int, MetricsDict]], optional
            Evaluation metrics of the clients, by client ID and round number.
        centralized : dict[int, MetricsDict], optional
            Evaluation metrics on the server, by round number.
        """
        self.fit = fit
        self.distributed = distributed
        self.centralized = centralized

    @classmethod
    def from_flwr(cls, flwr_history: FlwrHistory) -> "History":
        """Create a History object from a Flower History object.

        Parameters
        ----------
        flwr_history : flwr.server.history.History
            The Flower History object.

        Returns
        -------
        History
            The Eiffel History object.
        """
        fit = {}
        distributed = {}
        centralized = {}

        for cid, metrics in flwr_history.metrics_distributed_fit.items():
            fit[cid] = {}
            for round, json_metrics in metrics:
                fit[cid][round] = json.loads(str(json_metrics))

        for cid, metrics in flwr_history.metrics_distributed.items():
            distributed[cid] = {}
            for round, json_metrics in metrics:
                distributed[cid][round] = json.loads(str(json_metrics))

        for round, json_metrics in flwr_history.metrics_centralized:
            centralized[round] = json.loads(str(json_metrics))

        return cls(fit=fit, distributed=distributed, centralized=centralized)

    def save(
        self, key: str, path: Path | str | None = None, filename: str = "metrics.json"
    ) -> None:
        """Save the history to a JSON file.

        Parameters
        ----------
        key : str, optional
            The key to use to save the history. If `None`, the history is saved as
            `history.json`. Defaults to `None`.
        path : Path | str, optional
            The path to the file to save the history to. If `None`, the history is saved
            in the current working directory. Defaults to `None`.
        """
        content = getattr(self, key)

        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)

        file = path / filename
        file.write_text(json.dumps(content, indent=4))


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
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "missrate": fn / (fn + tp) if (fn + tp) != 0 else 0,
        "fallout": fp / (fp + tn) if (fp + tn) != 0 else 0,
    }
