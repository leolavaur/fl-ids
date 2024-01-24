"""Metrics for evaluating model performance."""

import json
import numbers
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

import numpy as np
import pandas as pd
from flwr.server.history import History as FlwrHistory
from schema import And, Schema, Use
from schema import Optional as SchemaOpt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from eiffel.utils.typing import EiffelCID, MetricsDict, NDArray

ScopeName = str

GLOBAL_METRICS = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "missrate",
    "fallout",
    "loss",
]

TARGET_METRICS = ["recall", "missrate"]

MetricsSchema = Schema(
    {
        Use(int): {
            str: {
                And(str, lambda s: s in GLOBAL_METRICS): And(
                    Use(float), lambda f: 0 <= f <= 1
                )
            }
        }
    }
)


class Results:
    """Results of a model training session.

    The Results object contains the training and evaluation metrics of a model training
    session.
    """

    def __init__(
        self,
        fit: dict[EiffelCID, dict] = {},
        distributed: dict[EiffelCID, dict] = {},
        centralized: dict | dict = dict(),
    ) -> None:
        """Initialize a Results object.

        Parameters
        ----------
        fit : dict[EiffelCID, dict[int, MetricsDict]], optional
            Training metrics of the clients, by client ID and round number.
        distributed : dict[EiffelCID, dict[int, MetricsDict]], optional
            Evaluation metrics of the clients, by client ID and round number.
        centralized : dict[int, MetricsDict], optional
            Evaluation metrics on the server, by round number.
        """
        self.fit = (
            fit
            if all(isinstance(value, dict) for value in fit.values())
            else {k: dict(v) for k, v in fit.items()}
        )
        self.distributed = (
            distributed
            if all(isinstance(value, dict) for value in distributed.values())
            else {k: dict(v) for k, v in distributed.items()}
        )
        self.centralized = (
            centralized if isinstance(centralized, dict) else dict(centralized)
        )

    @classmethod
    def from_flwr(cls, flwr_history: FlwrHistory) -> "Results":
        """Create a Results object from a Flower History object.

        Parameters
        ----------
        flwr_history : flwr.server.history.History
            The Flower History object.

        Returns
        -------
        Results
            The Eiffel Results object.
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

    @classmethod
    def from_path(cls, result_path: str | Path) -> "Results":
        """Create a Results object from a previous saved results.

        Parameters
        ----------
        path : str
            The path to the results.

        Returns
        -------
        Results
            The Eiffel Results object.
        """
        path = Path(result_path)
        fit, distributed, centralized = {}, {}, {}
        if (fit_path := path / "fit.json").exists():
            fit = json.loads(fit_path.read_text())
        if (distributed_path := path / "distributed.json").exists():
            distributed = json.loads(distributed_path.read_text())
        if (centralized_path := path / "centralized.json").exists():
            centralized = json.loads(centralized_path.read_text())

        return cls(fit=fit, distributed=distributed, centralized=centralized)

    def save(
        self,
        key: Optional[str] = None,
        path: Optional[Path | str] = None,
        filename: Optional[str] = None,
    ) -> None:
        """Save the history to a JSON file.

        Parameters
        ----------
        key : str
            The key to the metrics to save.
        path : Path | str | None, optional
        """
        if key is None:
            content = self.__dict__
        else:
            content = getattr(self, key)

        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)

        if filename is None:
            filename = f"{key or 'metrics'}.json"

        file = path / filename
        file.write_text(json.dumps(content, indent=4))

    def scope(self, scope: ScopeName) -> "Results":
        """Return a ScopedResults object with the given scope."""
        return Results(
            fit={
                cid: meta_series.scope(scope) for cid, meta_series in self.fit.items()
            },
            distributed={
                cid: meta_series.scope(scope)
                for cid, meta_series in self.distributed.items()
            },
            centralized=self.centralized.scope(scope),
        )
