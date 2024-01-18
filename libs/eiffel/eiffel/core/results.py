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


class Series(dict[int, MetricsDict]):
    """A dictionary of metrics, by round number."""

    def __init__(self, *args, **kwargs):
        """Initialize a Series."""
        d = dict(*args, **kwargs)
        try:
            super().__init__({int(k): v for k, v in d.items()})
        except ValueError:
            raise ValueError("Invalid key in Series, must be an int")

    def __setitem__(self, __key: Any, __value: Any) -> None:
        """Set an item in the dictionary."""
        try:
            return super().__setitem__(int(__key), MetricsDict(__value))
        except ValueError:
            raise ValueError("Invalid typing: key must be an int, value a MetricsDict")

    def metric(self, metric: str) -> list[float]:
        """Get the given metric."""
        return [metrics[metric] for metrics in self.values()]


class MetaSeries(dict[int, dict[ScopeName, MetricsDict]]):
    """A dictionary of Series, by round and scope name.

    The scope is a "class" filter for the metrics. For example, 'Bot' contains metrics
    for the 'Bot' class. 'global' contains metrics for the whole dataset.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a MetaSeries."""
        d = dict(*args, **kwargs)
        try:
            super().__init__({int(k): v for k, v in d.items()})
        except ValueError:
            raise ValueError("Invalid key in Series, must be an int")

    def scope(self, scope: ScopeName) -> Series:
        """Return a Series for the given scope."""
        return Series(
            {
                round: metrics[scope]
                for round, metrics in self.items()
                if scope in metrics
            }
        )


class Results:
    """Results of a model training session.

    The Results object contains the training and evaluation metrics of a model training
    session.
    """

    def __init__(
        self,
        fit: dict[EiffelCID, MetaSeries] = {},
        distributed: dict[EiffelCID, MetaSeries] = {},
        centralized: dict | MetaSeries = MetaSeries(),
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
            if all(isinstance(value, MetaSeries) for value in fit.values())
            else {k: MetaSeries(v) for k, v in fit.items()}
        )
        self.distributed = (
            distributed
            if all(isinstance(value, MetaSeries) for value in distributed.values())
            else {k: MetaSeries(v) for k, v in distributed.items()}
        )
        self.centralized = (
            centralized
            if isinstance(centralized, MetaSeries)
            else MetaSeries(centralized)
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

        return cls(fit=fit, distributed=distributed, centralized=Series(centralized))

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

    def scope(self, scope: ScopeName) -> "ScopedResults":
        """Return a ScopedResults object with the given scope."""
        return ScopedResults(
            fit={
                cid: meta_series.scope(scope) for cid, meta_series in self.fit.items()
            },
            distributed={
                cid: meta_series.scope(scope)
                for cid, meta_series in self.distributed.items()
            },
            centralized=self.centralized.scope(scope),
        )


class ScopedResults:
    """Results of a model training session for a given scope.

    The scoped results are a view over the original results, with a given scope. Where a
    Results object contains MetaSeries with different scopes, the ScopedResults object
    will only contain Series with the given scope.
    """

    def __init__(
        self,
        fit: dict[EiffelCID, Series] = {},
        distributed: dict[EiffelCID, Series] = {},
        centralized: dict | Series = Series(),
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
            if all(isinstance(value, Series) for value in fit.values())
            else {k: Series(v) for k, v in fit.items()}
        )
        self.distributed = (
            distributed
            if all(isinstance(value, Series) for value in distributed.values())
            else {k: Series(v) for k, v in distributed.items()}
        )
        self.centralized = (
            centralized if isinstance(centralized, Series) else Series(centralized)
        )
