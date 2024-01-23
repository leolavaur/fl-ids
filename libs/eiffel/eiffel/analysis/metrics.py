"""Utilities to process Eiffel metrics."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from eiffel.core.results import MetaSeries, Results, Series


def average(ss: list[Series | MetaSeries] | dict[str, Series | MetaSeries]) -> Series:
    """Compute the mean, metric per metric and for each round, of a list of Series.

    Parameters
    ----------
    ss : List[Series] | dict[str, Series]
        Series to average. If a dict is given, the values are considered to be the
        Series, and the keys are ignored.

    Returns
    -------
    Series
        The mean of the Series.
    """
    lst = ss if isinstance(ss, list) else list(ss.values())

    if len(lst) == 0:
        raise ValueError("lst must not be empty")

    for s in lst:
        if not isinstance(s, Series):
            raise ValueError("ss must contain only Series")
        if not s.keys() == lst[0].keys():
            raise ValueError("all Series must have the same rounds")
        if not next(iter(s.values())).keys() == next(iter(lst[0].values())).keys():
            raise ValueError("all Series must have the same metrics")

    rounds = lst[0].keys()
    metrics = next(iter(lst[0].values())).keys()

    m_bar = Series()
    for round in rounds:
        m_bar[round] = {}
        for metric in metrics:
            m_bar[round][metric] = float(np.mean([s[round][metric] for s in lst]))

    return dict_avg(ss)


def dict_avg(ss: list[dict]) -> dict:
    """Recursively average a list of dicts."""
    d = {}
    for k in ss[0].keys():
        if isinstance(ss[0][k], dict):
            d[k] = dict_avg([s[k] for s in ss])
        else:
            d[k] = np.mean([s[k] for s in ss])
    return d


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


def metrics_from_confmat(*conf: int) -> dict[str, float]:
    """Translate a confusion matrix into metrics.

    Parameters
    ----------
    conf : tuple[int]
        The confusion matrix, under the form (tn, fp, fn, tp).

    Returns
    -------
    dict[str, float]
        Dictionary with the evaluation metrics (accuracy, precision, recall, f1,
        missrate, fallout).
    """
    tn, fp, fn, tp = conf

    return {
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "precision": float(tp / (tp + fp)) if (tp + fp) != 0 else 0,
        "recall": float(tp / (tp + fn)) if (tp + fn) != 0 else 0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) != 0 else 0,
        "missrate": float(fn / (fn + tp)) if (fn + tp) != 0 else 0,
        "fallout": float(fp / (fp + tn)) if (fp + tn) != 0 else 0,
    }


def load_attr_metric(
    path: str, attr: str, metric="accuracy", with_malicious=False
) -> list[float]:
    """Load the metrics from the given path."""
    res = getattr(Results.from_path(path).scope("global"), attr)
    if not with_malicious:
        res = {k: v for k, v in res.items() if "malicious" not in k}
    return average(res).metric(metric)


def load_metrics(
    *paths: str, metric="accuracy", with_malicious=False
) -> dict[str, tuple[list[float], list[float]]]:
    """Load the metrics from the given paths."""
    metrics = {}
    for path in paths:
        p = Path(path)
        res = Results.from_path(path).scope("global")
        metrics[p.name] = (
            load_attr_metric(path, "fit", metric, with_malicious),
            load_attr_metric(path, "distributed", metric, with_malicious),
        )
    return metrics


def search_metrics(path: str, sort: bool = True, **conditions: str) -> list[str]:
    """
    Search the metrics from the given path.

    Parameters
    ----------
    path : str
        The path to the metrics.
    conditions : dict[str, str]
        The conditions to search the metrics. The keys are the names of hydra options
        that have been set, and the values are the values of the options. For example,
        if conditions is set to `{"distribution": "10-0"}`, then the fonctions will load
        all experiments that have been run with `distribution=10-0`. Values can also be
        regex patterns. For example, if conditions is set to `{"distribution":
        "10-.*"}`, then the fonctions will load all experiments that have been run with
        `distribution` starting with `10-`.

    Returns
    -------
    list[str]
        The paths to the metrics.
    """
    for _, v in conditions.items():
        v = str(v)
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"{path} is not a directory.")

    metrics: list[str] = []

    for d in p.iterdir():
        if not d.is_dir():
            continue
        options = {
            k.strip("+"): v for k, v in [p.split("=") for p in d.name.split(",")]
        }
        if all(
            re.match(v, options.get(k, "")) is not None for k, v in conditions.items()
        ):
            metrics.append(d.as_posix())

    return sorted(metrics) if sort else metrics


def avg(cond: str, lines: dict[str, list[float]]) -> dict[str, list[float]]:
    """Average the lines on the specified condition."""
    # extract the existing values of the condition
    values: list[str] = []
    for k in lines:
        conditions = k.split(",")
        for c in conditions:
            if c.startswith(cond):
                values.append(c.split("=")[1])
    values = list(set(values))

    new_lines = {}
    for variant in values:
        for k in lines:
            if f"{cond}={variant}" in k:
                # remove the condition from the name
                new_name = ",".join(
                    [c for c in k.split(",") if c != f"{cond}={variant}"]
                )
                if f"{cond}={variant}" not in new_lines:
                    new_lines[f"{cond}={variant}"] = {}
                new_lines[f"{cond}={variant}"][new_name] = lines[k]

    avgs = {}
    for name in next(iter(new_lines.values())):
        avgs[name] = []
        for variant in new_lines:
            avgs[name].append(new_lines[variant][name])
        avgs[name] = [sum(m) / len(m) for m in zip(*avgs[name])]
    return avgs


def choices(path: str) -> dict[str, list[str]]:
    """Return the available choices for each condition."""
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"{path} is not a directory.")

    choices: dict[str, list[str]] = {}

    for d in p.iterdir():
        if not d.is_dir():
            continue
        options = {
            k.strip("+"): v for k, v in [p.split("=") for p in d.name.split(",")]
        }
        for k, v in options.items():
            if k not in choices:
                choices[k] = []
            if v not in choices[k]:
                choices[k].append(v)

    return choices


def display_choices(d: dict[str, list[str]]) -> None:
    """Display the choices."""
    display(
        HTML(
            "<style>table td, table th, table tr {text-align:left !important;}</style>"
            + "<table><tr><th>Key</th><th>Values</th></tr>"
            + "".join(
                f"<tr><td>{k}</td><td>{', '.join(v)}</td></tr>" for k, v in d.items()
            )
            + "</table>"
        )
    )
