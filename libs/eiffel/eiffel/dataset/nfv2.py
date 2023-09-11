"""NF-V2 Dataset utilities.

This module contains functions to load and prepare the NF-V2 dataset for
Deep Learning applications. The NF-V2 dataset is a collection of 4 datasets
with a standardised set of features. The datasets are:
    * CSE-CIC-IDS-2018
    * UNSW-NB15
    * ToN-IoT
    * Bot-IoT

The NF-V2 dataset is available at:
    https://staff.itee.uq.edu.au/marius/NIDS_datasets/

Part of the code in this module is based on the code from Bertoli et al. (2022),
who tested Federated Learning on the NF-V2 dataset.

The code is available at:
    https://github.com/c2dc/fl-unsup-nids

References
----------
    * Sarhan, M., Layeghy, S. & Portmann, M., Towards a Standard Feature Set for
      Network Intrusion Detection System Datasets. Mobile Netw Appl (2021).
      https://doi.org/10.1007/s11036-021-01843-0 
    * Bertoli, G., Junior, L., Santos, A., & Saotome, O., Generalizing intrusion
      detection for heterogeneous networks: A stacked-unsupervised federated
      learning approach. arXiv preprint arxiv:2209.00721 (2022).
      https://arxiv.org/abs/2209.00721
"""

import operator
from pathlib import Path
from tempfile import gettempdir
from typing import List, Optional, Tuple, overload

import numpy as np
import pandas as pd
from omegaconf import ListConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from eiffel.dataset import DEFAULT_SEARCH_PATH, Dataset
from eiffel.utils.logging import logged

from ..utils.poisoning import PoisonOp

# Shortcuts names for dataset paths.
# ----------------------------------
# In the data directory, datasets are organised as follows:
#   .../nfv2
#       ├── origin
#       │   ├── NF-BoT-IoT-v2.csv.gz
#       │   ├── NF-CSE-CIC-IDS2018-v2.csv.gz
#       │   ├── NF-ToN-IoT-v2.csv.gz
#       │   └── NF-UNSW-NB15-v2.csv.gz
#       ├── reduced
#       │   ├── botiot_reduced.csv.gz
#       │   ├── cicids_reduced.csv.gz
#       │   ├── toniot_reduced.csv.gz
#       │   └── nb15_reduced.csv.gz
#       └── sampled
#           └── ...
DATASET_KEYS = {
    "origin/botiot": "origin/NF-BoT-IoT-v2.csv.gz",
    "origin/cicids": "origin/NF-CSE-CIC-IDS2018-v2.csv.gz",
    "origin/toniot": "origin/NF-ToN-IoT-v2.csv.gz",
    "origin/nb15": "origin/NF-UNSW-NB15-v2.csv.gz",
    "reduced/botiot": "reduced/botiot_reduced.csv.gz",
    "reduced/cicids": "reduced/cicids_reduced.csv.gz",
    "reduced/toniot": "reduced/toniot_reduced.csv.gz",
    "reduced/nb15": "reduced/nb15_reduced.csv.gz",
    "sampled/botiot": "sampled/botiot_sampled.csv.gz",
    "sampled/cicids": "sampled/cicids_sampled.csv.gz",
    "sampled/toniot": "sampled/toniot_sampled.csv.gz",
    "sampled/nb15": "sampled/nb15_sampled.csv.gz",
}

# Columns to drop from the dataset.
# ---------------------------------
# The sampled and reduced datasets contain an additional column called `Dataset` which
# must be dropped as well. Columns can either be completely discarded or saved in the
# metadata array (m).
RM_COLS = [
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "Label",
    "Attack",
]


class NFV2(Dataset):
    """NF-V2 Dataset."""

    def poison(
        self: Dataset,
        ratio: float,
        op: PoisonOp,
        target_classes: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> int:
        """Poison a dataset by apply a function to a given number of samples.

        Parameters
        ----------
        n: int
            Number of samples to poison in the target. If `target` is None, the whole
            dataset is poisoned.
        op: PoisonOp
            Poisoning operation to apply. Either PoisonOp.INC or PoisonOp.DEC.
        target_classes: Optional[List[str]]
            List of classes to poison. If None, all classes are poisoned, including
            benign samples.
        seed: Optional[int]
            Seed for reproducibility.

        Returns
        -------
        int
            The number of samples that have been modified.
        """
        d = self.copy()

        assert target_classes is None or (
            isinstance(target_classes, List | ListConfig)
            and all(isinstance(c, str) for c in target_classes)
        ), "Invalid value for `target_classes`. Must be a list of strings or None."

        if target_classes is None:
            # If targeted means all dataset (including benign samples)
            # target is a boolean Series using the same index as the dataset
            target = pd.Series(True, index=d.y.index)

        elif target_classes == ["*"]:
            # If targeted means all attacks (excluding benign samples)
            target = d.m["Attack"] != "Benign"

        else:
            target = d.m["Attack"].isin(target_classes)

        n = np.ceil(sum(target) * ratio).astype(int)
        if n > sum(target):
            raise ValueError(
                f"Invalid value for `ratio`: ratio * len(target) = {n}."
                "Must be less or equal to len(target)."
            )

        if len(target) != len(self):
            raise ValueError(
                "Invalid value for `target`. Must be of the same length as the dataset."
            )

        if target.dtype != bool:
            raise ValueError("Invalid value for `target`. Must be a boolean Series.")

        # get poisoning metadata
        if "Poisoned" not in d.m.columns:
            d.m["Poisoned"] = False

        if op == PoisonOp.DEC:
            target = target & d.m["Poisoned"]
        else:
            target = target & ~d.m["Poisoned"]

        # indices of the samples to poison (cap n at the number of available samples)
        n = min(n, sum(target))
        idx = d.y[target].sample(n=n, random_state=seed).index.to_list()

        # apply the poisoning operation
        d.y.loc[idx] = d.y[idx].apply(operator.not_)
        d.y = d.y.astype(int)
        if op == PoisonOp.DEC:
            d.m.loc[idx, "Poisoned"] = False
        else:
            d.m.loc[idx, "Poisoned"] = True

        # save
        self.X = d.X
        self.y = d.y
        self.m = d.m

        # clean up
        del target
        del d

        return len(idx)


def load_data(
    key: str,
    search_path: str | Path | None = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> NFV2:
    """Load a dataset.

    This function is overloaded to allow different output types depending on the
    given parameters. The following output types are possible:

    - `Dataset` if no split is performed.
    - `Tuple[Dataset, Dataset]` if `test_ratio` is given. The first element is the
        training set, the second element is the testing set.
    - `List[Dataset]` if `n_partition` is given. The dataset is split into
        `n_partition`.
    - `List[Tuple[Dataset, Dataset]]` if `test_ratio` and `n_partition` are given.
        The dataset is split into training and testing sets, which are then split
        into `n_partition` depending on the `common_test` parameter.

    Parameters
    ----------
    key : str
        Key of the dataset to load. Can be a shortcut key or a path to a CSV file.
    test_ratio : float, optional
        Ratio of the testing set. If given, the dataset is split into a training and
        a testing set.
    n_partitions : int, optional
        Number of partitions to split the dataset into. If given, the dataset is
        split into `n_partition` partitions.
    common_test : bool, optional
        If `True`, `test_ratio` is given and `n_partition` is greater than 1, the
        testing set is the same for all partitions.
    base_path : str or Path, optional
        Path to the directory containing the dataset. If not given, the dataset is
        loaded from the default path.
    seed : int, optional
        Seed for shuffling the dataset.
    shuffle : bool, optional
        If `True`, the dataset is shuffled before being split.

    Returns
    -------
    Union
        Depending on the parameters, the function returns a single dataset, a tuple
        of two datasets, a list of datasets or a list of tuples of two datasets.

    Raises
    ------
    NotImplementedError
        If the function is not implemented by the child class.
    FileNotFoundError
        If the dataset is not found at the given path.
    """
    # PATH MANAGEMENT
    # ---------------

    # Assume key is a path to a CSV file
    # Allow not to specify the extension
    if search_path is None:
        if (
            Path(key).exists()
            or Path(key + ".csv").exists()
            or Path(key + ".csv.gz").exists()
        ):
            # If the path is reachable, use it
            if Path(key).is_absolute():
                base = Path("/")
            else:
                base = Path(".")

            path = base / Path(key)

        elif key in DATASET_KEYS:
            # if the path is not reachable, check if it is a shortcut key
            base = Path(DEFAULT_SEARCH_PATH)
        else:
            # Else, assume the path is relative to the default base path
            base = Path(DEFAULT_SEARCH_PATH)
            path = base / Path(key)

    else:
        # Assume the path is relative to the given base path
        base = Path(search_path)
        path = base / Path(key)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{key}' not found."
            " Either check your inputs, or download the dataset first.",
            {
                "key": key,
                "base": base,
                "current": Path(".").absolute(),
                "path": path.resolve().absolute(),
            },
        )

    df = pd.read_csv(path, low_memory=True)

    # DATA PREPROCESSING
    # ------------------
    # shuffle the dataset
    if shuffle:
        df = df.sample(frac=1, random_state=seed)

    # drop the "Dataset" column if it exists
    if "Dataset" in df.columns:
        df = df.drop(columns=["Dataset"])

    # select the columns to compose the Dataset object
    X = df.drop(columns=RM_COLS)
    y = df["Label"]
    m = df[RM_COLS]

    # convert classes to numerical values
    X = pd.get_dummies(X)

    # normalize the data
    scaler = MinMaxScaler()
    scaler.fit(X)
    X[X.columns] = scaler.transform(X)

    return NFV2(X, y, m)
