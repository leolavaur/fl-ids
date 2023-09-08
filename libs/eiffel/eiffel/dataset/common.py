"""Common dataset classes and functions.

This module contains the common dataset classes and functions used by the Eiffel
framework. It provides a unified interface for loading datasets, regardless of their
format.
"""

# Required for type annotations to work with Python 3.7+, and especially forward
# references:
# class C:
#     def foo(self) -> "C":
#         return self
from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import List, Optional, Tuple, overload, Hashable

import ray
import numpy as np
import pandas as pd
from keras.utils import Sequence
from sklearn.model_selection import train_test_split

from ..engine.logging import logged
from ..engine.poisoning import PoisonOp

@ray.remote
class DatasetHolder(dict):

    def poison(self, key: Hashable) -> None:



class DatasetFacade(metaclass=ABCMeta):
    """Abstract class for a dataset management "Facade".

    A Facade is a design pattern that provides a unified interface to a set of
    interfaces in a subsystem. It defines a higher-level interface that makes the
    subsystem easier to use.

    This class provides a unified interface for managing datasets. To provide a dataset
    to the Eiffel framework, a child class of DatasetFacade must be implemented. Its
    `load_data` method MUST be implemented, and refered to in the configuration file.

    The other methods are optional, and can be implemented if needed. Especially, the
    `poison` is required the use case involves poisoning the dataset.
    """

    # Default path to the datasets.
    # -----------------------------
    # The dataset is downloaded to this path if it is not found.
    # On Linux, the default path is `/tmp/eiffel-data/`.
    # In the directory, datasets are organised as follows:
    #   /tmp/trustfids-data/nfv2
    #   ├── dataset_name
    #   │   └── dataset_file.csv.gz
    #   ├── dataset_collection
    #   │   ├── dataset_name
    #   │   │   └── dataset_file.csv.gz
    #   │   ├── dataset_name_2
    #   │   │   └── dataset_file_2.csv.gz
    #   │   └── ...
    #   └── ...
    DEFAULT_BASE_PATH = Path(gettempdir()) / "eiffel-data"

    @overload
    def load_data(
        cls,
        key: str,
        test_ratio: None = None,
        n_partitions: None = None,
        common_test: bool = False,
        base_path: str | Path | None = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dataset:
        ...

    @overload
    def load_data(
        cls,
        key: str,
        test_ratio: float,
        n_partitions: None = None,
        common_test: bool = False,
        base_path: str | Path | None = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
    ) -> Tuple[Dataset, Dataset]:
        ...

    @overload
    def load_data(
        cls,
        key: str,
        test_ratio: None = None,
        n_partitions: int = 0,
        common_test: bool = False,
        base_path: str | Path | None = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
    ) -> List[Dataset]:
        ...

    @overload
    def load_data(
        cls,
        key: str,
        test_ratio: float,
        n_partitions: int,
        common_test: bool = False,
        base_path: str | Path | None = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
    ) -> List[Tuple[Dataset, Dataset]]:
        ...

    @abstractmethod
    @classmethod
    def load_data(
        cls,
        key: str,
        test_ratio: Optional[float] = None,
        n_partitions: Optional[int] = None,
        common_test: bool = False,
        base_path: str | Path | None = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
    ) -> (
        Dataset
        | Tuple[Dataset, Dataset]
        | List[Dataset]
        | List[Tuple[Dataset, Dataset]]
    ):
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
        raise NotImplementedError(f"{cls}: Load function not implemented.")

    def _preprocess(cls, df: pd.DataFrame, shuffle) -> Dataset:
        """Preprocess a dataset.

        This function is called after loading the dataset, and before optionally
        splitting it into partitions, or training and testing sets. This function MUST
        be implemented by the child class.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to preprocess.

        Returns
        -------
        Dataset
            The preprocessed dataset.

        Raises
        ------
        NotImplementedError
            If the function is not implemented by the child class.
        """
        raise NotImplementedError(f"{cls}: Preprocess function not implemented.")

    def download(cls, path: Optional[str] = None) -> None:
        """Download the dataset to a given path.

        If no path is given, the dataset is downloaded to the default path.
        If the dataset is already present at the given path, no action is taken.

        Parameters
        ----------
        path : str, optional
            Path to the directory where to download the dataset, by default None.

        Raises
        ------
        NotImplementedError
            If the function is not implemented by the child class.
        FileNotFoundError
            If the given path does not exist.
        """
        raise NotImplementedError(f"{cls}: Download function not implemented.")

    def poison(
        cls,
        dataset: Dataset,
        ratio: float,
        op: PoisonOp,
        target_classes: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dataset, int]:
        """Poison a dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to poison.
        ratio : float
            Ratio of the dataset to poison.
        op : PoisonOp
            Poisoning operation to apply.
        target_classes : List[str], optional
            List of target classes to poison. If None, all classes are poisoned.
        seed : int, optional
            Seed for shuffling the dataset.

        Returns
        -------
        Dataset
            The poisoned dataset.
        int
            Number of samples poisoned.
        """
        raise NotImplementedError(f"{cls}: Poison function not implemented.")


class BatchLoader(Sequence):
    """Generator of batches for training."""

    X: pd.DataFrame
    target: pd.DataFrame | pd.Series | None

    batch_size: int

    def __init__(
        self,
        batch_size: int,
        X: pd.DataFrame,
        target: pd.DataFrame | pd.Series | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        """Initialise the BatchLoader."""
        self.batch_size = batch_size

        self.X = X
        self.target = target if target is not None else X.copy()

        if shuffle:
            indices = np.arange(len(X))
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(indices)

            self.X = X.iloc[indices]
            self.target = self.target.iloc[indices]

    def __len__(self):
        """Return the number of batches."""
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        """Return the batch at the given index."""
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch_size.
        high = min(low + self.batch_size, len(self.X))

        batch_x = self.X[low:high]
        if self.target is None:
            return batch_x

        batch_target = self.target[low:high]

        # A Sequence should apparently return a tuple of NumPy arrays, as DataFrames
        # cause errors in the fit() method.
        return batch_x.to_numpy(), batch_target.to_numpy()


@dataclass
class Dataset:
    """Dataset class."""

    X: pd.DataFrame
    y: pd.Series
    m: pd.DataFrame

    def to_tuple(self):
        """Return the dataset as a tuple.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
            Tuple of the dataset.
        """
        return self.X, self.y, self.m

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.X)

    def __eq__(self, other):
        """Compare two datasets."""
        return (
            self.X.equals(other.X) and self.y.equals(other.y) and self.m.equals(other.m)
        )

    def __getitem__(self, key: int | slice | list):
        """Return the given slice of X, y and m.

        Parameters
        ----------
        key : int | slice | list
            Index or slice to return.

        Returns
        -------
        Dataset
            Dataset containing the given slice.
        """
        assert isinstance(key, int) or isinstance(key, slice) or isinstance(key, list)
        return Dataset(self.X[key], self.y[key], self.m[key])

    def to_sequence(
        self,
        batch_size: int,
        target: int | None = None,
        seed: int | None = None,
        shuffle: bool = False,
    ) -> BatchLoader:
        """Convert the dataset to a BatchLoader object.

        Parameters
        ----------
        batch_size : int
            Size of the batches.
        target : int, optional
            Target to use for the batches, defaults to None. 0 for X, 1 for y, 2 for m.

        Returns
        -------
        BatchLoader
            The dataset as a batch sequence that can be processed by the Keras API.

        """
        if target is None:
            return BatchLoader(batch_size, self.X, seed=seed, shuffle=shuffle)
        if 0 <= target <= 2:
            return BatchLoader(
                batch_size, self.X, self[target], seed=seed, shuffle=shuffle
            )
        raise IndexError("If not None, parameter `target` must be in [0, 2]")

    def split(
        self,
        at: float,
        seed: int | None = None,
        stratify: Optional[pd.Series] = None,
    ) -> Tuple[Dataset, Dataset]:
        """Split the dataset into a training and a test set.

        Parameters
        ----------
        at : float
            Ratio where to split the dataset. Must be in [0, 1]. The first dataset will
            contain `at`% of the samples, the second one will contain the remaining.
        seed : int, optional
            Seed for the random number generator, by default None.
        strat_series : pd.Series, optional
            Series to use for stratification, by default None.

        Returns
        -------
        Tuple[Dataset, Dataset]
            Tuple of the training and test sets.
        """
        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            *self,
            train_size=at,
            random_state=seed,
            stratify=np.array(stratify) if stratify is not None else None,
        )

        return (
            Dataset(X_train, y_train, m_train),
            Dataset(X_test, y_test, m_test),
        )

    def copy(self):
        """Return a copy of the dataset."""
        return Dataset(self.X.copy(), self.y.copy(), self.m.copy())

    def partition(self, n_partition: int) -> List[Dataset]:
        """Partition the dataset into n partitions.

        Parameters
        ----------
        n_partition : int
            Number of partitions.

        Returns
        -------
            List of Datasets.
        """
        partition_size = math.floor(len(self.X) / n_partition)
        partitions = []
        for i in range(n_partition):
            idx_from, idx_to = i * partition_size, (i + 1) * partition_size

            partitions.append(self[idx_from:idx_to].copy())

        return partitions
