"""Common dataset classes and functions.

This module contains the common dataset classes and functions used by the Eiffel
framework. It provides a unified interface for loading datasets, regardless of their
format.
"""


import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Hashable, List, Optional, Tuple, overload

import numpy as np
import pandas as pd
import ray
from keras.utils import Sequence
from sklearn.model_selection import train_test_split

from ..utils.poisoning import PoisonOp


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

    def __getitem__(self, key: int | slice | list) -> "Dataset":
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
        return self.__class__(self.X[key], self.y[key], self.m[key])

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
    ) -> Tuple["Dataset", "Dataset"]:
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
            self.__class__(X_train, y_train, m_train),
            self.__class__(X_test, y_test, m_test),
        )

    def copy(self) -> "Dataset":
        """Return a copy of the dataset."""
        return self.__class__(self.X.copy(), self.y.copy(), self.m.copy())

    def shuffle(self, seed: int | None = None):
        """Shuffle the dataset."""
        indices = np.arange(len(self.X))
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

        self.X = self.X.iloc[indices]
        self.y = self.y.iloc[indices]
        self.m = self.m.iloc[indices]

    def partition(self, n_partition: int) -> List["Dataset"]:
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

    def poison(
        self,
        ratio: float,
        op: PoisonOp,
        target_classes: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> int:
        """Increase or decrease the proportion of poisoned samples in the dataset.

        This function MUST be implemented by the child class to allow poisoning.

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

        Raises
        ------
        NotImplementedError
            If the function is not implemented by the child class.
        """
        raise NotImplementedError(
            f"{self.__class__}.poison():  function not implemented."
        )


@ray.remote
class DatasetHolder(dict[Hashable, Dataset]):
    """Dataset holder to store datasets in the Ray object store.

    This class is used to store stateful datasets in the Ray object store. It is used to
    preserve the state of the dataset between the different steps of the pipeline, as
    Flower clients (up tp 1.5.0 at least) are ephemeral and do not preserve their state.

    The class is a subclass of `dict`, and can be used as such. Datasets are stored
    using the dict API (e.g. `dataset_holder["cicids"]` externally, and using
    `self["cicids"]` inside the object).

    Attributes
    ----------
    facade : DatasetFacade
        The facade used to manage the datasets.
    """

    def poison(self, key: Hashable, *args, **kwargs) -> int:
        """Poison the held dataset."""
        n = self[key].poison(*args, **kwargs)
        return n
