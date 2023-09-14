"""Dataset partitioners."""
import math
from abc import ABCMeta
from dataclasses import dataclass, field

import numpy as np

from .dataset import Dataset


@dataclass
class Partitioner(metaclass=ABCMeta):
    """Abstract partitioner class."""

    dataset: Dataset
    n_partitions: int
    partitions: list[Dataset] = field(init=False)

    def __post_init__(self) -> None:
        """Check if __init__ has been correclty overridden."""
        if (
            not self.partitions
            or not isinstance(self.partitions, list)
            or not all(isinstance(p, Dataset) for p in self.partitions)
        ):
            raise NotImplementedError(
                f"{self.__class__}.__init__(): "
                "function not implemented, or not implemented correctly."
            )

    def __len__(self) -> int:
        """Return the number of partitions."""
        return len(self.partitions)

    def __getitem__(self, idx: int | slice) -> Dataset:
        """Return the dataset at the given index."""
        return self.partitions[idx]

    def all(self) -> Dataset:
        """Return all the partitions."""
        return self.partitions


class DumbPartitioner(Partitioner):
    """Dumb partitioner class.

    The dumb partitioner does so by splitting the dataset into `n_partitions` chunks
    based on their indices. There is no guarantee that the partitions will be balanced.
    """

    def __init__(self, dataset: Dataset, n_partitions: int) -> None:
        """Initialize the IID partitioner."""
        partition_size = math.floor(len(dataset.X) / n_partitions)
        self.partitions = []
        for i in range(n_partitions):
            idx_from, idx_to = i * partition_size, (i + 1) * partition_size

            self.partitions.append(dataset[idx_from:idx_to])

        super().__init__(dataset, n_partitions)


class IIDPartitioner(Partitioner):
    """IID partitioner class.

    The IID partitioner ensures that the partitions are balanced using
    `train_test_split()` from scikit-learn.
    """

    def __init__(
        self, dataset: Dataset, n_partitions: int, class_column: str, df_key: str = "m"
    ) -> None:
        """Initialize the IID partitioner."""
        if not hasattr(dataset, df_key):
            raise KeyError(f"Dataset does not contain a DataFrame with key {df_key}")

        if class_column not in getattr(dataset, df_key).columns:
            raise KeyError(f"Dataset does not contain a column named {class_column}")

        self.partitions = []

        for _ in range(n_partitions):
            if n_partitions == 1:
                self.partitions.append(dataset)
                break

            ratio = 1 / n_partitions
            n_partitions -= 1

            d, rest = dataset.split(
                at=ratio, stratify=getattr(dataset, df_key)[class_column]
            )

            self.partitions.append(d)
            dataset = rest

        super().__init__(dataset, n_partitions)


class NIIDClassPartitioner(Partitioner):
    """NIID partitioner class."""

    def __init__(
        self,
        dataset: Dataset,
        n_partitions: int,
        class_column: str,
        preserved_classes: list[str],
        n_drop: int = 1,
        df_key: str = "m",
    ) -> None:
        """Initialize the NIID partitioner.

        This partitioner will select `n_drop` classes from each partition, except for
        the ones that are identified in `preserved_classes`, and drop all the samples
        that belong to these classes. Note that droping samples means that partitions
        will be smaller than `len(dataset) / n_partitions`.

        Parameters
        ----------
        dataset : Dataset
            Dataset to partition.
        n_partition : int
            Number of partitions.
        class_column : str
            Name of the column containing the class labels.
        preserved_classes : list[str]
            List of class values to preserve. Class dropping will not be applied to
            these classes.
        n_drop : int, optional
            Number of classes to drop per client, by default 1.
        df_key : str, optional
            Key of the DataFrame in the Dataset object containing the `class_column`, by
            default "m".

        Raises
        ------
        ValueError
            If the number of classes to drop is greater than the number of classes in
            the dataset minus the number of classes to preserve.
        KeyError
            If the class column is not present in the dataset, if the class column is
            not a column of the DataFrame, or if the class values are not present in the
            dataset.
        """
        if not hasattr(dataset, df_key):
            raise KeyError(f"Dataset does not contain a DataFrame with key {df_key}")

        if class_column not in getattr(dataset, df_key).columns:
            raise KeyError(f"Dataset does not contain a column named {class_column}")

        available_classes = getattr(dataset, df_key)[class_column].unique()

        if not preserved_classes or any(
            c not in available_classes for c in preserved_classes
        ):
            raise KeyError(
                f"Dataset does not contain all the class values in {preserved_classes}"
            )

        if n_drop > (len(available_classes) - len(preserved_classes)):
            raise ValueError(
                f"Cannot drop {n_drop} classes, only "
                f"{len(available_classes) - len(preserved_classes)} "
                "classes are available."
            )

        dropable = [c for c in available_classes if c not in preserved_classes]

        self.partitions = []
        pt = IIDPartitioner(dataset, n_partitions, class_column, df_key=df_key)
        parts = pt.all()
        for p in parts:
            drop = np.random.choice(dropable, n_drop, replace=False)
            mask = getattr(p, df_key)[class_column].isin(drop)
            p.drop(mask[mask].index)
            self.partitions.append(p)

        super().__init__(dataset, n_partitions)
