"""Trust-FIDS dataset module.

Each dataset submodule must implement the `load_data` function, which loads a given
dataset in memory, and optionally a `download` function, which downloads the dataset.
"""

from .common import BatchLoader, Dataset
