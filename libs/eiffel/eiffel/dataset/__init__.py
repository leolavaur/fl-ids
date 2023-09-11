"""Trust-FIDS dataset module.

Each dataset submodule must implement the `load_data` function, which loads a given
dataset in memory, and optionally a `download` function, which downloads the dataset.
"""

from pathlib import Path
from tempfile import gettempdir

from .common import Dataset, DatasetHolder

DEFAULT_SEARCH_PATH = Path(gettempdir()) / "eiffel-data"
