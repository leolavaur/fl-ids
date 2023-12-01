"""Typing definitions for Eiffel."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray as NDArray

EiffelCID = str
DictKey = str
DictVal = str | int | float | bool | bytes
ConfigDict = MetricsDict = dict[DictKey, DictVal]
