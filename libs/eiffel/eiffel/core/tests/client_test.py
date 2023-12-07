"""Tests for eiffel.core.client."""

from eiffel.core.client import EiffelClient
from eiffel.datasets.nfv2 import load_data
from eiffel.datasets.dataset import DatasetHandle


def mk_mock_client():
    """Make a mock client."""

    train, test = load_data("../../data/nfv2/sampled/cicids.csv.gz")
    data_holder = DatasetHandle.remote({"train": train, "test": test})


def test_evaluate():
    """Test the evaluate method.

    The evaluate method should return the metrics on the test set.
    """
