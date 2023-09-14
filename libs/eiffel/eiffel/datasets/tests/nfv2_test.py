"""Tests for dataset/nfv2.py."""

import tempfile

import numpy as np
import pandas as pd
import pytest

from eiffel.datasets import DEFAULT_SEARCH_PATH
from eiffel.datasets.dataset import Dataset
from eiffel.datasets.nfv2 import RM_COLS, NFV2Dataset, load_data
from eiffel.datasets.poisoning import PoisonOp, PoisonTask


def test_load_data():
    """Test load_data()."""
    # Test0: load the dataset without downloading
    try:
        d = load_data("origin/botiot")
    except FileNotFoundError as e:
        message, args = e.args
        assert "origin/botiot" in message
        assert args["key"] == "origin/botiot"
        assert args["base"] == DEFAULT_SEARCH_PATH
    else:
        assert False, "load_data() should raise FileNotFoundError"

    # mock the dataset with random data in a temporary directory
    cols = RM_COLS + ["col1", "col2"]

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = f"{tmpdir}/nfv2.csv"

        mock_df = pd.DataFrame(np.random.rand(100, len(cols)), columns=cols)
        # fill the "Attack" column with random values in {"Benign", "Botnet", "Dos",
        # "DDoS"}
        mock_df["Attack"] = np.random.choice(
            ["Benign", "Botnet", "Dos", "DDoS"], size=len(mock_df)
        )
        mock_df = mock_df.astype({"Attack": "category"})
        mock_df["Label"] = mock_df["Attack"] == "Benign"
        mock_df.to_csv(data_path, index=False)

        # Test1: load the whole dataset
        d = load_data(data_path)

        assert isinstance(d, Dataset)
        assert len(d) == len(mock_df)
        assert set(d.X.columns) == set(cols) - set(RM_COLS)


def test_poison():
    """Test poison()."""
    np.random.seed(1138)

    # Make a mock dataset
    m = pd.DataFrame()
    m["Attack"] = np.random.choice(["Benign", "Botnet", "DoS", "DDoS"], size=100)
    y = pd.Series(m["Attack"] != "Benign", name="Label")

    mock_d = NFV2Dataset(
        pd.DataFrame(np.random.rand(100, 10), columns=[f"col{i}" for i in range(10)]),
        y,
        m,
    )
    dos_n = sum(mock_d.m["Attack"] == "DoS")

    # Test1: poisoning on 10% of target
    n = mock_d.poison(
        *PoisonTask(0.1),
        target_classes=["DoS"],
        seed=1138,
    )
    p_dos_n = sum((mock_d.m["Attack"] == "DoS") & ~mock_d.y)
    assert n == p_dos_n == np.ceil(0.1 * dos_n)  # ceil because of ceil in `poison()`

    # Test2: poisoning on 10% of target; again -> 20% should be poisoned
    n = mock_d.poison(
        *PoisonTask(0.1),
        target_classes=["DoS"],
        seed=1138,
    )
    p_dos_n = sum((mock_d.m["Attack"] == "DoS") & ~mock_d.y)
    assert p_dos_n == np.ceil(0.2 * dos_n)

    # Test3: decrease poisoning by 10% of target -> 10% should be poisoned
    n = mock_d.poison(
        *PoisonTask(0.1, PoisonOp.DEC),
        target_classes=["DoS"],
        seed=1138,
    )
    p_dos_n = sum((mock_d.m["Attack"] == "DoS") & ~mock_d.y)
    assert p_dos_n == np.ceil(0.1 * dos_n)


if __name__ == "__main__":
    pytest.main([__file__])
