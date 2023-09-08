"""Tests for dataset/nfv2.py."""

import tempfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from trustfids.dataset.common import Dataset
from trustfids.dataset.nfv2 import DEFAULT_BASE_PATH, RM_COLS, load_data, poison
from trustfids.dataset.poisoning import PoisonIns, PoisonOp, PoisonTask


def test_load_data():
    """Test load_data()."""

    # Test0: load the dataset without downloading
    try:
        d = load_data("origin/botiot")
    except FileNotFoundError as e:
        message, args = e.args
        assert "origin/botiot" in message
        assert args["name"] == "origin/botiot"
        assert args["base"] == DEFAULT_BASE_PATH
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

        assert type(d) == Dataset
        assert len(d) == len(mock_df)
        assert set(d.X.columns) == set(cols) - set(RM_COLS)

        # Test2: load with train/test split
        train, test = load_data(data_path, test_ratio=0.2)
        train2, test2 = load_data(data_path, test_ratio=0.2, seed=1138)
        train3, test3 = load_data(data_path, test_ratio=0.2, seed=1138)

        assert len(train) == 0.8 * len(mock_df)
        assert len(test) == 0.2 * len(mock_df)

        assert len(train) == len(train2) == len(train3)
        assert len(test) == len(test2) == len(test3)

        assert not train.X.equals(train2.X) and not test.X.equals(test2.X)
        assert train2.X.equals(train3.X) and test2.X.equals(test3.X)

        assert type(train) == Dataset

        # Test3: load with partition
        datasets = load_data(data_path, n_partitions=10)
        datasets2 = load_data(data_path, n_partitions=10, seed=1138)
        datasets3 = load_data(data_path, n_partitions=10, seed=1138)

        assert len(datasets) == len(datasets2) == 10

        assert not datasets[0].X.equals(datasets2[0].X)
        assert datasets2[0].X.equals(datasets3[0].X)

        # assert type(datasets) == List[Dataset]
        assert type(datasets) == list
        assert type(datasets[0]) == Dataset

        # Test4: load with partition and train/test split
        datasets = load_data(data_path, n_partitions=10, test_ratio=0.2)
        datasets2 = load_data(data_path, n_partitions=10, test_ratio=0.2, seed=1138)
        datasets3 = load_data(data_path, n_partitions=10, test_ratio=0.2, seed=1138)

        assert len(datasets) == len(datasets2) == len(datasets3) == 10

        train, test = datasets[0]
        train2, test2 = datasets2[0]
        train3, test3 = datasets3[0]

        assert len(train) == len(train2) == len(train3) == 0.8 * len(mock_df) / 10
        assert len(test) == len(test2) == len(test3) == 0.2 * len(mock_df) / 10

        assert not train.X.equals(train2.X) and not test.X.equals(test2.X)
        assert train2.X.equals(train3.X) and test2.X.equals(test3.X)

        # assert type(datasets) == List[Tuple[Dataset, Dataset]]
        assert type(datasets) == list
        assert type(datasets[0]) == tuple
        assert type(datasets[0][0]) == Dataset


def test_poison():
    """Test poison()."""
    np.random.seed(1138)

    # Make a mock dataset
    m = pd.DataFrame()
    m["Attack"] = np.random.choice(["Benign", "Botnet", "DoS", "DDoS"], size=100)
    y = pd.Series(m["Attack"] != "Benign", name="Label")

    mock_d = Dataset(
        pd.DataFrame(np.random.rand(100, 10), columns=[f"col{i}" for i in range(10)]),
        y,
        m,
    )
    dos_n = sum(mock_d.m["Attack"] == "DoS")

    # Test1: poisoning on 10% of target
    p, n = poison(
        mock_d,
        *PoisonTask(0.1),
        target_classes=["DoS"],
        seed=1138,
    )
    p_dos_n = sum(p.y[p.m["Attack"] == "DoS"])
    assert len(p) == len(mock_d)
    assert p_dos_n == np.floor(0.9 * dos_n)  # floor because of ceil in `poison()`

    # Test2: poisoning on 10% of target; again -> 20% should be poisoned
    p, n = poison(
        p,
        *PoisonTask(0.1),
        target_classes=["DoS"],
        seed=1138,
    )
    p_dos_n = sum(p.y[p.m["Attack"] == "DoS"])
    assert len(p) == len(mock_d)
    assert p_dos_n == np.floor(0.8 * dos_n)

    # Test3: decrease poisoning by 10% of target -> 10% should be poisoned
    p, n = poison(
        p,
        *PoisonTask(0.1, PoisonOp.DEC),
        target_classes=["DoS"],
        seed=1138,
    )
    p_dos_n = sum(p.y[p.m["Attack"] == "DoS"])
    assert len(p) == len(mock_d)
    assert p_dos_n == np.floor(0.9 * dos_n)


if __name__ == "__main__":
    test_poison()
