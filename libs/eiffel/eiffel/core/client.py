"""Eiffel client API."""

import logging
from functools import reduce
from typing import Callable, Optional, cast

import numpy as np
import ray
from flwr.client import NumPyClient
from flwr.common import Config, Scalar
from keras.callbacks import History
from tensorflow import keras

from eiffel.core.metrics import metrics_from_preds
from eiffel.datasets.dataset import Dataset, DatasetHolder
from eiffel.datasets.poisoning import PoisonIns
from eiffel.utils.logging import VerbLevel
from eiffel.utils.typing import EiffelCID, MetricsDict, NDArray

from .pool import Pool

logger = logging.getLogger(__name__)


class EiffelClient(NumPyClient):
    """Eiffel client.

    Attributes
    ----------
    cid : EiffelCID
        The client ID.
    data_holder : DatasetHolder
        A reference to the datasets, living in the Ray object store.
    model : keras.Model
        The model to train.
    verbose : VerbLevel
        The verbosity level.
    seed : Optional[int]
        The seed to use for random number generation.
    poison_ins : Optional[PoisonIns]
        The poisoning instructions, if any.
    """

    cid: EiffelCID
    data_holder: DatasetHolder
    model: keras.Model
    poison_ins: Optional[PoisonIns]

    def __init__(
        self,
        cid: EiffelCID,
        data_holder: DatasetHolder,
        model: keras.Model,
        verbose: VerbLevel = VerbLevel.SILENT,
        seed: Optional[int] = None,
        poison_ins: Optional[PoisonIns] = None,
    ) -> None:
        """Initialize the EiffelClient."""
        self.cid = cid
        self.data_holder = data_holder
        self.model = model
        self.verbose = verbose
        self.seed = seed
        self.poison_ins = poison_ins

    def get_parameters(self) -> list[NDArray]:
        """Return the current parameters.

        Returns
        -------
        list[NDArray]
            Current model parameters.
        """
        return self.model.get_weights()

    def fit(
        self, parameters: list[NDArray], config: Config
    ) -> tuple[list[NDArray], int, MetricsDict]:
        """Fit the model to the local data set.

        Parameters
        ----------
        parameters : list[NDArray]
            The initial parameters to train on, generally those of the global model.
        config : Config
            The configuration for the training.

        Returns
        -------
        list[NDArray]
            The updated parameters.
        int
            The number of examples used for training.
        MetricsDict
            The metrics collected during training.
        """
        if self.poison_ins is not None:
            if "round" not in config:
                logger.warning(
                    f"{self.cid}: No round number provided, skipping poisoning."
                )
            elif config["round"] in self.poison_ins.tasks:
                task = self.poison_ins.tasks[config["round"]]
                self.data_holder.poison.remote(
                    task.fraction, task.op, self.poison_ins.target, self.seed
                )
                logger.debug(f"{self.cid}: Poisoned the dataset.")

        train_set: Dataset = ray.get(self.data_holder.get.remote("train"))
        self.model.set_weights(parameters)
        hist: History = self.model.fit(
            train_set.to_sequence(
                config["batch_size"], target=1, seed=self.seed, shuffle=True
            ),
            epochs=int(config["epochs"]),
            verbose=self.verbose,
        )
        del train_set
        return (
            self.model.get_weights(),
            len(train_set),
            {
                "accuracy": hist.history["accuracy"][-1],
                "loss": hist.history["loss"][-1],
            },
        )

    def evaluate(
        self, parameters: list[NDArray], config: Config
    ) -> tuple[float, int, MetricsDict]:
        """Evaluate the model on the local data set.

        Parameters
        ----------
        parameters : list[NDArray]
            The parameters of the model to evaluate.
        config : Config
            The configuration for the evaluation.

        Returns
        -------
        float
            The loss of the model during evaluation.
        int
            The number of samples used for evaluation.
        MetricsDict
            The metrics collected during evaluation.
        """
        batch_size = int(config["batch_size"])

        self.model.set_weights(parameters)

        test_set: Dataset = ray.get(self.data_holder.get.remote("test"))

        loss = self.model.evaluate(
            test_set.to_sequence(batch_size, target=1, seed=self.seed, shuffle=True),
            verbose=self.verbose,
        )

        # Do not shuffle the test set for inference, otherwise we cannot compare y_pred
        # with y_true.
        inferences: NDArray = self.model.predict(
            test_set.to_sequence(batch_size, target=1), verbose=self.verbose
        )

        y_pred = np.around(inferences).astype(int).reshape(-1)

        y_true = test_set.y.to_numpy().astype(int)

        metrics = metrics_from_preds(y_true, y_pred)
        metrics["loss"] = loss

        return loss, metrics, y_pred


def mk_client_fn(pools: list[Pool]) -> Callable[[EiffelCID], EiffelClient]:
    """Return a function which creates a client based on its CID.

    Parameters
    ----------
    pools : list[Pool]
        A list of Eiffel pools, which will be used to create the clients.

    Returns
    -------
    Callable[[EiffelCID], EiffelClient]
        A function which takes a client ID and returns a client.
    """
    if any((not p.deployed()) for p in pools):
        raise RuntimeError("Not all pools are deployed.")

    def mk_client(cid: EiffelCID) -> EiffelClient:
        """Create a client."""
        pool = next(p for p in pools if cid in p)
        return EiffelClient(
            cid,
            pool.holders[cid],
            pool.model_fn(),
            seed=pool.seed,
            poison_ins=pool.attack,
        )

    return mk_client
