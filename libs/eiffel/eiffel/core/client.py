"""Eiffel client API."""

import itertools
import json
import logging
from copy import deepcopy
from functools import reduce
from typing import Callable, Optional, cast

import numpy as np
import ray
from flwr.client import NumPyClient
from flwr.common import Config, Scalar
from keras.callbacks import History
from tensorflow import keras

from eiffel.core.metrics import metrics_from_preds
from eiffel.datasets.dataset import Dataset, DatasetHandle
from eiffel.datasets.poisoning import PoisonIns, PoisonTask
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
    data_holder: DatasetHandle
    model: keras.Model
    poison_ins: Optional[PoisonIns]

    def __init__(
        self,
        cid: EiffelCID,
        data_holder: DatasetHandle,
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

    def get_parameters(self, config: Config) -> list[NDArray]:
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
            elif self.poison_ins.tasks is None:
                logger.debug(
                    f"{self.cid}: No poisoning tasks provided, skipping poisoning."
                )
            elif config["round"] in self.poison_ins.tasks:
                self.poison(self.poison_ins.tasks[config["round"]])
                logger.debug(f"{self.cid}: Poisoned the dataset.")

        train_set: Dataset = ray.get(self.data_holder.get.remote("train"))
        self.model.set_weights(parameters)
        hist: History = self.model.fit(
            train_set.to_sequence(
                config["batch_size"], target=1, seed=self.seed, shuffle=True
            ),
            epochs=int(config["num_epochs"]),
            verbose=0,
        )

        return (
            self.model.get_weights(),
            len(train_set),
            {
                "accuracy": hist.history["accuracy"][-1],
                "loss": hist.history["loss"][-1],
                "_cid": self.cid,
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

        output = self.model.evaluate(
            test_set.to_sequence(batch_size, target=1, seed=self.seed, shuffle=True),
            verbose=self.verbose,
        )
        if isinstance(output, list):
            output = dict(zip(self.model.metrics_names, output))
            loss = output["loss"]
        else:
            loss = output

        # Do not shuffle the test set for inference, otherwise we cannot compare y_pred
        # with y_true.
        inferences: NDArray = self.model.predict(
            test_set.to_sequence(batch_size, target=1), verbose=self.verbose
        )

        y_pred = np.around(inferences).astype(int).reshape(-1)

        y_true = test_set.y.to_numpy().astype(int)

        metrics = metrics_from_preds(y_true, y_pred)
        metrics["loss"] = loss
        metrics["_cid"] = self.cid

        if "stats" in config and config["stats"]:
            # Compute attack-wise statistics. Only enabled for the last evaluation round.
            attack_stats = []
            class_df = test_set.m["Attack"]
            for attack in (a for a in class_df.unique() if a != "Benign"):
                attack_filter = class_df == attack

                count = len(test_set.m[attack_filter])
                correct = len(test_set.m[(attack_filter) & (test_set.y == y_pred)])
                missed = len(test_set.m[(attack_filter) & (test_set.y != y_pred)])

                attack_stats.append(
                    {
                        "attack": attack,
                        "count": count,
                        "correct": correct,
                        "missed": missed,
                        "percentage": f"{correct / count:.2%}",
                    }
                )

            metrics["attack_stats"] = json.dumps(attack_stats)

        return loss, len(test_set), metrics

    def poison(self, task: PoisonTask) -> None:
        """Poison the dataset.

        Parameters
        ----------
        task : PoisonTask
            The poisoning task to perform.
        """
        assert self.poison_ins is not None

        self.data_holder.poison.remote(
            "train", task.fraction, task.operation, self.poison_ins.target, self.seed
        )
        if self.poison_ins.poison_eval:
            self.data_holder.poison.remote(
                "test", task.fraction, task.operation, self.poison_ins.target, self.seed
            )


def mk_client(
    cid: EiffelCID,
    mappings: dict[EiffelCID, tuple[ray.ObjectRef, PoisonIns, keras.Model]],
    seed: int,
    attack: Optional[PoisonIns] = None,
) -> EiffelClient:
    """Return a client based on its CID."""
    if cid not in mappings:
        raise ValueError(f"Client `{cid}` not found in mappings.")

    handle, attack, model_fn = mappings[cid]

    return EiffelClient(
        cid,
        handle,
        model_fn(ray.get(handle.get.remote("train")).X.shape[1]),
        seed=seed,
        poison_ins=attack,
    )
