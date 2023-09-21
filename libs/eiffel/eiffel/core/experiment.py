"""Eiffel engine."""

import functools
from copy import deepcopy
from functools import reduce
from typing import Callable

import psutil
import ray
import tensorflow as tf
from flwr.client import ClientLike
from flwr.server import Server, ServerConfig
from flwr.server.history import History
from flwr.server.strategy import Strategy
from flwr.simulation import start_simulation
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, open_dict

from eiffel.core.errors import ConfigError
from eiffel.utils.typing import ConfigDict

from .client import get_client, mk_client_fn
from .pool import Pool


class Experiment:
    """Eiffel experiment.

    Attributes
    ----------
    server : flwr.server.Server
        The Flower server. It administrates the entire FL process, and is responsible
        for aggregating the models, based on the function provided in the
        `flwr.server.Strategy` object.
    strategy : flwr.server.Strategy
        The strategy used by the Flower server to aggregate the models. It is passed to
        Flower's `start_simulation` function.
    pools : list[Pool]
        The different client pools. Each pool is a collection of clients that share the
        same dataset and attack type.
    n_clients : int
        The total number of clients in the experiment.
    """

    n_clients: int
    n_rounds: int
    n_concurrent: int
    seed: int

    server: Server = None
    strategy: Strategy = None
    pools: list[Pool] = []

    def __init__(self, config: DictConfig):
        """Initialize the experiment.

        The `expriment` object is a DictConfig object containing all configuration to
        instantiate an Eiffel experiment. Notably, it should contain three ListConfig
        objects:

        - `pools`: the list of client pools. Each pool is a DictConfig object
            containing the configuration of the pool. At the very least, it should
            contain the number of clients in the pools as: `{n_benign: int, n_malicious:
            int}`.
        - `datasets`: the list of datasets used by the clients. Each dataset is a
          DictConfig object that can be passed to Hydra's instantiation logic for a
          `load_data` fonction.
        - `attacks`: the attack configuration as: `{type: str, profile: str}`.

        The number of pools is defined by the length of the `pools` list. If a single
        DictConfig is provided, ie. if the length of the list is 1, then the attack or
        dataset is used by all pools. Otherwise, the length of the list should be equal
        to the number of pools.

        Parameters
        ----------
        config : omegaconf.DictConfig
            The configuration of the experiment. This is the configuration aggregated by
            Hydra from the command line arguments and the configuration files.
        """
        self.seed = config.seed
        self.n_rounds = config.num_rounds

        pools = obj_to_list(config.pools)
        attacks = obj_to_list(config.attacks, expected_length=len(config.pools))
        datasets = obj_to_list(config.datasets, expected_length=len(config.pools))

        pools_mapping = zip(pools, attacks, datasets)

        for pool, attack, dataset in pools_mapping:
            if isinstance(attack, DictConfig):
                attack = dict(attack)
            if not hasattr(attack, "n_rounds"):
                attack["n_rounds"] = config.num_rounds

            if hasattr(pool, "_target_"):
                pool = instantiate(pool, dataset=dataset, attack=attack)
            else:
                pool = Pool(
                    dataset=dataset,
                    model_fn=instantiate(config.model),
                    attack=attack,
                    **pool,
                )
            self.pools.append(pool)

        self.n_clients = sum([len(p) for p in self.pools])
        self.n_concurrent = config.get("n_concurrent", self.n_clients)

        self.strategy = instantiate(
            config.strategy,
            min_fit_clients=self.n_clients,
            min_evaluate_clients=self.n_clients,
            on_fit_config_fn=mk_config_fn(
                {
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                }
            ),
            on_evaluate_config_fn=mk_config_fn({"batch_size": config.batch_size}),
        )

        if "server" in config and config.server is not None:
            self.server = instantiate(config.server)

    def run(self, **ray_kwargs) -> History:
        """Run the experiment."""
        init_kwargs = (ray_kwargs or {}) | {
            "ignore_reinit_error": True,
            # "local_mode": True,
        }
        ray.init(**init_kwargs)

        for pool in self.pools:
            pool.deploy()

        mappings = reduce(lambda a, b: a | b, [p.gen_mappings() for p in self.pools])

        fn = functools.partial(
            get_client,
            mappings=mappings,
            seed=self.seed,
        )

        return start_simulation(
            client_fn=fn,
            num_clients=self.n_clients,
            config=ServerConfig(num_rounds=self.n_rounds),
            strategy=self.strategy,
            # client_resources=compute_client_resources(self.n_concurrent),
            actor_kwargs={
                # Enable GPU growth upon actor init
                # does nothing if `num_gpus` in client_resources is 0.0
                "on_actor_init_fn": enable_tf_gpu_growth
            },
            clients_ids=reduce(lambda a, b: a + b, [p.ids for p in self.pools]),
            server=self.server,
            keep_initialised=True,
        )


def mk_config_fn(config: ConfigDict) -> Callable[[int], ConfigDict]:
    """Return a function which creates a config for the given round."""
    return lambda r: config | {"round": r}


def compute_client_resources(n_concurrent: int) -> tuple[float, float]:
    """Compute the number of CPUs and GPUs to allocate to each client."""
    available_cpus = psutil.cpu_count()
    available_gpus = len(tf.config.list_physical_devices("GPU"))
    return {
        "num_cpus": max(1, available_cpus / n_concurrent),
        "num_gpus": available_gpus / n_concurrent,
    }


def obj_to_list(
    config_obj: ListConfig | DictConfig,
    expected_length: int = 0,
) -> list:
    """Convert a DictConfig or ListConfig object to a list."""
    if not isinstance(config_obj, (ListConfig, DictConfig)):
        raise ConfigError(
            f"Invalid config object: {type(config_obj)}. Expected a list or dictionary."
        )

    if isinstance(config_obj, DictConfig):
        config_obj = [config_obj]

    if expected_length > 0:
        if len(config_obj) > 1 and len(config_obj) != expected_length:
            raise ConfigError(
                "The number of items in config_obj should be equal to"
                f" {expected_length}, or 1."
            )

        elif len(config_obj) == 1:
            config_obj = list(config_obj) * expected_length

    return config_obj
