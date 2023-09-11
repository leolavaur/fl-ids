"""Eiffel engine."""

from flwr.client import ClientLike
from flwr.server import Server
from flwr.server.history import History
from hydra.utils import instantiate
from omegaconf import DictConfig

from ..dataset.common import Dataset
from ..utils.poisoning import PoisonIns


class Experiment:
    """Eiffel experiment.

    Attributes
    ----------
    server : flwr.server.Server
        The Flower server. It administrates the entire FL process, and is responsible
        for aggregating the models, based on the function provided in the
        `flwr.server.Strategy` object.
    clients : dict[str, flwr.client.ClientLike]
        The Flower clients. They are responsible for training the models based on their
        given datasets.
    datasets : list[Dataset]
        The datasets used by the clients. The datasets are loaded from the `load_data`
        function that is provided in the configuration (under the form of a
        `omegaconf.DictConfig` object).
    """

    server: Server
    clients: dict[str, ClientLike]
    datasets: list[Dataset]

    def __init__(self, config: DictConfig):
        """Initialize the experiment.

        Parameters
        ----------
        config : omegaconf.DictConfig
            The configuration of the experiment. This is the configuration aggregated by
            Hydra from the command line arguments and the configuration files. It must
            contain the following keys:

            * server: The configuration of the Flower server. It is passed to an
                `hydra.utils.instantiate` call.
            * clients: The configuration of the Flower clients. It contains
                client-related options, including `fn` (passed to `hydra.utils.call`) to
                instantiate clients, and `class` to define the actual ClientLike object
                (passed to `hydra.utils.instantiate`). Both are optional, and default
                values are provided by Eiffel.
            * dataset: The configuration of the datasets. This object contains the
              `load_data` function that is passed then to a call to `hydra.utils.call`.

            Additionally, the configuration can contain the following keys:

            * strategy: The configuration of the strategy used by the Flower server. It
                is passed to an `hydra.utils.instantiate` call. Defaults to None.
        """
        self.server = config.server
        self.clients = config.clients
        self.datasets = config.datasets

    def run(self) -> History:
        pass
