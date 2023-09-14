"""Client pool API for Eiffel."""


from typing import Optional

from flwr.client import ClientLike
from ray import ObjectRef

from eiffel.datasets.dataset import Dataset
from eiffel.datasets.poisoning import PoisonIns

from ..utils.typing import EiffelCID


class Pool:
    """Pool of clients.

    A pool is a collection of clients that share the same dataset and attack type.

    Attributes
    ----------
    clients : dict[EiffelCID, flwr.client.ClientLike]
        The clients in the pool.
    shards : dict[EiffelCID, ray.ObjectRef]
        The DatasetHodlers references for each client, as datasets are stored in the Ray
        object store.
    """

    clients: dict[EiffelCID, ClientLike]
    shards: dict[EiffelCID, ObjectRef]

    def __init__(
        self,
        dataset: Dataset | str,
        benign: int,
        malicious: int = 0,
        attack: Optional[dict | PoisonIns] = None,
    ) -> None:
        """Initialize the pool.

        Parameters
        ----------
        dataset : Dataset | str
            The dataset used by the clients. If a string is provided, it should be a
            path a `load_data` function that returns a `Dataset` object. Otherwise, it
            should be a `Dataset` object.
        benign : int
            The number of benign clients in the pool.
        malicious : int, optional
            The number of malicious clients in the pool. Defaults to 0.
        attack : Optional[dict | PoisonIns], optional
            The attack to perform. If a dictionary is provided, it should be a valid
            dictionary
        """
        pass
