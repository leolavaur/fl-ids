import pickle
from functools import partial
from pathlib import Path

from eiffel.datasets.nfv2 import load_data
from similarity.module import AttackPartitioner  # local

SEED = 1138

cicids = load_data(
    path="../../data/nfv2/sampled/cicids.csv.gz",
    key="cicids",
    _default_target=["DoS"],
    seed=SEED,
)
botiot = load_data(
    path="../../data/nfv2/sampled/botiot.csv.gz",
    key="botiot",
    _default_target=["Reconnaissance"],
    seed=SEED,
)
nb15 = load_data(
    path="../../data/nfv2/sampled/nb15.csv.gz",
    key="nb15",
    _default_target=["Analysis"],
    seed=SEED,
)
toniot = load_data(
    path="../../data/nfv2/sampled/toniot.csv.gz",
    key="toniot",
    _default_target=["injection"],
    seed=SEED,
)


partial_partitioner = partial(
    AttackPartitioner,
    n_partitions=10,
    class_column="Attack",
    preserved_classes=["Benign", "injection"],
    seed=SEED,
)


train, test = cicids.split(at=0.8, seed=SEED)
for scheme in [
    "iid_full",
    "iid_drop_1",
    "iid_drop_2",
    "iid_keep_1",
    "kmeans_full",
    "kmeans_drop_1",
    "kmeans_drop_2",
    "kmeans_keep_1",
]:
    benign_strategy, attack_strategy = scheme.split("_", maxsplit=1)
    kwargs = {}
    if len((t := attack_strategy.split("_"))) == 2:
        key, value = t
        kwargs = {f"n_{key}": int(value)}

    partitioner = partial_partitioner(
        benign_strategy=benign_strategy, attacker_target=None, **kwargs
    )

    partitioner.load(train)
    shards = partitioner.all()

    for i, shard in enumerate(shards):
        dir = Path(f"../../data/nfv2/partitions/cicids/{scheme}")
        if not dir.exists():
            dir.mkdir(parents=True)

        pickle.dump(shard, (dir / f"shard_{i}.pkl").open("wb"))
