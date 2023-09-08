"""Entrypoint for the Eiffel CLI."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from .engine.logging import logged


@logged
def test_fn():  # noqa: D103
    print("I'm doing something!")


@hydra.main(version_base="1.3", config_path="conf", config_name="eiffel")
def main(cfg: DictConfig):
    """Entrypoint for the Eiffel CLI."""
    log = logging.getLogger(__name__)

    test_fn()

    log.info("Starting Eiffel")
    if cfg.is_empty():
        log.critical("Empty configuration.")
        exit(1)

    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
