import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="conf", config_name="eiffel")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(instantiate(cfg.strategy))


if __name__ == "__main__":
    main()
