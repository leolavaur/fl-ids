import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="config", config_name="eiffel")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
