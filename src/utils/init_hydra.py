import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="config")
def init_hydra(cfg: DictConfig) -> DictConfig:
    """
    Initialize Hydra configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        DictConfig: Hydra configuration object.
    """
    # Print the configuration for debugging purposes
    print(OmegaConf.to_yaml(cfg))
    return cfg

if __name__ == "__main__":
    init_hydra()