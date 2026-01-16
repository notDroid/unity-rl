import hydra
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import torch

@hydra.main(version_base=None, config_path="configs", config_name="3dball_ppo")
def main(cfg: DictConfig):
    verbose = cfg.setdefault("verbose", True)
    continue_ = cfg.setdefault("continue_", True)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = cfg.setdefault("device", default_device)

    if verbose:
        print(f"Using Config \"{HydraConfig.get().job.config_name}\":\n")
        print("------------------------------------------------------")
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print("------------------------------------------------------\n")

    Algo = get_class(cfg.algo._target_)
    algo = Algo(cfg)
    algo.run()

if __name__ == "__main__":
    main()
