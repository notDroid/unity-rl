import hydra
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import torch

@hydra.main(version_base=None, config_path="configs", config_name="3dball_ppo")
def main(cfg: DictConfig):
    Algo = get_class(cfg.algo._target_)
    algo = Algo(cfg)
    
    verbose = False
    continue_ = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "verbose" in cfg: verbose = cfg.verbose
    if "continue_" in cfg: continue_ = cfg.continue_
    if "device" in cfg: device = cfg.device

    if verbose:
        print(f"Using Config \"{HydraConfig.get().job.config_name}\":\n")
        print("------------------------------------------------------")
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print("------------------------------------------------------\n")

    algo.run(verbose, continue_, device)

if __name__ == "__main__":
    main()
