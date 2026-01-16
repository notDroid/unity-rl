import hydra
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
import torch
OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    verbose = cfg.setdefault("verbose", True)
    cfg.setdefault("continue_", True)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.setdefault("device", default_device)

    if verbose:
        print(f"Using Config \"{HydraConfig.get().job.config_name}\":\n")
        print("------------------------------------------------------")
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print("------------------------------------------------------\n")

    algo = get_class(cfg.algo.runner)()
    algo.run(cfg)

if __name__ == "__main__":
    main()
