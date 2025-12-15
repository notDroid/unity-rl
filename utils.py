import os
import torch

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from runners.ppo import make_ppo_agent

todict = lambda x: OmegaConf.to_container(x, resolve=True)

def ppo_load_config(directory_path, save_name, config_file="config/config.yaml", **kwargs):
    # Load config
    config_path = os.path.join(directory_path, config_file)
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)

    # Get model
    model = make_ppo_agent(config)

    # Load Params
    save_path = os.path.join(directory_path, save_name)
    state_obj = torch.load(save_path, **kwargs)
    model.load_state_dict(state_obj["model_state_dict"])

    return model
    