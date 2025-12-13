import os
import torch

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from rlkit.modules import PolicyWrapper, ValueWrapper, PPOLossModule

todict = lambda x: OmegaConf.to_container(x, resolve=True)

def ppo_load_config(directory_path, save_name, config_file="config/config.yaml", **kwargs):
    # Load config
    config_path = os.path.join(directory_path, config_file)
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)

    # Get model
    Model = get_class(config.model._target_)
    model_config = todict(config.model.params)

    # Policy
    policy_config = model_config.copy()
    if config.env.action.type == "continuous":
        policy_config["out_features"] *= 2
    policy_base = Model(**policy_config)
    policy = PolicyWrapper(policy_base, policy_type=config.env.action.type)

    # Value
    value_config = model_config.copy()
    value_config["out_features"] = 1
    value_base = Model(**value_config)
    value = ValueWrapper(value_base)

    # Load Params
    save_path = os.path.join(directory_path, save_name)
    state_obj = torch.load(save_path, **kwargs)
    policy.load_state_dict(state_obj["policy_state_dict"])
    value.load_state_dict(state_obj["value_state_dict"])

    return policy, value
    