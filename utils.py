import os
import torch

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from runners.ppo import make_ppo_agent

from rlkit.utils import download_from_hf_hub, upload_to_hf_hub

todict = lambda x: OmegaConf.to_container(x, resolve=True)

def ppo_load_config(directory_path, save_name=None, config_file="config/config.yaml", **kwargs):
    # Load config
    config_path = os.path.join(directory_path, config_file)
    config = OmegaConf.load(config_path)

    # Get model
    model = make_ppo_agent(config)

    # Load Params
    if save_name:
        save_path = os.path.join(directory_path, save_name)
        state_obj = torch.load(save_path, **kwargs)
        model.load_state_dict(state_obj["model_state_dict"])

    return model
    
def ppo_upload_model(directory_path, save_name, config_file="config/config.yaml", repo_id='notnotDroid/unity-rl', **kwargs):
    config_path = os.path.join(directory_path, config_file)
    save_path = os.path.join(directory_path, save_name)

    upload_to_hf_hub(local_path=config_path, repo_id=repo_id, remote_path=config_path, **kwargs)
    upload_to_hf_hub(local_path=save_path, repo_id=repo_id, remote_path=save_path, **kwargs)

def ppo_download_model(directory_path, save_name, config_file="config/config.yaml", repo_id='notnotDroid/unity-rl', **kwargs):
    config_path = os.path.join(directory_path, config_file)
    save_path = os.path.join(directory_path, save_name)

    download_from_hf_hub(local_path=config_path, repo_id=repo_id, remote_path=config_path, **kwargs)
    download_from_hf_hub(local_path=save_path, repo_id=repo_id, remote_path=save_path, **kwargs)

def PPOAgent(environment_name, config_name, run_name, save_type='models', config_file="config/config.yaml", repo_id='notnotDroid/unity-rl', **kwargs):
    directory_path = os.path.join("experiments", environment_name, 'ppo', config_name)

    if save_type == 'models': suffix = '.pt'
    elif save_type == 'ckpts': suffix = '.ckpt'
    else: raise RuntimeError("save type expected to be \"models\" or \"ckpts\"")
    save_name = os.path.join(save_type, run_name + suffix)

    config_path = os.path.join(directory_path, config_file)
    save_path = os.path.join(directory_path, save_name)
    
    # Download if needed
    if not os.path.exists(config_path) or not os.path.exists(save_path):
        ppo_download_model(directory_path, save_name, config_file, repo_id, **kwargs)

    # Create model
    return ppo_load_config(directory_path, save_name, config_file)