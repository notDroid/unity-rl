import os
import torch

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from runners.ppo import make_ppo_agent

from rlkit.utils import download_from_hf_hub, upload_to_hf_hub
from huggingface_hub import HfFileSystem

todict = lambda x: OmegaConf.to_container(x, resolve=True)

DEFAULT_REPO_ID = 'notnotDroid/unity-rl'

def ppo_load_config(directory_path, save_name=None, config_file="config/config.yaml", **kwargs):
    # Load config
    config_path = os.path.join(directory_path, config_file)
    config = OmegaConf.load(config_path)

    # Get model
    model = make_ppo_agent(config)

    # Load Params
    if save_name:
        save_path = os.path.join(directory_path, save_name)
        state_obj = torch.load(save_path, map_location=torch.device('cpu'), **kwargs)
        model.load_state_dict(state_obj["model_state_dict"])

    return model
    
def ppo_upload_model(directory_path, save_name, config_file="config/config.yaml", repo_id=DEFAULT_REPO_ID, **kwargs):
    config_path = os.path.join(directory_path, config_file)
    save_path = os.path.join(directory_path, save_name)

    upload_to_hf_hub(local_path=config_path, repo_id=repo_id, remote_path=config_path, **kwargs)
    upload_to_hf_hub(local_path=save_path, repo_id=repo_id, remote_path=save_path, **kwargs)

def ppo_download_model(directory_path, save_name, config_file="config/config.yaml", repo_id=DEFAULT_REPO_ID, **kwargs):
    config_path = os.path.join(directory_path, config_file)
    save_path = os.path.join(directory_path, save_name)

    download_from_hf_hub(local_path=config_path, repo_id=repo_id, remote_path=config_path, **kwargs)
    download_from_hf_hub(local_path=save_path, repo_id=repo_id, remote_path=save_path, **kwargs)

def PPOAgent(environment_name, config_name, run_name, save_type='models', config_file="config/config.yaml", repo_id=DEFAULT_REPO_ID, **kwargs):
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

def _print_tree_recursive(node, indent=""):
    """Helper to visualize the dictionary structure."""
    if isinstance(node, list):  # Leaf node (Runs)
        for i, run in enumerate(node):
            is_last = (i == len(node) - 1)
            prefix = "└── " if is_last else "├── "
            print(f"{indent}{prefix}{run} (Run)")
    elif isinstance(node, dict):
        keys = list(node.keys())
        for i, k in enumerate(keys):
            is_last = (i == len(keys) - 1)
            prefix = "└── " if is_last else "├── "
            print(f"{indent}{prefix}{k}")
            
            next_indent = indent + ("    " if is_last else "│   ")
            _print_tree_recursive(node[k], next_indent)

def get_repo_tree(repo_id=DEFAULT_REPO_ID, filters=None, verbose=False):
    """
    Fetches the experiment tree from Hugging Face and returns it as a nested dictionary.
    
    Structure: {Env: {Algo: {Config: [Run1, Run2]}}}
    
    Args:
        repo_id (str): Hugging Face Repository ID.
        filters (list): List of filters to traverse specific subtrees (e.g., ['3DBall', 'ppo']).
        verbose (bool): If True, prints the visual tree structure to stdout.
        
    Returns:
        dict: A nested dictionary representing the file structure.
    """
    fs = HfFileSystem()
    base_path = f"{repo_id}/experiments"
    
    # Determine where to start searching
    search_path = base_path
    if filters:
        # Sanitize filters to ensure no empty strings
        valid_filters = [f for f in filters if f]
        if valid_filters:
            search_path = f"{base_path}/{'/'.join(valid_filters)}"

    def _build_tree(current_path):
        # Determine depth relative to the root 'experiments' folder
        # Depth 0=Root, 1=Env, 2=Algo, 3=Config -> (Look for models)
        if current_path == base_path:
            depth = 0
        else:
            rel = current_path[len(base_path):].strip("/")
            depth = len(rel.split("/"))

        # Base Case: At Config level (Depth 3), look for 'models' folder
        if depth == 3:
            models_path = f"{current_path}/models"
            runs = []
            if fs.exists(models_path):
                try:
                    files = fs.ls(models_path, detail=False)
                    runs = [os.path.basename(f).replace('.pt', '') for f in files if f.endswith('.pt')]
                except Exception:
                    pass
            return runs

        # Recursive Case: List directories
        tree = {}
        try:
            paths = fs.ls(current_path, detail=False)
            for p in paths:
                name = os.path.basename(p)
                # Ignore system files/folders if any
                if name.startswith('.'): continue
                
                child = _build_tree(p)
                # Only include non-empty branches
                if child: 
                    tree[name] = child
        except Exception:
            pass # Path might not exist or be a file
            
        return tree

    # 1. Build the object
    if verbose: print(f"\n[Fetching tree from: {search_path} ...]")
    tree_object = _build_tree(search_path)
    
    # 2. Print if requested
    if verbose:
        if not tree_object:
            print(f"No entries found at {search_path}")
        else:
            root_label = filters[-1] if filters else "experiments"
            print(f"{root_label}")
            _print_tree_recursive(tree_object)
        print("")

    return tree_object