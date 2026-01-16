import os

from hydra.utils import get_class, instantiate, get_method
from omegaconf import DictConfig, OmegaConf

from rlkit.envs import UnityEnv
from rlkit.utils import plot_results
from huggingface_hub import upload_file

import torch
from torch import nn

todict = lambda x: OmegaConf.to_container(x, resolve=True)

def create_trainer(config: DictConfig):
    device = torch.device(config.get("device", "cpu"))

    # 1. Environment
    env_config = todict(config.env.params)
    env_fn = OmegaConf(config.env.env_fn)
    create_env = lambda: env_fn(config.env.name, **env_config)

    # 2. Agent
    agent_builder = get_class(config.agent.builder)()
    agent = agent_builder.build(config).to(device)

    # 3. Loss Module
    loss_builder = get_class(config.loss.builder)()
    loss_module = loss_builder.build(config, agent).to(device)
    # 4. State
    state_builder = get_class(config.state.builder)()
    state = state_builder.build(config, agent, loss_module)

    # 5. Trainer
    trainer_builder = get_class(config.trainer.builder)()
    ppo = trainer_builder.build(config, create_env, state)

    return ppo

class PPORunner:    
    def run(self, config: DictConfig):
        # 1. Create Trainer
        ppo = create_trainer(config)
            
        # (Optional) HuggingFace Syncing
        logger = ppo.state.get('logger', None)
        checkpointer = ppo.state.get('checkpointer', None)
        sync_interval = config.get("hf_sync_interval", None)
        sync_interval = None if sync_interval == 0 else sync_interval
        repo_id = config.get("repo_id", 'notnotDroid/unity-rl')

        # Upload config
        if sync_interval:
            config_path = os.path.join(config.dir, "config", "config.yaml")
            upload_file(path_or_fileobj=config_path, path_in_repo=config_path, repo_id=repo_id, repo_type="model", commit_message="config upload")

        ### Run PPO
        for gen in range(ppo.config.start_generation, ppo.config.generations):
            ppo.step(gen)
            if sync_interval is not None and ((gen+1) % sync_interval == 0):
                if logger: logger.sync_to_hub()
                if checkpointer: checkpointer.sync_to_hub()
        

        # 4. Save Results
        model_path = None
        model_path = config.get("model_path", None)
        if model_path: 
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ppo.close(model_path)

        # Final HuggingFace Sync
        if sync_interval and model_path:
            upload_file(path_or_fileobj=model_path, path_in_repo=model_path, repo_id=repo_id, repo_type="model", commit_message="model upload")
        if sync_interval and logger: logger.sync_to_hub()
        
        # Plot Results
        if "results_path" in config and logger:
            results_path = config.results_path
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            plot_results(logger.dataframe(), results_path, log_index="timestep") # Assuming timestep is a provided key