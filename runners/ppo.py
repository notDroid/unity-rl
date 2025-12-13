import os
import torch

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass

from rlkit.modules import PolicyWrapper, ValueWrapper, PPOLossModule
from rlkit.templates import PPOBasic, PPOTrainConfig, PPOState, ppo_log_keys
from rlkit.envs import UnityEnv
from rlkit.utils import plot_results
from torchinfo import summary

todict = lambda x: OmegaConf.to_container(x, resolve=True)

class PPORunner:
    def __init__(self, config: DictConfig):
        self.config = config

    def run(self, verbose=False, continue_=True, device="cpu"):
        
        # Environment and Train Config
        env_config = todict(self.config.env.params)
        create_env = lambda: UnityEnv(self.config.env.name, **env_config)

        train_config = todict(self.config.trainer.params)
        train_config = PPOTrainConfig(**train_config)

        ### 1. Model
        Model = get_class(self.config.model._target_)
        model_config = todict(self.config.model.params)

        # Policy
        policy_config = model_config.copy()
        if self.config.env.action.type == "continuous":
            policy_config["out_features"] *= 2
        policy_base = Model(**policy_config)
        policy = PolicyWrapper(policy_base, policy_type=self.config.env.action.type).to(device)

        # Value
        value_config = model_config.copy()
        value_config["out_features"] = 1
        value_base = Model(**value_config)
        value = ValueWrapper(value_base).to(device)

        if verbose:
            summary(policy_base, input_size=(1, model_config["in_features"]))

        ### 2. State

        # Loss Module
        ppo_algo_config = todict(self.config.algo.params)
        loss_module = PPOLossModule(policy, value, **ppo_algo_config).to(device)

        # Optimizer
        optimizer_config = todict(self.config.optimizer.params)
        optimizer = get_class(self.config.optimizer._target_)(loss_module.parameters(), **optimizer_config)

        # Utils
        checkpointer = None
        logger = None
        scaler = None
        metric_module = None
        lr_scheduler = None
        if "checkpointer" in self.config: checkpointer = instantiate(self.config.checkpointer)
        if "logger" in self.config: 
            log_keys = list(self.config.logger.get("keys", ppo_log_keys))
            if "lr_scheduler" in self.config: log_keys.append("lr")
            logger = instantiate(self.config.logger, keys=log_keys)
        if "scaler" in self.config: scaler = instantiate(self.config.scaler)
        if "metric_module" in self.config: metric_module = instantiate(self.config.metric_module)
        if "lr_scheduler" in self.config: 
            lr_scheduler = instantiate(self.config.lr_scheduler, optimizer=optimizer, total_epochs=train_config.generations)
            

        # Load Past State
        start_generation = 0
        if continue_:
            if checkpointer:
                checkpoint = checkpointer.load_progress()
                if checkpoint:
                    start_generation = int(checkpoint["generation"])
                    policy.load_state_dict(checkpoint["policy_state_dict"])
                    value.load_state_dict(checkpoint["value_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if "scaler_state_dict" in checkpoint:
                        if not scaler: scaler = torch.amp.GradScaler(enabled=(train_config.amp_dtype == torch.float16))
                        scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    if logger:
                        logger.revert("generation", start_generation)
                    if "lr_scheduler_state_dict" in checkpoint:
                        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                    print("CHECKPOINT FOUND, STARTING FROM GENERATION:", start_generation)
                else:
                    checkpointer.reset()
                    if logger: logger.reset()
            else:
                if logger:
                    logger.revert()
                    df = logger.dataframe()
                    if len(df) > 0:
                        start_generation = int(df["generation"].iloc[-1])
                        if lr_scheduler: 
                            lr_scheduler = instantiate(
                                self.config.lr_scheduler, optimizer=optimizer, total_epochs=train_config.generations,
                                last_epoch=start_generation-1,
                            )
                            initial_lrs = lr_scheduler.get_lr() 
                            for param_group, lr in zip(optimizer.param_groups, initial_lrs):
                                param_group['lr'] = lr
                else:
                    print("WARNING: potentially undefined behavior when continue=True with no logger or checkpointer, use with caution")
        else:
            if logger: logger.reset()
            if checkpointer: checkpointer.reset()
        train_config.start_generation = start_generation

        state = PPOState(
            policy=policy,
            value=value,
            optimizer=optimizer,
            loss_module=loss_module,

            checkpointer=checkpointer,
            logger=logger,
            scaler=scaler,
            metric_module=metric_module,
            lr_scheduler=lr_scheduler,
        )

        ### 3. Run PPO
        ppo = PPOBasic(create_env, train_config, state)
        ppo.run(verbose)

        # 4. Save Results
        model_path = None
        if "model_path" in self.config: 
            model_path = self.config.model_path
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ppo.close(model_path)
        
        if "results_path" in self.config and logger:
            results_path = self.config.results_path
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            plot_results(results_path, logger, log_index="timestep") # Assuming timestep is a provided key