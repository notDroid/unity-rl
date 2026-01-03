import os
import torch

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass

from torchrl.modules import ActorValueOperator, ValueOperator, ActorCriticWrapper
from tensordict.nn import TensorDictModule
from rlkit.templates import PPOBasic, PPOTrainConfig, PPOState, ppo_log_keys
from rlkit.envs import UnityEnv
from rlkit.utils import plot_results
from huggingface_hub import upload_file
from torchinfo import summary

from torch import nn

from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, OneHotCategorical
from tensordict.nn.distributions import NormalParamExtractor
from rlkit.modules import ClipPPOInverseLoss, ActorValueInverseOperator
from rlkit.models.utils import CatWrapper

from torchrl.objectives import ClipPPOLoss, ValueEstimators

todict = lambda x: OmegaConf.to_container(x, resolve=True)

def make_ppo_agent(config, verbose=False, device="cpu"):
     # Trunk (if present)
    in_keys = list(config.env.observation.observation_keys) if "observation_keys" in config.env.observation else ["observation"]
    trunk = None
    if "trunk" in config:
        trunk = instantiate(config.trunk).to(device)
    head_in_keys = in_keys if not trunk else ["hidden_features"]

    # Head
    Model = get_class(config.model._target_)
    model_config = todict(config.model.params)

    # Policy
    policy_config = model_config.copy()
    if config.env.action.type == "continuous":
        policy_config["out_features"] *= 2
    policy_base = Model(**policy_config)
    policy = create_policy(policy_base, policy_type=config.env.action.type, in_keys=head_in_keys).to(device)

    # Value
    value_config = model_config.copy()
    value_config["out_features"] = 1
    value_base = Model(**value_config)
    value = ValueOperator(value_base, in_keys=head_in_keys).to(device)

    # Inverse Dynamics (optional)
    inverse = None
    if config.algo.params.get("inverse_coef", 0) > 0:
        inverse_config = model_config.copy()
        inverse_config["in_features"] = 2 * model_config["in_features"]
        inverse_base = Model(**inverse_config)
        inverse_base = CatWrapper(inverse_base)
        inverse = TensorDictModule(inverse_base, in_keys=head_in_keys + [("next", key) for key in head_in_keys], out_keys=["predict_action"]).to(device)

    # Assemble Together
    if trunk:
        if not inverse:
            model = ActorValueOperator(
                common_operator=TensorDictModule(trunk, in_keys=in_keys, out_keys=["hidden_features"]),
                policy_operator=policy,
                value_operator=value,
            )
        else:
            model = ActorValueInverseOperator(
                trunk_net=trunk,
                policy_operator=policy,
                value_operator=value,
                inverse_operator=inverse,
                observation_keys=in_keys,
                hidden_keys=["hidden_features"],
            )
    else:
        model = ActorCriticWrapper(
            policy_operator=policy,
            value_operator=value,
        )
        if inverse: raise NotImplementedError("Inverse dynamics with no trunk is not supported.")

    if verbose:
        if not trunk:
            try: summary(policy_base, input_size=(1, model_config["in_features"]))
            except: pass
    
    return model.to(device)

class PPORunner:
    def __init__(self, config: DictConfig):
        self.config = config

    def run(self, verbose=False, continue_=True, device="cpu"):
        # Environment and Train Config
        env_config = todict(self.config.env.params)
        create_env = lambda: UnityEnv(self.config.env.name, **env_config)

        train_config = todict(self.config.trainer.params)
        train_config['device'] = device
        train_config = PPOTrainConfig(**train_config)

        ### 1. Model
        model = make_ppo_agent(self.config, device=device)

        ### 2. State

        # Loss Module
        ppo_algo_config = todict(self.config.algo.params)
        loss_module = create_loss_module(model, **ppo_algo_config).to(device)

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
            if self.config.algo.params.get("inverse_coef", 0) > 0: log_keys.append("inverse_loss")
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
                    model.load_state_dict(checkpoint["model_state_dict"])
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
            model=model,
            optimizer=optimizer,
            loss_module=loss_module,

            checkpointer=checkpointer,
            logger=logger,
            scaler=scaler,
            metric_module=metric_module,
            lr_scheduler=lr_scheduler,
        )
        sync_interval = self.config.get("hf_sync_interval", None)
        sync_interval = None if sync_interval == 0 else sync_interval
        repo_id = self.config.get("repo_id", 'notnotDroid/unity-rl')
        if sync_interval:
            config_path = os.path.join(self.config.dir, "config", "config.yaml")
            upload_file(path_or_fileobj=config_path, path_in_repo=config_path, repo_id=repo_id, repo_type="model", commit_message="config upload")
            

        ### 3. Run PPO
        ppo = PPOBasic(create_env=create_env, ppo_config=train_config, ppo_state=state, verbose=verbose)
        # ppo.run()
        for gen in range(ppo.config.start_generation, ppo.config.generations):
            ppo.step(gen)
            if sync_interval is not None and ((gen+1) % sync_interval == 0):
                if logger: logger.sync_to_hub()
                if checkpointer: checkpointer.sync_to_hub()
        

        # 4. Save Results
        model_path = None
        model_path = self.config.get("model_path", None)
        if model_path: 
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ppo.close(model_path)

        if sync_interval and model_path:
            upload_file(path_or_fileobj=model_path, path_in_repo=model_path, repo_id=repo_id, repo_type="model", commit_message="model upload")
        if sync_interval and logger: logger.sync_to_hub()
        
        if "results_path" in self.config and logger:
            results_path = self.config.results_path
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            plot_results(logger.dataframe(), results_path, log_index="timestep") # Assuming timestep is a provided key


def create_continuous_policy(model, in_keys=["observation"]):
    model = nn.Sequential(
        model,
        NormalParamExtractor()
    )
    model = TensorDictModule(model, in_keys=in_keys, out_keys=["loc", "scale"])
    
    policy = ProbabilisticActor(
        module=model,  
        distribution_class=TanhNormal,

        in_keys=["loc", "scale"],
        out_keys=["action"],

        return_log_prob=True,
        log_prob_key="log_prob",
        cache_dist=True,
    )

    return policy

def create_discrete_policy(model, in_keys=["observation"]):
    model = TensorDictModule(model, in_keys=in_keys, out_keys=["logits"])
    
    policy = ProbabilisticActor(
        module=model,  
        distribution_class=OneHotCategorical,

        in_keys=["logits"],
        out_keys=["action"],

        return_log_prob=True,
        log_prob_key="log_prob",
        cache_dist=True,
    )

    return policy

def create_policy(model, policy_type, **kwargs):
    if policy_type == 'continuous':
        return create_continuous_policy(model, **kwargs)
    if policy_type == 'discrete':
        return create_discrete_policy(model, **kwargs)
    raise KeyError(f"Unknown policy_type: {policy_type}, expected one of [\"continuous\", \"discrete\"]")

def create_loss_module(model, epsilon, entropy_coef, gamma, lmbda=0.95, value_coef=1, inverse_coef=0, inverse_loss_fn="cross_entropy"):
    if inverse_coef == 0:
        loss_module = ClipPPOLoss(
            actor_network=model.get_policy_head(),
            critic_network=model.get_value_head(),
            clip_epsilon=epsilon,
            loss_critic_type="smooth_l1",
            entropy_coeff=entropy_coef,
            value_coeff=value_coef,
        )
    else:
        loss_module = ClipPPOInverseLoss(
            actor_network=model.get_policy_head(),
            critic_network=model.get_value_head(),
            inverse_dynamics_network=model.get_inverse_head(),
            clip_epsilon=epsilon,
            loss_critic_type="smooth_l1",
            entropy_coeff=entropy_coef,
            value_coeff=value_coef,
            inverse_dynamics_coeff=inverse_coef,
            inverse_loss_fn=inverse_loss_fn,
        )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, 
        gamma=gamma, lmbda=lmbda, 
        shifted=True, average_gae=True,
    )

    # Defaults
    loss_module.set_keys(
        # From value estimator
        advantage='advantage',
        value_target='value_target', 
        value='state_value', 
        # From policy (should match ProbabilisticActor, this is correctly set automatically key not specified)
        sample_log_prob='log_prob',
        action='action', 
        # For value estimator
        reward='reward', 
        done='done', 
        terminated='terminated',
    )

    return loss_module