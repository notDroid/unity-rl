from torch import nn

from rlkit.models import MLP
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.objectives import ClipPPOLoss, ValueEstimators


def create_policy(model_config):
    model_config = model_config.copy()
    model_config["out_features"] *= 2
    model = MLP(**model_config)

    model = nn.Sequential(
        model,
        NormalParamExtractor()
    )
    model = TensorDictModule(model, in_keys=["observation"], out_keys=["loc", "scale"])
    
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

def create_value(model_config):
    # Remove out_features from config
    model_config = model_config.copy()
    model_config["out_features"] = 1

    model = MLP(**model_config)
    value = TensorDictModule(model, in_keys=["observation"], out_keys=["state_value"])
    return value


def make_loss_module(policy, value, epsilon, entropy_coef, gamma, lmbda):
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value,
        clip_epsilon=epsilon,
        loss_critic_type="smooth_l1", # Default
        entropy_coeff=entropy_coef,
        normalize_advantage=True,
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, 
        # time_dim=-1,
        gamma=gamma, lmbda=lmbda, 
        shifted=True,
        auto_reset_env=True,
    )

    # All defaults
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