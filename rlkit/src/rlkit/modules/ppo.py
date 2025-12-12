from torch import nn

from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, OneHotCategorical
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.objectives import ClipPPOLoss, ValueEstimators

def ContinuousPolicyWrapper(model):
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

def DiscretePolicyWrapper(model):
    model = TensorDictModule(model, in_keys=["observation"], out_keys=["logits"])
    
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

def PolicyWrapper(model, policy_type):
    if policy_type == 'continuous':
        return ContinuousPolicyWrapper(model)
    if policy_type == 'discrete':
        return DiscretePolicyWrapper(model)
    raise KeyError(f"Unknown policy_type: {policy_type}, expected one of [\"continuous\", \"discrete\"]")

def ValueWrapper(model):
    value = TensorDictModule(model, in_keys=["observation"], out_keys=["state_value"])
    return value

def PPOLossModule(policy, value, epsilon, entropy_coef, gamma, lmbda=0.95, value_coef=1):
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value,
        clip_epsilon=epsilon,
        loss_critic_type="smooth_l1", # Default
        entropy_coeff=entropy_coef,
        value_coeff=value_coef,
        # normalize_advantage=True,
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, 
        # time_dim=-1,
        gamma=gamma, lmbda=lmbda, 
        shifted=True, average_gae=True,
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