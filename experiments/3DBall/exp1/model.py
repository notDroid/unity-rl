from torch import nn
from model_util import MLP

from torchrl.modules import ProbabilisticActor, TanhNormal

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

OBSERVATION_KEY = "VectorSensor_size8"
ACTION_KEY = "continuous_action"

def create_policy(config, observation_key=OBSERVATION_KEY, action_key=ACTION_KEY):
    config = config.copy()
    config["out_features"] *= 2 # Double output dim, for loc and scale
    model = MLP(**config)

    normal_params_model = nn.Sequential(
        model,
        NormalParamExtractor()
    )
    logits_model = TensorDictModule(normal_params_model, in_keys=[("agents", observation_key)], out_keys=[("agents", "loc"), ("agents", "scale")])
    policy = ProbabilisticActor(
        module=logits_model,  
        distribution_class=TanhNormal,

        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", action_key)],

        return_log_prob=True,
        log_prob_key=("agents", "log_prob"),
        cache_dist=True,
    )

    return policy

def create_value(config, observation_key=OBSERVATION_KEY):
    # Remove out_features from config
    config = config.copy()
    config["out_features"] = 1

    model = MLP(**config)
    value = TensorDictModule(model, in_keys=[("agents", observation_key)], out_keys=[("agents", "state_value")])
    return value

