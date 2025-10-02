import config
from model_util import MLP
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torch.distributions import OneHotCategorical

import config


def create_policy(model_config):
    model_config = model_config.copy()
    model = MLP(**model_config)
    logits_model = TensorDictModule(model, in_keys=[(config.ROOT_KEY, config.OBSERVATION_KEY)], out_keys=[(config.ROOT_KEY, "logits")])
    policy = ProbabilisticActor(
        module=logits_model,  
        distribution_class=OneHotCategorical,

        in_keys=[(config.ROOT_KEY, "logits")],
        out_keys=[(config.ROOT_KEY, config.ACTION_KEY)],

        return_log_prob=True,
        log_prob_key=(config.ROOT_KEY, "log_prob"),
        cache_dist=True,
    )

    return policy

def create_value(model_config):
    # Remove out_features from config
    model_config = model_config.copy()
    model_config["out_features"] = 1

    model = MLP(**model_config)
    value = TensorDictModule(model, in_keys=[(config.ROOT_KEY, config.OBSERVATION_KEY)], out_keys=[(config.ROOT_KEY, "state_value")])
    return value