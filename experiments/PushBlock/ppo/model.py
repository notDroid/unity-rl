import config
from model_util import MLP
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torch.distributions import OneHotCategorical


def create_policy(model_config):
    model_config = model_config.copy()
    model = MLP(**model_config)
    logits_model = TensorDictModule(model, in_keys=[("agents", "observation")], out_keys=[("agents", "logits")])
    policy = ProbabilisticActor(
        module=logits_model,  
        distribution_class=OneHotCategorical,

        in_keys=[("agents", "logits")],
        out_keys=[("agents", config.ACTION_KEY)],

        return_log_prob=True,
        log_prob_key=("agents", "log_prob"),
        cache_dist=True,
    )

    return policy

def create_value(model_config):
    # Remove out_features from config
    model_config = model_config.copy()
    model_config["out_features"] = 1

    model = MLP(**model_config)
    value = TensorDictModule(model, in_keys=[("agents", "observation")], out_keys=[("agents", "state_value")])
    return value