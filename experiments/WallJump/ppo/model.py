from model_util import MLP
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, OneHotCategorical

# import config


def create_policy(model_config):
    model_config = model_config.copy()
    model = MLP(**model_config)
    logits_model = TensorDictModule(model, in_keys=["observation"], out_keys=["logits"])
    policy = ProbabilisticActor(
        module=logits_model,  
        distribution_class=OneHotCategorical,

        in_keys=["logits"],
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