from logging import config
from omegaconf import DictConfig, OmegaConf
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from rlkit.modules import ClipPPOInverseLoss
from hydra.utils import get_class
todict = lambda x: OmegaConf.to_container(x, resolve=True)

class PPOLossBuilder:
    DEFAULT_LOSS_PARAMS = {
        "epsilon": 0.2,
        "entropy_coeff": 0.01,
        "critic_coeff": 1.0,
    }
    DEFAULT_VALUE_ESTIMATOR_PARAMS = {
        "gamma": 0.99,
        "lmbda": 0.95,
    }

    def build(self, config: DictConfig, agent):
        return self._make_ppo_loss_module(config, agent)

    def _make_ppo_loss_module(self, config, agent):
        # Assemble Parameters
        loss_params = todict(config.loss.loss_params)
        loss_params = {**self.DEFAULT_LOSS_PARAMS, **loss_params}
        if 'include_inverse' in config.loss: include_inverse = config.loss.include_inverse
        else: include_inverse = hasattr(agent, "get_inverse_operator")
        
        # Loss Module
        loss_cls = get_class(config.loss.loss)
        loss_module = loss_cls(
            actor_network=agent.get_policy_operator(),
            critic_network=agent.get_value_operator(),
            **({"inverse_network": agent.get_inverse_operator() } if include_inverse else {}),
            **loss_params
        )

        # Value Estimator
        value_estimator_params = todict(config.loss.value_estimator_params) if 'value_estimator_params' in config.loss else {}
        value_estimator_params = {**self.DEFAULT_VALUE_ESTIMATOR_PARAMS, **value_estimator_params}
        if 'value_estimator' in config.loss:
            value_estimator_cls = get_class(config.loss.value_estimator)
            loss_module.make_value_estimator(value_estimator_cls)
        else:
            # Default to GAE
            loss_module.make_value_estimator(
                ValueEstimators.GAE, 
                shifted=True, average_gae=True,
                **value_estimator_params,
            )
        return loss_module