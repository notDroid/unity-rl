from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from torchrl.modules import ActorValueOperator, ValueOperator, ActorCriticWrapper
from tensordict.nn import TensorDictModule

from torch import nn

from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, OneHotCategorical
from tensordict.nn.distributions import NormalParamExtractor
from rlkit.modules import ActorValueInverseOperator
from rlkit.models.utils import CatWrapper
todict = lambda x: OmegaConf.to_container(x, resolve=True)

class ActorCriticBuilder:
    def build(self, config: DictConfig):
        return self._make_agent(config)

    def _make_agent(self, config: DictConfig):
            # Determine keys
            obs_keys = config.agent.get('obs_keys') or config.env.observation.get('observation_keys') or ["observation"]
            action_type = config.agent.get('action_type', config.env.action.type)
            hidden_keys = config.agent.get('hidden_keys', ['hidden_features'])  
            head_in_keys = config.agent.get('head_in_keys') or (obs_keys if not config.get('trunk_net') else ['hidden_features'])
            obs_keys = list(obs_keys); hidden_keys = list(hidden_keys); head_in_keys = list(head_in_keys)

            # 1. Make neural nets

            # (optional) duplicate head config
            if config.agent.get('head_config'):
                config.agent.policy_net = config.agent.head_config
                config.agent.value_net = config.agent.head_config
                if config.agent.get('include_inverse_head', None):
                    config.agent.inverse_net = config.agent.head_config
            # (optional) infer output dimensions
            if config.agent.get('head_config') or config.agent.get('infer_out_dim'):
                # **Depends on head having out_features defined**
                config.agent.policy_net.out_features = config.env.action.dim * (2 if action_type == "continuous" else 1)
                config.agent.value_net.out_features = 1
                if config.get('inverse_net', None):
                    config.agent.inverse_net.out_features = config.env.action.dim

            trunk_net = instantiate(config.agent.trunk_net) if config.agent.get('trunk_net') else None
            policy_net = instantiate(config.agent.policy_net)
            value_net = instantiate(config.agent.value_net)

            # Optional heads
            inverse_net = None
            if "inverse_net" in config.agent and config.agent.inverse_net:
                inverse_net = instantiate(config.agent.inverse_net)

            # 2. Build operators 
            if config.agent.get('trunk_module'):
                trunk_module = instantiate(
                    config.agent.trunk_module,
                    in_keys=obs_keys,
                    out_keys=hidden_keys,
                )
            elif not inverse_net and trunk_net:
                trunk_module = TensorDictModule(trunk_net, in_keys=obs_keys, out_keys=hidden_keys)
            policy_params = todict(config.agent.policy_params) if config.agent.get('policy_params') else {}
            policy_operator = self._build_policy(policy_net, action_type, head_in_keys, policy_params)
            value_operator = ValueOperator(value_net, in_keys=head_in_keys)

            # Optional heads
            inverse_operator = None
            if inverse_net: inverse_operator = self._build_inverse(inverse_net, head_in_keys, config)

            # 3. Assemble Together
            if trunk_net:
                if inverse_operator:
                    agent = ActorValueInverseOperator(
                        trunk_net=trunk_net,
                        policy_operator=policy_operator,
                        value_operator=value_operator,
                        inverse_operator=inverse_operator,
                        observation_keys=obs_keys,
                        hidden_keys=hidden_keys,
                    )
                else:
                    agent = ActorValueOperator(
                        common_operator=trunk_module,
                        policy_operator=policy_operator,
                        value_operator=value_operator,
                    )
            else:
                # Without Trunk
                agent = ActorCriticWrapper(
                    policy_operator=policy_operator,
                    value_operator=value_operator,
                )

            return agent

    def _build_policy(self, model, action_type, in_keys, policy_params):
        if action_type == "continuous":
            model = nn.Sequential(model, NormalParamExtractor())
            module = TensorDictModule(model, in_keys=in_keys, out_keys=["loc", "scale"])
            return ProbabilisticActor(
                module=module,
                in_keys=["loc", "scale"],
                # out_keys=["action"],
                distribution_class=TanhNormal,
                return_log_prob=True,
                log_prob_key="log_prob",
                cache_dist=True,
                **policy_params
            )
        elif action_type == "discrete":
            module = TensorDictModule(model, in_keys=in_keys, out_keys=["logits"])
            return ProbabilisticActor(
                module=module,
                in_keys=["logits"],
                # out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
                log_prob_key="log_prob",
                cache_dist=True,
                **policy_params
            )
        
    def _build_inverse(self, model, in_keys, config):
        # Inverse takes in [features_t, features_t+1] outputs action
        if config.agent.get('cat_inverse_inputs', True):
            model = CatWrapper(model)
        return TensorDictModule(
            model, 
            in_keys=in_keys +  [("next", in_key) for in_key in in_keys], 
            out_keys=config.agent.get('inverse_out_keys',["predict_action"])
        )