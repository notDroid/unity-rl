import torch
from torchrl.envs import Transform, EnvBase
from torchrl.envs.transforms.utils import _set_missing_tolerance
from torchrl.envs import Stack
from .unnest import UnnestTransform


class SetBatchDim(Transform):
    # Only works on a environment without batch size
    invertible = True
    
    def __init__(self, batch_size):
        super().__init__()
        self._batch_size = batch_size

    def _call(self, next_tensordict):
        next_tensordict.batch_size = self._batch_size
        return next_tensordict

    forward = _call

    def _reset(self, tensordict, tensordict_reset):
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset
    
    def _inv_call(self, in_tensordict):
        in_tensordict.batch_size = torch.Size([])
        return in_tensordict

    def transform_env_batch_size(self, batch_size):
        return self._batch_size
    
    def transform_observation_spec(self, observation_spec):
        observation_spec.shape = self._batch_size
        return observation_spec
    
    def transform_reward_spec(self, reward_spec):
        reward_spec.shape = self._batch_size
        return reward_spec
    
    def transform_done_spec(self, done_spec):
        done_spec.shape = self._batch_size
        return done_spec
    
    def transform_action_spec(self, action_spec):
        action_spec.shape = self._batch_size
        return action_spec

'''
For batched single agent environments.
'''
def append_batch_transform(env: EnvBase) -> EnvBase:
    agent_root_key = env.observation_keys[0][0]
    agents = list(env.action_spec[agent_root_key].keys())
    agent_list = [(agent_root_key, agent) for agent in agents]
    
    # 1. STACK AGENTS TOGETHER
    temp_key = "agents"
    # Stack
    stack = Stack(
        in_keys=agent_list, 
        out_key=(temp_key,), 
        in_key_inv=(temp_key,), 
        out_keys_inv=agent_list
    )
    # Batch dim
    batch = SetBatchDim(
        batch_size=torch.Size([len(agents)])
    )

    env = env.append_transform(stack).append_transform(batch)

    # 2. Unnest
    keys = [*env.observation_keys, *env.action_keys, *env.reward_keys, *env.done_keys]
    for i, key in enumerate(keys):
        keys[i] = key[1]
    unnest = UnnestTransform(temp_key, out_keys=keys)

    return env.append_transform(unnest)