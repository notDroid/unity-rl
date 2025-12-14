# Environment
import os
from rlkit.envs.env_utils import create_unity_env
import torch


# Utility
from torchrl.envs.transforms import ExcludeTransform, ObservationNorm, ClipTransform, RewardScaling
from rlkit.envs.transforms import append_batch_transform, InvertibleCatTensors, RenameAction, SoftResetWrapper
from torchrl.envs import Transform, EnvBase, check_env_specs
from torchrl.envs.transforms.utils import _set_missing_tolerance
from torchrl.envs import Stack
from rlkit.envs.transforms import UnnestTransform, SetBatchDim
from torchrl.data.tensor_specs import BoxList, OneHot, Composite, MultiOneHot

REGISTERED_NAME = "GridWorld"

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
    print(keys)
    unnest = UnnestTransform(temp_key, out_keys=keys)

    return env.append_transform(unnest)

# class FixGridWorld(Transform):
#     def __init__(self, action_key):
#         # Monitor the action key in both directions (input and output)
#         super().__init__(
#             in_keys=[action_key], 
#             out_keys=[action_key],
#             in_keys_inv=[action_key], 
#             out_keys_inv=[action_key]
#         )

#     def _inv_call(self, in_tensordict):
#         # Fixes action before sending to Unity
#         return self._fix_nested(in_tensordict, self.in_keys_inv[0])

#     def _call(self, out_tensordict):
#         # Fixes the "echoed" action returned by Unity before Rollout stacks it
#         return self._fix_nested(out_tensordict, self.in_keys[0])

class FixGridWorld(Transform):
    def __init__(self, action_key):
        super().__init__(in_keys_inv=[action_key], out_keys_inv=[action_key])

    def _inv_call(self, in_tensordict):
        # print(in_tensordict)
        # action_key = self.in_keys_inv[0]
        # action = in_tensordict.get(action_key)
        # action = torch.stack(action.unbind())
        # in_tensordict.set(action_key, action)
        # print(in_tensordict.get(action_key))
        return in_tensordict
    

    # def transform_action_spec(self, action_spec):
    #     if isinstance(action_spec, Composite):
    #         action_spec = action_spec[self.in_keys_inv[0]]
    #     n = action_spec.shape[1]
    #     action_spec = MultiOneHot(n, shape=action_spec.shape, device=action_spec.device, dtype=action_spec.dtype)
    #     return action_spec

def create_base_gridworld_env(path=None, graphics=False, **kwargs):
    registered_name = None if path else REGISTERED_NAME
    env = create_unity_env(path, registered_name, graphics, **kwargs)
    env = append_batch_transform(env)
    env = env.append_transform(
        FixGridWorld(env.action_key)
    )

    # env = env.append_transform(
    #     InvertibleCatTensors(in_keys=env.observation_keys, out_key="observation")
    # )
    # env = env.append_transform(
    #     ExcludeTransform("group_reward")
    # )
    return env

ENV_NAME = 'GridWorld'
ENV_PATH = os.path.join("built_envs", f"{ENV_NAME}.app")
env = create_base_gridworld_env(ENV_PATH, graphics=False)
data = env.rollout(1, break_when_any_done=False)
print("Success")