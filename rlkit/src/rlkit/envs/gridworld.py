from torchrl.envs.transforms import ExcludeTransform
from rlkit.envs.transforms import append_batch_transform, InvertibleCatTensors, RenameAction, SoftResetWrapper, FlattenMultiOneHot
from torchrl.envs.transforms import ExcludeTransform, ObservationNorm, ClipTransform, RewardScaling, RenameTransform
from torchrl.data.tensor_specs import BoxList, OneHot, Composite
from torchrl.envs import Transform
import torch

from .env_utils import create_unity_env

REGISTERED_NAME = "GridWorld"

class FixGridWorldSpec(Transform):
    def __init__(self, action_key):
        super().__init__(in_keys_inv=[action_key], out_keys_inv=[action_key])

    def _inv_call(self, in_tensordict):
        action_key = self.in_keys_inv[0]
        action = in_tensordict.get(action_key)

        action = torch.nested.nested_tensor(list(action.unbind(0)))
        in_tensordict.set(action_key, action)
        return in_tensordict

    def transform_action_spec(self, action_spec):
        action_key = self.in_keys_inv[0]
        action_spec = action_spec[action_key]
        shape = action_spec.shape
        n = shape[1]
        action_spec = OneHot(n, shape=action_spec.shape, device=action_spec.device, dtype=action_spec.dtype)
        action_spec = Composite({action_key: action_spec})
        action_spec.batch_size = torch.Size([shape[0]])
        return action_spec

def create_base_gridworld_env(path=None, graphics=False, **kwargs):
    registered_name = None if path else REGISTERED_NAME
    env = create_unity_env(path, registered_name, graphics, **kwargs)
    env = append_batch_transform(env)
    env = env.append_transform(
        FixGridWorldSpec(env.action_key)
    )

    env = env.append_transform(
        RenameTransform(in_keys=["RenderTextureSensor", "VectorlSensor"], out_keys=["visual_observation", "indicator"])
    )
    env = env.append_transform(
        ExcludeTransform("group_reward")
    )
    env = env.append_transform(
        RenameAction(env.action_key, "action")
    )
    return env