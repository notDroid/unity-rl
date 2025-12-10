from torchrl.envs.transforms import ExcludeTransform
from rlkit.envs.transforms import append_batch_transform, InvertibleCatTensors, RenameAction, SoftResetWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from .env_utils import create_unity_env

REGISTERED_NAME = "PushBlock"

def create_base_pushblock_env(path=None, graphics=False, **kwargs):
    registered_name = None if path else REGISTERED_NAME
    env = create_unity_env(path, registered_name, graphics, **kwargs)
    env = append_batch_transform(env)

    env = env.append_transform(
        InvertibleCatTensors(in_keys=env.observation_keys, out_key="observation")
    )
    env = env.append_transform(
        ExcludeTransform("group_reward")
    )
    env = env.append_transform(
        RenameAction(env.action_key, "action")
    )

    return env