from torchrl.envs.transforms import ExcludeTransform, ObservationNorm, ClipTransform, RewardScaling
from rlkit.envs.transforms import append_batch_transform, InvertibleCatTensors, RenameAction, SoftResetWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from .env_utils import create_unity_env

REGISTERED_NAME = "Crawler"

def create_base_crawler_env(path=None, graphics=False, **kwargs):
    registered_name = None if path else REGISTERED_NAME
    env = create_unity_env(path, registered_name, graphics, **kwargs)
    env = append_batch_transform(env)

    # Found constants by inspection of observation mean and std (from 2000 steps of random policy)
    env = env.append_transform(
        ObservationNorm(loc=1.5, scale=5.0, standard_normal=True, in_keys=["VectorSensor_size32"])
    )
    env = env.append_transform(
        InvertibleCatTensors(in_keys=env.observation_keys, out_key="observation")
    )
    env = env.append_transform(
        ExcludeTransform("group_reward")
    )
    env = env.append_transform(
        RenameAction(env.action_key, "action")
    )
    env = env.append_transform(
        ClipTransform(in_keys=["observation"], low=-5.0, high=5.0)
    )
    env = env.append_transform(
        RewardScaling(loc=0.0, scale=10.0, standard_normal=True),
    )
    return env
