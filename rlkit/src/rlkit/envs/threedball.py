from rlkit.envs.transforms import append_batch_transform, InvertibleCatTensors, RenameAction, SoftResetWrapper, UnityRandomizerTransform
from torchrl.envs.transforms import ExcludeTransform, ObservationNorm, ClipTransform, RewardScaling

from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from .env_utils import *

REGISTERED_NAME = "3DBall"

def create_base_3dball_env(path=None, graphics=False, **kwargs):
    registered_name = None if path else REGISTERED_NAME
    env = create_unity_env(path=path, registered_name=registered_name, graphics=graphics, **kwargs)
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
    env = env.append_transform(
        ClipTransform(in_keys=["observation"], low=-5.0, high=5.0)
    )
    
    return env

def create_randomized_3dball_env(path=None, graphics=False, time_scale = 1, interval=100, verbose=False, **kwargs):
    env_param_channel = EnvironmentParametersChannel()
    env = make_env(create_base_3dball_env, path, graphics, time_scale, use_soft_reset=False, side_channels=[env_param_channel])
    
    # Randomized Param Ranges
    params = {
        "scale": (0.2, 5),
        "gravity": (4, 105),
        "mass": (0.1, 20),
    }
    
    env = env.append_transform(
        UnityRandomizerTransform(interval=interval, env_param_channel=env_param_channel, params=params, verbose=verbose)
    )

    env = SoftResetWrapper(env)

    return env