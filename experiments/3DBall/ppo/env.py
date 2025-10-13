# Util and Machine Learning
import numpy as np

# Environment
from torchrl.envs.libs import UnityMLAgentsEnv
from torchrl.envs.transforms import ExcludeTransform, ObservationNorm, ClipTransform, RewardScaling
from rlkit.transforms import append_batch_transform, InvertibleCatTensors, RenameAction, SoftResetWrapper, UnityRandomizerTransform
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from config import ENV_PATH

def _create_unity_env(graphics, **kwargs):
        from mlagents_envs import environment # Force load for multiproccessing
        env = UnityMLAgentsEnv(
            file_name=ENV_PATH, 
            # registered_name="3DBall",
            worker_id=np.random.randint(10000), 
            no_graphics=(not graphics), **kwargs,
            device="cpu",
        )
        return env

def create_unity_env(graphics=False, **kwargs):
    try: env.close()
    except: pass    
    
    # Gambling
    ATTEMPTS = 3
    for i in range(ATTEMPTS):
        try:
            env = _create_unity_env(graphics, **kwargs)
            break
        except: pass
        if i + 1 == ATTEMPTS: raise RuntimeError("Failed to load environment")

    return env

def create_base_env(graphics=False, **kwargs):
    env = create_unity_env(graphics, **kwargs)
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

def create_env(graphics=False, time_scale = 1, **kwargs):
    engine_config_channel = EngineConfigurationChannel()
    env = create_base_env(graphics, **kwargs, side_channels=[engine_config_channel])
    if time_scale != 1:
        engine_config_channel.set_configuration_parameters(
            time_scale = time_scale,
            target_frame_rate = 30*time_scale if graphics else None,
            capture_frame_rate = 30*time_scale if graphics else None,
        )

    env = SoftResetWrapper(env)

    return env
    

def create_randomized_env(graphics=False, time_scale = 1, interval=100, verbose=False, **kwargs):
    env_param_channel = EnvironmentParametersChannel()
    engine_config_channel = EngineConfigurationChannel()
    env = create_base_env(graphics=graphics, **kwargs, side_channels=[env_param_channel, engine_config_channel])

    # Time scale
    if time_scale != 1:
        engine_config_channel.set_configuration_parameters(
            time_scale = time_scale,
            target_frame_rate = 30*time_scale if graphics else None,
            capture_frame_rate = 30*time_scale if graphics else None,
        )
    
    # Randomized
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