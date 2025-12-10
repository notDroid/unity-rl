import numpy as np
from torchrl.envs.libs import UnityMLAgentsEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from rlkit.envs.transforms import SoftResetWrapper


PORTS = 10_000
ATTEMPTS = 3

def _create_unity_env(path=None, registered_name=None, graphics=False, **kwargs):
        from mlagents_envs import environment # Force load for multiproccessing
        env = UnityMLAgentsEnv(
            file_name=path, 
            registered_name=registered_name,
            worker_id=np.random.randint(PORTS), 
            no_graphics=(not graphics), **kwargs,
            device="cpu",
        )
        return env

def create_unity_env(path=None, registered_name=None, graphics=False, **kwargs):
    # try: env.close()
    # except: pass    

    if path and registered_name: print("Cannot have both path and registered_name")
    if not path and not registered_name: print("Must have at least one of path or registered_name")
    
    # Gambling
    for i in range(ATTEMPTS):
        try:
            env = _create_unity_env(path, registered_name, graphics, **kwargs)
            break
        except: pass
        if i + 1 == ATTEMPTS: raise RuntimeError("Failed to load environment, call _create_unity_env() directly to find issue")

    return env

def make_env(create_base_env, path=None, graphics=False, time_scale=1.0, use_soft_reset=True, **kwargs):
    # Time Scale
    side_channels = kwargs.get('side_channels', [])
    if time_scale != 1.0:
        engine_config = EngineConfigurationChannel()
        engine_config.set_configuration_parameters(
            time_scale=time_scale,
            target_frame_rate=30 * time_scale if graphics else None,
            capture_frame_rate=30 * time_scale if graphics else None
        )
        side_channels.append(engine_config)
    kwargs['side_channels'] = side_channels

    # Make
    env = create_base_env(path, graphics, **kwargs)

    # Soft Reset
    if use_soft_reset:
        env = SoftResetWrapper(env)

    return env