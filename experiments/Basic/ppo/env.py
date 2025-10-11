# Util and Machine Learning
import numpy as np

# Environment
from torchrl.envs import EnvBase
from torchrl.envs.libs import UnityMLAgentsEnv
from torchrl.envs.transforms import ExcludeTransform, RenameTransform
from rlkit.transforms import append_batch_transform, InvertibleCatTensors, RenameAction, SoftResetWrapper, UnnestTransform
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from config import ENV_PATH

def _create_unity_env(graphics, **kwargs):
        from mlagents_envs import environment # Force load for multiproccessing
        env = UnityMLAgentsEnv(
            # file_name=ENV_PATH, 
            registered_name="Basic",
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

    keys = [*env.observation_keys, *env.action_keys, *env.reward_keys, *env.done_keys]
    for i, key in enumerate(keys):
        keys[i] = key[-1]

    env = env.append_transform(
        UnnestTransform(('group_0', 'agent_0'), keys),
    )
    env = env.append_transform(
        UnnestTransform(('group_0'), []),
    )

    env = env.append_transform(
        RenameTransform(["Basic"], ["observation"])
    )
    env = env.append_transform(
        ExcludeTransform("group_reward")
    )
    env = env.append_transform(
        RenameAction(env.action_key, "action")
    )

    return env

class HardResetWrapper(EnvBase):
    """Recreates the environment on every reset call"""
    def __init__(self, create_env):
        env = create_env()
        super().__init__(device=env.device, batch_size=env.batch_size)
        self.env = env
        self.create_env = create_env
        self._passthrough_specs()

    def _reset(self, tensordict=None, **kwargs):
        self.env.close()
        self.env = self.create_env()
        return self.env._reset(tensordict, **kwargs)
    
    # Passthrough
    def _step(self, tensordict): return self.env._step(tensordict)
    def _set_seed(self, *args, **kwargs): return self.env.set_seed(*args, **kwargs)
    def _passthrough_specs(self):
        self.observation_spec = self.env.observation_spec
        self.action_spec = self.env.action_spec
        self.reward_spec = self.env.reward_spec
        self.done_spec = self.env.done_spec

def create_env(graphics=False, time_scale = 1, **kwargs):
    def env_fn():
        engine_config_channel = EngineConfigurationChannel()
        env = create_base_env(graphics, **kwargs, side_channels=[engine_config_channel])
        if time_scale != 1:
            engine_config_channel.set_configuration_parameters(
                time_scale = time_scale,
                target_frame_rate = 30*time_scale if graphics else None,
                capture_frame_rate = 30*time_scale if graphics else None,
            )
        return env

    env = HardResetWrapper(env_fn)

    return env
