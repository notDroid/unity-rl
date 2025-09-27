import numpy as np
from torchrl.envs.libs import UnityMLAgentsEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torchrl.envs import TransformedEnv, Stack, ExcludeTransform
from config import ENV_PATH


### The Primitive Unity Environment

def create_unity_env(graphics=False, **kwargs):
    env = TransformedEnv(UnityMLAgentsEnv(
        file_name=ENV_PATH, worker_id=np.random.randint(10000), no_graphics=(not graphics), 
        device="cpu",
        **kwargs,
    ))

    return env

# Batch agent keys into a shared agent dimension
def batch_agents(env, out_key="agents"):
    agent_root_key = env.observation_keys[0][0]
    agents = list(env.action_spec[agent_root_key].keys())
    
    # Create transform
    stack = Stack(
        in_keys=[(agent_root_key, agent) for agent in agents], 
        out_key=(out_key,), 
        in_key_inv=(out_key,), 
        out_keys_inv=[(agent_root_key, agent) for agent in agents]
    )

    env.append_transform(stack)
    return env


### Minimum Usable Version of Unity Environment

def create_base_env(graphics=False, **kwargs):
    env = create_unity_env(graphics, **kwargs)
    env = batch_agents(env)
    return env


### Practical Version of Unity Environment (For training and inference)

def create_env(graphics=False, time_scale=1, **kwargs):
    # Time scale
    if time_scale != 1:
        engine_config_channel = EngineConfigurationChannel()
        env = create_base_env(graphics, **kwargs, side_channels=[engine_config_channel])
        engine_config_channel.set_configuration_parameters(time_scale=time_scale)
    else:
        env = create_base_env(graphics, **kwargs)

    # Exclude group reward
    env.append_transform(
        ExcludeTransform(("agents", "group_reward"))
    )
    return env
