from .crawler import create_base_crawler_env
from .pushblock import create_base_pushblock_env
from .threedball import create_base_3dball_env
from .walker import create_base_walker_env
from .walljump import create_base_walljump_env
from .worm import create_base_worm_env
from.gridworld import create_base_gridworld_env
from .env_utils import make_env

env_fn_map = {
    "Crawler": create_base_crawler_env,
    "PushBlock": create_base_pushblock_env,
    "3DBall": create_base_3dball_env,
    "Walker": create_base_walker_env,
    "WallJump": create_base_walljump_env,
    "Worm": create_base_worm_env,
    "GridWorld": create_base_gridworld_env,
}

registered_env_list = list(env_fn_map.keys())

"""
Top-level factory function for creating Unity Environments.

Args:
    name: The key to look up the environment function in env_fn_map (e.g. "Crawler": create_base_crawler_env).
    path: Path to the Unity executable.
    graphics: Whether to render graphics.
    time_scale: Speed of the simulation.
    **kwargs: Additional arguments passed to the unity environment (passed to UnityMLAgentsEnv).
    
Returns:
    An instantiated TorchRL environment.
"""
def UnityEnv(name, path=None, graphics=False, time_scale=1.0, **kwargs):
    if name not in env_fn_map: 
        print(f"Invalid Environment \"{name}\". Expected one of: {registered_env_list}")

    env_fn = env_fn_map[name]
    env = make_env(env_fn, path, graphics, time_scale, use_soft_reset=True, **kwargs)
    return env
