from .crawler import create_base_crawler_env
from .pushblock import create_base_pushblock_env
from .threedball import create_base_3dball_env
from .walker import create_base_walker_env
from .walljump import create_base_walljump_env
from .worm import create_base_worm_env
from .env_utils import make_env

env_fn_map = {
    "Crawler": create_base_crawler_env,
    "PushBlock": create_base_pushblock_env,
    "3DBall": create_base_3dball_env,
    "Walker": create_base_walker_env,
    "WallJump": create_base_walljump_env,
    "Worm": create_base_worm_env,
}

registered_env_list = list(env_fn_map.keys())

def UnityEnv(name, path=None, graphics=False, time_scale=1.0, **kwargs):
    if name not in env_fn_map: 
        print(f"Invalid Environment \"{name}\". Expected one of: {registered_env_list}")

    env_fn = env_fn_map[name]
    env = make_env(env_fn, path, graphics, time_scale, use_soft_reset=True, **kwargs)
    return env
