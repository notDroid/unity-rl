from torchrl.envs.transforms import ExcludeTransform
from rlkit.envs.transforms import append_batch_transform, InvertibleCatTensors, RenameAction, SoftResetWrapper, FlattenMultiOneHot
from torchrl.envs.transforms import ExcludeTransform, ObservationNorm, ClipTransform, RewardScaling, RenameTransform

from .env_utils import create_unity_env

REGISTERED_NAME = "Worm"

def create_base_worm_env(path=None, graphics=False, **kwargs):
    registered_name = None if path else REGISTERED_NAME
    env = create_unity_env(path, registered_name, graphics, **kwargs)
    env = append_batch_transform(env)

    env = env.append_transform(
        RenameTransform(in_keys=env.observation_keys, out_keys=["observation"])
    )
    env = env.append_transform(
        ExcludeTransform("group_reward")
    )
    env = env.append_transform(
        RenameAction(env.action_key, "action")
    )
    env = env.append_transform(
        ClipTransform(in_keys=["observation"], low=-3.0, high=3.0)
    )
    env = env.append_transform(
        RewardScaling(loc=0.0, scale=10.0, standard_normal=True),
    )
    return env