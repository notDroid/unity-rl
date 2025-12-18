from torchrl.collectors.utils import split_trajectories
from torch.distributions import Categorical, Normal, Independent

def reduce_dim(data, dim):
    dims = len(data.shape)
    if dims > dim: data = data.squeeze(dim=list(range(dim, dims)))
    return data

def categorical_entropy(data, logits_key="logits", **kwargs):
    logits = data[logits_key]
    action_dim = logits.shape[-1]

    # Compute entropy along batch dim
    entropy = Categorical(logits=logits.reshape(-1, action_dim)).entropy()
    return entropy.mean().cpu().item()

def normal_entropy(data, loc_key="loc", scale_key="scale", **kwargs):
    loc, scale = data[loc_key], data[scale_key]
    action_dim = loc.shape[-1]

    # Entropy
    entropy = Independent(Normal(
        loc.reshape(-1, action_dim), scale.reshape(-1, action_dim)
    ), 1).entropy()
    return entropy.mean().cpu().item()

def approx_entropy(data, log_prob_key="log_prob", **kwargs):
    return -data[log_prob_key].mean().cpu().item()

def reward_info(traj_data, reward_key="reward", mask_key="mask", **kwargs):
    # Expected shape: [Tr, T]
    reward = traj_data["next", reward_key]
    mask = traj_data[mask_key].to(reward.dtype)

    # Account for extra dims
    reward = reduce_dim(reward, dim=2)
    mask = reduce_dim(mask, dim=2)

    # Avg Return: Per trajectory reward sum -> mean return
    # Avg Reward: Sum of valid steps -> mean num timesteps
    return (reward * mask).sum(dim=-1).mean().cpu().item(), mask.sum(dim=-1).mean().cpu().item()

class SimpleMetricModule:
    mode_map = {
        "categorical": categorical_entropy,
        "approx": approx_entropy,
        "normal": normal_entropy,
    }

    # Set any keys in kwargs
    def __init__(self, mode, **kwargs):
        self.keys = kwargs
        if mode not in self.mode_map: raise KeyError(f"Invalid mode: \"{mode}\". Valid modes: {list(self.mode_map.keys())}")
        self.entropy_fn = self.mode_map[mode]

    def __call__(self, data, **kwargs):
        traj_data = split_trajectories(data, **kwargs)
        args = {
            "data": data,
            "traj_data": traj_data,
            **self.keys,
        }

        # Auto search for mask key if not provided
        if "mask_key" not in args:
            if "mask" in traj_data: args["mask_key"] = "mask"
            elif "collector" in traj_data: args["mask_key"] = ("collector", "mask")
            else: raise RuntimeError(f"Couldn't find mask key in: {traj_data}, set mask_key=...")
        
        return_, episode_length = reward_info(**args)
        return {
            "return": return_,
            "episode_length": episode_length,
            "entropy": self.entropy_fn(**args)
        }

