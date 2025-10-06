from tensordict import TensorDict
from torch import tensor
from torch.distributions import Categorical, Normal, Independent


def ppo_loss_td_to_dict(loss_data, weight):
    # Hard coded keys, values
    keys = ["value_loss", "explained_variance", "policy_loss", "kl_approx", "clip_fraction", "ESS"]
    values = ["loss_critic", "explained_variance", "loss_objective", "kl_approx", "clip_fraction", "ESS"]

    return {
        key: (loss_data[value].detach().mean().item(), weight) for key, value in zip(keys, values)
    }

def reduce_dim(data: tensor, dim: int):
    dims = len(data.shape)
    if dims > dim: data = data.squeeze(dim=list(range(dim, dims)))
    return data

def compute_categorical_entropy(traj_data, logits_key="logits", mask_key="mask"):
    # Expected shape: [Tr, T, A]
    logits = traj_data[logits_key]
    mask = traj_data[mask_key]

    # Account for extra dims
    logits = reduce_dim(logits, dim=3)
    mask = reduce_dim(mask, dim=2)

    # Compute entropy along batch dim
    entropy = Categorical(logits=logits.reshape(-1, logits.shape[-1])).entropy().reshape(mask.shape)
    # Masked entropy
    entropy = entropy.nan_to_num_(0, 0, 0) * mask
    # Average per trajectories
    entropy = entropy.sum(dim=-1) / mask.sum(dim=-1)
    return entropy.mean().cpu().item()

def compute_normal_entropy(traj_data, loc_key="loc", scale_key="scale", mask_key="mask"):
    # Expected shape: [Tr, T, A]
    loc, scale = traj_data[loc_key], traj_data[scale_key]
    mask = traj_data[mask_key]

    # Account for extra dims
    loc, scale = reduce_dim(loc, dim=3), reduce_dim(scale, dim=3)
    mask = reduce_dim(mask, dim=2)
    action_dim = loc.shape[-1]

    # Entropy
    entropy = Independent(Normal(
        loc.reshape(-1, action_dim), scale.reshape(-1, action_dim)
    ), 1).entropy().reshape(mask.shape)
    # Masked entropy
    entropy = entropy.nan_to_num_(0, 0, 0) * mask
    # Average per trajectories
    entropy = entropy.sum(dim=-1) / mask.sum(dim=-1)
    return entropy.mean().cpu().item()

def compute_approx_entropy(traj_data, log_prob_key="log_prob", mask_key="mask"):
    # Expected shape: [Tr, T]
    log_prob = traj_data[log_prob_key]
    mask = traj_data[mask_key]

    # Account for extra dims
    log_prob = reduce_dim(log_prob, dim=2)
    mask = reduce_dim(mask, dim=2)

    # Masked entropy
    entropy = (- log_prob * mask).sum(dim=-1) / mask.sum(dim=-1)
    return entropy.mean()

    

# Computes average return and episode length
def compute_traj_reward_info(traj_data: TensorDict, reward_key="reward", mask_key="mask"):
    # Expected shape: [Tr, T]
    reward = traj_data["next", reward_key]
    mask = traj_data[mask_key].to(reward.dtype)

    # Account for extra dims
    reward = reduce_dim(reward, dim=2)
    mask = reduce_dim(mask, dim=2)

    # Avg Return: Per trajectory reward sum -> mean return
    # Avg Reward: Sum of valid steps -> mean num timesteps
    return (reward * mask).sum(dim=-1).mean().cpu().item(), mask.sum(dim=-1).mean().cpu().item()