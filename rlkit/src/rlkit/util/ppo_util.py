def ppo_loss_td_to_dict(loss_data, weight):
    # Hard coded keys, values
    keys = ["value_loss", "explained_variance", "policy_loss", "kl_approx", "clip_fraction", "ESS"]
    values = ["loss_critic", "explained_variance", "loss_objective", "kl_approx", "clip_fraction", "ESS"]

    return {
        key: (loss_data[value].detach().mean().item(), weight) for key, value in zip(keys, values)
    }
