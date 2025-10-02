import os
import pandas as pd
import matplotlib.pyplot as plt
from train_util import Logger, compute_trajectory_metrics, compute_single_trajectory_metrics

import torch
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
import config

# Plot dataframe
def plot(name):
    # Get dataframe
    logger = Logger(keys = config.LOG_KEYS, log_path=config.LOG_PATH, name=name)
    df = logger.dataframe()
    df = df.set_index(config.LOG_INDEX)

    # Plot it
    rows = (len(df.columns) + 3) // 4
    df.plot(subplots=True, layout=(rows,4), figsize=(15, int(rows * 7/3)))
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, f"{name}.png"))

# Play single game
def play(create_env, policy, timestamps):
    env = create_env()

    with torch.no_grad():
        tensordict_data = env.rollout(timestamps, policy=policy, auto_cast_to_device=True, break_when_any_done=False).to(policy.device)
        env.close()
        metrics = compute_single_trajectory_metrics(tensordict_data)
    
    return metrics

def test(create_env, policy, timestamps, workers=1):
    device = policy.device

    # Create Collecter + Replay Buffer
    if workers > 1:
        collector = MultiSyncDataCollector(
            [create_env]*workers, policy, 
            frames_per_batch=timestamps, 
            total_frames=timestamps, 
            env_device="cpu", device=device, storing_device=device, 
            update_at_each_batch=True
        )
    else:
        collector = SyncDataCollector(
            create_env, policy, 
            frames_per_batch=timestamps, 
            total_frames=timestamps, 
            env_device="cpu", device=device, storing_device=device,
        )

    data = None
    policy.eval()
    for i, tensordict_data in enumerate(collector):
        data = tensordict_data
        break
    try:
        collector.shutdown()
    except:
        pass

    data = data.to(device)
    with torch.no_grad():
        metrics = compute_trajectory_metrics(data)
    return metrics