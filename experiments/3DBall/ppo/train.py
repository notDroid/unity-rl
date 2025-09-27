# Torch
import torch
from torch import nn
from torch import optim

# TorchRL data
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from train_util import make_loss_module, loss_dict, compute_trajectory_metrics, Checkpointer, Logger, Stopwatch


'''
PPO Train Function

**DESCRIPTION**:
Does a simple PPO train loop:
    - Does checkpointing and logging, seamless continues. Can later increase timestamps, and other train loop params to extend training.
    - Does mixed precision training in torch.float16 on cuda device.
    - Supports multiple workers by multiprocessing during data collection.
        - Notably this will decrease the amount of timesteps per worker from generation_size to generation_size / workers.

**USAGE:**:
# Single train call
train(create_env, policy, value, train_config)

# Change config over train calls
for i in range(3):
    train_config["lr"] = lr_start * (10**(-i))
    train_config["timestamps"] =  timestamps_per_stage * (i + 1)
    train(create_env, policy, value, train_config, continue_training=(i!=0))


**EXPLANATION OF IMPLEMENTATION DECISIONS**:
I tried to make it modular so I can drop it into other projects, but not too modular as to increase complexity too much.
I decided to go with this approach:
DO NOT import config directly, instead info is passed through train_config.
imagine train_util functions as PARAMETERS to the train function. 
    - You can replace the implementation of make_loss_module, loss_dict, compute_trajectory_metrics, Checkpointer, Logger.
This makes it so that this PPO train function will work in different environments as long as you do the work of implementing train_util functions yourself.
The train_util functions should import environment config information, specifically make_loss_module, loss_dict, and compute_trajectory_metrics. 
The checkpointer and logger can also be changed, but its already portable enough to use in other projects without modifications.
'''

def train(create_env, policy, value, train_config, continue_training=True):
    ### LOAD CONFIG
    # Training Loop Params
    workers = train_config["workers"]
    device, storage_device = train_config["device"], train_config["storage_device"]
    timestamps, generation_size = train_config["timestamps"], train_config["generation_size"]
    epochs, minibatch_size = train_config["epochs"], train_config["minibatch_size"]
    # RL Params
    epsilon = train_config["epsilon"]
    gamma, gae_lambda = train_config["gamma"], train_config["gae_lambda"]
    entropy_coef = train_config["entropy_coef"]
    # Optimizer Params
    max_grad_norm = train_config["max_grad_norm"]
    lr = train_config["lr"]
    # Checkpoint and Log params
    checkpoint_interval, log_interval = train_config["checkpoint_interval"], train_config["log_interval"]
    ckpt_path, log_path, model_path = train_config["ckpt_path"], train_config["log_path"], train_config["model_path"]
    name = train_config["name"]
    log_keys, best_metric_key = train_config["log_keys"], train_config["best_metric_key"]

    # Mixed precision? + Move models to device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_dtype   = torch.float16 if device_type == "cuda" else torch.float32
    policy, value = policy.to(device), value.to(device)

    ### CREATE UTILILTY
    # Loss + Optimizer + Scaler
    loss_module = make_loss_module(policy, value, epsilon=epsilon, entropy_coef=entropy_coef, gamma=gamma, lmbda=gae_lambda)
    optimizer = optim.Adam(loss_module.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # Checkpointer + Logger
    checkpointer = Checkpointer(ckpt_path=ckpt_path, name=name)
    logger = Logger(keys=log_keys, log_path=log_path, name=name)
    
    # Reset/Continue
    start_generation = 0
    if not continue_training:
        checkpointer.reset()
        logger.reset()
    else:
        checkpoint = checkpointer.load_progress()
        if checkpoint:
            start_generation = int(checkpoint["generation"])
            policy.load_state_dict(checkpoint["policy_state_dict"])
            value.load_state_dict(checkpoint["value_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("CHECKPOINT FOUND. STARTING FROM GENERATION:", start_generation)
        else:
            print("CHECKPOINT NOT FOUND. STARTING FROM SCRATCH")

    # Create Collecter + Replay Buffer
    if workers > 1:
        collector = MultiSyncDataCollector(
            [create_env]*workers, policy, 
            frames_per_batch=generation_size, 
            total_frames=timestamps - start_generation*generation_size, 
            env_device="cpu", device=device, storing_device=storage_device, 
            update_at_each_batch=True
        )
    else:
        collector = SyncDataCollector(
            create_env, policy, 
            frames_per_batch=generation_size, 
            total_frames=timestamps - generation_size*start_generation, 
            env_device="cpu", device=device, storing_device=storage_device,
        )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(generation_size, device=storage_device), 
        sampler=SamplerWithoutReplacement(), 
        batch_size=minibatch_size,
    )

    # Watches
    short_watch = Stopwatch()
    long_watch = Stopwatch()

    ### TRAINING LOOP
    short_watch.start(); long_watch.start()
    policy.eval(); value.eval()

    # 0. Collect Trajectories
    for i, tensordict_data in enumerate(collector):
        logger.sum({"collection_time": short_watch.end()})

        # 1. Compute Advantages and Value Target and Collection Metrics
        tensordict_data = tensordict_data.to(device)
        with torch.no_grad():
            loss_module.value_estimator(tensordict_data)
            metrics = compute_trajectory_metrics(tensordict_data)
        logger.add(metrics)
            
        # 2. Minibatch Gradient Descent Loop
        short_watch.start()
        policy.train(); value.train()
        replay_buffer.empty(); replay_buffer.extend(tensordict_data.reshape(-1))
        for epoch in range(epochs):
            for j, batch in enumerate(replay_buffer):
                # 2.1 Optimization Step
                batch = batch.to(device)
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=(amp_dtype == torch.float16)):
                    loss_data = loss_module(batch)
                    loss = loss_data["loss_objective"] + loss_data["loss_critic"] + loss_data["loss_entropy"]
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=max_grad_norm)

                scaler.step(optimizer)
                scaler.update()

                # 2.2 Accumulate Train Metrics
                weight = float(batch.batch_size[0])
                logger.accumulate(loss_dict(loss_data, weight))          
        policy.eval(); value.eval()
        logger.sum({"train_time": short_watch.end()})

        # 3. Log Progress
        logger.sum({"generation": 1})
        gen = start_generation + i + 1
        if (log_interval > 0) and (gen % log_interval == 0):
            logger.sum({"time": long_watch.end()})
            long_watch.start()
            logger.next(print_row=True)

        # 4. Checkpoint Progress
        if (checkpoint_interval > 0) and (gen % checkpoint_interval == 0):
            metric = metrics[best_metric_key]
            checkpointer.save_progress(metric_key=best_metric_key,
            state_obj={
                "generation": gen,
                "policy_state_dict": policy.state_dict(),
                "value_state_dict": value.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                best_metric_key: metric,
            })

        # 0. Go Back to Collection Phase
        short_watch.start()
    checkpointer.copy_model('latest', model_path, ('policy_state_dict', 'value_state_dict'))