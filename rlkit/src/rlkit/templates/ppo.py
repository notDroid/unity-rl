# Torch
import torch
from torch import nn

# TorchRL
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data import ReplayBuffer, LazyMemmapStorage, SliceSamplerWithoutReplacement, SamplerWithoutReplacement

# Util
import numpy as np
from operator import itemgetter as get
from .ppo_config import PPOConfig
from rlkit.utils import Stopwatch, Checkpointer, LoggerBase, SimpleMetricModule
from dataclasses import dataclass
from typing import Optional

### PPO State to Save

@dataclass
class PPOState:
    policy: ProbabilisticActor
    value: TensorDictModule
    optimizer: Optimizer
    loss_module: ClipPPOLoss

    checkpointer: Optional[Checkpointer] = None
    logger: Optional[LoggerBase] = None
    scaler: Optional[torch.amp.GradScaler] = None
    metric_module: Optional[SimpleMetricModule] = None
    lr_scheduler: Optional[LRScheduler] = None

### Stateless Helpers

class PPOCollectorModule:
    def __init__(self, create_env: callable, ppo_config: PPOConfig, ppo_state: PPOState):
        self.create_env = create_env
        self.config = ppo_config
        self.state = ppo_state

        # -------------- Collector Utility --------------
        if self.config.workers > 1:
            self.collector = MultiSyncDataCollector(
                [self.create_env]*self.config.workers, self.state.policy, 
                frames_per_batch=self.config.collector_buffer_size, 
                total_frames=self.config.timesteps, 
                env_device="cpu", device=self.config.device, storing_device=self.config.storage_device, 
                reset_at_each_iter=False,
                update_at_each_batch=False, # Manually update policy weights
            )
        else:
            self.collector = SyncDataCollector(
                self.create_env, self.state.policy, 
                frames_per_batch=self.config.collector_buffer_size, 
                total_frames=self.config.timesteps, 
                env_device="cpu", device=self.config.device, storing_device=self.config.storage_device,
                reset_at_each_iter=False,
            )

        self.buffer = ReplayBuffer(
            storage=LazyMemmapStorage(
                self.config.generation_size, 
                device=self.config.storage_device, 
                ndim=self.config.env_batch_dim + 1 + int(self.config.workers > 1)
            ),
            sampler=SliceSamplerWithoutReplacement(
                slice_len = self.config.slice_len,
                shuffle=False, strict_length=False, 
                end_key=("next", "done"), # Maybe let this be provided
            ),
            batch_size=self.config.n_slices * self.config.slice_len,
        )

    def step(self):
        if self.config.workers > 1: self.collector.update_policy_weights_() # Update weights manually
        self.state.policy.eval(); self.state.value.eval()
        self.buffer.empty()

        # Buffer in memory then move to memory mapped storage in loop
        for j in range(self.config.collector_iters_per_gen):
            data = self.collector.next()
            self.buffer.extend(data)

class PPOTrainModule:
    def __init__(self, ppo_config: PPOConfig, ppo_state: PPOState):
        self.config = ppo_config
        self.state = ppo_state

        self.buffer = ReplayBuffer(
            storage=LazyMemmapStorage(self.config.generation_size, device=self.config.storage_device), 
            sampler=SamplerWithoutReplacement(), 
            batch_size=self.config.minibatch_size
        )

        # Fill state default values
        if self.state.scaler is None:
            self.state.scaler = torch.amp.GradScaler(enabled=(self.config.amp_dtype == torch.float16))

    def step(self, batch, epoch, j):
        # -------------- a. Compute Loss --------------
        with torch.autocast(device_type=self.config.device_type, dtype=self.config.amp_dtype, enabled=(self.config.amp_dtype==torch.float16)):
            loss_data = self.state.loss_module(batch)
            loss = loss_data["loss_objective"].mean() + loss_data["loss_critic"].mean() + loss_data["loss_entropy"].mean()

        # -------------- b. KL Safety Check --------------
        kl_approx = loss_data["kl_approx"].mean().cpu().item()
        if self.config.kl_soft_clip is not None and kl_approx > self.config.kl_soft_clip:
            self.early_stop += 1
            if self.early_stop >= self.config.early_stop_threshold: 
                print(f"Early stopped at ({epoch}, {j})")
                return "stop"
        if self.config.kl_hard_clip is not None and kl_approx > self.config.kl_hard_clip:
            print(f"Skipping iteration ({epoch}, {j}) with kl > kl_hard_clip: {kl_approx} > {self.config.kl_hard_clip}")
            return "skip"
        
        # -------------- c. Optimization Step --------------
        self.state.optimizer.zero_grad(set_to_none=True)
        self.state.scaler.scale(loss).backward()

        self.state.scaler.unscale_(self.state.optimizer)
        nn.utils.clip_grad_norm_(self.state.loss_module.parameters(), max_norm=self.config.max_grad_norm)

        self.state.scaler.step(self.state.optimizer)
        self.state.scaler.update()

        # -------------- d. Metric Update --------------
        weight = float(batch.batch_size[0])
        metrics = self.ppo_loss_td_to_dict(loss_data, weight)
        return metrics
    
    @staticmethod
    def ppo_loss_td_to_dict(loss_data, weight):
        # Hard coded keys, values
        keys = ["value_loss", "explained_variance", "policy_loss", "kl_approx", "clip_fraction", "ESS"]
        values = ["loss_critic", "explained_variance", "loss_objective", "kl_approx", "clip_fraction", "ESS"]

        return {
            key: (loss_data[value].detach().mean().item(), weight) for key, value in zip(keys, values)
        }
    
    def train(self):
        self.state.policy.train(); self.state.value.train()
        self.early_stop = 0

        for epoch in range(self.config.epochs):
            for j, batch in enumerate(self.buffer):
                batch = batch.to(self.config.device)
                metrics = self.step(batch, epoch, j)

                if metrics == "stop": 
                    self.buffer.empty()
                    return
                if metrics == "skip": continue

                if self.state.logger: self.state.logger.acc(metrics, mode='ema')
        self.buffer.empty()
    
class PPOAdvantageModule:
    def __init__(self, ppo_config: PPOConfig, ppo_state: PPOState, collect_module: PPOCollectorModule, train_module: PPOTrainModule):
        self.config = ppo_config
        self.state = ppo_state
        self.collect_module = collect_module
        self.train_module = train_module

        # Fill state default
        if not self.state.metric_module: self.state.metric_module = SimpleMetricModule(mode="approx")

    def step(self):    
        self.state.policy.eval(); self.state.value.eval()
        self.train_module.buffer.empty()
        self.metrics = dict()

        for j, batch in enumerate(self.collect_module.buffer):
            batch = batch.to(self.config.device)
        
            with torch.no_grad():
                self.state.loss_module.value_estimator(batch)
                metrics = self.state.metric_module(batch)
            batch["advantage"].clamp_(-self.config.adv_clip, self.config.adv_clip)
            
            if self.state.logger: self.state.logger.acc(metrics, mode='avg')
            self.accumulate_metrics(metrics)
            self.train_module.buffer.extend(batch.reshape(-1).cpu())
        self.collect_module.buffer.empty()
        return self.get_metrics()

    def accumulate_metrics(self, next_metrics):
        for (key, value) in next_metrics.items():
            if key not in self.metrics: 
                self.metrics[key] = {"value": value, "n": 1}
            else:
                self.metrics[key]["value"] += value
                self.metrics[key]["n"] += 1
        
    def get_metrics(self):
        final_metrics = dict()
        for (key, metric) in self.metrics.items():
            final_metrics[key] = metric["value"] / metric["n"]
        return final_metrics

class PPOBasic:
    def __init__(self, create_env: callable, ppo_config: PPOConfig, ppo_state: PPOState):
        self.create_env = create_env
        self.config = ppo_config
        self.state = ppo_state

        # Stateless Helper Modules
        self.collect_module = PPOCollectorModule(create_env, ppo_config, ppo_state)
        self.train_module = PPOTrainModule(ppo_config, ppo_state)
        self.adv_module = PPOAdvantageModule(ppo_config, ppo_state, self.collect_module, self.train_module)

        # Utility
        self.short_watch = Stopwatch()
        self.long_watch = Stopwatch()

    def step(self, gen):
        self.long_watch.start()

        # 1. Collect Trajectory Dataset
        self.short_watch.start()
        self.collect_module.step()
        if self.state.logger: 
            self.state.logger.add({"collection_time": self.short_watch.end()})

        # 2. Compute Advantage + Value Targets
        metrics = self.adv_module.step()

        # 3. Train Loop
        self.short_watch.start()
        self.train_module.train()
        if self.state.logger: 
            self.state.logger.add({"train_time": self.short_watch.end()})

        # 4. Log and Checkpoint
        if self.state.logger:
            self.state.logger.add({"time": self.long_watch.end()})
            self._log_step()
        if self.state.checkpointer and (gen % self.config.checkpoint_interval) == 0:
            self._ckpt_step(gen, metrics)

        return metrics

    def _ckpt_step(self, gen, metrics):
        state_obj = {
            "generation": gen + 1,
            "policy_state_dict": self.state.policy.state_dict(),
            "value_state_dict": self.state.value.state_dict(),
            "optimizer_state_dict": self.state.optimizer.state_dict(),
            "scaler_state_dict": self.state.scaler.state_dict(),
        }
        if self.config.best_metric_key:
            state_obj[self.config.best_metric_key] =  metrics[self.config.best_metric_key]
        
        self.state.checkpointer.save_progress(state_obj)

    def _log_step(self):
        if self.state.logger: self.state.logger.add({"generation": 1})
        if self.state.logger: self.state.logger.add({"timestep": self.config.generation_size})

        if self.state.lr_scheduler:
            lr = self.state.lr_scheduler.get_last_lr()[0]
            self.state.logger.acc({"lr": lr})
            self.state.lr_scheduler.step()

        self.state.logger.next(print_row=True)

    def close(self, save=False):
        try: self.collect_module.collector.shutdown()
        except: pass
        if save and self.state.checkpointer: 
            self.state.checkpointer.copy_model(self.config.model_path, 'latest', ('policy_state_dict', 'value_state_dict'))

    def model(self): 
        return self.state.policy, self.state.value
    
    def history(self):
        return self.state.logger.dataframe()
    
    def run(self):
        for gen in range(self.config.start_generation, self.config.generations):
            self.step(gen)
        self.close(save=True)