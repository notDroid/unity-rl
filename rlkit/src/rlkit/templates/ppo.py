# Torch
import torch
from torch import nn

# TorchRL
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data import ReplayBuffer, LazyMemmapStorage, SliceSamplerWithoutReplacement, SamplerWithoutReplacement
from torchrl.modules import ActorValueOperator, ValueOperator, ActorCriticWrapper

# Util
from .ppo_config import PPOTrainConfig
from .ppo_state import PPOState
from rlkit.utils import Stopwatch, Checkpointer, LoggerBase, SimpleMetricModule, atomic_torch_save

### Stateless Helpers
class PPOCollectorModule:
    def __init__(self, create_env: callable, ppo_config: PPOTrainConfig, ppo_state: PPOState):
        self.create_env = create_env
        self.config = ppo_config
        self.state = ppo_state

        # -------------- Collector Utility --------------
        if self.config.workers > 1:
            self.collector = MultiSyncDataCollector(
                [self.create_env]*self.config.workers, self.state.model.get_policy_operator(), 
                frames_per_batch=self.config.collector_buffer_size, 
                total_frames=-1, 
                env_device="cpu", device=self.config.device, storing_device=self.config.storage_device, 
                reset_at_each_iter=False,
                update_at_each_batch=False, # Manually update policy weights
            )
        else:
            self.collector = SyncDataCollector(
                self.create_env, self.state.model.get_policy_operator(), 
                frames_per_batch=self.config.collector_buffer_size, 
                total_frames=-1, 
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
        self.state.model.eval()
        self.buffer.empty()

        # Buffer in memory then move to memory mapped storage in loop
        for j in range(self.config.collector_iters_per_gen):
            data = self.collector.next()
            self.buffer.extend(data)

class PPOTrainModule:
    def __init__(self, ppo_config: PPOTrainConfig, ppo_state: PPOState):
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

        self.trunk = None
        if ppo_config.trunk:
            self.trunk = self.state.model.module[0]

    def step(self, batch, epoch, j):
        # -------------- a. Compute Loss --------------
        with torch.autocast(device_type=self.config.device_type, dtype=self.config.amp_dtype, enabled=(self.config.amp_dtype!=torch.float32)):
            # Trunk forward pass if exists
            if self.trunk:
                batch = self.trunk(batch)

            # Head forward pass and loss computation
            loss_data = self.state.loss_module(batch)
            loss = sum(v.mean() for k, v in loss_data.items() if k.startswith("loss_"))

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
        keys = ["value_loss", "explained_variance", "policy_loss", "kl_approx", "clip_fraction", "ESS", "inverse_loss"]
        values = ["loss_critic", "explained_variance", "loss_objective", "kl_approx", "clip_fraction", "ESS", "loss_inverse_dynamics"]

        return {
            key: (loss_data[value].detach().mean().item(), weight) for key, value in zip(keys, values) if value in loss_data
        }
    
    def train(self):
        self.state.model.train()
        self.early_stop = 0

        for epoch in range(self.config.epochs):
            for j, batch in enumerate(self.buffer):
                batch = batch.to(self.config.device)
                metrics = self.step(batch, epoch, j)

                if metrics == "stop": 
                    self.buffer.empty()
                    self.update_lr()
                    return
                if metrics == "skip": continue

                if self.state.logger: self.state.logger.acc(metrics, mode='ema', beta=0.95)
        self.buffer.empty()
        self.update_lr()

    def update_lr(self):
        if self.state.lr_scheduler:
            lr = self.state.lr_scheduler.get_last_lr()[0]
            if self.state.logger: self.state.logger.acc({"lr": lr})
            self.state.lr_scheduler.step()
    
class PPOAdvantageModule:
    def __init__(self, ppo_config: PPOTrainConfig, ppo_state: PPOState, collect_module: PPOCollectorModule, train_module: PPOTrainModule):
        self.config = ppo_config
        self.state = ppo_state
        self.collect_module = collect_module
        self.train_module = train_module

        # Fill state default
        if not self.state.metric_module: self.state.metric_module = SimpleMetricModule(mode="approx")

    def step(self):    
        self.state.model.eval()
        self.train_module.buffer.empty()
        self.metrics = dict()

        # print(self.collect_module.buffer)
        for j, batch in enumerate(self.collect_module.buffer):
            batch = batch.to(self.config.device)
            # print(batch)
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
    def __init__(self, create_env: callable, ppo_config: PPOTrainConfig, ppo_state: PPOState, verbose=False):
        self.create_env = create_env
        self.config = ppo_config
        self.state = ppo_state
        self.verbose = verbose

        # Stateless Helper Modules
        self.collect_module = PPOCollectorModule(create_env, ppo_config, ppo_state)
        self.train_module = PPOTrainModule(ppo_config, ppo_state)
        self.adv_module = PPOAdvantageModule(ppo_config, ppo_state, self.collect_module, self.train_module)

        # Utility
        self.short_watch = Stopwatch()
        self.long_watch = Stopwatch()

    def step(self, gen):
        self.long_watch.start()

        if self.verbose: print(f"[{gen+1}/{self.config.generations}] Starting Generation")
        # 1. Collect Trajectory Dataset
        self.short_watch.start()
        self.collect_module.step()
        collection_time = self.short_watch.end()
        if self.state.logger: 
            self.state.logger.add({"collection_time": collection_time})
        if self.verbose: print(f"[{gen+1}/{self.config.generations}] Collected Data in {collection_time}")
        
        # 2. Compute Advantage + Value Targets
        metrics = self.adv_module.step()

        # 3. Train Loop
        self.short_watch.start()
        self.train_module.train()
        train_time = self.short_watch.end()
        if self.state.logger: 
            self.state.logger.add({"train_time": train_time})
        if self.verbose: print(f"[{gen+1}/{self.config.generations}] Trained in {train_time}")

        # 4. Log and Checkpoint
        if self.state.logger and ((gen+1) % self.config.log_interval) == 0:
            self.state.logger.add({"time": self.long_watch.end()})
            self._log_step()
        if self.state.checkpointer and ((gen+1) % self.config.checkpoint_interval) == 0:
            self._ckpt_step(gen, metrics)
        if self.verbose and not self.state.logger: print(f"[{gen+1}/{self.config.generations}] Step Results: {metrics}")

        return metrics

    def _ckpt_step(self, gen, metrics):
        state_obj = {
            "generation": gen + 1,
            "model_state_dict": self.state.model.state_dict(),
            "optimizer_state_dict": self.state.optimizer.state_dict(),
            "scaler_state_dict": self.state.scaler.state_dict(),
        }
        if self.config.best_metric_key:
            state_obj[self.config.best_metric_key] =  metrics[self.config.best_metric_key]
        if self.state.lr_scheduler:
            state_obj["lr_scheduler_state_dict"] = self.state.lr_scheduler.state_dict()
        
        self.state.checkpointer.save_progress(state_obj)

    def _log_step(self):
        if self.state.logger: self.state.logger.add({"generation": 1})
        if self.state.logger: self.state.logger.add({"timestep": self.config.generation_size})

        self.state.logger.next(print_row=self.verbose)

    def close(self, model_path=None):
        try: self.collect_module.collector.shutdown()
        except: pass
        if model_path: 
            state_obj = {
                "model_state_dict": self.state.model.state_dict(),
            }
            atomic_torch_save(state_obj, path=model_path)

    def model(self): 
        return self.state.model
    
    def history(self):
        return self.state.logger.dataframe()
    
    def run(self):
        for gen in range(self.state.start_generation, self.config.generations):
            self.step(gen)