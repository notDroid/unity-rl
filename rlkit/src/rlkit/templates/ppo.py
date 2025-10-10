# Torch
import torch
from torch import nn

# TorchRL
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torch.optim import Optimizer
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data import ReplayBuffer, LazyMemmapStorage, SliceSamplerWithoutReplacement, SamplerWithoutReplacement


# Util
import numpy as np
from operator import itemgetter as get
from rlkit.util import Stopwatch, Checkpointer, Logger, SimpleMetricModule

def ppo_loss_td_to_dict(loss_data, weight):
    # Hard coded keys, values
    keys = ["value_loss", "explained_variance", "policy_loss", "kl_approx", "clip_fraction", "ESS"]
    values = ["loss_critic", "explained_variance", "loss_objective", "kl_approx", "clip_fraction", "ESS"]

    return {
        key: (loss_data[value].detach().mean().item(), weight) for key, value in zip(keys, values)
    }

ADV_CLIP = 5

class PPOTrainer:
    def __init__(self, create_env: callable, train_config: dict):
        self.create_env = create_env
        self._load_config(train_config)
        self.loaded = False

    def _load_config(self, train_config: dict):
        self.train_config = train_config
        # -------------- RL Params --------------
        # (self.epsilon, self.gamma, self.gae_lambda, self.entropy_coef) = get(
        #     "epsilon", "gamma", "gae_lambda", "entropy_coef"
        # )(train_config)
        
        # -------------- Collection Params --------------
        (self.workers, self.device, self.storage_device, self.generations,
        self.generation_size, self.collector_buffer_size) = get(
            "workers", "device", "storage_device", "generations",
            "generation_size", "collector_buffer_size",
        )(train_config)

        # -------------- Advantage Phase Params --------------
        (self.n_slices, self.slice_len) = get(
            "n_slices", "slice_len",
        )(train_config)

        # -------------- Training Loop Params --------------
        (self.epochs, self.minibatch_size, self.kl_soft_clip, 
        self.kl_hard_clip, self.early_stop_threshold) = get(
            "epochs", "minibatch_size", 
            "kl_soft_clip", "kl_hard_clip", "early_stop_threshold"
        )(train_config)

        # -------------- Optimizer Params --------------
        (self.max_grad_norm, self.lr) = get("max_grad_norm", "lr")(train_config)

        # -------------- Checkpoint and Log params --------------
        (self.checkpoint_interval, self.log_interval, self.model_path, self.best_metric_key) = get(
            "checkpoint_interval", "log_interval", "model_path", "best_metric_key"
        )(train_config)

        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.amp_dtype   = torch.float16 if self.device_type == "cuda" else torch.float32

    def _init_utils(self):
        # -------------- Watches + Metrics --------------
        self.short_watch = Stopwatch()
        self.long_watch = Stopwatch()
        self.metric_module = SimpleMetricModule(mode="approx")

        # -------------- Replay Buffers --------------
        self.collect_replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(self.generation_size, device=self.storage_device, ndim=2 + int(self.workers > 1)),
            sampler=SliceSamplerWithoutReplacement(
                slice_len = self.slice_len,
                shuffle=False, strict_length=False, 
                end_key=("next", "done")
            ),
            batch_size=self.n_slices * self.slice_len,
        )
        self.train_replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(self.generation_size, device=self.storage_device), 
            sampler=SamplerWithoutReplacement(), 
            batch_size=self.minibatch_size
        )

        # -------------- Collectors --------------
        self.timesteps = self.generation_size * (self.generations - self.start_generation)
        self.collector_iters_per_gen = int(np.ceil(self.generation_size / self.collector_buffer_size))
        if self.workers > 1:
            self.collector = MultiSyncDataCollector([self.create_env]*self.workers, self.policy, 
                frames_per_batch=self.collector_buffer_size, 
                total_frames=self.timesteps, 
                env_device="cpu", device=self.device, storing_device=self.storage_device, 
                reset_at_each_iter=False,
                update_at_each_batch=False, # Manually update
            )
        else:
            self.collector = SyncDataCollector(
                self.create_env, self.policy, 
                frames_per_batch=self.collector_buffer_size, 
                total_frames=self.timesteps, 
                env_device="cpu", device=self.device, storing_device=self.storage_device,
                reset_at_each_iter=False,
            )

    def load_state(self, 
        policy: ProbabilisticActor, value: TensorDictModule, optimizer: Optimizer, 
        loss_module: ClipPPOLoss, checkpointer: Checkpointer=None, logger: Logger=None, 
        scaler: torch.amp.GradScaler=None, start_generation=0,
        ):

        self.loaded = True
        self.policy = policy
        self.value = value
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.checkpointer = checkpointer
        self.logger = logger
        # Scaler is optionally provided (only if it has state)
        self.scaler = scaler
        if scaler is None:
            self.scaler = torch.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))
        self.start_generation = start_generation

    def _collect(self):
        # Set up
        if self.workers > 1: self.collector.update_policy_weights_() # Update weights manually
        self.policy.eval(); self.value.eval()
        self.short_watch.start(); self.collect_replay_buffer.empty()

        # Buffer in memory then move to memory mapped storage in loop
        for j in range(self.collector_iters_per_gen):
            data = self.collector.next()
            self.collect_replay_buffer.extend(data)
        
        # Finish up
        collection_time = self.short_watch.end()
        if self.logger: self.logger.add({"collection_time": collection_time})

    def _prepare_train(self):
        self.train_replay_buffer.empty()
        for j, batch in enumerate(self.collect_replay_buffer):
            batch = batch.to(self.device)
        
            with torch.no_grad():
                self.loss_module.value_estimator(batch)
                metrics = self.metric_module(batch)
            batch["advantage"].clamp_(-ADV_CLIP, ADV_CLIP)
            
            if self.logger: self.logger.acc(metrics, mode='avg')
            self.train_replay_buffer.extend(batch.reshape(-1).cpu())
        self.collect_replay_buffer.empty()
        return metrics

    def _train_loop(self):
        self.short_watch.start()
        self.policy.train(); self.value.train()
        self.early_stop = 0
        for epoch in range(self.epochs):
            for j, batch in enumerate(self.train_replay_buffer):
                batch = batch.to(self.device)
                self._train_step(batch, epoch, j)
                
                if self.kl_soft_clip is not None and self.early_stop >= self.early_stop_threshold: break
            if self.kl_soft_clip is not None and self.early_stop >= self.early_stop_threshold: break
        self.train_replay_buffer.empty()
        train_time = self.short_watch.end()
        if self.logger: self.logger.add({"train_time": train_time})
        if self.logger: self.logger.add({"generation": 1})
        if self.logger: self.logger.add({"timestep": self.generation_size})
        
    def _train_step(self, batch, epoch, j):
        # -------------- a. Compute Loss --------------
        with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=(self.amp_dtype==torch.float16)):
            loss_data = self.loss_module(batch)
            loss = loss_data["loss_objective"] + loss_data["loss_critic"] + loss_data["loss_entropy"]

        # -------------- b. KL Safety Check --------------
        kl_approx = loss_data["kl_approx"].mean().cpu().item()
        if self.kl_soft_clip is not None and kl_approx > self.kl_soft_clip:
            self.early_stop += 1
            if self.early_stop >= self.early_stop_threshold: 
                print(f"Early stopped at ({epoch}, {j})")
                return
        if self.kl_hard_clip is not None and kl_approx > self.kl_hard_clip:
            print(f"Skipping iteration ({epoch}, {j}) with kl > kl_hard_clip: {kl_approx} > {self.kl_hard_clip}")
            return
        
        # -------------- c. Optimization Step --------------
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=self.max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # -------------- d. Metric Update --------------
        weight = float(batch.batch_size[0])
        if self.logger: self.logger.acc(ppo_loss_td_to_dict(loss_data, weight), mode='ema')
        return
    
    def _log_step(self):
        self.logger.add({"time": self.long_watch.end()})
        self.long_watch.start()
        self.logger.next(print_row=True)

    def _ckpt_step(self, metrics: dict):
        state_obj={
            "generation": self.gen + 1,
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        if self.best_metric_key:
            state_obj[self.best_metric_key] =  metrics[self.best_metric_key]
        
        self.checkpointer.save_progress(state_obj)

    def train(self):
        if not self.loaded: raise RuntimeError("Load state using load_state()")
        self._init_utils()
        self.long_watch.start()
        for gen in range(self.start_generation, self.generations):
            self.gen = gen

            # 1. Collect Trajectories
            self._collect()

            # 2. Compute Advantages/Value Targets/Metrics
            metrics = self._prepare_train()

            # 3. Minibatch Gradient Descent
            self._train_loop()

            # 4. Save Results
            if self.logger and (gen % self.log_interval) == 0:
                self._log_step()
            if self.checkpointer and (gen % self.checkpoint_interval) == 0:
                self._ckpt_step(metrics)
        self.close(save=True)

    def close(self, save=False):
        try: self.collector.shutdown()
        except: pass
        if save and self.checkpointer: self.checkpointer.copy_model('latest', self.model_path, ('policy_state_dict', 'value_state_dict'))

    def model(self): 
        return self.policy, self.value
    
    def history(self):
        return self.logger.dataframe()
    