from dataclasses import dataclass, field
from math import ceil

import torch
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ActorValueOperator, ActorCriticWrapper
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchrl.objectives import ClipPPOLoss

from typing import Optional
from rlkit.utils import Checkpointer, LoggerBase, SimpleMetricModule


'''
Required Config:
    1. generations, generation_size
    2. n_slices, slice_len
        - slice_len is window for (advantage, value target) estimation, n_slices*slice_len batch size
    3. epochs, minibatch_size
        - per gen train loop config
    
Optional Config:
    1. workers, device, storage_device, collector_buffer_size
        - lower collector buffer size when out of RAM
    2. kl_soft_clip, kl_hard_clip, early_stop_threshold
        - soft_clip + early_stop_threshold is used to early stop the train loop, hard_clip skips iterations
    3. checkpoint_interval, log_interval, model_path, best_metric_key
    4. max_grad_norm, adv_clip
'''

@dataclass
class PPOTrainConfig:
    # Required parameters
    generations: int
    generation_size: int
    n_slices: int
    slice_len: int
    epochs: int
    minibatch_size: int

    # Optional Start Generation
    start_generation: int = 0

    # Optional parameters with defaults    
    workers: int = 1
    device: str = "cpu"
    storage_device: str = "cpu"
    collector_buffer_size: int | None = None
    env_batch_dim: int = 0

    kl_soft_clip: float | None = None
    kl_hard_clip: float | None = None
    early_stop_threshold: int | None = None
    
    checkpoint_interval: int = 1
    log_interval: int = 1
    model_path: str | None = None
    best_metric_key: str | None = None

    max_grad_norm: float = 1.0
    adv_clip: float = 5.0

    # Derived Attributes
    timesteps: int = field(init=False)
    collector_iters_per_gen: int = field(init=False)
    device_type: str = field(init=False)
    amp_dtype: torch.dtype = field(init=False)

    def __post_init__(self):
        if self.collector_buffer_size is None:
            self.collector_buffer_size = self.generation_size

        self.timesteps = self.generation_size * (self.generations - self.start_generation)
        self.collector_iters_per_gen = int(ceil(self.generation_size / self.collector_buffer_size))

        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.amp_dtype = torch.float16 if self.device_type == "cuda" else torch.float32

# PPO State to Save
@dataclass
class PPOState:
    model: ActorValueOperator | ActorCriticWrapper
    loss_module: ClipPPOLoss
    optimizer: Optimizer

    checkpointer: Optional[Checkpointer] = None
    logger: Optional[LoggerBase] = None
    scaler: Optional[torch.amp.GradScaler] = None
    metric_module: Optional[SimpleMetricModule] = None
    lr_scheduler: Optional[LRScheduler] = None