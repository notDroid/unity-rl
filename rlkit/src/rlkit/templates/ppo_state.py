from dataclasses import dataclass, field
import torch

from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ActorValueOperator, ActorCriticWrapper
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchrl.objectives import ClipPPOLoss

from typing import Optional
from rlkit.utils import Checkpointer, LoggerBase, SimpleMetricModule

from .ppo_config import PPOTrainConfig

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

    # Optional
    start_generation: int = 0

class PPOStateManager:
    def __init__(self, train_config: PPOTrainConfig, state: PPOState):
        self.state = state
        self.train_config = train_config

    def restore_checkpoint(self, checkpoint):
        # Restore required components
        self.state.start_generation = int(checkpoint["generation"])
        self.state.model.load_state_dict(checkpoint["model_state_dict"])
        self.state.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore optional components
        if "scaler_state_dict" in checkpoint:
            if self.state.scaler is not None:
                scaler = self.state.scaler
            else:
                scaler = torch.amp.GradScaler(enabled=(self.train_config.amp_dtype == torch.float16))
                self.state.scaler = scaler
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.state.logger is not None:
            self.state.logger.revert("generation", self.state.start_generation)

        if self.state.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.state.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        return self.state
            
    def reset_state(self):
        # Reset required components
        self.state.start_generation = 0

        # Reset optional components
        if self.state.logger is not None:
            self.state.logger.reset()
        if self.state.checkpointer is not None:
            self.state.checkpointer.reset()

        return self.state