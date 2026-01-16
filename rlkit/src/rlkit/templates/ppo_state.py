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
    agent: ActorValueOperator | ActorCriticWrapper
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
    HANDLED_COMPONENTS = {'start_generation', 'model', 'loss_module', 'optimizer', 'checkpointer', 'logger', 'scaler', 'lr_scheduler'}

    def __init__(self, train_config: PPOTrainConfig):
        self.train_config = train_config

    def restore_checkpoint(self, state: dict, checkpoint):
        # Restore required components
        state['start_generation'] = int(checkpoint["generation"])
        state['model'].load_state_dict(checkpoint["model_state_dict"])
        state['optimizer'].load_state_dict(checkpoint["optimizer_state_dict"])


        # Restore optional components
        if "scaler_state_dict" in checkpoint:
            if state.get('scaler', None) is not None:
                scaler = state['scaler']
            else:
                scaler = torch.amp.GradScaler(enabled=(self.train_config.amp_dtype == torch.float16))
                state['scaler'] = scaler
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if state.get('logger', None) is not None:
            state['logger'].revert("generation", state['start_generation'])

        if state.get('lr_scheduler', None) is not None and "lr_scheduler_state_dict" in checkpoint:
            state['lr_scheduler'].load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # Try restoring any other components
        for key in checkpoint.keys():
            suffix_len = len('_state_dict')
            if key.endswith('_state_dict') and key[:-suffix_len] not in self.HANDLED_COMPONENTS:
                component = state[key[:-suffix_len]]
                if hasattr(component, 'load_state_dict'):
                    component.load_state_dict(checkpoint[key])

        return state
            
    def reset_state(self, state: dict):
        # Reset required components
        state['start_generation'] = 0

        # Reset optional components
        if state.get('logger', None) is not None:
            state['logger'].reset()
        if state.get('checkpointer', None) is not None:
            state['checkpointer'].reset()

        return state