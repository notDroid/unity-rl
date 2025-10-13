import torch
import math
from torch import nn
from torch import Tensor

class AutomaticEntropyModule(nn.Module):
    def __init__(self, alpha_init, target_entropy):
        super().__init__()
        self.target_entropy = target_entropy
        self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(alpha_init)))

    def forward(self, log_prob: Tensor) -> Tensor:
        alpha_loss = (-self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        return alpha_loss

    @property
    def alpha(self):
        return self.log_alpha.exp()