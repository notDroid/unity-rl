import torch
import torch.nn as nn
from .mlp import MLP

class CatWrapper(nn.Module):
    def __init__(self, module, dim=-1):
        super().__init__()
        self.add_module("inner", module)
        self.dim = dim

    def forward(self, *args):
        x = torch.cat(args, dim=self.dim)
        x = self.inner(x)
        return x