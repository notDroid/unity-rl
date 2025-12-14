from .conv_blocks import CondCXBlock, LayerNorm2d
from .mlp import MLPBlock
import torch
from torch import nn

class ConditionalCNN(nn.Module):
    # Assuming time is actually a one hot indicator
    def __init__(self, in_channels, out_dim, depths, dims, time_dim):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(time_dim, dims[-1], bias=False), MLPBlock(dims[-1]),
        )

        # Build Backbone
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            # Down
            if i != 0: 
                self.downs.append(nn.Sequential(
                    LayerNorm2d(prev_dim), 
                    nn.Conv2d(prev_dim, dim, kernel_size=2, stride=2),
                ))

            # Stage
            stage = nn.ModuleList([CondCXBlock(dim, time_dim=dims[-1]) for _ in range(depth)])
            self.stages.append(stage)
            
            prev_dim = dim

        # Project Out
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], out_dim)
        
    def forward(self, x, t):
        x = self.stem(x)
        t = self.time_embedding(t)

        for i in range(len(self.stages)):
            if i != 0:
                x = self.downs[i-1](x)
            stage = self.stages[i]
            for block in stage:
                x = block(x, t)
        
        x = self.norm(x.mean([-2, -1]))
        return self.head(x)
    
class CondVisionFeatureModel(nn.Module):
    def __init__(self, cnn_model, mlp_model):
        super().__init__()
        self.cnn_model = cnn_model
        self.mlp_model = mlp_model
    
    def forward(self, x, t):
        return self.mlp_model(self.cnn_model(x, t))