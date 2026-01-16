import torch
import torch.nn as nn

class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.SiLU()):
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * out_features, bias=False)
        self.act = activation

    def forward(self, x):
        x = self.linear(x)
        x, gate = torch.chunk(x, chunks=2, dim=-1)
        return x * self.act(gate)

class MLPBlock(nn.Module):
    def __init__(self, features, expansion_factor=8/3):
        super().__init__()
        expanded_features = int(features * expansion_factor)

        self.norm = nn.RMSNorm((features,))
        self.geglu = GatedLinear(features, expanded_features)
        self.proj_down = nn.Linear(expanded_features, features, bias=False)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        x = self.geglu(x)
        x = self.proj_down(x)

        return x + residual
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, n_blocks = 1, norm_layer=nn.RMSNorm, **kwargs):
        super().__init__()
        self.proj_in = nn.Linear(in_features, hidden_dim)

        self.mlp_blocks = nn.ModuleList([MLPBlock(hidden_dim, **kwargs) for _ in range(n_blocks)])
        
        self.proj_out = nn.Sequential(
            norm_layer((hidden_dim,)),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        x = self.proj_in(x)

        for layer in self.mlp_blocks:
            x = layer(x)

        return self.proj_out(x)