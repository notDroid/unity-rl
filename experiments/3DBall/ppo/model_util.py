import torch
import torch.nn as nn

class GeGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * out_features)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        x = self.linear(x)
        x, gate = torch.chunk(x, chunks=2, dim=-1)
        return x * self.gelu(gate)

class MLPBlock(nn.Module):
    def __init__(self, features, expansion_factor=8/3):
        super().__init__()
        expanded_features = int(features * expansion_factor)

        self.norm = nn.RMSNorm((features,))
        self.geglu = GeGLU(features, expanded_features)
        self.proj_down = nn.Linear(expanded_features, features)

        # TODO: Add last layer feature scaling intialized from eps (starts from ~identity)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        x = self.geglu(x)
        x = self.proj_down(x)

        return x + residual
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, n_blocks = 1):
        super().__init__()
        self.proj_in = nn.Linear(in_features, hidden_dim)

        self.mlp_blocks = nn.ModuleList([MLPBlock(hidden_dim) for _ in range(n_blocks)])
        
        self.proj_out = nn.Sequential(
            nn.RMSNorm((hidden_dim,)),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        x = self.proj_in(x)

        for layer in self.mlp_blocks:
            x = layer(x)

        x = self.proj_out(x)
        return x
