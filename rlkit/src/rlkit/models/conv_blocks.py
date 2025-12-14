# Code modified from https://github.com/facebookresearch/sam2
import torch
from torch import nn

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class CXBlock(nn.Module):
    """ConvNeXt Block.
    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        residual = x

        # 1. Space Context Part
        x = self.dwconv(x)

        # 2. Patch Processing Part
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = residual + x
        return x
    
class AdaLayerNorm(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.linear = nn.Linear(time_dim, 2*channels)

        # zero init
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, t):
        # Normalize
        x = self.norm(x)

        # Get (scale, shift)
        t = self.linear(t).unsqueeze(-2).unsqueeze(-2)
        scale, shift = torch.chunk(t, 2, dim=-1)

        # Modulate
        x = x * (1 + scale) + shift
        return x
    
class CondCXBlock(nn.Module):
    """Conditional ConvNeXt Block.
    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        time_dim,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.adaln = AdaLayerNorm(dim, time_dim)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, t):
        residual = x

        # 1. Space Context Part
        x = self.dwconv(x)

        # 2. Time Modulation
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.adaln(x, t)

        # 3. Patch Processing Part
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = residual + x
        return x