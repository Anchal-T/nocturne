from typing import List, Optional

import torch
import torch.nn as nn

from .noisy_layer import NoisyLinear


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.GroupNorm(min(8, out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


class ResidualMLPBlock(nn.Module):
    """A residual block for 1D features, inspired by Scaling CRL."""
    def __init__(self, channels: int, linear_cls: type, use_silu: bool = True):
        super().__init__()
        act_cls = nn.SiLU if use_silu else nn.ReLU
        
        # Following Scaling CRL paper: 4 layers per residual block
        self.layers = nn.Sequential(
            linear_cls(channels, channels),
            nn.LayerNorm(channels),
            act_cls(),
            linear_cls(channels, channels),
            nn.LayerNorm(channels),
            act_cls(),
            linear_cls(channels, channels),
            nn.LayerNorm(channels),
            act_cls(),
            linear_cls(channels, channels),
            nn.LayerNorm(channels),
            act_cls(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class ResidualMLP(nn.Module):
    """An MLP head that relies on stacking residual blocks for depth.
    
    If mlp_depth = 2, behaves like a standard 2-layer MLP (1 hidden layer).
    If mlp_depth > 3, builds floor(mlp_depth / 4) residual blocks.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, 
                 mlp_depth: int, linear_cls: type):
        super().__init__()
        
        act_cls = nn.SiLU
        self.initial_layer = nn.Sequential(
            linear_cls(in_features, hidden_features),
            nn.LayerNorm(hidden_features) if mlp_depth > 3 else nn.Identity(),
            act_cls()
        )
        
        num_residual_blocks = mlp_depth // 4
        self.residual_blocks = nn.Sequential(*[
            ResidualMLPBlock(hidden_features, linear_cls, use_silu=True)
            for _ in range(num_residual_blocks)
        ])
        
        self.final_layer = linear_cls(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        return self.final_layer(x)


def _parse_hidden_layers(
    hidden_layers: Optional[List[int]],
    encoder_dim: Optional[int],
    head_dim: Optional[int],
) -> tuple:
    """Resolve encoder_dim and head_dim from either explicit args or legacy list.

    Legacy format was a 4-element list [unused, encoder_dim, unused, head_dim].
    New code should pass encoder_dim and head_dim directly.
    """
    if encoder_dim is not None and head_dim is not None:
        return encoder_dim, head_dim

    hl = hidden_layers or [128, 128, 128, 32]
    if len(hl) != 4:
        raise ValueError(
            f"hidden_layers must have 4 elements (legacy format), got {len(hl)}. "
            "Prefer using encoder_dim and head_dim directly."
        )
    return hl[1], hl[3]


class QNetwork(nn.Module):
    """Q-network with residual CNN encoder and NoisyNet dueling heads.

    The occupancy-grid portion of the observation is processed by a residual
    CNN encoder; its output is concatenated with the remaining ego/path/TTZ
    features before passing through NoisyLinear dueling streams.

    Args:
        obs_dim: Total observation dimension (grid + extra features).
        n_actions: Number of discrete actions.
        grid_size: Number of elements in the flattened occupancy grid.
        grid_channels: Number of occupancy-grid channels.
        grid_rows: Height of the occupancy grid.
        grid_cols: Width of the occupancy grid.
        dueling: Whether to use dueling architecture.
        encoder_dim: Output width of the CNN encoder's final linear layer.
        head_dim: Hidden width of the advantage/value stream heads.
        hidden_layers: Legacy 4-element list [_, encoder_dim, _, head_dim].
            Ignored when encoder_dim and head_dim are provided explicitly.
        noisy: Whether to use NoisyLinear in the heads for exploration.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        grid_size: int,
        hidden_layers: Optional[List[int]] = None,
        grid_channels: int = 1,
        grid_rows: int = 25,
        grid_cols: int = 14,
        dueling: bool = True,
        encoder_dim: Optional[int] = None,
        head_dim: Optional[int] = None,
        noisy: bool = True,
        mlp_depth: int = 2,
    ):
        super().__init__()
        enc_dim, hd_dim = _parse_hidden_layers(hidden_layers, encoder_dim, head_dim)

        # Store for checkpoint compatibility
        self.hidden_layers = [enc_dim, enc_dim, enc_dim, hd_dim]
        self.grid_size = grid_size
        self.grid_channels = grid_channels
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.dueling = dueling
        self.noisy = noisy
        self.mlp_depth = mlp_depth

        extra_dim = obs_dim - grid_size
        linear_cls = NoisyLinear if noisy else nn.Linear

        conv_backbone = nn.Sequential(
            nn.Unflatten(1, (grid_channels, grid_rows, grid_cols)),
            nn.Conv2d(grid_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(8, 16), 16),
            nn.SiLU(),
            ResidualBlock(16, 16),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = conv_backbone(torch.zeros(1, grid_size)).shape[1]

        self.grid_encoder = nn.Sequential(
            *conv_backbone,
            nn.Linear(flat_dim, enc_dim),
            nn.SiLU(),
        )

        head_input_dim = enc_dim + extra_dim
        if self.dueling:
            self.advantage_head = ResidualMLP(
                in_features=head_input_dim,
                hidden_features=hd_dim,
                out_features=n_actions,
                mlp_depth=self.mlp_depth,
                linear_cls=linear_cls,
            )
            self.value_head = ResidualMLP(
                in_features=head_input_dim,
                hidden_features=hd_dim,
                out_features=1,
                mlp_depth=self.mlp_depth,
                linear_cls=linear_cls,
            )
        else:
            self.head = ResidualMLP(
                in_features=head_input_dim,
                hidden_features=hd_dim,
                out_features=n_actions,
                mlp_depth=self.mlp_depth,
                linear_cls=linear_cls,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid_features = self.grid_encoder(x[:, :self.grid_size])
        combined = torch.cat([grid_features, x[:, self.grid_size:]], dim=1)
        if self.dueling:
            advantage = self.advantage_head(combined)
            value = self.value_head(combined)
            return value + advantage - advantage.mean(dim=-1, keepdim=True)
        return self.head(combined)

    def reset_noise(self) -> None:
        """Reset noise in all NoisyLinear layers (call before each forward in training)."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
