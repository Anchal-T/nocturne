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

        extra_dim = obs_dim - grid_size
        linear_cls = NoisyLinear if noisy else nn.Linear

        conv_backbone = nn.Sequential(
            nn.Unflatten(1, (grid_channels, grid_rows, grid_cols)),
            nn.Conv2d(grid_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(8, 16), 16),
            nn.ReLU(inplace=True),
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
            nn.ReLU(),
        )

        head_input_dim = enc_dim + extra_dim
        if self.dueling:
            self.advantage_head = nn.Sequential(
                linear_cls(head_input_dim, hd_dim),
                nn.ReLU(),
                linear_cls(hd_dim, n_actions),
            )
            self.value_head = nn.Sequential(
                linear_cls(head_input_dim, hd_dim),
                nn.ReLU(),
                linear_cls(hd_dim, 1),
            )
        else:
            self.head = nn.Sequential(
                linear_cls(head_input_dim, hd_dim),
                nn.ReLU(),
                linear_cls(hd_dim, n_actions),
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
