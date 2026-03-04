from typing import List, Optional

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu(out)


class QNetwork(nn.Module):
    """
    Q-network.

    A residual CNN encoder with BatchNorm processes the flattened occupancy-grid
    portion of the
    observation; its output is concatenated with the remaining ego/path/TTZ
    features before passing through a small MLP head.

        ``hidden_layers`` is a 4-element list to preserve config compatibility:
      [0] unused
      [1] encoder output width  (Linear after the conv stack)
      [2] unused
            [3] stream hidden width
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        grid_size: int,
        hidden_layers: Optional[List[int]] = None,
        grid_rows: int = 25,
        grid_cols: int = 14,
        dueling: bool = True,
    ):
        super().__init__()
        hidden_layers = hidden_layers or [128, 128, 128, 32]
        assert len(hidden_layers) == 4

        self.hidden_layers = hidden_layers
        self.grid_size = grid_size
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.dueling = dueling
        extra_dim = obs_dim - grid_size

        # Build the residual backbone first so we can probe its output size with
        # a dummy tensor for arbitrary occupancy-grid shapes.
        conv_backbone = nn.Sequential(
            nn.Unflatten(1, (1, grid_rows, grid_cols)),
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
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
            nn.Linear(flat_dim, hidden_layers[1]),
            nn.ReLU(),
        )

        head_input_dim = hidden_layers[1] + extra_dim
        if self.dueling:
            self.advantage_head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_layers[3]),
                nn.ReLU(),
                nn.Linear(hidden_layers[3], n_actions),
            )
            self.value_head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_layers[3]),
                nn.ReLU(),
                nn.Linear(hidden_layers[3], 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_layers[3]),
                nn.ReLU(),
                nn.Linear(hidden_layers[3], n_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid_input = x[:, : self.grid_size]
        extra_input = x[:, self.grid_size :]
        grid_features = self.grid_encoder(grid_input)
        combined = torch.cat([grid_features, extra_input], dim=1)
        if self.dueling:
            advantage = self.advantage_head(combined)
            value = self.value_head(combined)
            return value + advantage - advantage.mean(dim=-1, keepdim=True)
        return self.head(combined)
