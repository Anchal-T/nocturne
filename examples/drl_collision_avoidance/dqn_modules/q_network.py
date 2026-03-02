from typing import List, Optional

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q-network.

    A CNN encoder processes the flattened occupancy-grid portion of the
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

        # Build the conv backbone first so we can probe its output size with a
        # dummy tensor rather than hard-coding 32 * 6 * 3 for a fixed 25×14 grid.
        conv_backbone = nn.Sequential(
            nn.Unflatten(1, (1, grid_rows, grid_cols)),
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
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
