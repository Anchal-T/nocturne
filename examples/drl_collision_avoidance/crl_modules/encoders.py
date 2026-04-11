"""
CRL Encoders: SA-encoder φ(s,a) and G-encoder ψ(g).

Architecture follows the Scaling CRL paper (Wang et al. 2025):
  - LeCun uniform initialization
  - LayerNorm + Swish (SiLU) activations
  - Residual blocks with 4 dense layers each
  - Configurable depth (total layers = multiple of 4)
  - Output: 64-dim embedding (raw, no L2 normalization)

Q(s,a,g) = -||φ(s,a) - ψ(g)||₂
"""

import math

import torch
import torch.nn as nn


def lecun_init_(tensor: torch.Tensor) -> torch.Tensor:
    """LeCun uniform init: Uniform(-sqrt(1/fan_in), sqrt(1/fan_in)).
    Matches the paper's variance_scaling(1/3, 'fan_in', 'uniform').
    """
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    bound = math.sqrt(1.0 / max(1, fan_in))
    return nn.init.uniform_(tensor, -bound, bound)


def _apply_lecun_init(module: nn.Module) -> None:
    """Apply LeCun init to all Linear layers in a module."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            lecun_init_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ResidualBlock(nn.Module):
    """4-layer residual block: matches paper's residual_block().

    Each block: Linear→LN→SiLU repeated 4 times, with identity skip.
    """

    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        _apply_lecun_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class SAEncoder(nn.Module):
    """State-Action encoder: φ(s, a) → R^embed_dim.

    Takes the concatenation of state (grid + ego features) and action,
    passes through an initial projection then `depth // 4` residual blocks,
    and outputs a fixed-size embedding.

    Args:
        state_dim:  Dimension of the state-only observation (no goal features).
                    For nocturne with 20×10 grid: 3*20*10 + 2 + 3 = 605.
        action_dim: Dimension of the (tanh-squashed) continuous action. = 2.
        width:      Hidden width of residual blocks.
        depth:      Total number of residual layers (must be multiple of 4).
                    E.g. depth=16 → 4 residual blocks (16 layers).
        embed_dim:  Output embedding dimension.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        width: int = 256,
        depth: int = 16,
        embed_dim: int = 64,
    ):
        super().__init__()
        num_blocks = max(1, depth // 4)
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim + action_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(width, embed_dim)
        _apply_lecun_init(self)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class GEncoder(nn.Module):
    """Goal encoder: ψ(g) → R^embed_dim.

    Takes a 2-dim ego-frame goal displacement [long/DIST_NORM, lat/DIST_NORM]
    and produces a fixed-size embedding.

    Args:
        goal_dim:  Dimension of the goal vector. = 2 for nocturne.
        width:     Hidden width.
        depth:     Total residual layers (multiple of 4).
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        goal_dim: int,
        width: int = 256,
        depth: int = 16,
        embed_dim: int = 64,
    ):
        super().__init__()
        num_blocks = max(1, depth // 4)
        self.input_proj = nn.Sequential(
            nn.Linear(goal_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(width, embed_dim)
        _apply_lecun_init(self)

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(goal)
        x = self.blocks(x)
        return self.output_proj(x)
