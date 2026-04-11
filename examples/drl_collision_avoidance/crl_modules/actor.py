"""
CRL Actor: SAC-style stochastic continuous actor.

Architecture matches the Scaling CRL paper:
  - Input: concat([state, goal])  (state_dim + goal_dim)
  - Residual blocks with LayerNorm + SiLU
  - Outputs (mean, log_std) for tanh-squashed continuous actions
  - LeCun uniform initialization
"""

import torch
import torch.nn as nn

from .encoders import ResidualBlock, _apply_lecun_init

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class ContinuousActor(nn.Module):
    """SAC-style stochastic actor for goal-conditioned CRL.

    Takes (state, goal) and outputs (mean, log_std) for a tanh-squashed
    Gaussian. The action is in [-1, 1]^action_dim; the caller scales to
    the environment's throttle/steer range.

    Args:
        state_dim:  Pure state dimension (no goal features). = 605 for nocturne.
        goal_dim:   Goal encoding dimension. = 2 for nocturne.
        action_dim: Output action dimension. = 2 (throttle, steer).
        width:      Hidden width of residual blocks.
        depth:      Total residual layers (multiple of 4).
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        width: int = 256,
        depth: int = 16,
    ):
        super().__init__()
        num_blocks = max(1, depth // 4)
        input_dim = state_dim + goal_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(num_blocks)])
        self.mean_head = nn.Linear(width, action_dim)
        self.log_std_head = nn.Linear(width, action_dim)
        _apply_lecun_init(self)

    def forward(self, state: torch.Tensor, goal: torch.Tensor):
        """Return (mean, log_std) without squashing."""
        x = torch.cat([state, goal], dim=-1)
        x = self.input_proj(x)
        x = self.blocks(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        # Clamp log_std to [LOG_STD_MIN, LOG_STD_MAX] via smooth tanh rescaling
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def sample(self, state: torch.Tensor, goal: torch.Tensor):
        """Sample action with reparameterisation; return (action_tanh, log_prob).

        log_prob accounts for the tanh squashing Jacobian:
            log π(a|s,g) = log N(x; μ, σ) - Σ log(1 - tanh²(x_i) + ε)
        """
        mean, log_std = self(state, goal)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        x_t = mean + std * eps
        action = torch.tanh(x_t)
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(x_t)
            - torch.log(1.0 - action.pow(2) + 1e-6)
        ).sum(dim=-1)
        return action, log_prob

    @torch.no_grad()
    def deterministic_action(
        self, state: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """Return tanh(mean) without sampling."""
        mean, _ = self(state, goal)
        return torch.tanh(mean)
