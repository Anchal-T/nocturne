"""
CRL Agent: Contrastive RL with Hindsight Experience Replay.

Implements the Scaling CRL training objective (Wang et al. 2025):
  - InfoNCE critic loss: classifies whether (s,a) pairs belong to trajectories
    leading to goal g.  Q(s,a,g) = -||φ(s,a) - ψ(g)||₂.
  - Logsumexp regularisation (from paper): λ * mean(logsumexp(logits)²).
  - SAC-style actor: maximises Q(s, π(s,g), g) - α * log π(a|s,g).
  - Auto-tuned entropy coefficient α.

Usage:
    agent = CRLAgent(CRLAgentConfig(state_dim=605, ...))
    # in env loop:
    action = agent.select_action(state, goal)
    agent.replay_buffer.add(state, action, ego_info, done, env_id=0)
    # periodically:
    metrics = agent.train_step()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .actor import ContinuousActor
from .encoders import GEncoder, SAEncoder
from .her_buffer import HERReplayBuffer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CRLAgentConfig:
    """All hyperparameters for a CRL agent instance.

    Derived dimensions are computed by the caller from the environment.

    Attributes:
        state_dim:                Pure state dimension (no goal). For nocturne
                                  this is 605 = 600 (occupancy grid 3×20×10)
                                  + 2 (heading, speed) + 3 (TTZ features).
        action_dim:               Continuous action dimension (= 2 for nocturne:
                                  throttle and steer, both tanh-squashed to
                                  [-1, 1]).
        goal_dim:                 Goal encoding dimension (= 2 for nocturne:
                                  longitudinal and lateral ego-frame displacement
                                  normalised by DIST_NORM).
        critic_depth:             Total number of residual layers used by the
                                  SA- and G-encoders.
        actor_depth:              Total number of residual layers used by the
                                  actor network.
        network_width:            Hidden channel width shared by all networks.
        embed_dim:                Output embedding dimension for both encoders.
                                  The InfoNCE logits matrix is (B, B) over this
                                  space via ||φ(s,a) - ψ(g)||₂.
        critic_lr:                Learning rate for the joint SA/G encoder
                                  optimiser.
        actor_lr:                 Learning rate for the actor optimiser.
        alpha_lr:                 Learning rate for the log-α optimiser.
        batch_size:               Training batch size drawn from the HER buffer.
        gamma:                    Discount factor; also used as the geometric
                                  weight in HER future sampling.
        target_entropy_factor:    target_entropy = -factor * action_dim.
        logsumexp_penalty_coeff:  Weight λ of the logsumexp² regularisation
                                  term added to the InfoNCE critic loss.
        her_max_episodes:         Maximum number of completed episodes retained
                                  in the HER ring buffer.
        min_replay_episodes:      Minimum completed episodes before any gradient
                                  step is taken.
        num_envs:                 Number of parallel environments feeding this
                                  agent (one in-progress episode tracked per env).
        device:                   PyTorch device string, e.g. ``"cuda"`` or
                                  ``"cpu"``.
    """

    # --- Observation / action dimensions ---
    state_dim: int = 605  # pure state dim (no goal)
    action_dim: int = 2  # continuous action dim
    goal_dim: int = 2  # goal encoding dim

    # --- Network architecture ---
    critic_depth: int = 16  # total residual layers for SA/G encoders
    actor_depth: int = 16  # total residual layers for actor
    network_width: int = 256  # hidden width shared across all sub-networks
    embed_dim: int = 64  # encoder output dimension

    # --- Optimisation ---
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 512
    gamma: float = 0.99
    target_entropy_factor: float = 0.5  # target_entropy = -factor * action_dim
    logsumexp_penalty_coeff: float = 0.1  # λ for logsumexp² regularisation

    # --- HER buffer ---
    her_max_episodes: int = 50_000
    min_replay_episodes: int = 200  # minimum episodes before training starts
    num_envs: int = 1

    # --- Runtime ---
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CRLAgent:
    """Contrastive RL agent with HER for nocturne driving tasks.

    The agent couples three learnable components:

    1. **SA-Encoder** φ(s, a) → ℝ^embed_dim — encodes (state, action) pairs.
    2. **G-Encoder**  ψ(g)    → ℝ^embed_dim — encodes goals.
    3. **Actor**      π(a|s, g) — Gaussian policy conditioned on state and goal,
       with a tanh squashing layer so actions lie in [-1, 1]².

    Training alternates three gradient steps per call to :meth:`train_step`:

    * **Critic step** – InfoNCE loss over a (B×B) logit matrix plus a
      logsumexp² penalty that prevents score collapse.
    * **Actor step** – SAC-style objective:
      ``mean(α·log π(a|s,g) − Q(s, π(s,g), g))``.
    * **Alpha step** – Dual ascent on the temperature parameter α to enforce
      a minimum-entropy constraint.

    Public interface::

        agent = CRLAgent(CRLAgentConfig(state_dim=605))

        # Environment interaction (numpy in, numpy out)
        action = agent.select_action(state, goal)
        agent.replay_buffer.add(state, action, ego_info, done, env_id=0)

        # Training
        metrics = agent.train_step()   # None until buffer is ready

        # Persistence
        agent.save("checkpoints/crl_step_100k.pt")
        agent.load("checkpoints/crl_step_100k.pt")

        # Diagnostics
        print(agent.param_count())
    """

    def __init__(self, config: CRLAgentConfig) -> None:
        self.config = config

        # Resolve device: fall back to CPU if CUDA is requested but unavailable.
        if config.device != "cpu" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        # ------------------------------------------------------------------ #
        # Networks                                                             #
        # ------------------------------------------------------------------ #

        self.sa_encoder: SAEncoder = SAEncoder(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            width=config.network_width,
            depth=config.critic_depth,
            embed_dim=config.embed_dim,
        ).to(self.device)

        self.g_encoder: GEncoder = GEncoder(
            goal_dim=config.goal_dim,
            width=config.network_width,
            depth=config.critic_depth,
            embed_dim=config.embed_dim,
        ).to(self.device)

        self.actor: ContinuousActor = ContinuousActor(
            state_dim=config.state_dim,
            goal_dim=config.goal_dim,
            action_dim=config.action_dim,
            width=config.network_width,
            depth=config.actor_depth,
        ).to(self.device)

        # ------------------------------------------------------------------ #
        # Entropy coefficient α (auto-tuned via dual ascent on log α)         #
        # ------------------------------------------------------------------ #

        # target_entropy < 0: the policy must be at least this uncertain.
        self.target_entropy: float = -config.target_entropy_factor * float(
            config.action_dim
        )

        # log α is unconstrained; α = exp(log α) is always positive.
        self.log_alpha: torch.Tensor = torch.tensor(
            0.0,
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

        # ------------------------------------------------------------------ #
        # Optimisers                                                           #
        # ------------------------------------------------------------------ #

        # Critic optimiser covers both encoders jointly so that the distance
        # metric is updated in a coordinated fashion.
        self.critic_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters()),
            lr=config.critic_lr,
        )

        self.actor_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
        )

        self.alpha_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=config.alpha_lr,
        )

        # ------------------------------------------------------------------ #
        # Replay buffer                                                        #
        # ------------------------------------------------------------------ #

        self.replay_buffer: HERReplayBuffer = HERReplayBuffer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            goal_dim=config.goal_dim,
            max_episodes=config.her_max_episodes,
            gamma=config.gamma,
            num_envs=config.num_envs,
        )

        # ------------------------------------------------------------------ #
        # Counters                                                             #
        # ------------------------------------------------------------------ #

        self.train_steps: int = 0

    # ---------------------------------------------------------------------- #
    # Properties                                                               #
    # ---------------------------------------------------------------------- #

    @property
    def alpha(self) -> float:
        """Current value of the entropy coefficient α (always > 0)."""
        return float(self.log_alpha.exp().item())

    # ---------------------------------------------------------------------- #
    # Action selection (numpy interface for environment interaction)           #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Return a tanh-squashed action ∈ [-1, 1]^action_dim (numpy).

        Args:
            state:         Pure state vector of shape (state_dim,).
            goal:          Goal vector of shape (goal_dim,).
            deterministic: If True, return the mean action without noise.

        Returns:
            action: (action_dim,) numpy float32 array.
        """
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
        g = torch.FloatTensor(goal).unsqueeze(0).to(self.device)  # (1, goal_dim)

        self.actor.eval()
        if deterministic:
            action = self.actor.deterministic_action(s, g)  # (1, action_dim)
        else:
            action, _ = self.actor.sample(s, g)  # (1, action_dim)
        self.actor.train()

        return action.squeeze(0).cpu().numpy()  # (action_dim,)

    @torch.no_grad()
    def select_action_batch(
        self,
        states: np.ndarray,
        goals: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Batch action selection for multiple environments.

        Args:
            states:        (N, state_dim) float32 array.
            goals:         (N, goal_dim)  float32 array.
            deterministic: If True, return mean actions without noise.

        Returns:
            actions: (N, action_dim) numpy float32 array.
        """
        s = torch.FloatTensor(states).to(self.device)  # (N, state_dim)
        g = torch.FloatTensor(goals).to(self.device)  # (N, goal_dim)

        self.actor.eval()
        if deterministic:
            actions = self.actor.deterministic_action(s, g)  # (N, action_dim)
        else:
            actions, _ = self.actor.sample(s, g)  # (N, action_dim)
        self.actor.train()

        return actions.cpu().numpy()  # (N, action_dim)

    # ---------------------------------------------------------------------- #
    # Training step                                                            #
    # ---------------------------------------------------------------------- #

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one full gradient update across all three sub-objectives.

        Returns ``None`` if the replay buffer does not yet contain enough
        episodes to start training (controlled by
        ``config.min_replay_episodes``).  Otherwise returns a metrics dict
        with scalar losses and diagnostics.

        Training sub-steps (in order):
            1. **Critic** — InfoNCE loss on (B×B) logit matrix, plus a
               logsumexp² regularisation penalty.
            2. **Actor** — SAC entropy-regularised policy gradient using
               critic values computed with full gradient flow through the
               encoders, so the actor can learn from the Q signal.
            3. **Alpha** — Dual update on the temperature log α to enforce
               the target entropy constraint.

        Returns:
            Dict with keys:
                - ``critic_loss``   : InfoNCE loss (+ penalty).
                - ``actor_loss``    : Policy gradient loss.
                - ``alpha_loss``    : Temperature dual loss.
                - ``alpha``         : Current entropy coefficient value.
                - ``log_prob_mean`` : Mean log-probability of sampled actions.
                - ``logsumexp_mean``: Mean of logsumexp(logits) row — useful
                  for diagnosing score collapse.
        """
        if self.replay_buffer.num_episodes < self.config.min_replay_episodes:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return None

        # Move batch tensors to device.
        obs_t = torch.FloatTensor(batch["obs"]).to(self.device)  # (B, state_dim)
        act_t = torch.FloatTensor(batch["action"]).to(self.device)  # (B, action_dim)
        goal_t = torch.FloatTensor(batch["goal"]).to(self.device)  # (B, goal_dim)

        # ------------------------------------------------------------------ #
        # 1. Critic loss (InfoNCE + logsumexp² regularisation)               #
        # ------------------------------------------------------------------ #

        # Encode (state, action) pairs → φ(s_i, a_i), shape (B, embed_dim).
        sa_repr = self.sa_encoder(obs_t, act_t)
        # Encode goals → ψ(g_j), shape (B, embed_dim).
        g_repr = self.g_encoder(goal_t)

        # Pairwise squared L2 differences: (B, B, embed_dim).
        # logits[i, j] = Q(s_i, a_i, g_j) = -||φ_i - ψ_j||₂
        diff = sa_repr.unsqueeze(1) - g_repr.unsqueeze(0)  # (B, B, embed_dim)
        # Add a small ε inside the sqrt for numerical stability.
        logits = -torch.sqrt((diff**2).sum(dim=-1) + 1e-8)  # (B, B)

        # InfoNCE: diagonal entries are the positives (s_i, a_i) → g_i.
        # loss = -mean( logits[i,i] - logsumexp_j( logits[i,j] ) )
        logsumexp_val = torch.logsumexp(logits, dim=1)  # (B,)
        nce_loss = -torch.mean(torch.diag(logits) - logsumexp_val)

        # Logsumexp² regularisation — prevents the scores from collapsing to
        # uniformly large values which would satisfy InfoNCE trivially.
        lse_penalty = self.config.logsumexp_penalty_coeff * torch.mean(logsumexp_val**2)

        critic_loss = nce_loss + lse_penalty

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------------------------------------ #
        # 2. Actor loss (SAC entropy-regularised policy gradient)             #
        # ------------------------------------------------------------------ #

        # Sample fresh actions from the current policy.
        action_new, log_prob = self.actor.sample(obs_t, goal_t)  # (B, action_dim), (B,)

        # Q-values for the freshly sampled actions.  Gradients flow through
        # the encoders so the actor can learn from the Q signal.  Any encoder
        # gradients accumulated here sit in .grad until they are zeroed by
        # critic_optimizer.zero_grad(set_to_none=True) at the start of the
        # next critic step, so they never corrupt a critic update.
        sa_repr_new = self.sa_encoder(obs_t, action_new)  # (B, embed_dim)
        g_repr_new = self.g_encoder(goal_t)  # (B, embed_dim)

        # Diagonal distances: Q(s_i, π(s_i,g_i), g_i), shape (B,).
        q_values = -torch.sqrt(
            ((sa_repr_new - g_repr_new) ** 2).sum(dim=-1) + 1e-8
        )  # (B,)

        # Actor objective: maximise E[Q - α·log π] ≡ minimise E[α·log π - Q].
        actor_loss = torch.mean(self.log_alpha.exp().detach() * log_prob - q_values)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------------------------------------ #
        # 3. Alpha (entropy coefficient) loss                                 #
        # ------------------------------------------------------------------ #

        # Dual gradient ascent: increase α if entropy < target, else decrease.
        # log α · (-log π - target_entropy)  →  minimised w.r.t. log α.
        alpha_loss = torch.mean(
            self.log_alpha.exp() * (-log_prob.detach() - self.target_entropy)
        )

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ------------------------------------------------------------------ #
        # Book-keeping                                                         #
        # ------------------------------------------------------------------ #

        self.train_steps += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": self.alpha,
            "log_prob_mean": float(log_prob.mean().item()),
            "logsumexp_mean": float(logsumexp_val.mean().item()),
        }

    # ---------------------------------------------------------------------- #
    # Persistence                                                              #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Serialise all learnable state to a single checkpoint file.

        Args:
            path: Destination file path (parent directories are created
                  automatically if they do not exist).
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        torch.save(
            {
                "sa_encoder": self.sa_encoder.state_dict(),
                "g_encoder": self.g_encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "log_alpha": self.log_alpha.data,
                "train_steps": self.train_steps,
                "config": self.config,
            },
            path,
        )
        print(f"[CRLAgent] Saved checkpoint → {path}")

    def load(self, path: str) -> None:
        """Restore all learnable state from a checkpoint file.

        The checkpoint must have been produced by :meth:`save`.  Mismatched
        architecture configs will raise an error at ``load_state_dict`` time.

        Args:
            path: Path to the ``.pt`` checkpoint file.
        """
        ckpt = torch.load(path, map_location=self.device)

        self.sa_encoder.load_state_dict(ckpt["sa_encoder"])
        self.g_encoder.load_state_dict(ckpt["g_encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        self.train_steps = int(ckpt.get("train_steps", 0))

        print(f"[CRLAgent] Loaded checkpoint ← {path} (train_steps={self.train_steps})")

    # ---------------------------------------------------------------------- #
    # Diagnostics                                                              #
    # ---------------------------------------------------------------------- #

    def param_count(self) -> Dict[str, int]:
        """Return the trainable parameter counts for each sub-network.

        Returns:
            Dict with keys ``'sa_encoder'``, ``'g_encoder'``, ``'actor'``,
            and ``'total'``.
        """

        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "sa_encoder": _count(self.sa_encoder),
            "g_encoder": _count(self.g_encoder),
            "actor": _count(self.actor),
        }
        counts["total"] = counts["sa_encoder"] + counts["g_encoder"] + counts["actor"]
        return counts

    def __repr__(self) -> str:
        pc = self.param_count()
        return (
            f"CRLAgent("
            f"state_dim={self.config.state_dim}, "
            f"action_dim={self.config.action_dim}, "
            f"goal_dim={self.config.goal_dim}, "
            f"embed_dim={self.config.embed_dim}, "
            f"total_params={pc['total']:,}, "
            f"train_steps={self.train_steps}, "
            f"device={self.device}"
            f")"
        )
