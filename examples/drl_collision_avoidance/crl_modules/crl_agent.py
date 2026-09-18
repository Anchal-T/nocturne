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
from nocturne.utils.distributed import (
    DistInfo,
    all_reduce_scalar,
    dummy_loss,
    unwrap_module,
    wrap_ddp,
)


def _reset_module_params(module: nn.Module) -> None:
    """Re-apply LeCun init for Linear leaves; identity for LayerNorm."""
    if isinstance(module, nn.Linear):
        from .encoders import lecun_init_

        lecun_init_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


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
    find_unused_parameters: bool = False
    # Floor α≈0.08 (exp(-2.5)); prevents entropy collapse that freezes InfoNCE.
    log_alpha_min: float = -2.5
    log_alpha_max: float = 2.0
    grad_clip_norm: float = 10.0
    # HER buffer is never checkpointed; stale Adam moments on a fresh buffer
    # immediately drive InfoNCE to ln(B). Reset critic/α optimisers on load.
    reset_critic_optimizer_on_load: bool = True
    # Also re-init SA/G weights on load. A good policy refill produces
    # near-homogeneous HER goals; fine-tuning the old metric collapses to
    # ln(B) within ~100 steps. Actor + α are kept. From-scratch critic
    # re-learns (as in the original 0→30k run).
    reinit_critic_on_load: bool = True
    # Extra: if α was already collapsed in the checkpoint, same SA/G re-init.
    reinit_critic_on_collapsed_alpha: bool = False
    # After SA/G re-init, hold λ=0 for this many critic steps so InfoNCE can
    # form a metric before logsumexp² pulls all logits toward 0 (→ ln(B) trap).
    lse_penalty_warmup_steps: int = 5000
    # During buffer refill after resume, mix uniform random actions so HER
    # goals stay diverse (a strong actor alone yields near-homogeneous futures).
    resume_random_action_prob: float = 0.5
    # Once the HER buffer first hits min_replay after a resume+reinit, run this
    # many consecutive train_steps before returning to the cheap 1-step/ep
    # schedule. Empirically ~200 steps moves InfoNCE off ln(B); live 1/ep
    # alone stayed pinned.
    critic_burst_steps_on_resume: int = 2000


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

    def __init__(self, config: CRLAgentConfig, dist_info=None) -> None:
        self.config = config
        self.dist_info = dist_info or DistInfo(device=torch.device(config.device))

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

        find_unused = bool(getattr(config, "find_unused_parameters", True))
        self.sa_encoder = wrap_ddp(
            self.sa_encoder, self.dist_info, find_unused_parameters=find_unused)
        self.g_encoder = wrap_ddp(
            self.g_encoder, self.dist_info, find_unused_parameters=find_unused)
        self.actor = wrap_ddp(
            self.actor, self.dist_info, find_unused_parameters=find_unused)

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
        # When >0, logsumexp penalty is disabled (set on critic re-init).
        self._lse_warmup_remaining: int = 0
        # True after loading a resume checkpoint (enables random-action mix).
        self._resume_buffer_warmup: bool = False
        # Set on load; cleared after the one-shot critic burst.
        self._pending_critic_burst: bool = False

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
        actor = unwrap_module(self.actor)
        if deterministic:
            action = actor.deterministic_action(s, g)  # (1, action_dim)
        else:
            action, _ = actor.sample(s, g)  # (1, action_dim)
        self.actor.train()

        return action.squeeze(0).cpu().numpy()  # (action_dim,)

    def _sample_actor(self, state, goal):
        """Sample tanh-squashed actions using the DDP-wrapped actor forward."""
        mean, log_std = self.actor(state, goal)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        x_t = mean + std * eps
        action = torch.tanh(x_t)
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(x_t)
            - torch.log(1.0 - action.pow(2) + 1e-5)
        ).sum(dim=-1)
        return action, log_prob

    @torch.inference_mode()
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
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        g = torch.as_tensor(goals, dtype=torch.float32, device=self.device)
        actor = unwrap_module(self.actor)
        if deterministic:
            actions = actor.deterministic_action(s, g)
        else:
            actions, _ = actor.sample(s, g)
        return actions.cpu().numpy()

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

        Under DDP, every rank participates in the same number of backwards
        even when a local HER buffer is not ready (dummy loss). Negatives
        are rank-local; gradients are averaged.
        """
        ready = (
            self.replay_buffer.num_episodes >= self.config.min_replay_episodes
        )
        batch = self.replay_buffer.sample(self.config.batch_size) if ready else None
        ready = ready and batch is not None
        if self.dist_info.is_distributed:
            global_ready = all_reduce_scalar(
                1.0 if ready else 0.0, self.device, op="min"
            )
            if global_ready < 1.0:
                loss = (
                    dummy_loss(self.sa_encoder)
                    + dummy_loss(self.g_encoder)
                    + dummy_loss(self.actor)
                )
                loss.backward()
                self.critic_optimizer.zero_grad(set_to_none=True)
                self.actor_optimizer.zero_grad(set_to_none=True)
                return None
        elif not ready:
            return None

        # Move batch tensors to device.
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        goal_t = torch.as_tensor(batch["goal"], dtype=torch.float32, device=self.device)

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
        # Held at 0 for a few thousand steps after SA/G re-init so the metric
        # can form before λ·lse² dominates and drives logits → −ln(B).
        if self._lse_warmup_remaining > 0:
            lse_coeff = 0.0
            self._lse_warmup_remaining -= 1
        else:
            lse_coeff = self.config.logsumexp_penalty_coeff
        lse_penalty = lse_coeff * torch.mean(logsumexp_val**2)

        critic_loss = nce_loss + lse_penalty

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters()),
            self.config.grad_clip_norm,
        )
        self.critic_optimizer.step()

        # ------------------------------------------------------------------ #
        # 2. Actor loss (SAC entropy-regularised policy gradient)             #
        # ------------------------------------------------------------------ #

        # Sample fresh actions from the current policy.
        # Sample fresh actions from the current policy via DDP-wrapped forward.
        action_new, log_prob = self._sample_actor(obs_t, goal_t)

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
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.grad_clip_norm
        )
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
        with torch.no_grad():
            self.log_alpha.clamp_(
                self.config.log_alpha_min, self.config.log_alpha_max
            )
        if self.dist_info.is_distributed:
            import torch.distributed as dist
            dist.broadcast(self.log_alpha.data, src=0)

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

    def save(self, path: str, episodes_completed: int = 0) -> None:
        """Serialise all learnable state to a single checkpoint file.

        Args:
            path: Destination file path (parent directories are created
                  automatically if they do not exist).
            episodes_completed: Episode count to restore on resume.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload = {
            "sa_encoder": unwrap_module(self.sa_encoder).state_dict(),
            "g_encoder": unwrap_module(self.g_encoder).state_dict(),
            "actor": unwrap_module(self.actor).state_dict(),
            "log_alpha": self.log_alpha.data,
            "train_steps": self.train_steps,
            "episodes_completed": int(episodes_completed),
            "config": self.config,
            "world_size": self.dist_info.world_size,
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }
        torch.save(payload, path)
        if self.dist_info.is_rank0:
            print(f"[CRLAgent] Saved checkpoint → {path}")

    def load(self, path: str) -> int:
        """Restore all learnable state from a checkpoint file.

        The checkpoint must have been produced by :meth:`save`.  Mismatched
        architecture configs will raise an error at ``load_state_dict`` time.

        Args:
            path: Path to the ``.pt`` checkpoint file.

        Returns:
            Episode count stored in the checkpoint, or parsed from
            ``crl_epN.pth`` if the file predates that field.
        """
        ckpt = torch.load(path, map_location=self.device)

        unwrap_module(self.sa_encoder).load_state_dict(ckpt["sa_encoder"])
        unwrap_module(self.g_encoder).load_state_dict(ckpt["g_encoder"])
        unwrap_module(self.actor).load_state_dict(ckpt["actor"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        self.train_steps = int(ckpt.get("train_steps", 0))

        # Actor optimiser transfers fine; critic/α moments do not — the HER
        # buffer is wiped on resume, so restoring Adam state drives InfoNCE
        # to ln(B) within a few dozen steps (seen on every prior 30k resume).
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if (
            not self.config.reset_critic_optimizer_on_load
            and "critic_optimizer" in ckpt
        ):
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        if (
            not self.config.reset_critic_optimizer_on_load
            and "alpha_optimizer" in ckpt
        ):
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        if self.config.reset_critic_optimizer_on_load and self.dist_info.is_rank0:
            print(
                "[CRLAgent] Reset critic/α Adam state on load "
                "(HER buffer not restored)"
            )

        # Clamp to floor/ceiling — never snap α back to 1.0 (that caused instability).
        alpha_before = float(self.log_alpha.exp().item())
        with torch.no_grad():
            self.log_alpha.clamp_(
                self.config.log_alpha_min, self.config.log_alpha_max
            )
        alpha_after = float(self.log_alpha.exp().item())
        if abs(alpha_before - alpha_after) > 1e-6 and self.dist_info.is_rank0:
            print(
                f"[CRLAgent] Clamped α {alpha_before:.4f} → {alpha_after:.4f} "
                f"(floor={self.config.log_alpha_min:.2f})"
            )

        should_reinit_critic = bool(self.config.reinit_critic_on_load) or (
            self.config.reinit_critic_on_collapsed_alpha and alpha_before < 0.05
        )
        if should_reinit_critic:
            unwrap_module(self.sa_encoder).apply(_reset_module_params)
            unwrap_module(self.g_encoder).apply(_reset_module_params)
            # Shrink output projections so initial −L2 logits are not huge;
            # otherwise λ·lse² dominates before InfoNCE can learn.
            for enc in (self.sa_encoder, self.g_encoder):
                out = unwrap_module(enc).output_proj
                with torch.no_grad():
                    out.weight.mul_(0.1)
                    if out.bias is not None:
                        out.bias.mul_(0.1)
            self.critic_optimizer = torch.optim.Adam(
                list(self.sa_encoder.parameters())
                + list(self.g_encoder.parameters()),
                lr=self.config.critic_lr,
            )
            self._lse_warmup_remaining = int(self.config.lse_penalty_warmup_steps)
            if self.dist_info.is_rank0:
                reason = (
                    "resume without HER buffer"
                    if self.config.reinit_critic_on_load
                    else "collapsed α on load"
                )
                print(
                    f"[CRLAgent] Re-initialised SA/G encoders + critic optimiser "
                    f"({reason}; lse warmup={self._lse_warmup_remaining} steps)"
                )

        self._resume_buffer_warmup = True
        self._pending_critic_burst = bool(
            self.config.reinit_critic_on_load
            or self.config.reinit_critic_on_collapsed_alpha
        ) and int(self.config.critic_burst_steps_on_resume) > 0

        episodes_completed = int(ckpt.get("episodes_completed", 0))
        if episodes_completed <= 0:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem.startswith("crl_ep"):
                suffix = stem[len("crl_ep") :]
                if suffix.isdigit():
                    episodes_completed = int(suffix)

        if self.dist_info.is_rank0:
            print(
                f"[CRLAgent] Loaded checkpoint ← {path} "
                f"(train_steps={self.train_steps}, "
                f"episodes={episodes_completed})"
            )
        return episodes_completed

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
