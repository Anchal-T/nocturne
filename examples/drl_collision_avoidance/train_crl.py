"""
CRL training script for nocturne collision avoidance.

Trains a CRLAgent (Contrastive RL + HER) using self-supervised goal-conditioned
learning.  No hand-crafted reward is used; the agent learns to reach goals via
the InfoNCE contrastive objective applied to HER-relabelled episodes.

Usage:
    cd nocturne
    python -m examples.drl_collision_avoidance.train_crl \
        scenario_path=/path/to/scenarios \
        crl.critic_depth=64

To run depth-scaling experiments, use the preset configs:
    python -m examples.drl_collision_avoidance.train_crl \
        --config-name crl_config_depth64 scenario_path=...
"""

from __future__ import annotations

import os
import signal
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg_dict(cfg) -> Dict[str, Any]:
    if hasattr(cfg, "_content"):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def _make_crl_env(cfg_dict: Dict[str, Any], env_index: int):
    """Factory for a CRLCollisionAvoidanceEnv with a unique seed."""
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from examples.drl_collision_avoidance.crl_collision_avoidance_env import (
        CRLCollisionAvoidanceEnv,
    )

    env_cfg = dict(cfg_dict)
    env_cfg["seed"] = int(cfg_dict.get("seed", 42)) + env_index * 1000
    env_cfg.setdefault("single_agent_mode", True)
    env_cfg.setdefault("max_num_vehicles", 1)
    return CRLCollisionAvoidanceEnv(env_cfg)


def _resolve_device(cfg_dict: Dict[str, Any]) -> str:
    device = str(cfg_dict.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[train_crl] CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return device


def _print_header(config, agent) -> None:
    counts = agent.param_count()
    print("\n" + "=" * 60)
    print("CRL Training — Scaling CRL (Wang et al. 2025)")
    print(f"  critic_depth : {config.critic_depth}")
    print(f"  actor_depth  : {config.actor_depth}")
    print(f"  network_width: {config.network_width}")
    print(f"  embed_dim    : {config.embed_dim}")
    print(f"  batch_size   : {config.batch_size}")
    print(f"  state_dim    : {config.state_dim}")
    print(f"  goal_dim     : {config.goal_dim}")
    print(f"  action_dim   : {config.action_dim}")
    print(
        f"  Parameters   : SA={counts['sa_encoder']:,}  G={counts['g_encoder']:,}  "
        f"Actor={counts['actor']:,}  Total={counts['total']:,}"
    )
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _run_crl_training(
    envs: List,
    agent,
    crl_cfg: Dict[str, Any],
    writer,
    running_state: Dict,
) -> None:
    """Main episode-based CRL training loop.

    For each episode:
      1. Collect a complete episode using the stochastic actor.
      2. Store every step in the HER replay buffer.
      3. After the episode, run ``gradient_steps_per_episode`` train steps.
    """
    total_episodes = int(crl_cfg["total_episodes"])
    num_envs = len(envs)
    log_interval = int(crl_cfg["log_interval"])
    save_interval = int(crl_cfg["save_interval"])
    checkpoint_dir = str(crl_cfg["checkpoint_dir"])
    gradient_steps_per_ep = int(crl_cfg.get("gradient_steps_per_episode", 10))
    min_replay_episodes = int(crl_cfg.get("min_replay_episodes", 200))

    os.makedirs(checkpoint_dir, exist_ok=True)

    episodes_completed = 0
    global_step = 0
    start_time = time.time()

    # Rolling statistics windows
    reward_window: Deque[float] = deque(maxlen=log_interval)
    goal_window: Deque[float] = deque(maxlen=log_interval)
    collision_window: Deque[float] = deque(maxlen=log_interval)
    ep_length_window: Deque[int] = deque(maxlen=log_interval)
    latest_metrics: Optional[Dict[str, float]] = None

    print("Collecting initial episodes to fill HER buffer ...\n")

    while episodes_completed < total_episodes and running_state["running"]:
        env = envs[episodes_completed % num_envs]
        env_id = episodes_completed % num_envs

        # ---- Episode collection ----------------------------------------
        obs, _ = env.reset()
        state, goal = env.get_state_goal(obs)
        x, y, cos_h, sin_h = env.get_ego_info()

        ep_steps = 0
        ep_goal = False
        ep_collided = False
        done = False

        while not done and running_state["running"]:
            action = agent.select_action(state, goal)

            next_obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state, next_goal = env.get_state_goal(next_obs)

            next_x = info["ego_x"]
            next_y = info["ego_y"]
            next_cos_h = info["ego_cos_h"]
            next_sin_h = info["ego_sin_h"]

            ego_info = np.array([x, y, cos_h, sin_h], dtype=np.float32)
            agent.replay_buffer.add(
                state=state,
                action=action,
                ego_info=ego_info,
                done=done,
                env_id=env_id,
            )

            state = next_state
            goal = next_goal
            x, y, cos_h, sin_h = next_x, next_y, next_cos_h, next_sin_h

            ep_steps += 1
            global_step += 1
            ep_goal = bool(info.get("goal_achieved", False))
            ep_collided = bool(info.get("collided", False))

        episodes_completed += 1
        reward_window.append(0.0)  # CRL has no env reward; placeholder
        goal_window.append(float(ep_goal))
        collision_window.append(float(ep_collided))
        ep_length_window.append(ep_steps)

        # ---- Gradient steps --------------------------------------------
        if agent.replay_buffer.num_episodes >= min_replay_episodes:
            for _ in range(gradient_steps_per_ep):
                metrics = agent.train_step()
                if metrics is not None:
                    latest_metrics = metrics

        # ---- Logging ---------------------------------------------------
        if episodes_completed % log_interval == 0:
            elapsed = max(time.time() - start_time, 1e-3)
            eps_per_s = episodes_completed / elapsed
            goal_rate = float(np.mean(goal_window)) if goal_window else 0.0
            coll_rate = float(np.mean(collision_window)) if collision_window else 0.0
            avg_ep_len = float(np.mean(ep_length_window)) if ep_length_window else 0.0

            loss_str = ""
            if latest_metrics:
                loss_str = (
                    f" | critic={latest_metrics['critic_loss']:.4f}"
                    f" | actor={latest_metrics['actor_loss']:.4f}"
                    f" | alpha={latest_metrics['alpha']:.4f}"
                    f" | lse={latest_metrics['logsumexp_mean']:.2f}"
                )

            print(
                f"Ep {episodes_completed:6d} | "
                f"goal_rate={goal_rate:.3f} | "
                f"coll_rate={coll_rate:.3f} | "
                f"avg_len={avg_ep_len:.1f} | "
                f"buf_eps={agent.replay_buffer.num_episodes:5d} | "
                f"train_steps={agent.train_steps:6d}"
                f"{loss_str} | "
                f"eps/s={eps_per_s:.2f}"
            )

            if writer is not None and latest_metrics is not None:
                for k, v in latest_metrics.items():
                    writer.add_scalar(f"train/{k}", v, episodes_completed)
                writer.add_scalar("train/goal_rate", goal_rate, episodes_completed)
                writer.add_scalar("train/collision_rate", coll_rate, episodes_completed)
                writer.add_scalar("train/avg_ep_length", avg_ep_len, episodes_completed)
                writer.add_scalar(
                    "train/buffer_episodes",
                    agent.replay_buffer.num_episodes,
                    episodes_completed,
                )
                writer.add_scalar("train/global_steps", global_step, episodes_completed)

        # ---- Checkpoint ------------------------------------------------
        if episodes_completed % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"crl_ep{episodes_completed}.pth")
            agent.save(ckpt_path)

    print(f"\nTraining complete. {episodes_completed} episodes, {global_step} steps.")
    final_path = os.path.join(checkpoint_dir, "crl_final.pth")
    agent.save(final_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _make_writer(cfg_dict):
    try:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = os.path.join(
            cfg_dict.get("crl", {}).get("checkpoint_dir", "checkpoints/crl"),
            "tb_logs",
        )
        return SummaryWriter(log_dir=log_dir)
    except ImportError:
        return None


@hydra.main(config_path="../../cfgs/drl_collision_avoidance", config_name="crl_config")
def main(cfg) -> None:
    from examples.drl_collision_avoidance.crl_modules.crl_agent import (
        CRLAgent,
        CRLAgentConfig,
    )

    cfg_dict = _make_cfg_dict(cfg)
    crl_cfg = cfg_dict["crl"]
    device = _resolve_device(cfg_dict)

    running_state = {"running": True}

    def _signal_handler(sig, frame):
        print("\n[train_crl] Interrupted, stopping ...")
        running_state["running"] = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # -- Build one env to probe dimensions --
    probe_env = _make_crl_env(cfg_dict, 0)
    _obs_probe, _ = probe_env.reset()
    state_dim = probe_env.state_dim
    goal_dim = probe_env.goal_dim
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()
    del probe_env

    num_envs = int(crl_cfg.get("num_envs", 8))

    # -- Build CRLAgent --
    agent_config = CRLAgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        critic_depth=int(crl_cfg.get("critic_depth", 16)),
        actor_depth=int(crl_cfg.get("actor_depth", 16)),
        network_width=int(crl_cfg.get("network_width", 256)),
        embed_dim=int(crl_cfg.get("embed_dim", 64)),
        critic_lr=float(crl_cfg.get("critic_lr", 3e-4)),
        actor_lr=float(crl_cfg.get("actor_lr", 3e-4)),
        alpha_lr=float(crl_cfg.get("alpha_lr", 3e-4)),
        batch_size=int(crl_cfg.get("batch_size", 512)),
        gamma=float(crl_cfg.get("gamma", 0.99)),
        target_entropy_factor=float(crl_cfg.get("target_entropy_factor", 0.5)),
        logsumexp_penalty_coeff=float(crl_cfg.get("logsumexp_penalty_coeff", 0.1)),
        her_max_episodes=int(crl_cfg.get("her_max_episodes", 50_000)),
        min_replay_episodes=int(crl_cfg.get("min_replay_episodes", 200)),
        num_envs=num_envs,
        device=device,
    )
    agent = CRLAgent(agent_config)
    _print_header(agent_config, agent)

    # -- Build environments --
    envs = [_make_crl_env(cfg_dict, i) for i in range(num_envs)]
    writer = _make_writer(cfg_dict)

    # -- Optional: resume from checkpoint --
    resume = crl_cfg.get("resume_checkpoint", None)
    if resume:
        agent.load(resume)

    try:
        _run_crl_training(envs, agent, crl_cfg, writer, running_state)
    finally:
        for env in envs:
            env.close()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
