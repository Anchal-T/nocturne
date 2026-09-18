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
from typing import Any, Deque, Dict, Optional

import hydra
import numpy as np
import torch
from cfgs.config import set_display_window
from examples.drl_collision_avoidance.scenario_utils import (
    apply_scenario_path_defaults,
)
from examples.drl_collision_avoidance.vec_env import DummyVecEnv, SubprocVecEnv
from nocturne.utils.distributed import (
    all_reduce_scalar,
    cleanup_distributed,
    init_distributed,
    rank_offset_seed,
    reduce_metrics,
)

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


def _ego_from_info(info: Optional[Dict[str, Any]]) -> np.ndarray:
    info = info or {}
    return np.array(
        [
            float(info.get("ego_x", 0.0)),
            float(info.get("ego_y", 0.0)),
            float(info.get("ego_cos_h", 1.0)),
            float(info.get("ego_sin_h", 0.0)),
        ],
        dtype=np.float32,
    )


def _make_vec_env(cfg_dict: Dict[str, Any], num_envs: int, mode: str):
    env_fns = [lambda i=i: _make_crl_env(cfg_dict, i) for i in range(num_envs)]
    if num_envs <= 1 or str(mode).lower() in ("dummy", "sequential"):
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def _run_crl_training(
    vec_env,
    agent,
    crl_cfg: Dict[str, Any],
    writer,
    running_state: Dict,
    state_dim: int,
    start_episode: int = 0,
) -> None:
    """Collect from rank-local envs, then train.

    ``vec_env`` steps environments together. Subprocess mode overlaps C++
    simulation across CPU cores instead of running them one after another.
    """
    total_episodes = int(crl_cfg["total_episodes"])
    num_envs = int(vec_env.n_envs)
    log_interval = int(crl_cfg["log_interval"])
    save_interval = int(crl_cfg["save_interval"])
    checkpoint_dir = str(crl_cfg["checkpoint_dir"])
    gradient_steps_per_ep = int(crl_cfg.get("gradient_steps_per_episode", 10))
    max_grad_steps = int(crl_cfg.get("max_grad_steps_per_iter", 8))
    dist_info = getattr(agent, "dist_info", None)
    is_rank0 = dist_info is None or dist_info.is_rank0
    is_distributed = dist_info is not None and dist_info.is_distributed
    world_size = int(getattr(dist_info, "world_size", 1) or 1)

    if is_rank0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    episodes_completed = int(start_episode)
    local_new_episodes = 0
    global_step = 0
    start_time = time.time()

    reward_window: Deque[float] = deque(maxlen=log_interval)
    goal_window: Deque[float] = deque(maxlen=log_interval)
    collision_window: Deque[float] = deque(maxlen=log_interval)
    ep_length_window: Deque[int] = deque(maxlen=log_interval)
    latest_metrics: Optional[Dict[str, float]] = None

    if is_rank0:
        if episodes_completed:
            print(
                f"Resuming at episode {episodes_completed}/{total_episodes} "
                f"({num_envs} envs/rank × {world_size} ranks, "
                f"{gradient_steps_per_ep} grad steps/ep)\n"
            )
        print("Collecting initial episodes to fill HER buffer ...\n")

    obs = vec_env.reset()
    reset_infos = getattr(vec_env, "last_reset_infos", [{} for _ in range(num_envs)])
    ep_steps = np.zeros(num_envs, dtype=np.int32)
    ep_goal = np.zeros(num_envs, dtype=bool)
    ep_collided = np.zeros(num_envs, dtype=bool)
    ego = np.stack([_ego_from_info(info) for info in reset_infos])
    finished_this_step = 0

    while running_state["running"]:
        local_done = episodes_completed >= total_episodes
        if is_distributed:
            all_done = all_reduce_scalar(
                1.0 if local_done else 0.0,
                agent.device,
                op="min",
            )
            if all_done >= 1.0:
                break
        elif local_done:
            break

        if not local_done:
            states = obs[:, :state_dim]
            goals = obs[:, state_dim:]
            actions = agent.select_action_batch(states, goals)
            # After resume the HER buffer is empty; a strong actor alone fills
            # it with near-homogeneous successes and InfoNCE cannot learn.
            # Mix uniform random actions until min_replay is reached.
            rand_p = float(getattr(agent.config, "resume_random_action_prob", 0.0))
            if (
                getattr(agent, "_resume_buffer_warmup", False)
                and agent.replay_buffer.num_episodes
                < int(crl_cfg.get("min_replay_episodes", 2000))
                and rand_p > 0.0
            ):
                mask = np.random.rand(num_envs) < rand_p
                if mask.any():
                    actions = actions.copy()
                    actions[mask] = np.random.uniform(
                        -1.0, 1.0, size=(int(mask.sum()), actions.shape[1])
                    ).astype(np.float32)
            elif (
                getattr(agent, "_resume_buffer_warmup", False)
                and agent.replay_buffer.num_episodes
                >= int(crl_cfg.get("min_replay_episodes", 2000))
            ):
                agent._resume_buffer_warmup = False
            vec_env.step_async(actions)

        # GPU updates from the previous vec-step run while workers simulate.
        local_finished = float(finished_this_step)
        if is_distributed:
            synced_finished = int(all_reduce_scalar(
                local_finished, agent.device, op="max"))
        else:
            synced_finished = finished_this_step
        train_count = min(
            max_grad_steps, gradient_steps_per_ep * synced_finished
        )
        for _ in range(train_count):
            metrics = agent.train_step()
            if metrics is not None:
                latest_metrics = metrics

        # One-shot critic burst after resume: consecutive steps move InfoNCE
        # off ln(B) (offline: ~200 suffice). Live 1-step/ep stayed pinned.
        burst_n = int(crl_cfg.get("critic_burst_steps_on_resume", 0) or 0)
        if (
            burst_n > 0
            and getattr(agent, "_pending_critic_burst", False)
            and agent.replay_buffer.num_episodes
            >= int(crl_cfg.get("min_replay_episodes", 2000))
        ):
            if is_rank0:
                print(
                    f"[train_crl] Critic burst: {burst_n} consecutive train "
                    f"steps (buf={agent.replay_buffer.num_episodes}) ..."
                )
            for i in range(burst_n):
                metrics = agent.train_step()
                if metrics is not None:
                    latest_metrics = metrics
                if (
                    is_rank0
                    and metrics is not None
                    and (i + 1) % 500 == 0
                ):
                    print(
                        f"  burst {i+1}/{burst_n} "
                        f"critic={metrics['critic_loss']:.4f} "
                        f"alpha={metrics['alpha']:.4f}"
                    )
            agent._pending_critic_burst = False
            if is_rank0 and latest_metrics is not None:
                print(
                    f"[train_crl] Critic burst done: "
                    f"critic={latest_metrics['critic_loss']:.4f} "
                    f"alpha={latest_metrics['alpha']:.4f}"
                )

        if latest_metrics is not None and is_distributed:
            latest_metrics = reduce_metrics(latest_metrics, agent.device)

        if not local_done:
            next_obs, _rewards, dones, infos = vec_env.step_wait()
            finished_this_step = 0

            for env_id in range(num_envs):
                info = infos[env_id]
                done = bool(dones[env_id])
                agent.replay_buffer.add(
                    state=states[env_id],
                    action=actions[env_id],
                    ego_info=np.array(ego[env_id], dtype=np.float32, copy=True),
                    done=done,
                    env_id=env_id,
                )
                ep_steps[env_id] += 1
                global_step += 1
                ep_goal[env_id] = bool(info.get("goal_achieved", False))
                ep_collided[env_id] = bool(info.get("collided", False))
                if done:
                    local_new_episodes += 1
                    finished_this_step += 1
                    reward_window.append(0.0)
                    goal_window.append(float(ep_goal[env_id]))
                    collision_window.append(float(ep_collided[env_id]))
                    ep_length_window.append(int(ep_steps[env_id]))
                    ep_steps[env_id] = 0
                    ep_goal[env_id] = False
                    ep_collided[env_id] = False
                    ego[env_id] = _ego_from_info(info.get("_reset_info"))
                else:
                    ego[env_id] = _ego_from_info(info)
            obs = next_obs

        prev_episodes = episodes_completed
        if is_distributed:
            global_new = int(
                all_reduce_scalar(float(local_new_episodes), agent.device, op="sum")
            )
            episodes_completed = start_episode + global_new
        else:
            episodes_completed = start_episode + local_new_episodes
        running_state["episodes"] = episodes_completed

        crossed_log = (
            finished_this_step > 0
            and log_interval > 0
            and (episodes_completed // log_interval) > (prev_episodes // log_interval)
        )
        if crossed_log:
            elapsed = max(time.time() - start_time, 1e-3)
            collected = max(episodes_completed - start_episode, 0)
            eps_per_s = collected / elapsed
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

            if is_rank0:
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

        crossed_save = (
            finished_this_step > 0
            and save_interval > 0
            and (episodes_completed // save_interval) > (prev_episodes // save_interval)
        )
        if is_rank0 and crossed_save:
            ckpt_path = os.path.join(checkpoint_dir, f"crl_ep{episodes_completed}.pth")
            agent.save(ckpt_path, episodes_completed=episodes_completed)
            latest_path = os.path.join(checkpoint_dir, "crl_latest.pth")
            agent.save(latest_path, episodes_completed=episodes_completed)

    if is_rank0:
        print(f"\nTraining complete. {episodes_completed} episodes, {global_step} steps.")
        final_path = os.path.join(checkpoint_dir, "crl_final.pth")
        agent.save(final_path, episodes_completed=episodes_completed)


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

    set_display_window()
    cfg_dict = _make_cfg_dict(cfg)
    dist_info = init_distributed(requested_device=str(cfg_dict.get("device", "cpu")))
    try:
        _run_crl_main(cfg_dict, dist_info)
    finally:
        cleanup_distributed()


def _run_crl_main(cfg_dict, dist_info) -> None:
    from examples.drl_collision_avoidance.crl_modules.crl_agent import (
        CRLAgent,
        CRLAgentConfig,
    )

    cfg_dict = apply_scenario_path_defaults(cfg_dict, default_split="train")
    cfg_dict["seed"] = rank_offset_seed(int(cfg_dict.get("seed", 42)), dist_info.rank)
    if dist_info.is_distributed:
        cfg_dict["device"] = str(dist_info.device)
    crl_cfg = cfg_dict["crl"]
    device = str(dist_info.device)

    running_state = {"running": True}

    def _signal_handler(sig, frame):
        print("\n[train_crl] Interrupted, stopping ...")
        running_state["running"] = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    probe_env = _make_crl_env(cfg_dict, 0)
    _obs_probe, _ = probe_env.reset()
    state_dim = probe_env.state_dim
    goal_dim = probe_env.goal_dim
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()
    del probe_env

    num_envs = int(crl_cfg.get("num_envs", 8))
    vec_mode = str(crl_cfg.get("vec_env", "subproc"))
    # Build workers before the CUDA agent so Linux fork does not copy a
    # live GPU context into each simulation process.
    vec_env = _make_vec_env(cfg_dict, num_envs, vec_mode)

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
        find_unused_parameters=bool(crl_cfg.get("find_unused_parameters", False)),
        log_alpha_min=float(crl_cfg.get("log_alpha_min", -2.5)),
        log_alpha_max=float(crl_cfg.get("log_alpha_max", 2.0)),
        reset_critic_optimizer_on_load=bool(
            crl_cfg.get("reset_critic_optimizer_on_load", True)
        ),
        reinit_critic_on_load=bool(crl_cfg.get("reinit_critic_on_load", True)),
        reinit_critic_on_collapsed_alpha=bool(
            crl_cfg.get("reinit_critic_on_collapsed_alpha", False)
        ),
        lse_penalty_warmup_steps=int(crl_cfg.get("lse_penalty_warmup_steps", 5000)),
        resume_random_action_prob=float(
            crl_cfg.get("resume_random_action_prob", 0.5)
        ),
        critic_burst_steps_on_resume=int(
            crl_cfg.get("critic_burst_steps_on_resume", 2000)
        ),
        device=device,
    )
    agent = CRLAgent(agent_config, dist_info=dist_info)
    if dist_info.is_rank0:
        _print_header(agent_config, agent)
        print(
            f"  Parallel envs : {num_envs} ({vec_mode}) × {dist_info.world_size} ranks, "
            f"grad steps/ep={crl_cfg.get('gradient_steps_per_episode', 2)}\n"
        )

    writer = _make_writer(cfg_dict) if dist_info.is_rank0 else None

    start_episode = int(crl_cfg.get("start_episode", 0) or 0)
    resume = crl_cfg.get("resume_checkpoint", None)
    if resume:
        loaded_episode = agent.load(resume)
        if start_episode <= 0:
            start_episode = loaded_episode

    try:
        _run_crl_training(
            vec_env,
            agent,
            crl_cfg,
            writer,
            running_state,
            state_dim,
            start_episode=start_episode,
        )
    finally:
        if dist_info.is_rank0:
            latest = int(running_state.get("episodes", start_episode) or start_episode)
            if latest > 0:
                latest_path = os.path.join(
                    str(crl_cfg["checkpoint_dir"]), "crl_latest.pth"
                )
                try:
                    agent.save(latest_path, episodes_completed=latest)
                except Exception as exc:
                    print(f"[train_crl] Could not save latest checkpoint: {exc}")
        vec_env.close()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
