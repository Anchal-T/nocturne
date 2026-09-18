# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate a saved R-MAPPO actor/critic without optimizer or ValueNorm state.

Example:
    python examples/on_policy_files/evaluate_ppo.py \
        --checkpoint /path/to/models \
        --scenario-path /path/to/formatted_json_v2_no_tl_valid \
        --episodes 100
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from algos.ppo.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from cfgs.config import PROCESSED_VALID_NO_TL, PROJECT_PATH, set_display_window
from examples.on_policy_files.nocturne_runner import (
    NocturneSharedRunner,
    _t2n,
    make_eval_env,
)
from nocturne.utils.distributed import DistInfo


def _load_cfg(args):
    config_dir = str(PROJECT_PATH / "cfgs")
    overrides = [
        f"scenario_path={args.scenario_path}",
        f"algorithm.device={args.device}",
        "algorithm.wandb=False",
        "algorithm.use_eval=False",
        "algorithm.use_render=False",
        "algorithm.n_rollout_threads=1",
        "algorithm.n_eval_rollout_threads=1",
        f"algorithm.use_centralized_V={args.use_centralized_v}",
        f"scenario.max_visible_road_points={args.max_visible_road_points}",
        f"subscriber.use_occlusion_features={args.use_occlusion_features}",
        f"subscriber.n_frames_stacked={args.n_frames_stacked}",
        f"max_num_vehicles={args.max_num_vehicles}",
        f"num_files={args.num_files}",
        f"scenario_cache_size={args.scenario_cache_size}",
        "scenario_pool_size=0",
        "wandb=False",
        "debug=True",
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)
    cfg.subscriber.keep_inactive_agents = True
    cfg.device = args.device
    return cfg


def evaluate(args):
    checkpoint = Path(args.checkpoint)
    actor_path = checkpoint / "actor.pt"
    critic_path = checkpoint / "critic.pt"
    if not actor_path.exists() or not critic_path.exists():
        raise FileNotFoundError(
            "checkpoint directory must contain actor.pt and critic.pt"
        )

    cfg = _load_cfg(args)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    envs = make_eval_env(cfg)
    results = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "split": args.scenario_path,
        "episodes": 0,
        "nan_action_batches": 0,
    }
    try:
        obs = envs.reset()
        num_agents = int(obs.shape[1])
        policy = R_MAPPOPolicy(
            cfg.algorithm,
            envs.observation_space[0],
            envs.observation_space[0] if not cfg.algorithm.use_centralized_V
            else envs.share_observation_space[0],
            envs.action_space[0],
            device=device,
            dist_info=DistInfo(device=device),
        )
        policy.actor_module().load_state_dict(
            torch.load(actor_path, map_location=device),
            strict=False,
        )
        policy.critic_module().load_state_dict(
            torch.load(critic_path, map_location=device),
            strict=False,
        )
        policy.actor_module().eval()
        policy.critic_module().eval()

        rnn_states = np.zeros(
            (1, num_agents, int(cfg.algorithm.recurrent_N),
             int(cfg.algorithm.hidden_size)),
            dtype=np.float32,
        )
        masks = np.ones((1, num_agents, 1), dtype=np.float32)
        episode_reward = np.zeros((num_agents, 1), dtype=np.float64)
        goals = np.zeros(num_agents, dtype=bool)
        collisions = np.zeros(num_agents, dtype=bool)
        ever_active = np.zeros(num_agents, dtype=bool)
        episode_rewards = []
        goal_rates = []
        collision_rates = []
        episode_lengths = []
        timeout_episodes = 0
        nan_action_batches = 0
        total_steps = 0
        episode_steps = 0
        max_steps = args.episodes * int(cfg.episode_length) * 3

        with torch.no_grad():
            while len(episode_rewards) < args.episodes:
                if total_steps >= max_steps:
                    raise RuntimeError(
                        "evaluation exceeded {} steps with {} completed episodes"
                        .format(max_steps, len(episode_rewards))
                    )
                actions, next_rnn_states = policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions_np = np.array(np.split(_t2n(actions), 1))
                rnn_states = np.array(np.split(_t2n(next_rnn_states), 1))
                nan_action_batches += int(not np.isfinite(actions_np).all())
                actions_env = NocturneSharedRunner._format_actions_for_env(
                    None, actions_np, envs.action_space[0]
                )
                obs, rewards, dones, infos = envs.step(actions_env)
                total_steps += 1
                episode_steps += 1
                episode_reward += np.asarray(rewards[0], dtype=np.float64)
                for agent_idx, info in enumerate(infos[0]):
                    if not info.get("inactive", False):
                        ever_active[agent_idx] = True
                    goals[agent_idx] |= bool(info.get("goal_achieved", False))
                    collisions[agent_idx] |= bool(info.get("collided", False))

                if not bool(np.all(dones, axis=1)[0]):
                    masks.fill(1.0)
                    masks[np.asarray(dones, dtype=bool)] = 0.0
                    continue

                if np.any(ever_active):
                    active_reward = float(
                        episode_reward.reshape(-1)[ever_active].mean()
                    )
                    goal_rates.append(float(goals[ever_active].mean()))
                    collision_rates.append(float(collisions[ever_active].mean()))
                else:
                    active_reward = float(episode_reward.mean())
                    goal_rates.append(0.0)
                    collision_rates.append(0.0)
                episode_rewards.append(active_reward)
                episode_lengths.append(episode_steps)
                timeout_episodes += int(not bool(np.any(goals | collisions)))
                episode_reward.fill(0.0)
                goals.fill(False)
                collisions.fill(False)
                ever_active.fill(False)
                rnn_states.fill(0.0)
                masks.fill(1.0)
                episode_steps = 0

        results.update({
            "episodes": len(episode_rewards),
            "agent_slots": num_agents,
            "steps": total_steps,
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "std_episode_reward": float(np.std(episode_rewards)),
            "mean_goal_rate_per_agent": float(np.mean(goal_rates)),
            "mean_collision_rate_per_agent": float(np.mean(collision_rates)),
            "episodes_with_any_goal": float(np.mean(np.asarray(goal_rates) > 0.0)),
            "episodes_with_any_collision": float(
                np.mean(np.asarray(collision_rates) > 0.0)
            ),
            "timeout_like_episodes": timeout_episodes,
            "mean_episode_length": float(np.mean(episode_lengths)),
            "nan_action_batches": nan_action_batches,
        })
    finally:
        envs.close()
    print(json.dumps(results, sort_keys=True))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario-path", default=PROCESSED_VALID_NO_TL)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-visible-road-points", type=int, default=8)
    parser.add_argument("--n-frames-stacked", type=int, default=1)
    parser.add_argument("--max-num-vehicles", type=int, default=20)
    parser.add_argument("--num-files", type=int, default=100)
    parser.add_argument("--scenario-cache-size", type=int, default=32)
    parser.add_argument("--use-occlusion-features", action="store_true")
    parser.add_argument("--use-centralized-v", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TMPDIR", str(Path.home() / "tmp"))
    try:
        set_display_window()
    except FileNotFoundError:
        pass
    evaluate(parse_args())
