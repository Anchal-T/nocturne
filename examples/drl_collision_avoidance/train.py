# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Deep RL training script for collision avoidance in Nocturne.

Run with:
    python -m examples.drl_collision_avoidance.train \
        drl.vec_env_mode=ray \
        drl.ray_address=auto \
        drl.num_workers=32 \
        drl.num_envs_per_worker=2
"""
import os

import hydra
from omegaconf import OmegaConf
from cfgs.config import set_display_window
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from nocturne.envs.wrappers import create_env


class NocturneMultiAgentEnv(MultiAgentEnv):
    """Thin wrapper that makes the Nocturne env look like a Ray MultiAgentEnv."""

    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, env_cfg):
        """Initialize the wrapper around a Nocturne BaseEnv.

        Args
        ----
            env_cfg (dict): configuration dict forwarded to :func:`create_env`.
        """
        self._skip_env_checking = True  # temporary fix for rllib env-checking issue
        super().__init__()
        self._env = create_env(env_cfg)

    def step(self, actions):
        """See superclass."""
        next_obs, rew, done, info = self._env.step(actions)
        return next_obs, rew, done, info

    def reset(self):
        """See superclass."""
        return self._env.reset()

    @property
    def observation_space(self):
        """See superclass."""
        return self._env.observation_space

    @property
    def action_space(self):
        """See superclass."""
        return self._env.action_space

    def render(self, mode=None):
        """See superclass."""
        return self._env.render()

    def seed(self, seed=None):
        """See superclass."""
        self._env.seed(seed)

    def __getattr__(self, name):
        """Delegate attribute lookup to the wrapped env."""
        return getattr(self._env, name)


def _make_env(env_cfg):
    """Factory function used when registering the env with RLlib."""
    return NocturneMultiAgentEnv(env_cfg)


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    """Train a PPO collision-avoidance policy with RLlib.

    Key DRL hyper-parameters are read from the ``drl`` config group and can be
    overridden from the command line, e.g.::

        python -m examples.drl_collision_avoidance.train \\
            drl.vec_env_mode=ray drl.ray_address=auto \\
            drl.num_workers=32 drl.num_envs_per_worker=2
    """
    set_display_window()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    drl_cfg = cfg_dict.get("drl", {})

    # ------------------------------------------------------------------ #
    # Ray initialization                                                   #
    # ------------------------------------------------------------------ #
    ray_address = drl_cfg.get("ray_address")
    vec_env_mode = drl_cfg.get("vec_env_mode", "local")

    if vec_env_mode == "ray" and ray_address is not None:
        ray.init(address=ray_address)
    elif vec_env_mode == "ray":
        ray.init()
    else:
        # local mode is handy for interactive debugging
        ray.init(local_mode=True)

    # ------------------------------------------------------------------ #
    # Worker and batch-size configuration                                  #
    # ------------------------------------------------------------------ #
    num_workers = int(drl_cfg.get("num_workers", 2))
    num_envs_per_worker = int(drl_cfg.get("num_envs_per_worker", 1))
    num_gpus = int(drl_cfg.get("num_gpus", 0))
    use_lstm = bool(drl_cfg.get("use_lstm", False))
    num_sgd_iter = int(drl_cfg.get("num_sgd_iter", 5))
    rollout_fragment_length = int(drl_cfg.get("rollout_fragment_length", 20))
    entropy_coeff = float(drl_cfg.get("entropy_coeff", 0.0))
    stop_episodes = int(drl_cfg.get("stop_episodes", 60000))
    checkpoint_freq = int(drl_cfg.get("checkpoint_freq", 100))

    # Compute sensible batch-size defaults if not explicitly set.
    effective_envs = max(num_workers * num_envs_per_worker, 1)
    train_batch_size = drl_cfg.get("train_batch_size") or max(
        100 * effective_envs, 512)
    sgd_minibatch_size = drl_cfg.get("sgd_minibatch_size") or max(
        int(train_batch_size / 4), 512)

    # ------------------------------------------------------------------ #
    # Results directory                                                    #
    # ------------------------------------------------------------------ #
    results_dir = drl_cfg.get("results_dir") or os.path.expanduser(
        "~/ray_results/nocturne")

    # ------------------------------------------------------------------ #
    # Environment registration                                             #
    # ------------------------------------------------------------------ #
    register_env("nocturne_collision_avoidance",
                 lambda env_cfg: _make_env(env_cfg))

    # ------------------------------------------------------------------ #
    # PPO training                                                         #
    # ------------------------------------------------------------------ #
    tune.run(
        "PPO",
        local_dir=results_dir,
        stop={"episodes_total": stop_episodes},
        checkpoint_freq=checkpoint_freq,
        config={
            # Environment
            "env": "nocturne_collision_avoidance",
            "env_config": cfg_dict,
            # Framework
            "framework": "torch",
            # Resources
            "num_gpus": num_gpus,
            "num_workers": num_workers,
            "num_envs_per_worker": num_envs_per_worker,
            # Normalise observations using a running mean/std filter.
            "observation_filter": "MeanStdFilter",
            # PPO hyper-parameters
            "entropy_coeff": entropy_coeff,
            "num_sgd_iter": num_sgd_iter,
            "train_batch_size": train_batch_size,
            "rollout_fragment_length": rollout_fragment_length,
            "sgd_minibatch_size": sgd_minibatch_size,
            # Multi-agent: all vehicles share a single policy.
            "multiagent": {
                "policies": {"shared_policy"},
                "policy_mapping_fn": (
                    lambda agent_id, episode, **kwargs: "shared_policy"),
                # Count individual agent steps toward the training batch.
                "count_steps_by": "agent_steps",
            },
            "model": {
                "use_lstm": use_lstm,
            },
            # Periodic evaluation
            "evaluation_interval": 50,
            "evaluation_duration": 1,
            "evaluation_num_workers": 0,
            "evaluation_config": {
                "record_env": "videos_eval",
                "render_env": True,
            },
        },
    )


if __name__ == "__main__":
    main()
