# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Example run script for RLlib."""
import os

import hydra
from omegaconf import OmegaConf
from cfgs.config import set_display_window
from nocturne.utils.ray_compat import patch_legacy_gym_monitor

patch_legacy_gym_monitor()

import ray
from ray import tune
from ray.tune.registry import register_env

from ray.rllib.policy.policy import PolicySpec

from examples.rllib_files.rllib_env import create_rllib_env


def _ray_version_tuple():
    parts = ray.__version__.split(".")
    try:
        return tuple(int(part) for part in parts[:2])
    except ValueError:
        return (0, 0)


def _rllib_gpu_config(num_gpus):
    """Use Ray 1.11's num_gpus learner API; never invent num_learners."""
    major, minor = _ray_version_tuple()
    if major > 2 or (major == 2 and minor >= 4):
        print(
            f"[rllib] Ray {ray.__version__} supports newer learner APIs; "
            "this integration still uses legacy num_gpus until retested."
        )
    elif (major, minor) != (1, 11):
        print(
            f"[rllib] Ray {ray.__version__} detected; configuring legacy "
            f"num_gpus={num_gpus} (validated against 1.11)."
        )
    return {"num_gpus": int(num_gpus)}


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    """Run RLlib example."""
    set_display_window()
    cfg = OmegaConf.to_container(cfg, resolve=True)
    rllib_cfg = cfg.get("rllib") or {}
    if cfg['debug']:
        ray.init(local_mode=True)
        num_workers = 0
        num_envs_per_worker = 1
        num_gpus = 0
        use_lstm = False
    else:
        num_workers = int(rllib_cfg.get("num_workers", 15))
        num_envs_per_worker = int(rllib_cfg.get("num_envs_per_worker", 5))
        num_gpus = int(rllib_cfg.get("num_gpus", 1))
        use_lstm = bool(rllib_cfg.get("use_lstm", True))

    register_env("nocturne", lambda env_cfg: create_rllib_env(env_cfg))
    probe_env = create_rllib_env(cfg)
    try:
        observation_space = probe_env.observation_space
        action_space = probe_env.action_space
    finally:
        probe_env.close()

    username = os.environ["USER"]
    local_dir = rllib_cfg.get("local_dir") or (
        f"/checkpoint/{username}/nocturne/ray_results")
    trainer_config = {
        "env": "nocturne",
        "env_config": cfg,
        "framework": "torch",
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "observation_filter": "MeanStdFilter",
        "entropy_coeff": float(rllib_cfg.get("entropy_coeff", 0.0)),
        "num_sgd_iter": int(rllib_cfg.get("num_sgd_iter", 5)),
        "train_batch_size":
        max(100 * max(num_workers, 1) * num_envs_per_worker, 512),
        "rollout_fragment_length":
        int(rllib_cfg.get("rollout_fragment_length", 20)),
        "sgd_minibatch_size":
        max(int(100 * max(num_workers, 1) * num_envs_per_worker / 4), 512),
        "multiagent": {
            "policies": {
                "shared_policy": PolicySpec(
                    policy_class=None,
                    observation_space=observation_space,
                    action_space=action_space,
                    config={},
                )
            },
            "policy_mapping_fn":
            (lambda agent_id, episode, **kwargs: "shared_policy"),
            "count_steps_by": "agent_steps",
        },
        "model": {
            "use_lstm": use_lstm
        },
        "evaluation_interval": int(rllib_cfg.get("evaluation_interval", 50)),
        "evaluation_duration": int(rllib_cfg.get("evaluation_duration", 1)),
        "evaluation_num_workers":
        int(rllib_cfg.get("evaluation_num_workers", 0)),
        "evaluation_config": {
            "record_env": rllib_cfg.get("record_env", "videos_test"),
            "render_env": bool(rllib_cfg.get("render_env", True)),
        },
    }
    trainer_config.update(_rllib_gpu_config(num_gpus))
    tune_kwargs = {
        "local_dir": local_dir,
        "stop": {
            "episodes_total": int(rllib_cfg.get("stop_episodes", 60000))
        },
        "checkpoint_freq": int(rllib_cfg.get("checkpoint_freq", 1000)),
        "checkpoint_at_end": True,
        "config": trainer_config,
    }
    resume_checkpoint = rllib_cfg.get("resume_checkpoint")
    if resume_checkpoint:
        tune_kwargs["restore"] = resume_checkpoint
    tune.run("PPO", **tune_kwargs)


if __name__ == "__main__":
    main()
