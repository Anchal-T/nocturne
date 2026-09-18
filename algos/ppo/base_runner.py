# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Code modified from https://github.com/marlbenchmark/on-policy
import os

import numpy as np
import torch
import wandb
from tensorboardX import SummaryWriter

from algos.ppo.utils.shared_buffer import SharedReplayBuffer
from nocturne.utils.distributed import DistInfo, unwrap_module


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


def _atomic_torch_save(obj, path):
    """Write ``obj`` to ``path`` by replacing a sibling temp file."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _try_load_state_dict(module, state, name):
    """Load a state dict, skipping incomplete or legacy payloads."""
    if module is None or not isinstance(state, dict) or not state:
        return
    try:
        module.load_state_dict(state, strict=False)
    except (RuntimeError, ValueError, TypeError) as exc:
        print("skipping incompatible {}: {}".format(name, exc))


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config):

        self.all_args = config["cfg.algo"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.dist_info = config.get("dist_info") or DistInfo(device=self.device)
        self.num_agents = config["num_agents"]
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]

        # parameters
        # self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        # self.episodes_per_thread = self.all_args.episodes_per_thread
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        elif self.dist_info.is_rank0:
            self.run_dir = config["logdir"]
            self.log_dir = str(self.run_dir / "logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / "models")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        else:
            self.run_dir = config["logdir"]
            self.log_dir = None
            self.writter = None
            self.save_dir = str(self.run_dir / "models")

        from algos.ppo.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        from algos.ppo.r_mappo.r_mappo import R_MAPPO as TrainAlgo

        share_observation_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )

        # policy network
        self.policy = Policy(
            self.all_args,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
            device=self.device,
            dist_info=self.dist_info,
        )

        if self.model_dir is not None:
            self._pending_lagrangian_state = None
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(
            self.all_args, self.policy, device=self.device, dist_info=self.dist_info
        )

        # Apply any Lagrangian state captured during restore (the trainer
        # owns the multiplier and cost value normalizer, but doesn't exist
        # until after `restore()` runs).
        if getattr(self, "_pending_lagrangian_state", None):
            self.trainer.load_lagrangian_state_dict(self._pending_lagrangian_state)
            self._pending_lagrangian_state = None
        if getattr(self, "_pending_trainer_state", None):
            state = self._pending_trainer_state
            for key, optimizer in (
                ("actor_optimizer", self.policy.actor_optimizer),
                ("critic_optimizer", self.policy.critic_optimizer),
            ):
                if key in state:
                    try:
                        optimizer.load_state_dict(state[key])
                    except (ValueError, RuntimeError) as exc:
                        print("skipping incompatible {}: {}".format(key, exc))
            if "cost_critic_optimizer" in state and getattr(
                self.policy, "use_lagrangian", False
            ):
                try:
                    self.policy.cost_critic_optimizer.load_state_dict(
                        state["cost_critic_optimizer"]
                    )
                except (ValueError, RuntimeError) as exc:
                    print("skipping incompatible cost_critic_optimizer: {}".format(exc))
            if "value_normalizer" in state and getattr(
                self.trainer, "value_normalizer", None
            ) is not None:
                _try_load_state_dict(
                    self.trainer.value_normalizer,
                    state["value_normalizer"],
                    "value_normalizer",
                )
            self._pending_trainer_state = None

        # buffer
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

        # Compute cost returns for PPO-Lagrangian.
        if getattr(self.trainer, "use_lagrangian", False):
            next_cost_values = self.trainer.policy.get_cost_values(
                np.concatenate(self.buffer.share_obs[-1]),
                np.concatenate(self.buffer.rnn_states_critic[-1]),
                np.concatenate(self.buffer.masks[-1]),
            )
            next_cost_values = np.array(
                np.split(_t2n(next_cost_values), self.n_rollout_threads)
            )
            self.buffer.compute_cost_returns(
                next_cost_values, self.trainer.cost_value_normalizer
            )

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        if not self.dist_info.is_rank0:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        _atomic_torch_save(
            unwrap_module(self.trainer.policy.actor).state_dict(),
            os.path.join(self.save_dir, "actor.pt"),
        )
        _atomic_torch_save(
            unwrap_module(self.trainer.policy.critic).state_dict(),
            os.path.join(self.save_dir, "critic.pt"),
        )
        if getattr(self.trainer.policy, "use_lagrangian", False):
            _atomic_torch_save(
                unwrap_module(self.trainer.policy.cost_critic).state_dict(),
                os.path.join(self.save_dir, "cost_critic.pt"),
            )
            _atomic_torch_save(
                self.trainer.lagrangian_state_dict(),
                os.path.join(self.save_dir, "lagrangian.pt"),
            )
        trainer_state = {
            "world_size": self.dist_info.world_size,
            "actor_optimizer": self.trainer.policy.actor_optimizer.state_dict(),
            "critic_optimizer": self.trainer.policy.critic_optimizer.state_dict(),
        }
        if getattr(self.trainer.policy, "use_lagrangian", False):
            trainer_state["cost_critic_optimizer"] = (
                self.trainer.policy.cost_critic_optimizer.state_dict()
            )
        if getattr(self.trainer, "value_normalizer", None) is not None:
            trainer_state["value_normalizer"] = (
                self.trainer.value_normalizer.state_dict()
            )
        _atomic_torch_save(
            trainer_state, os.path.join(self.save_dir, "trainer_state.pt")
        )

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(
            str(self.model_dir) + "/actor.pt",
            map_location=self.device,
        )
        _try_load_state_dict(
            unwrap_module(self.policy.actor),
            policy_actor_state_dict,
            "actor",
        )
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + "/critic.pt",
                map_location=self.device,
            )
            _try_load_state_dict(
                unwrap_module(self.policy.critic),
                policy_critic_state_dict,
                "critic",
            )
            if getattr(self.policy, "use_lagrangian", False):
                cost_critic_path = str(self.model_dir) + "/cost_critic.pt"
                if not os.path.exists(cost_critic_path):
                    raise FileNotFoundError(
                        f"use_lagrangian=True but cost_critic.pt was not found in {self.model_dir}"
                    )
                policy_cost_critic_state_dict = torch.load(
                    cost_critic_path,
                    map_location=self.device,
                )
                _try_load_state_dict(
                    unwrap_module(self.policy.cost_critic),
                    policy_cost_critic_state_dict,
                    "cost_critic",
                )

                lagrangian_path = str(self.model_dir) + "/lagrangian.pt"
                if not os.path.exists(lagrangian_path):
                    raise FileNotFoundError(
                        f"use_lagrangian=True but lagrangian.pt was not found in {self.model_dir}"
                    )
                self._pending_lagrangian_state = torch.load(
                    lagrangian_path,
                    map_location=self.device,
                )

        trainer_state_path = str(self.model_dir) + "/trainer_state.pt"
        if os.path.exists(trainer_state_path):
            try:
                self._pending_trainer_state = torch.load(
                    trainer_state_path,
                    map_location=self.device,
                )
            except (RuntimeError, OSError, EOFError, ValueError) as exc:
                print("skipping trainer_state.pt: {}".format(exc))

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            elif self.writter is not None:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                elif self.writter is not None:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
