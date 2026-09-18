# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Code modified from https://github.com/marlbenchmark/on-policy
"""Runner for PPO from https://github.com/marlbenchmark/on-policy."""
from pathlib import Path
import os
import time

import hydra
from cfgs.config import set_display_window
import imageio
import numpy as np
import setproctitle
import torch
import wandb

from algos.ppo.base_runner import Runner
from algos.ppo.env_wrappers import SubprocVecEnv, DummyVecEnv

from nocturne.envs.wrappers import create_ppo_env
from nocturne.utils.distributed import (
    all_reduce_scalar,
    broadcast_object,
    cleanup_distributed,
    control_barrier,
    init_distributed,
    rank_offset_seed,
    reduce_metrics,
)


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


def _env_seed(cfg, env_rank):
    """Rank-offset seed so no two processes collect the same scenarios."""
    dist_rank = int(
        os.environ.get("RANK", getattr(cfg, "dist_rank", 0))
    )
    return rank_offset_seed(cfg.seed, dist_rank) + env_rank * 1000


def make_train_env(cfg):
    """Construct a training environment."""

    def get_env_fn(rank):

        def init_env():
            env = create_ppo_env(cfg, rank)
            env.seed(_env_seed(cfg, rank))
            return env

        return init_env

    if cfg.algorithm.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(cfg.algorithm.n_rollout_threads)])


def make_eval_env(cfg):
    """Construct an eval environment."""

    def get_env_fn(rank):

        def init_env():
            env = create_ppo_env(cfg)
            env.seed(_env_seed(cfg, rank) + 10000)
            return env

        return init_env

    if cfg.algorithm.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(cfg.algorithm.n_eval_rollout_threads)])


def make_render_env(cfg):
    """Construct a rendering environment."""

    def get_env_fn(rank):

        def init_env():
            env = create_ppo_env(cfg)
            # TODO(eugenevinitsky) implement this
            env.seed(cfg.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(0)])


class NocturneSharedRunner(Runner):
    """
    Runner class to perform training, evaluation and data collection for the Nocturne envs.

    WARNING: Assumes a shared policy.
    """

    def __init__(self, config):
        """Initialize."""
        super(NocturneSharedRunner, self).__init__(config)
        self.cfg = config['cfg.algo']
        self.render_envs = config['render_envs']

    def _format_actions_for_env(self, actions, action_space):
        """Convert policy actions to the representation expected by env.step."""
        if action_space.__class__.__name__ == 'MultiDiscrete':
            for i in range(action_space.shape):
                uc_actions_env = np.eye(action_space.high[i] +
                                        1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env),
                                                 axis=2)
        elif action_space.__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(action_space.n)[actions], 2)
        elif action_space.__class__.__name__ == 'Box':
            actions_env = np.clip(actions, action_space.low,
                                  action_space.high).astype(np.float32)
        else:
            raise NotImplementedError
        return actions_env

    def run(self):
        """Run the training code."""
        self.warmup()

        start = time.time()
        world_size = max(1, int(self.dist_info.world_size))
        episodes = (int(self.num_env_steps) // self.episode_length //
                    self.n_rollout_threads // world_size)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            rollout_start = time.time()
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, cost_value_preds = self.collect(
                    step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_value_preds

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            control_barrier(self.dist_info)
            rollout_seconds = time.time() - rollout_start
            train_start = time.time()
            self.compute()
            train_infos = self.train()
            if getattr(self.trainer, 'use_lagrangian', False):
                # Per-episode cost: sum costs over the time dimension,
                # then average over rollout threads and agents, then ranks.
                mean_episode_cost = float(
                    self.buffer.costs.sum(axis=0).mean())
                mean_episode_cost = all_reduce_scalar(
                    mean_episode_cost, self.device, op="mean")
                train_infos['mean_episode_cost'] = mean_episode_cost
                train_infos['lagrangian_multiplier'] = (
                    self.trainer.update_lagrangian(mean_episode_cost))
            train_infos = reduce_metrics(train_infos, self.device)
            train_seconds = time.time() - train_start

            # post process: global env steps across all ranks
            total_num_steps = ((episode + 1) * self.episode_length *
                               self.n_rollout_threads * world_size)

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                control_barrier(self.dist_info)
                self.save()
                control_barrier(self.dist_info)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                fps = int(total_num_steps / max(end - start, 1e-8))
                if self.dist_info.is_rank0:
                    print(
                        "\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {} "
                        "(rollout {:.1f}s, train {:.1f}s).\n"
                        .format(self.algorithm_name, self.experiment_name,
                                episode + 1, episodes, total_num_steps,
                                self.num_env_steps, fps, rollout_seconds,
                                train_seconds))

                if self.use_wandb and self.dist_info.is_rank0:
                    wandb.log({'fps': fps}, step=total_num_steps)
                env_infos = {}
                for agent_id in range(self.num_agents):
                    idv_rews = []
                    idv_near_miss = []
                    for info in infos:
                        agent_info = info[agent_id]
                        if agent_info.get('inactive', False):
                            continue
                        if 'individual_reward' in agent_info:
                            idv_rews.append(agent_info['individual_reward'])
                        if 'near_miss' in agent_info:
                            idv_near_miss.append(agent_info['near_miss'])
                    agent_k = 'agent%i/individual_rewards' % agent_id
                    env_infos[agent_k] = idv_rews
                    if idv_near_miss:
                        env_infos['agent%i/near_miss' % agent_id] = (
                            idv_near_miss)

                # active_masks is stored at t+1, aligned with rewards at t.
                active = self.buffer.active_masks[1:]
                rew = self.buffer.rewards
                active_count = float(np.sum(active))
                if active_count > 0:
                    avg_step_reward = float(np.sum(rew * active) / active_count)
                    max_step_reward = float(np.max(rew * active))
                else:
                    avg_step_reward = float(np.mean(rew))
                    max_step_reward = float(np.max(rew))
                train_infos["average_episode_rewards"] = (
                    avg_step_reward * self.episode_length)
                train_infos["maximum_step_reward"] = (
                    all_reduce_scalar(max_step_reward, self.device, op="mean")
                )
                train_infos = reduce_metrics(train_infos, self.device)
                if self.dist_info.is_rank0:
                    print("average episode rewards is {}".format(
                        train_infos["average_episode_rewards"]))
                    print(f"maximum per step reward is {max_step_reward}")
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                control_barrier(self.dist_info)
                if self.dist_info.is_rank0:
                    self.eval(total_num_steps)
                control_barrier(self.dist_info)

            # Save videos only when rendering is explicitly enabled.
            if self.use_render and episode % self.cfg.render_interval == 0:
                control_barrier(self.dist_info)
                if self.dist_info.is_rank0:
                    self.render(total_num_steps)
                control_barrier(self.dist_info)

    def warmup(self):
        """Initialize the buffers."""
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents,
                                                            axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect rollout data."""
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        actions_env = self._format_actions_for_env(actions,
                                                   self.envs.action_space[0])

        # Collect cost value predictions for PPO-Lagrangian.
        cost_value_preds = None
        if getattr(self.trainer, 'use_lagrangian', False):
            cost_value = self.trainer.policy.get_cost_values(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]))
            cost_value_preds = np.array(
                np.split(_t2n(cost_value), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, cost_value_preds

    def insert(self, data):
        """Store the data in the buffers."""
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_value_preds = data
        costs = np.array([[[agent_info.get('cost', 0.0)]
                           for agent_info in env_info]
                          for env_info in infos],
                         dtype=np.float32)

        dones_env = np.all(dones, axis=1)
        inactive = np.array([[[1.0 if agent_info.get('inactive', False) else 0.0]
                              for agent_info in env_info]
                             for env_info in infos],
                            dtype=np.float32)

        rnn_states[dones_env] = np.zeros(((dones_env).sum(), self.num_agents,
                                          self.recurrent_N, self.hidden_size),
                                         dtype=np.float32)
        rnn_states_critic[dones_env] = np.zeros(
            ((dones_env).sum(), self.num_agents,
             *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
                        dtype=np.float32)
        masks[dones_env] = np.zeros(((dones_env).sum(), self.num_agents, 1),
                                    dtype=np.float32)

        # Padding / already-removed agents keep a dummy obs. Do not treat
        # them as active, including on the env-timeout step.
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
                               dtype=np.float32)
        active_masks[dones] = np.zeros(((dones).sum(), 1), dtype=np.float32)
        active_masks[inactive.astype(bool)] = 0.0
        if np.any(dones_env):
            active_masks[dones_env] = 1.0 - inactive[dones_env]

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents,
                                                            axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs,
                           obs,
                           rnn_states,
                           rnn_states_critic,
                           actions,
                           action_log_probs,
                           values,
                           rewards,
                           masks,
                           costs=costs,
                           cost_value_preds=cost_value_preds,
                           active_masks=active_masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        """Get the policy returns in deterministic mode."""
        eval_episode = 0

        eval_episode_rewards = []
        eval_goal_rates = []
        eval_collision_rates = []
        one_episode_rewards = [[] for _ in range(self.n_eval_rollout_threads)]
        episode_goals = [
            np.zeros(self.num_agents, dtype=bool)
            for _ in range(self.n_eval_rollout_threads)
        ]
        episode_collisions = [
            np.zeros(self.num_agents, dtype=bool)
            for _ in range(self.n_eval_rollout_threads)
        ]
        episode_active = [
            np.zeros(self.num_agents, dtype=bool)
            for _ in range(self.n_eval_rollout_threads)
        ]

        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N,
             self.hidden_size),
            dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1),
                             dtype=np.float32)

        while eval_episode < self.cfg.eval_episodes:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        deterministic=True)
            eval_actions = np.array(
                np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_actions_env = self._format_actions_for_env(
                eval_actions, self.eval_envs.action_space[0])
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env)

            for env_i, info_arr in enumerate(eval_infos):
                for agent_i, agent_info in enumerate(info_arr):
                    if not agent_info.get('inactive', False):
                        episode_active[env_i][agent_i] = True
                    episode_goals[env_i][agent_i] |= bool(
                        agent_info.get('goal_achieved', False))
                    episode_collisions[env_i][agent_i] |= bool(
                        agent_info.get('collided', False))

            for env_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[env_i].append(eval_rewards[env_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env] = np.zeros(
                ((eval_dones_env).sum(), self.num_agents, self.recurrent_N,
                 self.hidden_size),
                dtype=np.float32)

            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32)
            eval_masks[eval_dones_env] = np.zeros(
                ((eval_dones_env).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if not eval_dones_env[eval_i]:
                    continue
                rewards = np.asarray(one_episode_rewards[eval_i])
                active = episode_active[eval_i]
                if np.any(active):
                    agent_returns = rewards.sum(axis=0).reshape(-1)[:self.num_agents]
                    eval_episode_rewards.append(
                        float(agent_returns[active].mean()))
                    eval_goal_rates.append(
                        float(episode_goals[eval_i][active].mean()))
                    eval_collision_rates.append(
                        float(episode_collisions[eval_i][active].mean()))
                else:
                    eval_episode_rewards.append(
                        float(np.sum(rewards, axis=0).mean()))
                    eval_goal_rates.append(0.0)
                    eval_collision_rates.append(0.0)
                one_episode_rewards[eval_i] = []
                episode_goals[eval_i].fill(False)
                episode_collisions[eval_i].fill(False)
                episode_active[eval_i].fill(False)
                eval_episode += 1

        mean_reward = float(np.mean(eval_episode_rewards)) if eval_episode_rewards else 0.0
        mean_goal = float(np.mean(eval_goal_rates)) if eval_goal_rates else 0.0
        mean_collision = (
            float(np.mean(eval_collision_rates)) if eval_collision_rates else 0.0
        )
        metrics = {
            'eval_episode_rewards': mean_reward,
            'avg_eval_goals_achieved': mean_goal,
            'avg_eval_num_collisions': mean_collision,
        }
        print(
            "eval @ {} steps: reward={:.4f} goal_rate={:.4f} collision_rate={:.4f}"
            .format(total_num_steps, mean_reward, mean_goal, mean_collision))
        if self.use_wandb:
            wandb.log(metrics, step=total_num_steps)
        elif self.writter is not None:
            for key, value in metrics.items():
                self.writter.add_scalars(key, {key: value}, total_num_steps)
        return metrics

    @torch.no_grad()
    def render(self, total_num_steps):
        """Visualize the env."""
        envs = self.render_envs

        all_frames = []
        for episode in range(self.cfg.render_episodes):
            obs = envs.reset()
            if self.cfg.save_gifs:
                image = envs.envs[0].render('rgb_array')
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros(
                (1, self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32)
            masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            self.trainer.prep_rollout()
            for step in range(self.episode_length):
                calc_start = time.time()

                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True)
                actions = np.array(np.split(_t2n(action), 1))
                rnn_states = np.array(np.split(_t2n(rnn_states), 1))

                actions_env = self._format_actions_for_env(
                    actions, envs.action_space[0])

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones] = np.zeros(
                    ((dones).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32)
                masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
                masks[dones] = np.zeros(((dones).sum(), 1), dtype=np.float32)

                if self.cfg.save_gifs:
                    image = envs.envs[0].render('rgb_array')
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.cfg.ifi:
                        time.sleep(self.cfg.ifi - elapsed)
                else:
                    envs.render('human')

                if np.all(dones[0]):
                    break

            # note, every rendered episode is exactly the same since there's no randomness in the env and our actions
            # are deterministic
            # TODO(eugenevinitsky) why is this lower than the non-render reward?
            render_val = np.mean(np.sum(np.array(episode_rewards), axis=0))
            print("episode reward of rendered episode is: " + str(render_val))
            if self.use_wandb:
                wandb.log({'render_rew': render_val}, step=total_num_steps)

        if self.cfg.save_gifs:
            if self.use_wandb:
                np_arr = np.stack(all_frames).transpose((0, 3, 1, 2))
                wandb.log({"video": wandb.Video(np_arr, fps=4, format="gif")},
                          step=total_num_steps)
            # else:
            imageio.mimsave(os.getcwd() + '/render.gif',
                            all_frames,
                            duration=self.cfg.ifi)


@hydra.main(config_path='../../cfgs/', config_name='config')
def main(cfg):
    """Run the on-policy code."""
    set_display_window()
    dist_info = init_distributed(requested_device=cfg.algorithm.device)
    try:
        _run_training(cfg, dist_info)
    finally:
        cleanup_distributed()


def _run_training(cfg, dist_info):
    """Rank-aware training body used by both single-process and torchrun."""
    if dist_info.is_distributed:
        cfg.algorithm.distributed = True
        cfg.algorithm.device = str(dist_info.device)

    logdir = Path(os.getcwd())
    if cfg.wandb_id is not None:
        wandb_id = cfg.wandb_id
    else:
        wandb_id = wandb.util.generate_id()
        wandb_id = broadcast_object(wandb_id, src=0)
    wandb_mode = "disabled" if (cfg.debug or not cfg.wandb) else "online"

    if cfg.wandb:
        if dist_info.is_rank0:
            run = wandb.init(config=cfg,
                             project=cfg.wandb_name,
                             name=wandb_id,
                             group='ppov2_' + cfg.experiment,
                             resume="allow",
                             settings=wandb.Settings(start_method="fork"),
                             mode=wandb_mode)
            logdir = Path(run.dir)
        else:
            run = None
            cfg.algorithm.wandb = False
        logdir = Path(broadcast_object(str(logdir), src=0))
    else:
        if dist_info.is_rank0:
            if not logdir.exists():
                curr_run = 'run1'
            else:
                exst_run_nums = [
                    int(str(folder.name).split('run')[1])
                    for folder in logdir.iterdir()
                    if str(folder.name).startswith('run')
                ]
                if len(exst_run_nums) == 0:
                    curr_run = 'run1'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)
            logdir = logdir / curr_run
            if not logdir.exists():
                os.makedirs(str(logdir))
            logdir = Path(broadcast_object(str(logdir), src=0))
        else:
            logdir = Path(broadcast_object(None, src=0))

    if cfg.algorithm.algorithm_name == "rmappo":
        assert (cfg.algorithm.use_recurrent_policy
                or cfg.algorithm.use_naive_recurrent_policy), (
                    "check recurrent policy!")
    elif cfg.algorithm.algorithm_name == "mappo":
        assert (not cfg.algorithm.use_recurrent_policy
                and not cfg.algorithm.use_naive_recurrent_policy), (
                    "check recurrent policy!")
    else:
        raise NotImplementedError

    device = dist_info.device
    if dist_info.is_rank0:
        if device.type == "cuda":
            print("choose to use gpu...")
        else:
            print("choose to use cpu...")
    torch.set_num_threads(cfg.algorithm.n_training_threads)

    setproctitle.setproctitle(
        str(cfg.algorithm.algorithm_name) + "-" + str(cfg.experiment) +
        f"-rank{dist_info.rank}")

    seed = rank_offset_seed(cfg.algorithm.seed, dist_info.rank)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cfg.subscriber.keep_inactive_agents = True
    envs = make_train_env(cfg)
    eval_envs = make_eval_env(cfg) if cfg.algorithm.use_eval else envs
    render_envs = (make_render_env(cfg)
                   if cfg.algorithm.use_render else envs)
    num_agents = envs.reset().shape[1]

    config = {
        "cfg.algo": cfg.algorithm,
        "envs": envs,
        "eval_envs": eval_envs,
        "render_envs": render_envs,
        "num_agents": num_agents,
        "device": device,
        "logdir": logdir,
        "dist_info": dist_info,
    }

    runner = NocturneSharedRunner(config)
    runner.run()

    envs.close()
    if cfg.algorithm.use_eval and eval_envs is not envs:
        eval_envs.close()
    if cfg.algorithm.use_render and render_envs is not envs:
        render_envs.close()

    if cfg.wandb and run is not None:
        run.finish()
    elif runner.writter is not None:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == '__main__':
    main()
