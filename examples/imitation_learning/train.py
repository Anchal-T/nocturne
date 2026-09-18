# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Imitation learning training script (behavioral cloning)."""
from datetime import datetime
from pathlib import Path
import random
import json
import os
import re

import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        """Use a dependency-free iterator when tqdm is unavailable."""
        return iterable

from examples.imitation_learning.model import ImitationAgent
from examples.imitation_learning.waymo_data_loader import WaymoDataset
from nocturne.utils.distributed import (
    broadcast_object,
    cleanup_distributed,
    control_barrier,
    init_distributed,
    rank_offset_seed,
    reduce_metrics,
    unwrap_module,
    wrap_ddp,
)


def set_seed_everywhere(seed):
    """Ensure determinism."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().mean().cpu())
    return float(value)


def _atomic_torch_save(payload, path):
    """Save a checkpoint without exposing a partially written file."""
    path = Path(path)
    tmp_path = Path(str(path) + ".tmp")
    torch.save(payload, str(tmp_path))
    os.replace(str(tmp_path), str(path))


def _epoch_from_path(path):
    """Infer the completed epoch from legacy ``model_N.pth`` names."""
    match = re.search(r"(?:model|checkpoint)_(\d+)\.(?:pth|pt)$", str(path))
    return int(match.group(1)) if match else 0


def _load_resume_state(model, checkpoint_path, device):
    """Load either a legacy model or a resumable training checkpoint."""
    payload = torch.load(checkpoint_path, map_location=device)
    optimizer_state = None
    completed_epoch = _epoch_from_path(checkpoint_path)
    if isinstance(payload, nn.Module):
        model_state = payload.state_dict()
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        model_state = payload["model_state_dict"]
        optimizer_state = payload.get("optimizer_state_dict")
        completed_epoch = int(payload.get("completed_epoch", completed_epoch))
    elif isinstance(payload, dict):
        model_state = payload
    else:
        raise TypeError(
            "unsupported imitation checkpoint payload: {}".format(
                type(payload).__name__
            )
        )
    model.load_state_dict(model_state)
    return completed_epoch, optimizer_state


@hydra.main(config_path="../../cfgs/imitation", config_name="config")
def main(args):
    """Train an IL model."""
    dist_info = init_distributed(requested_device=args.device)
    try:
        _train_imitation(args, dist_info)
    finally:
        cleanup_distributed()


def _train_imitation(args, dist_info):
    seed = rank_offset_seed(args.seed, dist_info.rank)
    set_seed_everywhere(seed)
    device = dist_info.device
    args.device = str(device)

    # create dataset and dataloader
    if args.actions_are_positions:
        expert_bounds = [[-0.5, 3], [-3, 3], [-0.07, 0.07]]
        actions_discretizations = [21, 21, 21]
        actions_bounds = [[-0.5, 3], [-3, 3], [-0.07, 0.07]]
        mean_scalings = [3, 3, 0.07]
        std_devs = [0.1, 0.1, 0.02]
    else:
        expert_bounds = [[-6, 6], [-0.7, 0.7]]
        actions_bounds = expert_bounds
        actions_discretizations = [15, 43]
        mean_scalings = [3, 0.7]
        std_devs = [0.1, 0.02]

    dataloader_cfg = {
        'tmin': 0,
        'tmax': 90,
        'view_dist': args.view_dist,
        'view_angle': args.view_angle,
        'dt': 0.1,
        'expert_action_bounds': expert_bounds,
        'expert_position': args.actions_are_positions,
        'state_normalization': 100,
        'n_stacked_states': args.n_stacked_states,
        'dist_rank': dist_info.rank,
        'dist_world_size': dist_info.world_size,
    }
    scenario_cfg = {
        'start_time': 0,
        'allow_non_vehicles': True,
        'spawn_invalid_objects': True,
        'max_visible_road_points': args.max_visible_road_points,
        'sample_every_n': 1,
        'road_edge_first': False,
    }
    dataset = WaymoDataset(
        data_path=args.path,
        file_limit=args.num_files,
        dataloader_config=dataloader_cfg,
        scenario_config=scenario_cfg,
    )
    data_loader = iter(
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_cpus,
            pin_memory=device.type == "cuda",
        ))

    sample_state, _ = next(data_loader)
    n_states = int(broadcast_object(sample_state.shape[-1], src=0))

    model_cfg = {
        'n_inputs': n_states,
        'hidden_layers': [1024, 256, 128],
        'discrete': args.discrete,
        'mean_scalings': mean_scalings,
        'std_devs': std_devs,
        'actions_discretizations': actions_discretizations,
        'actions_bounds': actions_bounds,
        'device': str(device),
    }

    model = ImitationAgent(model_cfg).to(device)
    start_epoch = 0
    optimizer_state = None
    resume_from = getattr(args, "resume_from", None)
    if resume_from:
        start_epoch, optimizer_state = _load_resume_state(
            model, resume_from, device
        )
    model = wrap_ddp(
        model,
        dist_info,
        find_unused_parameters=False,
    )
    model.train()
    if dist_info.is_rank0:
        print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if dist_info.is_rank0 and resume_from:
        print(
            "Resumed {} completed epochs from {}".format(
                start_epoch, resume_from
            )
        )

    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if dist_info.is_rank0:
        exp_dir = Path.cwd() / Path('train_logs') / time_str
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_dir = Path(broadcast_object(str(exp_dir), src=0))
    else:
        exp_dir = Path(broadcast_object(None, src=0))

    configs_path = exp_dir / 'configs.json'
    configs = {
        'scenario_cfg': scenario_cfg,
        'dataloader_cfg': dataloader_cfg,
        'model_cfg': model_cfg,
        'world_size': dist_info.world_size,
    }
    if dist_info.is_rank0:
        with open(configs_path, 'w') as fp:
            json.dump(configs, fp, sort_keys=True, indent=4)
        print('Wrote configs at', configs_path)

    writer = None
    if args.write_to_tensorboard and dist_info.is_rank0:
        writer = SummaryWriter(log_dir=str(exp_dir))
    if args.wandb and dist_info.is_rank0:
        wandb.init(config=args,
                   project=args.wandb_project,
                   name=args.experiment,
                   group=args.experiment,
                   resume="allow",
                   settings=wandb.Settings(start_method="fork"),
                   mode="online")

    if dist_info.is_rank0:
        print('Exp dir created at', exp_dir)
        print(f'`tensorboard --logdir={exp_dir}`\n')
    batches_per_epoch = args.samples_per_epoch // args.batch_size
    for epoch in range(start_epoch, args.epochs):
        if dist_info.is_rank0:
            print(f'\nepoch {epoch+1}/{args.epochs}')
        n_samples = (epoch * args.batch_size * batches_per_epoch *
                     dist_info.world_size)

        progress = range(batches_per_epoch)
        if dist_info.is_rank0:
            progress = tqdm(progress, unit='batch')
        for i in progress:
            states, expert_actions = next(data_loader)
            states = states.to(device)
            expert_actions = expert_actions.to(device)

            if args.discrete:
                log_prob, expert_idxs = model(states,
                                              expert_action=expert_actions,
                                              return_indexes=True)
            else:
                log_prob = model(states, expert_action=expert_actions.float())
                dist = unwrap_module(model).dist(states)
            loss = -log_prob.mean()

            metrics_dict = {}

            optimizer.zero_grad()
            # Data generation has variable latency. Ensure every rank has
            # finished its forward pass before launching NCCL gradient work.
            control_barrier(dist_info)
            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            metrics_dict['train/grad_norm'] = total_norm
            metrics_dict['train/post_clip_grad_norm'] = torch.clamp(
                total_norm, max=1.0
            )
            optimizer.step()

            metrics_dict['train/loss'] = loss.item()

            if args.actions_are_positions:
                metrics_dict['train/x_logprob'] = log_prob[0]
                metrics_dict['train/y_logprob'] = log_prob[1]
                metrics_dict['train/steer_logprob'] = log_prob[2]
            else:
                metrics_dict['train/accel_logprob'] = log_prob[0]
                metrics_dict['train/steer_logprob'] = log_prob[1]

            if not model_cfg['discrete']:
                diff_actions = torch.mean(torch.abs(dist.mean -
                                                    expert_actions),
                                          axis=0)
                metrics_dict['train/accel_diff'] = diff_actions[0]
                metrics_dict['train/steer_diff'] = diff_actions[1]
                metrics_dict['train/l2_dist'] = torch.norm(
                    dist.mean - expert_actions.float())

            if model_cfg['discrete']:
                with torch.no_grad():
                    model_actions, model_idxs = unwrap_module(model)(
                        states,
                        deterministic=True,
                        return_indexes=True)
                accuracy = [
                    (model_idx == expert_idx).float().mean(axis=0)
                    for model_idx, expert_idx in zip(model_idxs, expert_idxs.T)
                ]
                if args.actions_are_positions:
                    metrics_dict['train/x_pos_acc'] = accuracy[0]
                    metrics_dict['train/y_pos_acc'] = accuracy[1]
                    metrics_dict['train/heading_acc'] = accuracy[2]
                else:
                    metrics_dict['train/accel_acc'] = accuracy[0]
                    metrics_dict['train/steer_acc'] = accuracy[1]

            metrics_dict = {
                key: _as_float(val)
                for key, val in metrics_dict.items()
            }
            metrics_dict = reduce_metrics(metrics_dict, device)
            n_samples += args.batch_size * dist_info.world_size

            if dist_info.is_rank0:
                for key, val in metrics_dict.items():
                    if writer is not None:
                        writer.add_scalar(key, val, n_samples)
                if args.wandb:
                    wandb.log(metrics_dict, step=n_samples)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            # Rank 1 must not begin another backward while rank 0 performs
            # slow checkpoint I/O on the shared filesystem.
            control_barrier(dist_info)
            if dist_info.is_rank0:
                model_path = exp_dir / f'model_{epoch+1}.pth'
                checkpoint_path = exp_dir / f'checkpoint_{epoch+1}.pt'
                _atomic_torch_save(unwrap_module(model), model_path)
                _atomic_torch_save(
                    {
                        "completed_epoch": epoch + 1,
                        "model_state_dict": unwrap_module(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_cfg": model_cfg,
                    },
                    checkpoint_path,
                )
                print(f'\nSaved model at {model_path}')
                print(f'Saved resumable checkpoint at {checkpoint_path}')
            control_barrier(dist_info)
        if args.discrete and dist_info.is_rank0:
            if args.actions_are_positions:
                print('xpos')
                print('model: ', model_idxs[0][0:10])
                print('expert: ', expert_idxs[0:10, 0])
                print('ypos')
                print('model: ', model_idxs[1][0:10])
                print('expert: ', expert_idxs[0:10, 1])
                print('steer')
                print('model: ', model_idxs[2][0:10])
                print('expert: ', expert_idxs[0:10, 2])
            else:
                print('accel')
                print('model: ', model_idxs[0][0:10])
                print('expert: ', expert_idxs[0:10, 0])
                print('steer')
                print('model: ', model_idxs[1][0:10])
                print('expert: ', expert_idxs[0:10, 1])

    control_barrier(dist_info)
    if dist_info.is_rank0:
        print('Done, exp dir is', exp_dir)
        if writer is not None:
            writer.flush()
            writer.close()
        if args.wandb:
            wandb.finish()


if __name__ == '__main__':
    main()
