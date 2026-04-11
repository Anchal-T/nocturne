import torch
import numpy as np

from examples.drl_collision_avoidance.dqn_modules.q_network import (
    QNetwork,
    _lecun_init_linear,
)
from examples.drl_collision_avoidance.vec_env import AsyncSubprocVecEnv


class _RemoteStub:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


def _make_async_vec_env(n_envs=4, num_envs_per_worker=2):
    env = AsyncSubprocVecEnv.__new__(AsyncSubprocVecEnv)
    env.n_envs = n_envs
    env.num_envs_per_worker = num_envs_per_worker
    env._action_buf = torch.full((n_envs,), -1.0, dtype=torch.float32)
    env._pending_workers = set()
    env.closed = True
    env.remotes = [
        _RemoteStub() for _ in range((n_envs + num_envs_per_worker - 1) // num_envs_per_worker)
    ]
    return env


def test_async_step_preserves_env_action_mapping():
    env = _make_async_vec_env()

    env.step_async(
        actions=np.array([30.0, 20.0], dtype=np.float32),
        env_ids=np.array([3, 2], dtype=np.int64),
    )

    assert env._action_buf.tolist() == [-1.0, -1.0, 20.0, 30.0]
    assert env.remotes[1].messages == [("step", None)]


def test_q_network_mlp_depth_two_uses_legacy_two_layer_head():
    grid_channels = 3
    grid_size = grid_channels * 25 * 14
    obs_dim = grid_size + 4 + 2 + 3

    net = QNetwork(
        obs_dim=obs_dim,
        n_actions=15,
        grid_size=grid_size,
        grid_channels=grid_channels,
        grid_rows=25,
        grid_cols=14,
        dueling=True,
        noisy=False,
        mlp_depth=2,
    )

    assert len(net.advantage_head.residual_blocks) == 0
    assert len(net.value_head.residual_blocks) == 0
    assert not any("residual_blocks" in key for key in net.state_dict())


def test_lecun_init_linear_uses_scaling_crl_uniform_bound():
    torch.manual_seed(0)
    layer = torch.nn.Linear(1, 4096, bias=True)

    _lecun_init_linear(layer)

    assert layer.weight.abs().max().item() <= 1.0 + 1e-6
    assert torch.count_nonzero(layer.bias).item() == 0
