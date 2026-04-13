import numpy as np
import torch

from examples.drl_collision_avoidance import train
from examples.drl_collision_avoidance.dqn_modules.q_network import (
    QNetwork,
    _lecun_init_linear,
)
from examples.drl_collision_avoidance.vec_env import AsyncSubprocVecEnv, RayAsyncVecEnv


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


class _AgentStub:
    def __init__(self, next_actions):
        self.next_actions = np.asarray(next_actions, dtype=np.int64)
        self.calls = []

    def select_action_batch(self, obs):
        self.calls.append(np.asarray(obs).copy())
        return self.next_actions.copy()


class _RayCollectorVecEnvStub:
    def __init__(self):
        self.step_wait_calls = []
        self.step_async_calls = []

    def step_wait(self, min_ready):
        self.step_wait_calls.append(min_ready)
        return (
            np.array([2, 3], dtype=np.int64),
            np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([False, True]),
            [{"done": False}, {"done": True}],
        )

    def step_async(self, actions, env_ids=None):
        self.step_async_calls.append(
            (np.asarray(actions).copy(), None if env_ids is None else np.asarray(env_ids).copy())
        )

    def step(self, actions):
        raise AssertionError("ray mode should use step_wait/step_async, not synchronous step()")


class _RayRemoteRecorder:
    def __init__(self):
        self.calls = []
        self.step = self

    def remote(self, actions):
        self.calls.append(np.asarray(actions).copy())
        return f"pending-{len(self.calls)}"


class _TrainingVecEnvStub:
    def __init__(self, obs):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.step_async_calls = []

    def reset(self):
        return self.obs.copy()

    def step_async(self, actions, env_ids=None):
        self.step_async_calls.append(
            (np.asarray(actions).copy(), None if env_ids is None else np.asarray(env_ids).copy())
        )


def test_collect_transition_batch_uses_async_api_for_ray_mode():
    vec_env = _RayCollectorVecEnvStub()
    agent = _AgentStub(next_actions=[7, 8])
    obs = np.zeros((4, 2), dtype=np.float32)
    current_obs = np.array(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        dtype=np.float32,
    )
    current_actions = np.array([0, 1, 2, 3], dtype=np.int64)

    step_batch, _, updated_obs, updated_actions, step_size = train._collect_transition_batch(
        vec_env=vec_env,
        vec_env_mode="ray",
        obs=obs,
        current_obs=current_obs,
        current_actions=current_actions,
        num_envs=4,
        agent=agent,
        min_ready_fraction=0.5,
    )

    ready_ids, prev_obs, prev_actions, next_obs, rewards, dones, infos = step_batch
    assert vec_env.step_wait_calls == [2]
    assert len(vec_env.step_async_calls) == 1
    dispatched_actions, dispatched_ids = vec_env.step_async_calls[0]
    np.testing.assert_array_equal(dispatched_actions, np.array([7, 8], dtype=np.int64))
    np.testing.assert_array_equal(dispatched_ids, np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(ready_ids, np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(prev_obs, np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32))
    np.testing.assert_array_equal(prev_actions, np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(next_obs, np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32))
    np.testing.assert_array_equal(rewards, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(dones, np.array([False, True]))
    assert infos == [{"done": False}, {"done": True}]
    np.testing.assert_array_equal(updated_obs[2:], next_obs)
    np.testing.assert_array_equal(updated_actions, np.array([0, 1, 7, 8], dtype=np.int64))
    assert step_size == 2


def test_ray_async_step_async_preserves_subset_action_alignment():
    env = RayAsyncVecEnv.__new__(RayAsyncVecEnv)
    env.n_envs = 4
    env.num_workers = 2
    env.num_envs_per_worker = 2
    env.closed = True
    env._pending = {}
    env._workers = [_RayRemoteRecorder(), _RayRemoteRecorder()]

    env.step_async(
        actions=np.array([20.0, 30.0], dtype=np.float32),
        env_ids=np.array([2, 3], dtype=np.int64),
    )

    assert env._workers[0].calls == []
    assert len(env._workers[1].calls) == 1
    np.testing.assert_array_equal(env._workers[1].calls[0], np.array([20.0, 30.0], dtype=np.float32))
    assert env._pending == {1: "pending-1"}


def test_run_training_loop_kicks_off_async_collection_for_ray_mode(monkeypatch):
    vec_env = _TrainingVecEnvStub(obs=[[1.0, 2.0], [3.0, 4.0]])
    agent = _AgentStub(next_actions=[5, 6])
    running_state = {"running": True}

    def fake_collect_transition_batch(**kwargs):
        return (
            (
                np.array([0], dtype=np.int64),
                np.array([[1.0, 2.0]], dtype=np.float32),
                np.array([5], dtype=np.int64),
                np.array([[7.0, 8.0]], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([True]),
                [{}],
            ),
            kwargs["obs"],
            kwargs["current_obs"],
            kwargs["current_actions"],
            1,
        )

    monkeypatch.setattr(train, "_collect_transition_batch", fake_collect_transition_batch)
    monkeypatch.setattr(train, "_submit_or_train_batch", lambda **kwargs: None)
    monkeypatch.setattr(train, "_process_done_episodes", lambda *args, **kwargs: 1)

    episodes_completed, global_step = train._run_training_loop(
        vec_env=vec_env,
        vec_env_mode="ray",
        agent=agent,
        learner=None,
        num_envs=2,
        num_episodes=1,
        min_replay=1,
        train_freq=1,
        min_ready_fraction=0.5,
        log_interval=1,
        save_interval=100,
        checkpoint_dir="/tmp",
        writer=None,
        start_time=0.0,
        running_state=running_state,
    )

    assert episodes_completed == 1
    assert global_step == 1
    assert len(vec_env.step_async_calls) == 1
    dispatched_actions, dispatched_ids = vec_env.step_async_calls[0]
    np.testing.assert_array_equal(dispatched_actions, np.array([5, 6], dtype=np.int64))
    assert dispatched_ids is None


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
