"""HER buffer sampling stays O(1) per transition and returns finite goals."""
import numpy as np

from examples.drl_collision_avoidance.crl_modules.her_buffer import HERReplayBuffer


def _fill(buffer, n_episodes, length=8):
    for ep in range(n_episodes):
        for t in range(length):
            state = np.full(4, ep + t * 0.01, dtype=np.float32)
            action = np.array([0.1, -0.2], dtype=np.float32)
            ego = np.array([t * 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
            buffer.add(state, action, ego, done=(t == length - 1), env_id=0)


def test_her_ring_buffer_caps_and_samples():
    buf = HERReplayBuffer(
        state_dim=4, action_dim=2, goal_dim=2, max_episodes=32, num_envs=1
    )
    _fill(buf, 80)
    assert buf.num_episodes == 32
    batch = buf.sample(64)
    assert batch is not None
    assert batch["obs"].shape == (64, 4)
    assert batch["action"].shape == (64, 2)
    assert batch["goal"].shape == (64, 2)
    assert np.isfinite(batch["goal"]).all()
