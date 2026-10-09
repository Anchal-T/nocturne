"""Compatibility helpers for legacy Ray releases."""

import numpy as np


def patch_legacy_gym_monitor() -> None:
    """Restore the Gym symbol required by Ray 1.x at import time."""
    try:
        import gym
    except ImportError:
        return

    if not hasattr(gym.wrappers, "Monitor"):
        class LegacyMonitor(gym.Wrapper):
            """No-op replacement used when recording is disabled."""

            def __init__(self, env, *args, **kwargs):
                super().__init__(env)

        gym.wrappers.Monitor = LegacyMonitor

    # Gym 0.23 stores a custom NumPy Generator when a space is seeded. Ray
    # 1.11 deep-copies policy spaces while building its multi-GPU model
    # towers, but that custom Generator cannot be deep-copied on this
    # Python/NumPy stack. A standard Generator has the same sampling API and
    # remains deepcopyable.
    if not getattr(gym.spaces.Space, "_no_generator_seed", False):
        def seed_without_generator(self, seed=None):
            self._np_random = np.random.default_rng(seed)
            return [seed]

        gym.spaces.Space.seed = seed_without_generator
        gym.spaces.Space._no_generator_seed = True
