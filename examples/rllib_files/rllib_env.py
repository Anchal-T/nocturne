"""RLlib adapter kept in an importable module for Ray worker processes."""

import gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from nocturne.envs.wrappers import create_env


def _to_legacy_gym_space(space):
    """Convert Gymnasium spaces to the Gym spaces expected by Ray 1.x."""
    if space.__class__.__name__ == "Discrete":
        return gym.spaces.Discrete(int(space.n))
    if space.__class__.__name__ == "Box":
        return gym.spaces.Box(
            np.asarray(space.low),
            np.asarray(space.high),
            dtype=space.dtype,
        )
    return space


class RLlibWrapperEnv(MultiAgentEnv):
    """Thin wrapper making our environment look like a MultiAgentEnv."""

    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, env):
        self._env = env
        self._observation_space = _to_legacy_gym_space(env.observation_space)
        self._action_space = _to_legacy_gym_space(env.action_space)
        self._skip_env_checking = True
        super().__init__()

    def step(self, actions):
        result = self._env.step(actions)
        if len(result) == 5:
            next_obs, rew, terminated, truncated, info = result
            done = {
                key: bool(terminated.get(key, False)
                          or truncated.get(key, False))
                for key in set(terminated) | set(truncated)
            }
            done["__all__"] = bool(
                terminated.get("__all__", False)
                or truncated.get("__all__", False)
            )
            return next_obs, rew, done, info
        return result

    def reset(self):
        result = self._env.reset()
        return result[0] if isinstance(result, tuple) else result

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def render(self, mode=None):
        return self._env.render()

    def seed(self, seed=None):
        self._env.seed(seed)

    def __getattr__(self, name):
        return getattr(self._env, name)


def create_rllib_env(cfg):
    """Create a Ray 1.x-compatible Nocturne multi-agent environment."""
    return RLlibWrapperEnv(create_env(cfg))
