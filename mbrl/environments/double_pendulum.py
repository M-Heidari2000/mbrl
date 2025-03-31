from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import DtypeObservation
from gymnasium import spaces


class DoublePendulum(gym.Env):

    def __init__(
        self,
        render_mode: Optional[str]=None,
        horizon: int=100,
        **kwargs,
    ):
        env = gym.make(
            "Reacher-v5",
            render_mode=render_mode,
            max_episode_steps=horizon,
            **kwargs,
        )

        self.wrapped_env = DtypeObservation(env=env, dtype=np.float32)

        self.horizon = horizon

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8, ),
            dtype=np.float32
        )

        self.action_space = self.wrapped_env.action_space

    def reset(self, *args, **kwargs):
        obs, info = self.wrapped_env.reset(*args, **kwargs)
        self.target = obs[[4, 5]]
        obs[[8, 9]] = obs[[8, 9]] + self.target
        obs = obs[[0, 1, 2, 3, 6, 7, 8, 9]]
        
        return obs, info

    def step(self, *args, **kwargs):
        return self.wrapped_env.step(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.wrapped_env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.wrapped_env.render(*args, **kwargs)