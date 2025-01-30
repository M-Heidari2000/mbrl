from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RescaleAction, TimeLimit
from gymnasium import spaces
from gymnasium.envs.classic_control.pendulum import PendulumEnv


class Pendulum(gym.Env):

    def __init__(
        self,
        render_mode: Optional[str]=None,
        horizon: int=100,
        g: float=10.0,
        heatmap_steps: float=0.1,
    ):
        self.env = PendulumEnv(
            render_mode=render_mode,
            g=g,
        )

        self.wrapped_env = TimeLimit(self.env, max_episode_steps=horizon)

        self.wrapped_env = RescaleAction(
            env=self.wrapped_env,
            min_action=-1.0,
            max_action=1.0,
        )

        self.state_space = spaces.Box(
            low=np.array([0.0, -8.0]),
            high=np.array([2*np.pi, 8.0]),
            shape=(2, ),
            dtype=np.float32,
        )

        self.heatmap_steps = heatmap_steps
        self._heatmap = np.zeros(
            np.ceil((self.state_space.high - self.state_space.low) / self.heatmap_steps).astype(np.int32)+1,
        )
        self.target = np.array([0.0, 0.0]).reshape(-1, 1)

    def manifold(s):
        assert s.shape[0] == 2
        e = np.stack([
            np.cos(s[0, :]),
            np.sin(s[0, :]),
            s[1, :],
        ], axis=0)
        return e
    
    @property
    def observation_space(self):
        return self.wrapped_env.observation_space
    
    @property
    def action_space(self):
        return self.wrapped_env.action_space

    @property
    def state(self):
        return self.env.state

    def reset(self, *args, **kwargs):
        return self.wrapped_env.reset(*args, **kwargs)
    
    def step(self, *args, **kwargs):
        # Update heatmap
        state_idx = tuple(
            np.floor((self.env.state - self.state_space.low) / self.heatmap_steps).astype(np.int32)
        )
        self._heatmap[state_idx] += 1

        return self.wrapped_env.step(self, *args, **kwargs)

    def close(self, *args, **kwargs):
        return self.wrapped_env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.wrapped_env.render(*args, **kwargs)

    def reset_heatmap(self):
        self._heatmap = self._heatmap * 0

    def get_heatmap(self):
        return self._heatmap

