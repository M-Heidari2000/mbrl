from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from gymnasium.wrappers import RescaleAction, TimeLimit, OrderEnforcing
from gymnasium import spaces
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize


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

        self.wrapped_env = OrderEnforcing(self.wrapped_env)

        self.wrapped_env = RescaleAction(
            env=self.wrapped_env,
            min_action=-1.0,
            max_action=1.0,
        )

        self.state_space = spaces.Box(
            low=np.array([-np.pi, -8.0]),
            high=np.array([np.pi, 8.0]),
            shape=(2, ),
            dtype=np.float32,
        )

        self.horizon = horizon

        self.heatmap_steps = heatmap_steps
        self._heatmap = np.zeros(
            np.ceil((self.state_space.high - self.state_space.low) / self.heatmap_steps).astype(np.int32)+1,
        )

        self.target = np.array([0.0, 0.0]).reshape(-1, 1)

    def manifold(self, s):
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
    def _state(self):
        th, thdot = self.env.state
        th = angle_normalize(th)
        return np.array([th, thdot]).reshape(-1, 1)

    def reset(self, *args, **kwargs):
        obs, info = self.wrapped_env.reset(*args, **kwargs)
        info["state"] = self._state.flatten()
        return obs, info

    def step(self, *args, **kwargs):
        
        # Update heatmap
        state_idx = tuple(
            np.floor((self._state.flatten() - self.state_space.low) / self.heatmap_steps).astype(np.int32)
        )
        self._heatmap[state_idx] += 1

        obs, reward, terminated, truncated, info = self.wrapped_env.step(*args, **kwargs)
        info["state"] = self._state.flatten()
        return obs, reward, terminated, truncated, info

    def close(self, *args, **kwargs):
        return self.wrapped_env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.wrapped_env.render(*args, **kwargs)

    def reset_heatmap(self):
        self._heatmap = self._heatmap * 0

    def show_heatmap(self, cmap: Optional[str]=None):
        h = self._heatmap.T[::-1]
        y, x = h.shape
        x_low, y_low = np.round(self.state_space.low, 2)
        x_hi, y_hi = np.round(self.state_space.high, 2)
        ax = sns.heatmap(
            h,
            xticklabels=[str(x_low)] + [None]*(x-2) + [str(x_hi)],
            yticklabels=[str(y_hi)] + [None]*(y-2) + [str(y_low)],
            cmap=cmap,
        )
        ax.set_xlabel("angle")
        ax.set_ylabel("angular velocity")
        ax.set_title("collected data")
        plt.show()