import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class Linear(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    def __init__(
        self,
        A,
        b,
        B,
        Q,
        R,
        Ns=None,
        No=None,
        render_mode: str=None,
        horizon: int= 1000,
        heatmap_steps: float=0.1,
    ):
        # Verify parameters' shapes
        assert A.shape == (1, 1)
        self.A = A.astype(np.float32)

        assert b.shape == (1, )
        # Convert to column vector
        self.b = b.astype(np.float32).reshape(-1, 1)

        assert B.shape[0] == 1
        self.action_dim = B.shape[1]
        self.B = B.astype(np.float32)
        
        assert Q.shape == (1, 1)
        self.Q = Q.astype(np.float32)
        assert R.shape == (self.action_dim, self.action_dim)
        self.R = R.astype(np.float32)

        self.Ns = Ns
        self.No = No
        if Ns is not None:
            assert Ns.shape == (1, 1)
            self.Ns = Ns.astype(np.float32)
        if No is not None:
            assert No.shape == (1, 1)
            self.No = No.astype(np.float32)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.horizon = horizon

        self.state_space = spaces.Box(
            low=-4.0,
            high=4.0,
            shape=(1, ),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim, ),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, ),
            dtype=np.float32,
        )

        self.target = (0.5 * (self.state_space.low + self.state_space.high)).reshape(-1, 1)

        self.heatmap_steps = heatmap_steps
        self._heatmap = np.zeros(
            np.ceil((self.state_space.high - self.state_space.low) / self.heatmap_steps).astype(np.int32) + 1,
        )

    def manifold(self, s):
        assert s.shape[0] == 1
        return s

    def _get_obs(self):
        obs = self.manifold(self._state)
        if self.No is not None:
            no = self.np_random.multivariate_normal(
                mean=np.zeros(self.observation_space.shape),
                cov=self.No,
            ).astype(np.float32).reshape(-1, 1)
            obs = obs + no
        return obs

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        
        super().reset(seed=seed)
        options = options or {}
        initial_state = options.get("initial_state")
        
        if initial_state is not None:
            assert initial_state.shape == self.state_space.shape
            self._state = initial_state.astype(np.float32).reshape(-1, 1)
        else:
            self._state = self.state_space.sample().reshape(-1, 1)
        
        self._step = 1
        observation = self._get_obs().flatten()
        info = {"state": self._state.copy().flatten()}

        return observation, info
            
    def step(
        self,
        action
    ):
        assert action.shape == self.action_space.shape
        action = action.astype(np.float32).reshape(-1, 1)

        # Update heatmap
        state_idx = tuple(
            np.floor((self._state.flatten() - self.state_space.low) / self.heatmap_steps).astype(np.int32)
        )
        self._heatmap[state_idx] += 1

        # Calculate reward for current state and action
        reward = -((self._state - self.target).T @ self.Q @ (self._state - self.target)) - (action.T @ self.R @ action)

        # Calculate Next step
        self._state = self.A @ self._state + self.b + self.B @ action
        if self.Ns is not None:
            ns = self.np_random.multivariate_normal(
                mean=np.zeros(self.state_space.shape),
                cov=self.Ns,
            ).astype(np.float32).reshape(-1, 1)
            self._state = self._state + ns

        # Check if the state is valid
        valid_state = (
            np.all(self.state_space.low < self._state.flatten()) and np.all(self._state.flatten() < self.state_space.high)
        )

        self._step += 1        
        info = {"state": self._state.copy().flatten()}
        truncated = bool(self._step >= self.horizon)
        terminated = not valid_state
        reward = reward.item()
        obs = self._get_obs().flatten()

        return obs, reward, terminated, truncated, info 

    def render(self):
        pass

    def reset_heatmap(self):
        self._heatmap = self._heatmap * 0

    def show_heatmap(self):
        s = np.arange(
            self.state_space.low.item(),
            self.state_space.high.item() + self.heatmap_steps,
            self.heatmap_steps,
        )
        plt.bar(
            x=s,
            height=self._heatmap,
            width=self.heatmap_steps,
            edgecolor="black",
            facecolor="#a1c9f4",
        )
        plt.xlabel("s")
        plt.ylabel("count")
        plt.title("collected data")
        plt.show()