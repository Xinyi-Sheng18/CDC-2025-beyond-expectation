# This environment is adapted from:
# https://github.com/baumanndominik/ergodic_rl
# Original license: MIT
# Reference: Baumann, D., et al. "Reinforcement learning with non-ergodic reward increments:
# robustness via ergodicity transformations," Transactions on Machine Learning Research, 2025, arXiv.
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CoinTossSimple(gym.Env):
    """
    A simplified coin toss environment:
    - Action: how much to bet (fraction of current wealth)
    - State: either 0 or 1 (random), representing environment condition
    - Reward: gain or loss based on state and bet
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=1e-10, high=0.99999, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Discrete(2)  # state ∈ {0, 1}
        self.initial_wealth = 100.0
        self.cum_reward = self.initial_wealth
        self.state = None
        self.max_episode_steps = 10_000
        self.episode_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cum_reward = self.initial_wealth
        self.state = np.random.randint(2)
        self.episode_steps = 0
        return self.state, {}  # observation, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        self.episode_steps += 1

        self.state = np.random.randint(2)

        if self.cum_reward < 1e-10:
            self.cum_reward = 1e-10

        if self.state == 0:
            reward = -0.4 * action * self.cum_reward
        else:
            reward = 0.5 * action * self.cum_reward

        self.cum_reward += reward

        done = self.episode_steps >= self.max_episode_steps
        obs = self.state
        info = {"wealth": self.cum_reward}

        return obs, reward.item(), done, False, info

    def render(self, mode="human"):
        print(f"Step: {self.episode_steps}, Wealth: {self.cum_reward}, State: {self.state}")

    def close(self):
        pass
