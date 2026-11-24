"""
Cross-Entropy Method (CEM) implementation.
"""

#from .gymnassium.gym_env import GridWorldEnv as env
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import numpy as np
import gymnasium as gym
EPISODES = 300
SEED = 42
BATCH = 50 # step for changing case

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, size: int = 5, render_mode: Optional[str] = None):
        self.size = size
        self.episodes = EPISODES
        self.seed = SEED
        self.batch = BATCH
        self.render_mode = render_mode
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._current_case = np.random.choice([0, 1, 2, 3])  # Randomly select one of the four cases
        self._current_current = np.random.uniform(0, 1)  # Randomly select a current value
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [row, col] coordinates
                "current_case": gym.spaces.Discrete(4),  # Current case (0, 1, 2, or 3)
                "current_value": gym.spaces.Box(0, 1, shape=(), dtype=float),  # Current value
                "current_gradient": gym.spaces.Box(-1, 1, shape=(), dtype=float),  # Current gradient
            }
        )
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }
        self._current_gradient = 0.0  # Initialize the gradient

    def _get_obs(self):
        return {"agent": self._agent_location.copy(), "current_case": self._current_case, "current_value": self._current_current, "current_gradient": self._current_gradient}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        return self._get_obs(), self._get_info()

    def step(self, action):
        # every step, we need to check if we need to change the case/dataset
        for step in range(BATCH):
            # Perform the action
            self._current_current = np.clip(self._current_current + (action - 1) * 0.1, 0, 1)  # Adjust the current value
            self._current_power = self._current_current ** 2  # Calculate the power
            self._current_case = np.random.choice([0, 1, 2, 3])  # Randomly select one of the four cases
            self._current_gradient = (self._current_current - 0.5) * 2  # Calculate the gradient
            direction = self._action_to_direction[action]
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            for row in range(self.size):
                line = ""
                for col in range(self.size):
                    if np.array_equal([row, col], self._agent_location):
                        line += "A "
                    elif np.array_equal([row, col], self._target_location):
                        line += "T "
                    else:
                        line += ". "
                print(line)
            print()

class Agent(GridWorldEnv): 
    def __init__(self, samples= 10, elite = 7, alpha = 0.1):
        super().__init__()
        self.samples = samples
        self.elite = elite
        self.alpha = alpha
        self.mean = 0.0
        self.std = 1.0

    """
    Sample actions from the current policy.
    Args:
        None
    Return:
        actions (np.array): New samples based on the current distribution.
    """
    def sampling(self):
        actions = np.random.normal(self.mean, self.std, self.samples)
        return actions

    """
    Update the policy distribution based on the elite samples.
    Args:
        elite_samples (np.array): The elite samples used to update the current and power of agent("position" in the graphic).
    Return:
        None
    """
    def update_distribution(self, elite_samples):
        new_mean = np.mean(elite_samples)
        new_std = np.std(elite_samples)
        self.mean = self.alpha * new_mean + (1 - self.alpha) * self.mean
        self.std = self.alpha * new_std + (1 - self.alpha) * self.std

    """
    Run the agent for a specified number of episodes.
    Args:
        n_episodes (int): Number of episodes to run the agent.
    Return:
        new_mean (float): Updated mean of the distribution.
        new_std (float): Updated standard deviation of the distribution.
        new_samples (np.array): New samples based on the updated distribution.
    """
    def run(self):
        actions = self.sampling()
        rewards = []
        self.reset()
        for action in actions:
            obs, reward, terminated, truncated, info = self.step(int(np.clip(action, 0, 3)))
            rewards.append(reward)
        elite_indices = np.argsort(rewards)[-self.elite:]
        elite_samples = actions[elite_indices]
        self.update_distribution(elite_samples)
        new_mean = self.mean
        new_std = self.std
        new_samples = self.sampling()
        return new_mean, new_std, new_samples

    """
    Plot of subplots showing the performance of the agent over one episode
    One plot with:
        - The power curve for the scenario currently in effect.
        - The points/samples that the algorithm is evaluating on the curve.
        - The model output (proposed current) at each time step.
        - The name of the method being executed.
    Another plot with:
        - On the side (or in a side panel):
        - Relevant gradient values (mean/variance).
        - Output from the policy network.
        - A visual indication of when the shading scenario changes.
    Args:
        None
    Return:
        Plot of the agent's performance over one episode.
    """
    def plot(self):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Agent, samples and line of case data
        case_path = Path(__file__).resolve().parents[1] / "src" / f"Caso{self._current_case + 1}.csv"
        curve = np.genfromtxt(case_path, delimiter=",", names=True, encoding="utf-8")
        x = curve["corriente"]
        y = curve["potencia"]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        axs[0].plot(x, y, label='Power Curve', color='blue')
        samples = np.clip(self.sampling(), x.min(), x.max())
        axs[0].scatter(samples, np.interp(samples, x, y), color='red', label='Samples')
        axs[0].set_title('Power Curve and Samples')
        axs[0].set_xlabel('Current')
        axs[0].set_ylabel('Power')
        axs[0].legend()

        # Metrics
        gradient_values = [self.mean - self.std, self.mean, self.mean + self.std]
        policy_output = [self.mean] * len(gradient_values)
        axs[1].bar(range(len(gradient_values)), gradient_values, color='green', alpha=0.6, label='Gradient Values')
        axs[1].plot(range(len(policy_output)), policy_output, color='orange', label='Policy Output', marker='o')
        axs[1].set_title('Gradient Values and Policy Output')
        axs[1].set_xlabel('Sample Index')
        axs[1].set_ylabel('Value')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    """
    Makes a lot of plots and unifies them into a GIF to visualize the agent's performance over time.
    """
    def make_gif(self):
        return 0

if __name__ == "__main__":
    # ========== Testing ==========
    env_ = GridWorldEnv()
    agent = Agent()
    agent.plot()