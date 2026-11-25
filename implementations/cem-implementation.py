"""
Cross-Entropy Method (CEM) implementation.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from gymnassium.gym_env import CVP_env


class Agent(): 
    def __init__(self, env, samples= 10, elite = 7, alpha = 0.1):
        super().__init__()
        self.env = env
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
        obs, info = self.reset()
        for action in actions:
            # CVP_env has 3 actions: 0, 1, and 2
            env_action = int(np.clip(np.rint(action), 0, self.env.action_space.n - 1))
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            rewards.append(reward)

            if terminated or truncated:
                break
        
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
        agent_current = np.clip(self.mean, x.min(), x.max())
        agent_power = np.interp(agent_current, x, y)
        axs[0].scatter(agent_current, agent_power, color='green', marker='s', label='CEM Agent')
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
    env_ = CVP_env()
    agent = Agent(env_)
    agent.plot()