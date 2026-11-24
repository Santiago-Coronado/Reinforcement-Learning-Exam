"""
Cross-Entropy Method (CEM) implementation.
"""

from gymnassium.gym_env import GridWorldEnv as env
import numpy as np
import matplotlib.pyplot as plt

class Agent(env): 
    def __init__(self, env, samples= 10, elite = 7, alpha = 0.1):
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
    def run(self, n_episodes= env.episodes):
        for episode in range(n_episodes):
            actions = self.sampling()
            rewards_episode = []
            for action in actions:
                obs, reward, terminated, truncated, info = self.env.step(int(action))
                rewards_episode.append(reward)
                if terminated or truncated:
                    break
            elite_indices = np.argsort(rewards_episode)[-self.elite:]
            elite_samples = actions[elite_indices]
            self.update_distribution(elite_samples)
            new_mean = self.mean
            new_std = self.std
            new_samples = self.sampling()
        return new_mean, new_std, new_samples

    """
    Plot the performance of the agent over episodes
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
        n_episodes (int): Number of episodes to run the agent.
    Return:
        Plot of the agent's performance over episodes.
    """
    def plot(self, n_episodes= env.episodes):
        return 0

    """
    Makes a lot of plots and unifies them into a GIF to visualize the agent's performance over time.
    """
    def make_gif(self):
        return 0