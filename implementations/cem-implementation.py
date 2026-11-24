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
    Plot the performance of the agent over episodes
    Args:
        n_episodes (int): Number of episodes to run the agent.
    Return:
        Plot with:
        - The power curve for the scenario currently in effect.
        - The points/samples that the algorithm is evaluating on the curve.
        - The model output (proposed current) at each time step.
        - The name of the method being executed.
        - On the side (or in a side panel):
            - Relevant gradient values (mean/variance).
            - Output from the policy network.
            - A visual indication of when the shading scenario changes.
    """
    def plot(self, n_episodes= env.episodes):
        return 0 
    """
    Makes a lot of plots and unifies them into a GIF to visualize the agent's performance over time.
    """
    def make_gif(self):
        return 0

    git commit -m "Filled some of the functions and added comments to see general behaviour of each function