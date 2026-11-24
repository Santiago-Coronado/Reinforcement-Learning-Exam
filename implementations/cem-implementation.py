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
    """
    def sampling(self):
        return 0

    """
    Update the policy distribution based on the elite samples.
    """
    def update_distribution(self, elite_samples):
        return 0

    """
    Plot the performance of the agent over episodes by making a GIF
    """
    def plot(self, n_episodes= env.episodes):
        return 0 
    