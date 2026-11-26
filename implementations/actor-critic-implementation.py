"""
Actor-Critic implementation 
"""
from io import BytesIO
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add the parent folder to the system path to allow imports from the implementations folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymnassium.gym_env import CVP_env

class Agent:
    def __init__(self, env, actor_learning_rate=0.01, critic_learning_rate=0.05, gamma=0.99): 
        self.env = env
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.action_count = env.action_space.n


    def probabilities(self, observation):
        return 0

    def action(self, observation):
        return 0

    def V(self, observation):
        return 0

    def train(self, episodes=500, seed=42):
        return 0

    def evaluate(self, frames=None, episode=1, seed=None):
        return 0

    def plot(self, step=None, episode=None, show=True, history=None):
        return 0

    def make_gif(self, output_path=None, episodes=1, duration=100):
        return 0