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

        self.policy_weights = np.zeros((3, self.action_count), dtype=float)
        self.value_weights = np.zeros(3, dtype=float)

    """
    Gives bias, current, and power features from an observation.
    Args:
        observation: The current state of the environment.
    Returns:
        A numpy array containing the features.
        1.0: bias to help with later calculations.
        float(observation[0]): current feature.
        float(observation[1]): power feature.
    """
    def features(self, observation):
            return np.array([1.0, float(observation[0]), float(observation[1])], dtype=float)
    
    """
    Get action probabilities from policy using a linear softmax policy.
    Args:
        observation: The current state of the environment.
    Returns:
        A numpy array containing the probabilities of each action.
    """
    def probabilities(self, observation):
        logits = self.features(observation) @ self.policy_weights
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        return probabilities / np.sum(probabilities)
    
    """
    Choose the most likely action from policy.
    Args:
        observation: The current state of the environment.
    Returns:
        The action to take.
    """
    def action(self, observation):
        probabilities = self.probabilities(observation)
        return int(np.random.choice(self.action_count, p=probabilities))

    """
    Get state value from critic.
    Args:
        observation: The current state of the environment.
    Returns:
        state value of the observation.
    """
    def V(self, observation):
        return float(self.features(observation) @ self.value_weights)

    def train(self, episodes=500, seed=42):
            episode_returns = []
            for episode_index in range(episodes):
                observation, _ = self.env.reset(seed=seed + episode_index)
                episode_return = 0.0
                terminated = False
                truncated = False
    
                while not (terminated or truncated):
                    state_features = self.features(observation)
                    action_probabilities = self.probabilities(observation)
                    action = int(np.random.choice(
                        self.action_count,
                        p=action_probabilities,
                    ))
                    next_observation, reward, terminated, truncated, _ = (
                        self.env.step(action)
                    )
    
                    baseline = self.V(observation)
                    next_value = (
                        0.0 if terminated else self.V(next_observation)
                    )
                    advantage = reward + self.gamma * next_value - baseline

                    # Update weights using advantage.
                    self.value_weights += (
                        self.critic_learning_rate * advantage * state_features
                    )
                    action_gradient = -action_probabilities
                    action_gradient[action] += 1.0 # Give action taken a boost 
                    self.policy_weights += (
                        self.actor_learning_rate
                        * advantage
                        * np.outer(state_features, action_gradient) 
                    )
    
                    observation = next_observation
                    episode_return += float(reward)
    
                episode_returns.append(episode_return)
    
            return episode_returns

    """
    Evaluate the policy on a given number of episodes. Also prepares data for visualization
    Args:
        frames: A list to store the frames of the evaluation.
        episode: The episode number.
        seed: The random seed for reproducibility.
    Returns:
        The total return of the evaluation.
    """
    def evaluate(self, frames=None, episode=1, seed=None):
        observation, info = self.env.reset(seed=seed)
        episode_return = 0.0
        step = 0
        visual_history = None

        if frames is not None:
            visual_history = [{
                "step": 0,
                "case": info["current_case"],
                "current": info["current"],
                "power": info["power"],
                "reward": 0.0,
                "action": None,
            }]
            frames.append(self.plot(
                step=step,
                episode=episode,
                show=False,
                history=visual_history,
            ))

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = self.action(observation, deterministic=True)
            observation, reward, terminated, truncated, info = self.env.step(action)
            episode_return += float(reward)
            step += 1

            if visual_history is not None:
                visual_history.append({
                    "step": step,
                    "case": info["evaluated_case"],
                    "current": info["evaluated_current"],
                    "power": info["evaluated_power"],
                    "reward": reward,
                    "action": action,
                })
                frames.append(self.plot(
                    step=step,
                    episode=episode,
                    show=False,
                    history=visual_history,
                ))

        return episode_return

    def plot(self, step=None, episode=None, show=True, history=None):
        return 0

    def make_gif(self, output_path=None, episodes=1, duration=100):
        return 0