"""
Cross-Entropy Method (CEM) implementation.
"""
from pathlib import Path
from io import BytesIO
import sys

# Add the parent folder to the system path to allow imports from the implementations folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from gymnassium.gym_env import CVP_env
from PIL import Image, ImageDraw


class Agent(): 
    def __init__(self, env, samples=10, elite=3, alpha=0.9):
        super().__init__()
        self.env = env
        self.samples = samples
        self.elite = elite
        self.alpha = alpha
        self.parameter_count = 2 * env.action_space.n
        self.mean = np.zeros(self.parameter_count)
        self.std = np.ones(self.parameter_count)

    """
    Sample candidate policy parameter vectors.
    Args:
        None
    Return:
        parameters (np.array): Samples from the current parameter distribution.
    """
    def sampling(self, count=None):
        if count is None:
            count = self.samples
        return np.random.normal(
            self.mean,
            self.std,
            size=(count, self.parameter_count)
        )

    """
    Update the parameter distribution using elite policies.
    Args:
        elite_parameters (np.array): Parameters of the highest-return policies.
    Return:
        None
    """
    def update_distribution(self, elite_parameters):
        new_mean = np.mean(elite_parameters, axis=0)
        new_std = np.std(elite_parameters, axis=0)
        self.mean = self.alpha * new_mean + (1 - self.alpha) * self.mean
        self.std = np.maximum(
            self.alpha * new_std + (1 - self.alpha) * self.std,
            0.01
        )

    def features(self, observation):
        return np.array([1.0, np.sign(float(observation[2]))])

    def action(self, observation, parameters):
        weights = parameters.reshape(2, self.env.action_space.n)
        logits = self.features(observation) @ weights
        return int(np.argmax(logits))

    def evaluate(self, parameters, frames=None, episode=1, seed=None):
        observation, info = self.env.reset(seed=seed)
        rewards = []
        step = 0

        terminated = False
        truncated = False
        while not (terminated or truncated):
            if frames is not None:
                frames.append(self.plot(step=step, episode=episode, show=False))

            action = self.action(observation, parameters)
            observation, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            step += 1

        if frames is not None:
            frames.append(self.plot(step=step, episode=episode, show=False))

        return float(np.sum(rewards))

    """
    Train the CEM algorithm.
    Args:
        iterations (int): Number of iterations to run.
        episodes_per_policy (int): Number of episodes to run per policy.
    Return:
        list: List of the best return achieved in each iteration.
    """
    def train(self, iterations=50, episodes_per_policy=1):
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if episodes_per_policy < 1:
            raise ValueError("episodes_per_policy must be at least 1")

        history = []
        for iteration in range(iterations):
            candidates = self.sampling()
            seeds = [
                iteration * episodes_per_policy + episode
                for episode in range(episodes_per_policy)
            ]
            returns = np.array([
                np.mean([
                    self.evaluate(parameters, seed=seed)
                    for seed in seeds
                ])
                for parameters in candidates
            ])
            elite_count = min(self.elite, len(candidates))
            elite_indices = np.argsort(returns)[-elite_count:]
            self.update_distribution(candidates[elite_indices])
            history.append(float(np.max(returns)))
        return history

    """
    Run the current mean policy for one episode.
    Args:
        frames (list, optional): Frames to append for visualization.
        episode (int): Episode number used in visualization captions.
    Return:
        float: Total return achieved in the episode.
    """
    def run(self, frames=None, episode=1):
        return self.evaluate(self.mean, frames=frames, episode=episode)

    """
    Plot of subplots showing the performance of the agent over one episode
    One plot with:
        - The power curve for the scenario currently in effect.
        - The environment's current operating point.
    Another plot with:
        - The action scores produced by the mean policy.
    Args:
        step (int, optional): Step number used in visualization captions.
        episode (int, optional): Episode number used in visualization captions.
        show (bool): Whether to display the plot.
    Return:
        Plot of the agent's performance over one episode.
    """
    def plot(self, step=None, episode=None, show=True):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        # Plot the power curve and the environment's current operating point
        curve = self.env.cases[self.env._current_case]
        x = curve[:, 0]
        y = curve[:, 1]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        axs[0].plot(x, y, label='Power Curve', color='blue')
        current = self.env._current_current
        power = self.env._current_power
        axs[0].scatter(current, power, s=100, color='green', marker='s', label='Environment state')
        axs[0].set_title(f'Power Curve (case {self.env._current_case})')
        axs[0].set_xlabel('Current')
        axs[0].set_ylabel('Power')
        axs[0].legend()

        # Plot the action scores produced by the mean policy
        weights = self.mean.reshape(2, self.env.action_space.n)
        gradient_sign = np.sign(self.env._current_gradient)
        action_logits = np.array([1.0, gradient_sign]) @ weights
        axs[1].bar(
            ['decrease', 'maintain', 'increase'],
            action_logits,
            color='green',
            alpha=0.6
        )
        axs[1].set_title('Mean Policy Action Scores')
        axs[1].set_ylabel('Action score')

        plt.tight_layout()
        buffer = BytesIO()
        try:
            fig.savefig(buffer, format='png', dpi=80)
            if show:
                plt.show()
            buffer.seek(0)
            with Image.open(buffer) as rendered:
                frame = rendered.convert('RGB')
        finally:
            plt.close(fig)
            buffer.close()

        if step is not None:
            caption = (
                f"Episode {episode} | Step {step} | "
                f"Case {self.env._current_case} | "
                f"Current {current:.3f} A | Power {power:.3f} W"
            )
            ImageDraw.Draw(frame).text((8, 8), caption, fill='black')
        return frame.quantize(colors=256)

    """
    Makes a lot of plots and unifies them into a GIF to visualize the agent's performance over time.
    """
    def make_gif(self, output_path=None, episodes=1, duration=100):
        if episodes < 1:
            raise ValueError("episodes must be at least 1")
        if duration < 1:
            raise ValueError("duration must be at least 1 millisecond")

        if output_path is None:
            output_path = Path(__file__).with_name("CEM_agent_performance.gif")
        else:
            output_path = Path(output_path)

        frames = []
        for episode in range(1, episodes + 1):
            self.run(frames=frames, episode=episode)

        frames[0].save(
            output_path,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        return output_path

if __name__ == "__main__":
    env = CVP_env()
    agent = Agent(env)
    agent.train()
    gif_path = agent.make_gif()
    print(f"Saved animation to {gif_path}")