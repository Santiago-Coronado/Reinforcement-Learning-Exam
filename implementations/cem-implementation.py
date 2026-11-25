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
    def sampling(self, count=None):
        if count is None:
            count = self.samples
        actions = np.random.normal(self.mean, self.std, count)
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
    def run(self, frames=None, episode=1):
        obs, info = self.env.reset()
        sampled_action_values = []
        rewards = []
        step = 0

        terminated = False
        truncated = False
        while not (terminated or truncated):
            candidate_actions = self.sampling()
            sampled_action = float(candidate_actions[0])
            if frames is not None:
                frames.append(self.plot(
                    step=step,
                    episode=episode,
                    show=False
                ))

            env_action = int(np.clip(
                np.rint(sampled_action), 0, self.env.action_space.n - 1
            ))
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            sampled_action_values.append(sampled_action)
            rewards.append(reward)
            step += 1

        if frames is not None:
            frames.append(self.plot(
                sample_currents=np.array([]),
                step=step,
                episode=episode,
                show=False
            ))

        elite_count = min(self.elite, len(rewards))
        elite_indices = np.argsort(rewards)[-elite_count:]
        elite_samples = np.asarray(sampled_action_values)[elite_indices]
        self.update_distribution(elite_samples)
        return float(np.sum(rewards))

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
    def plot(self, step=None, episode=None, show=True, sample_currents=None):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        # Plot the active CVP curve and the environment current operating point.
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
        sample_radius = 10 * self.env.current_increment
        sample_min = max(x.min(), current - sample_radius)
        sample_max = min(x.max(), current + sample_radius)
        if sample_currents is None:
            sample_currents = np.random.uniform(
                sample_min, sample_max, size=self.samples
            )
        sample_currents = np.asarray(sample_currents)
        if sample_currents.size:
            sample_currents = np.clip(sample_currents, x.min(), x.max())
            sample_powers = np.interp(sample_currents, x, y)
            axs[0].scatter(
                sample_currents,
                sample_powers,
                color='red',
                alpha=0.7,
                label='Nearby samples',
                zorder=3
            )
        axs[0].set_title(f'Power Curve (case {self.env._current_case})')
        axs[0].set_xlabel('Current')
        axs[0].set_ylabel('Power')
        axs[0].legend()

        action_distribution = [self.mean - self.std, self.mean, self.mean + self.std]
        axs[1].bar(
            ['mean - std', 'mean', 'mean + std'],
            action_distribution,
            color='green',
            alpha=0.6
        )
        axs[1].set_title('CEM Action Distribution')
        axs[1].set_ylabel('Sampled action')

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
    gif_path = agent.make_gif()
    print(f"Saved animation to {gif_path}")