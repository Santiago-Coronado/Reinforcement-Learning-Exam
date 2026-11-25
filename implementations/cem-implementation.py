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
from PIL import Image


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

    """
    Extract features from an observation.
    Args:
        observation (np.array): The current observation.
    Return:
        np.array: The extracted features.
    """
    def features(self, observation):
        return np.array([1.0, np.sign(float(observation[2]))])

    """
    Determine the action to take based on the current observation and policy parameters.
    Args:
        observation (np.array): The current observation.
        parameters (np.array): The current policy parameters.
    Return:
        int: The action to take.
    """
    def action(self, observation, parameters):
        weights = parameters.reshape(2, self.env.action_space.n)
        logits = self.features(observation) @ weights
        return int(np.argmax(logits))
    """
    Evaluate the performance of a policy.
    Args:
        parameters (np.array): The policy parameters to evaluate.
        frames (list): List of frames to save the evaluation.
        episode (int): The episode number.
        seed (int): The random seed.
    Return:
        float: The total reward obtained.
    """
    def evaluate(self, parameters, frames=None, episode=1, seed=None):
        observation, info = self.env.reset(seed=seed)
        rewards = []
        step = 0

        visual_history = None
        if frames is not None:
            visual_history = [{
                "step": 0,
                "case": info["current_case"],
                "current": info["current"],
                "power": info["power"],
                "reward": 0.0,
                "action": None
            }]
            frames.append(self.plot(
                step=step,
                episode=episode,
                show=False,
                history=visual_history
            ))

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = self.action(observation, parameters)
            observation, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            step += 1

            if visual_history is not None:
                visual_history.append({
                    "step": step,
                    "case": info["evaluated_case"],
                    "current": info["evaluated_current"],
                    "power": info["evaluated_power"],
                    "reward": reward,
                    "action": action
                })
                frames.append(self.plot(
                    step=step,
                    episode=episode,
                    show=False,
                    history=visual_history
                ))

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
    Plot the power curve and episode progress in a two-panel layout.
    Args:
        step (int, optional): Step number used in visualization captions.
        episode (int, optional): Episode number used in visualization captions.
        show (bool): Whether to display the plot.
        history (list, optional): Episode states accumulated for visualization.
    Return:
        Plot of the agent's performance over one episode.
    """
    def plot(self, step=None, episode=None, show=True, history=None):
        if history:
            data = history[-1]
            case = data["case"]
            current = data["current"]
            power = data["power"]
        else:
            case = self.env._current_case
            current = self.env._current_current
            power = self.env._current_power
            data = {
                "step": step if step is not None else 0,
                "action": None,
            }

        fig, (ax_curve, ax_progress) = plt.subplots(1, 2, figsize=(14, 5))
        reward_axis = ax_progress.twinx()

        all_currents = np.concatenate([curve[:, 0] for curve in self.env.cases])
        all_powers = np.concatenate([curve[:, 1] for curve in self.env.cases])
        x_min, x_max = all_currents.min(), all_currents.max()
        y_min, y_max = all_powers.min(), all_powers.max()

        curve = self.env.cases[case]
        ax_curve.plot(
            curve[:, 0],
            curve[:, 1],
            color="blue",
            label="P-I Curve",
            linewidth=2
        )

        if history:
            points = [
                (entry["current"], entry["power"])
                for entry in history
                if entry["case"] == case and entry["step"] > 0
            ]
            if points:
                points_array = np.array(points)
                ax_curve.scatter(
                    points_array[:, 0],
                    points_array[:, 1],
                    color="red",
                    label="Evaluated Points",
                    zorder=3
                )

        ax_curve.scatter(
            current,
            power,
            color="green",
            marker="D",
            s=100,
            label="Current Position",
            zorder=4
        )
        ax_curve.set_xlim(x_min, x_max)
        ax_curve.set_ylim(y_min, y_max * 1.1)
        ax_curve.set_title(f"Case {case} - Step {data['step']}")
        ax_curve.set_xlabel("Current I (A)")
        ax_curve.set_ylabel("Power P (W)")
        ax_curve.legend()
        ax_curve.grid(True)

        if history:
            steps = [entry["step"] for entry in history]
            powers = [entry["power"] for entry in history]
            rewards = [entry["reward"] for entry in history]
            ax_progress.plot(steps, powers, label="Power (W)")
            reward_axis.plot(steps, rewards, color="orange", label="Reward")

        ax_progress.set_xlabel("Step")
        ax_progress.set_ylabel("Power (W)")
        reward_axis.set_ylabel("Reward")
        ax_progress.set_title("Power and Reward Progress")
        ax_progress.grid(True)

        power_lines, power_labels = ax_progress.get_legend_handles_labels()
        reward_lines, reward_labels = reward_axis.get_legend_handles_labels()
        if power_lines or reward_lines:
            ax_progress.legend(
                power_lines + reward_lines,
                power_labels + reward_labels,
                loc="best"
            )

        action_text = "Initial" if data["action"] is None else str(data["action"])
        fig.suptitle(
            f"CEM | Episode {episode} | Case {case} | Step {data['step']} | "
            f"Current: {current:.3f} A | Power: {power:.3f} W | "
            f"Action: {action_text}"
        )
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

        return frame.quantize(colors=256)

    """
    Makes a lot of plots and unifies them into a GIF to visualize the agent's performance over time.
    Args:
        output_path (str, optional): Path to save the GIF. If None, saves to
            the default location.
        episodes (int): Number of episodes to run for the GIF.
        duration (int): Duration of each frame in milliseconds.
    Return:
        str: Path to the saved GIF.
    """
    def make_gif(self, output_path=None, episodes=1, duration=100):
        if episodes < 1:
            raise ValueError("episodes must be at least 1")
        if duration < 1:
            raise ValueError("duration must be at least 1 millisecond")

        if output_path is None:
            output_path = (
                Path(__file__).resolve().parents[1]
                / "gifs"
                / "CEM_agent_performance.gif"
            )
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