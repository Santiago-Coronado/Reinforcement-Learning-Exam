"""
Actor-Critic implementation 
"""
from io import BytesIO
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import softmax

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
        return softmax(logits)
    
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
            action_probabilities = self.probabilities(observation)
            visual_history = [{
                "step": 0,
                "case": info["current_case"],
                "current": info["current"],
                "power": info["power"],
                "reward": 0.0,
                "action": None,
                "policy_probabilities": action_probabilities.copy(),
                "critic_value": self.V(observation),
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
            action_probabilities = self.probabilities(observation)
            action = int(np.random.choice(
                self.action_count,
                p=action_probabilities,
            ))
            critic_value = self.V(observation)
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
                    "policy_probabilities": action_probabilities.copy(),
                    "critic_value": critic_value,
                })
                frames.append(self.plot(
                    step=step,
                    episode=episode,
                    show=False,
                    history=visual_history,
                ))

        return episode_return

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

        fig, (ax_curve, ax_progress) = plt.subplots(1, 2, figsize=(16, 5))
        reward_axis = ax_progress.twinx()
        policy_axis = ax_progress.twinx()
        policy_axis.spines["right"].set_position(("outward", 60))

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
            linewidth=2,
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
                    zorder=3,
                )

        ax_curve.scatter(
            current,
            power,
            color="green",
            marker="D",
            s=100,
            label="Current Position",
            zorder=4,
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
            critic_values = [entry["critic_value"] for entry in history]
            policy_probabilities = np.array([
                entry["policy_probabilities"] for entry in history
            ])
            ax_progress.plot(steps, powers, label="Power (W)")
            reward_axis.plot(steps, rewards, color="orange", label="Reward")
            reward_axis.plot(
                steps,
                critic_values,
                color="red",
                linestyle="--",
                label="Critic value",
            )
            for action, label in enumerate(("Decrease", "Hold", "Increase")):
                policy_axis.plot(
                    steps,
                    policy_probabilities[:, action],
                    linestyle=":",
                    label=f"Policy P({label})",
                )

        ax_progress.set_xlabel("Step")
        ax_progress.set_ylabel("Power (W)")
        reward_axis.set_ylabel("Reward / critic value")
        policy_axis.set_ylabel("Policy action probability")
        policy_axis.set_ylim(0.0, 1.0)
        ax_progress.set_title("Power and Reward Progress")
        ax_progress.grid(True)

        power_lines, power_labels = ax_progress.get_legend_handles_labels()
        reward_lines, reward_labels = reward_axis.get_legend_handles_labels()
        policy_lines, policy_labels = policy_axis.get_legend_handles_labels()
        if power_lines or reward_lines or policy_lines:
            ax_progress.legend(
                power_lines + reward_lines + policy_lines,
                power_labels + reward_labels + policy_labels,
                loc="best",
            )

        action_text = "Initial" if data["action"] is None else str(data["action"])
        fig.suptitle(
            f"Actor-Critic | Episode {episode} | Case {case} | Step {data['step']} | "
            f"Current: {current:.3f} A | Power: {power:.3f} W | "
            f"Action: {action_text}"
        )
        plt.tight_layout()

        buffer = BytesIO()
        try:
            fig.savefig(buffer, format="png", dpi=80)
            if show:
                plt.show()
            buffer.seek(0)
            with Image.open(buffer) as rendered:
                frame = rendered.convert("RGB")
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
                / "ActorCritic_agent_performance.gif"
            )
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frames = []
        for episode in range(1, episodes + 1):
            self.evaluate(frames=frames, episode=episode)

        frames[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        return output_path

if __name__ == "__main__":
    env = CVP_env()
    agent = Agent(env)
    agent.train()
    gif_path = agent.make_gif()
    print(f"Saved animation to {gif_path}")