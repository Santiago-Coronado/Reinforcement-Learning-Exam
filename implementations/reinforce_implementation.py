"""
Reinforce Implementation

Reinforce is a policy gradient method for reinforcement learning.
It is a Monte Carlo method that updates the policy parameters in the direction of the gradient of the expected return.
"""

# imports
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Policy Network - reinforce uses a simple feedforward neural network to represent the policy
class PolicyNetwork(nn.Module):
    """
    Policy Network for reinforce Agent
    A simple feedforward neural network that takes the state as input and outputs action probabilities.
    """
    def __init__(self, input_size = 3, hidden_size = 64, output_size = 3):
        """
        Initialize the Policy Network.

        Args:
            input_size (int): Size of the input layer (state dimension).
            hidden_size (int): Size of the hidden layer.
            output_size (int): Size of the output layer (number of actions).
        """
        super().__init__()

        # Layers of the neural network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The output logits for each action.
        """
        return self.network(state)

# REINFORCE Agent
class agent:
    """
    Agent implementing the REINFORCE algorithm for policy gradient reinforcement learning.
    """
    def __init__(self, input_size = 3,hidden_size = 64,output_size = 3,learning_rate = 0.001, gamma = 0.99):
        """
        Initialize the REINFORCE Agent.

        Args:
            input_size (int): Size of the input layer (state dimension).
            hidden_size (int): Size of the hidden layer.
            output_size (int): Size of the output layer (number of actions).
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
        """
        self.gamma = gamma
        self.policy = PolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = learning_rate)

    # Select an action based on the current policy
    def select_action(self, observation, deterministic = False):
        """
        Select an action based on the current policy.

        Args:
            observation: The current state observation.
            deterministic (bool): If True, select the action with the highest probability.
                                  If False, sample from the action distribution.

        Returns:
            action (int): The selected action.
            log_probability (torch.Tensor): The log probability of the selected action.
        """
        state = torch.FloatTensor(observation)
        logits = self.policy(state)
        distribution = Categorical(logits=logits)

        # Deterministic true is used for evaluation, while false is used for training
        if deterministic:
            action = torch.argmax(logits)
        else:
            action = distribution.sample()

        # Change to log probability to simplify the loss calculation in the update_policy function
        log_probability = distribution.log_prob(action)

        return action.item(), log_probability

    # Calculate discounted returns
    def calculate_returns(self, rewards):
        """
        Calculate discounted returns for a sequence of rewards.

        Args:
            rewards (list): A list of rewards received during an episode.

        Returns:
            torch.Tensor: A tensor of discounted returns.
        """
        returns = []
        discounted_return = 0.0

        # Using reversed to calculate the discounted return from the end of the episode to the beginning
        for reward in reversed(rewards):
            discounted_return = (reward + self.gamma * discounted_return)
            returns.insert(0, discounted_return)

        returns = torch.tensor(returns,dtype = torch.float32)

        # Normalize returns to improve training stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased = False) + 1e-8)

        return returns

    # Update the policy after completing an episode
    def update_policy(self, log_probabilities, rewards):
        """
        Update the policy network parameters based on the collected log probabilities and rewards.

        Args:
            log_probabilities (list): A list of log probabilities of the actions taken during the episode.
            rewards (list): A list of rewards received during the episode.

        Returns:
            float: The loss value after the policy update.
        """
        returns = self.calculate_returns(rewards)
        policy_loss = []

        for log_probability, discounted_return in zip(log_probabilities, returns):
            policy_loss.append(-log_probability * discounted_return)

        loss = torch.stack(policy_loss).mean()

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Avoid excessively large gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm = 1.0)
        self.optimizer.step()

        return loss.item()

    # Train the agent
    def train(self, env, episodes = 300):
        """
        Train the agent in the given environment.

        Args:
            env: The environment in which the agent will be trained.
            episodes (int): The number of episodes to train the agent.
        """
        history = []

        # For each episode, reset the environment and collect log probabilities and rewards until the episode ends
        for episode in range(episodes):
            observation, info = env.reset(seed = 42 + episode)
            log_probabilities = []
            rewards = []
            powers = []

            terminated = False
            truncated = False

            while not terminated and not truncated:
                action, log_probability = self.select_action(observation)
                observation, reward, terminated, truncated, info = (env.step(action))
                log_probabilities.append(log_probability)
                rewards.append(reward)
                powers.append(info["evaluated_power"])

            # Update policy after the episode ends
            loss = self.update_policy(log_probabilities, rewards)

            # Calculate total reward and average power for the episode
            total_reward = sum(rewards)
            average_power = np.mean(powers)

            history.append({
                "episode": episode + 1,
                "total_reward": total_reward,
                "average_power": average_power,
                "loss": loss,
                "evaluations": len(rewards)
            })

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {total_reward:.4f} | "
                f"Average Power: {average_power:.3f} W | "
                f"Loss: {loss:.6f}"
            )

        return history

# Evaluate the trained agent
def evaluate_agent(model_agent, env, seed = 42):
    """
    Evaluate the trained agent in the environment.

    Args:
        model_agent: The trained agent to evaluate.
        env: The environment to evaluate the agent in.
        seed (int): The random seed for reproducibility.

    Returns:
        list: A history of the evaluation, including steps, cases, currents, powers, rewards, actions, and case changes.
    """
    observation, info = env.reset(seed = seed)
    history = []
    total_reward = 0.0
    powers = []
    action_counts = {0: 0, 1: 0, 2: 0}

    terminated = False
    truncated = False

    while not terminated and not truncated:
        # Deterministic action selection for evaluation
        with torch.no_grad():
            action, _ = model_agent.select_action(
                observation,
                deterministic = True
            )

        observation, reward, terminated, truncated, info = (env.step(action))
        total_reward += reward

        powers.append(info["evaluated_power"])
        action_counts[action] += 1

        history.append({
            "step": len(history) + 1,
            "case": info["evaluated_case"],
            "current": info["evaluated_current"],
            "power": info["evaluated_power"],
            "reward": reward,
            "action": action,
            "case_changed": info["case_changed"]
        })

    print("\nEvaluation Results")
    print(f"Total Steps: {len(history)}")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Average Power: {np.mean(powers):.3f} W")

    print("\nAction Distribution:")

    action_names = {
        0: "Decrease",
        1: "Hold",
        2: "Increase"
    }

    for action, count in action_counts.items():
        print(
            f"Action {action} ({action_names[action]}): "
            f"{count} times "
            f"({count / len(history) * 100:.2f}%)"
        )

    return history

# Function to create a GIF of the evaluation
def create_gif(episode_history, env):
    """
    Create a GIF visualizing the evaluation of the agent.

    Args:
        episode_history (list): A history of the evaluation, including steps, cases, currents, powers, rewards, actions, and case changes.
        env: The environment used for evaluation.
    """
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    ax_curve = axes[0]
    ax_progress = axes[1]
    reward_axis = ax_progress.twinx()

    all_currents = np.concatenate([curve[:, 0] for curve in env.cases])
    all_powers = np.concatenate([curve[:, 1] for curve in env.cases])

    x_min, x_max = all_currents.min(), all_currents.max()
    y_min, y_max = all_powers.min(), all_powers.max()

    action_names = {
        0: "Decrease",
        1: "Hold",
        2: "Increase"
    }

    def update(frame):
        data = episode_history[frame]

        case = data["case"]
        current = data["current"]
        power = data["power"]

        ax_curve.clear()
        ax_progress.clear()
        reward_axis.clear()

        # P-I curve
        curve = env.cases[case]

        ax_curve.plot(
            curve[:, 0],
            curve[:, 1],
            color="blue",
            label="P-I Curve",
            linewidth=2
        )

        # Evaluated points from the same case
        points = [
            (x["current"], x["power"])
            for x in episode_history[:frame + 1]
            if x["case"] == case
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

        # Current position
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

        ax_curve.set_title(
            f"Case {case} - Step {data['step']}"
        )

        ax_curve.set_xlabel("Current I (A)")
        ax_curve.set_ylabel("Power P (W)")
        ax_curve.legend()
        ax_curve.grid(True)

        # Power and Reward Progress
        previous_data = episode_history[:frame + 1]

        steps = [x["step"] for x in previous_data]
        powers = [x["power"] for x in previous_data]
        rewards = [x["reward"] for x in previous_data]

        # Power (left axis)
        ax_progress.plot(
            steps,
            powers,
            color="blue",
            linewidth=2,
            label="Power (W)"
        )

        ax_progress.set_xlabel("Step")
        ax_progress.set_ylabel("Power (W)", color="blue")
        ax_progress.tick_params(axis="y", labelcolor="blue")

        # Reward (right axis)
        reward_axis.plot(
            steps,
            rewards,
            color="orange",
            linewidth=2,
            label="Reward"
        )

        reward_axis.set_ylabel("Reward", color="orange")
        reward_axis.tick_params(axis="y", labelcolor="orange")

        ax_progress.set_title("Power and Reward Progress")
        ax_progress.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax_progress.get_legend_handles_labels()
        lines2, labels2 = reward_axis.get_legend_handles_labels()

        ax_progress.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="best"
        )

        # Main title
        action_text = action_names[data["action"]]

        fig.suptitle(
            f"REINFORCE | Case {case} | Step {data['step']} | "
            f"Current: {current:.3f} A | "
            f"Power: {power:.3f} W | "
            f"Action: {action_text}"
        )

        plt.tight_layout()

    animation = FuncAnimation(
        fig,
        update,
        frames=len(episode_history),
        interval=300,
        repeat=True
    )

    animation.save(
        "reinforce.gif",
        writer=PillowWriter(fps=5)
    )

    plt.close(fig)

    print("GIF saved as reinforce.gif")

if __name__ == "__main__":

    from gymnassium.gym_env import CVP_env
    
    env = CVP_env(max_steps=300)

    reinforce_agent = agent(
        input_size=3,
        hidden_size=64,
        output_size=3,
        learning_rate=0.001,
        gamma=0.99
    )

    # Train
    history = reinforce_agent.train(
        env,
        episodes = 300
    )

    # Evaluate trained agent
    evaluation_history = evaluate_agent(
        reinforce_agent,
        env,
        seed = 42
    )

    # Generate GIF
    create_gif(evaluation_history, env)

    env.close()