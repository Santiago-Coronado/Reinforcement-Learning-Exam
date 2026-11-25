"""
Reinforce Implementation

Reinforce is a policy gradient method for reinforcement learning.
It is a Monte Carlo method that updates the policy parameters in the direction of the gradient of the expected return.
"""

# imports
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from gymnassium.gym_env import CVP_env

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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
            nn.Linear(hidden_size, hidden_size),
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
    def __init__(self, input_size = 3, hidden_size = 64, output_size = 3, learning_rate = 0.001, gamma = 0.999):
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
        discounted_return = 0

        # Using reversed to calculate the discounted return from the end of the episode to the beginning
        for reward in reversed(rewards):
            discounted_return = (reward + self.gamma * discounted_return)
            returns.insert(0, discounted_return)

        returns = torch.tensor(returns, dtype = torch.float32)

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

        loss = torch.stack(policy_loss).sum()

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Avoid excessively large gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm = 1.0)
        self.optimizer.step()

        return loss.item()

    # Train the agent
    def train(self, env, episodes = 1000, patience = 100, min_delta = 0.001):
        """
        Train the agent in the given environment.

        Args:
            env: The environment to train the agent in.
            episodes (int): The number of episodes to train for.
            patience (int): The number of episodes to wait for improvement before stopping.
            min_delta (float): The minimum change in the loss to qualify as an improvement.
        """
        history = []
        best_average_reward = -np.inf
        best_parameters = None
        episodes_without_improvement = 0

        # For each episode, reset the environment and collect experiences until the episode ends
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

            # Calculate the average reward last 50 episodes
            window = min(50, len(history))
            moving_average = np.mean([item["total_reward"] for item in history[-window:]])

            # Check if the moving average has improved enough to consider it a new best
            if moving_average > best_average_reward + min_delta:
                best_average_reward = moving_average
                best_parameters = copy.deepcopy(self.policy.state_dict())
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 1

            if (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{episodes} | "
                    f"Reward: {total_reward:.4f} | "
                    f"Average Power: {average_power:.3f} W | "
                    f"Moving Average: {moving_average:.4f} | "
                    f"Loss: {loss:.8f}"
                )

            # Stop if the moving average does not improve
            if episodes_without_improvement >= patience:
                print(f"Early stopping at episode {episode + 1}")
                break

        # Restore the best policy during training to ensure the agent retains the best learned parameters
        if best_parameters is not None:
            self.policy.load_state_dict(best_parameters)

        return history

# Evaluate the trained agent
def evaluate_agent(agent, env, seed = 42):
    """
    Evaluate the trained agent in the environment.

    Args:
        agent: The trained agent to evaluate.
        env: The environment to evaluate the agent in.
        seed (int): The random seed for reproducibility.

    Returns:
        list: A history of the evaluation, including steps, cases, currents, powers, rewards, actions, and case changes.
    """
    observation, info = env.reset(seed = seed)
    history = []
    total_reward = 0
    powers = []
    action_counts = {0: 0, 1: 0, 2: 0}

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = agent.select_action(observation, deterministic = True)

        with torch.no_grad():
            state = torch.FloatTensor(observation)
            logits = agent.policy(state)
            probabilities = torch.softmax(logits, dim=-1)

        print(
            f"Observation: {np.round(observation, 4)} | "
            f"Action: {action} | "
            f"Probabilities: {np.round(probabilities.numpy(), 4)}"
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
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Average Power: {np.mean(powers):.3f} W")

    print("\nAction Distribution:")

    for action, count in action_counts.items():
        print(
            f"Action {action}: {count} times "
            f"({count / len(history) * 100:.2f}%)"
        )

    return history

# Main function to train the agent on the environment
if __name__ == "__main__":

    env = CVP_env(max_steps = 300)

    reinforce_agent = agent(
        input_size = 3,
        hidden_size = 64,
        output_size = 3,
        learning_rate = 0.001,
        gamma = 0.999
    )

    history = reinforce_agent.train(
        env,
        episodes = 1000,
        patience = 100
    )

    evaluation_history = evaluate_agent(
        reinforce_agent,
        env
    )

    env.close()