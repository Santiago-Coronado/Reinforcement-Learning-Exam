"""
Multi-Layer Perceptron (MLP) with Cross-Entropy Method (CEM) implementation.

MLP = multi-layer perceptron, a type of feedforward artificial neural network.
CEM = cross-entropy method, a stochastic optimization technique used for training models.
"""

# imports
import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)

class agent:
    """
    Agent class that encapsulates the MLP model and Cross-Entropy Method (CEM) optimizer for reinforcement learning.
    """

    # MLP model class
    class MLP(nn.Module):
        """
        Multi-Layer Perceptron (MLP) model for the agent.
        """

        def __init__(self, input_size = 3, hidden_size = 16, output_size = 3):
            """
            Initializes the MLP model with the specified input size, hidden size, and output size.

            Args:
                input_size (int): The number of input features.
                hidden_size (int): The number of neurons in the hidden layer.
                output_size (int): The number of output features.
            """
            super().__init__()

            # Layers of the MLP
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

        def forward(self, x):
            """
            Defines the forward pass of the MLP model.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor after passing through the MLP layers.
            """
            return self.model(x)

    # CEM optimizer class
    class CEMOptimizer:
        """
        A class for the CEM (Cross-Entropy Method) optimizer.
        """

        def __init__(self, initial_parameters, population_size = 10, elite_size = 2, std = 0.1):
            """
            Initializes the CEMOptimizer with the specified parameters.

            Args:
                initial_parameters (torch.Tensor): The initial parameters for the optimization.
                population_size (int): The number of samples to generate in each iteration.
                elite_size (int): The number of top-performing samples to select as elites.
                std (float): The standard deviation for sampling new parameters.
            """
            self.mean = initial_parameters.clone()
            self.population_size = population_size
            self.elite_size = elite_size
            self.std = torch.ones_like(initial_parameters) * std

        def sample_population(self):
            """
            Samples a population of new parameters based on the current mean and standard deviation.
        
            Returns:
                torch.Tensor: A tensor containing the sampled population of new parameters.
            """
            population = []
        
            for _ in range(self.population_size):
                # Random noise
                noise = torch.randn_like(self.mean)
                # Sample new parameters
                new_params = self.mean + noise * self.std
                population.append(new_params)
        
            return torch.stack(population)
        
        def update(self, population, rewards):
            """
            Updates the mean and standard deviation of the optimizer based on the rewards of the sampled population.
        
            Args:
                population (torch.Tensor): The sampled population of new parameters.
                rewards (torch.Tensor): The rewards corresponding to each sample in the population.
            """
            # Select elite samples based on rewards
            elite_indices = torch.topk(rewards, self.elite_size).indices
            elite_samples = population[elite_indices]
        
            # Update mean and std based on elite samples
            self.mean = elite_samples.mean(dim=0)
            self.std = torch.std(elite_samples, dim=0) + 1e-6  # Add a small value to avoid zero std
        
            return elite_samples

    def __init__(self, input_size = 3, hidden_size = 16, output_size = 3, population_size = 10, elite_size = 2, std = 0.1):
        """
        Initializes the agent with an MLP model and a CEM optimizer.

        Args:
            input_size (int): The number of input features for the MLP.
            hidden_size (int): The number of neurons in the hidden layer of the MLP.
            output_size (int): The number of output features for the MLP.
            population_size (int): The number of samples to generate in each iteration for CEM.
            elite_size (int): The number of top-performing samples to select as elites for CEM.
            std (float): The standard deviation for sampling new parameters in CEM.
        """
        self.model = self.MLP(input_size, hidden_size, output_size)
        initial_parameters = self.get_parameters()
        self.cem_optimizer = self.CEMOptimizer(initial_parameters, population_size, elite_size, std)

    def get_parameters(self):
        """
        Retrieves the parameters of the MLP model as a single concatenated tensor.
    
        Returns:
            torch.Tensor: A single tensor containing all the parameters of the MLP model.
        """
        parameters = []
        for param in self.model.parameters():
            parameters.append(param.data.view(-1))
        return torch.cat(parameters)
    
    def set_parameters(self, new_params):
        """
        Sets the parameters of the MLP model from a single concatenated tensor.
    
        Args:
            new_params (torch.Tensor): A single tensor containing all the parameters to set in the MLP model.
        """
        current_index = 0
        for param in self.model.parameters():
            param_length = param.numel()
            param.data.copy_(new_params[current_index:current_index + param_length].view_as(param))
            current_index += param_length

    def sample_population(self):
        """
        Samples a population of new parameters using the CEM optimizer.
    
        Returns:
            torch.Tensor: A tensor containing the sampled population of new parameters.
        """
        return self.cem_optimizer.sample_population()

    def update(self, population, rewards):
        """
        Updates the CEM optimizer based on the rewards of the sampled population.
    
        Args:
            population (torch.Tensor): The sampled population of new parameters.
            rewards (torch.Tensor): The rewards corresponding to each sample in the population.
    
        Returns:
            torch.Tensor: A tensor containing the elite samples selected based on rewards.
        """
        return self.cem_optimizer.update(population, rewards)

    # Select action based on the observation
    def select_action(self, observation):
        """
        Selects an action based on the current observation.

        Args:
            observation: The current observation from the environment.

        Returns:
            int: The selected action.
        """
        # Convert observation to tensor and dimention 0 for batch
        obs_tensor = torch.tensor(observation, dtype = torch.float32).unsqueeze(0)

        # Use the MLP model to get the action logits and select the action with the highest logit
        with torch.no_grad():
            logits = self.model(obs_tensor)
            action = torch.argmax(logits, dim = 1)

        return action.item()

    def evaluate_candidate(self, env, parameters, seeds):
        """
        Evaluates a candidate solution by running it through the environment.

        Args:
            env: The environment to evaluate the candidate in.
            parameters (torch.Tensor): The parameters of the candidate solution.
            seeds (list): A list of random seeds to use for evaluation.

        Returns:
            float: The average reward obtained by evaluating the candidate.
        """
        # Set the parameters of the MLP model to the candidate parameters
        self.set_parameters(parameters)
        episode_rewards = []

        # For each seed, reset the environment and run an episode to evaluate the candidate
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            total_reward = 0
            terminated = False
            truncated = False

            # Run the episode until it is terminated or truncated and accumulate the total reward
            # to calculate the average reward for the candidate
            while not terminated and not truncated:
                action = self.select_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            episode_rewards.append(total_reward)

        return np.mean(episode_rewards)

    # Function to train the agent using the CEM optimizer
    def train(self, env, generations = 100):
        """
        Trains the CEM optimizer on the given environment.
    
        Args:
            env: The environment to train on.
            generations (int): The number of generations to train.
    
        Returns:
            list: A list of dictionaries containing the training history.
        """
        seeds = [42, 43, 44] # Test seeds for reproducibility
        best_reward = -np.inf
        best_parameters = None
        history = []

        # Loop through the specified number of generations to train the agent
        for generation in range(generations):
            population = self.sample_population()
            population_rewards = []

            # Evaluate each candidate in the population and store their rewards
            for candidate in population:
                reward = self.evaluate_candidate(env, candidate, seeds)
                population_rewards.append(reward)

            rewards_tensor = torch.tensor(population_rewards, dtype=torch.float32)

            self.update(population, rewards_tensor)

            # Find the best candidate in the current generation
            generation_best_index = np.argmax(population_rewards)
            generation_best_reward = population_rewards[generation_best_index]

            # Update the best parameters if the current generation's best reward is better than the previous best reward
            if generation_best_reward > best_reward:
                best_reward = generation_best_reward
                best_parameters = population[generation_best_index].clone()

            # Calculate the average reward for the current generation and store the training history
            average_reward = np.mean(population_rewards)

            history.append({
                "generation": generation + 1,
                "average_reward": average_reward,
                "best_reward": generation_best_reward,
                "global_best": best_reward
            })

            print(
                f"Generation {generation + 1}/{generations} | "
                f"Average Reward: {average_reward:.4f} | "
                f"Best Reward: {generation_best_reward:.4f} | "
                f"Global Best: {best_reward:.4f}"
            )

        self.set_parameters(best_parameters)

        return history

# Main function to train the agent on the environment
if __name__ == "__main__":
    
    from gymnassium.gym_env import CVP_env

    env = CVP_env(max_steps = 200)

    model_agent = agent(
        input_size = 3,
        hidden_size = 16,
        output_size = 3,
        population_size = 10,
        elite_size = 2,
        std = 0.1
    )

    history = model_agent.train(env, generations = 100)

    env.close()