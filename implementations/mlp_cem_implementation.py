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

# MLP class
class MLP_CEM(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with Cross-Entropy Method (CEM) implementation.
    This class defines a multi-layer perceptron model that can be trained using the cross-entropy method.
    """

    def __init__(self, input_size = 3, hidden_size = 16, output_size = 3):
        """
        Initializes the MLP_CEM model with the specified input size, hidden size, and output size.

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
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
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
            param.data.copy_(new_params[current_index:current_index + param_length].copy())
            current_index += param_length

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