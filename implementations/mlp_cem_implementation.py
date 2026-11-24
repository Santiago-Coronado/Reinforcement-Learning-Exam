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

# mlp-cem class
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

        # Define the layers of the MLP
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
            param.data.copy_(new_params[current_index:current_index + param_length].view(param.size()))
            current_index += param_length

# DATOS:
# 10 muestras separados uniformemente y checar la potencia de cada una y tomar esas observaciones
# para la entrada de MLP para predecir donde esta el máximo
# Mostrar sobre pesos, 10 versiones de pesos, 10 redes neuronales