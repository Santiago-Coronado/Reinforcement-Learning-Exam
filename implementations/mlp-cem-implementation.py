"""
Multi-Layer Perceptron (MLP) with Cross-Entropy Method (CEM) implementation.

MLP = multi-layer perceptron, a type of feedforward artificial neural network.
CEM = cross-entropy method, a stochastic optimization technique used for training models.
"""

# torch imports
import torch
import torch.nn as nn

torch.manual_seed(42)

# mlp-cem manual class
class MLP_CEM(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with Cross-Entropy Method (CEM) implementation.

    This class defines a multi-layer perceptron model that can be trained using the cross-entropy method.
    It inherits from the nn.Module class in PyTorch.

    Attributes:
        input_size (int): The number of input features.
        hidden_size (int): The number of neurons in the hidden layer.
        model (nn.Sequential): The sequential model containing the layers of the MLP.
    """

    def __init__(self, input_size, hidden_size):
        """
        Initializes the MLP_CEM model with the specified input size, hidden size, and output size.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define the layers of the MLP
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
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

# Example
input_size = 2
hidden_size = 16
observation = torch.tensor([120.0, 2.5])

# Create an instance of the MLP_CEM model
model = MLP_CEM(input_size, hidden_size)

# Perform a forward pass with the observation
output = model(observation)

# Print the output
print("Output of the MLP_CEM model:", output.item())

# DATOS:
# 10 muestras separados uniformemente y checar la potencia de cada una y tomar esas observaciones
# para la entrada de MLP para predecir donde esta el máximo
# Mostrar sobre pesos, 10 versiones de pesos, 10 redes neuronales