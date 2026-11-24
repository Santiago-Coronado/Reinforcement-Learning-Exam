import torch
from mlp_cem_implementation import MLP_CEM

# Create model
model = MLP_CEM()

# Example: [current, power, gradient]
observation = torch.tensor([0.5, 0.7, 0.2], dtype=torch.float32)

# Model output
output = model(observation)

print("Observation:", observation)
print("Network output:", output)

# Select action
action = torch.argmax(output).item()
print("Selected action:", action)

# Check parameters
parameters = model.get_parameters()
print("Number of parameters:", parameters.numel())