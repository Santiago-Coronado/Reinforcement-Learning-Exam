from mlp_cem_implementation import agent
import torch

model = agent()

# Check MLP parameters
parameters = model.get_parameters()
print("Parameters:", parameters.shape)

# Generate population
population = model.sample_population()
print("Population:", population.shape)

# Test MLP output
observation = torch.tensor(
    [0.5, 0.7, 0.2],
    dtype=torch.float32
)

output = model.model(observation)
action = torch.argmax(output).item()

print("MLP output:", output)
print("Selected action:", action)