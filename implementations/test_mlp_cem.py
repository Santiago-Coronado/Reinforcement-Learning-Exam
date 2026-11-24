import torch
from mlp_cem_implementation import MLP_CEM, CEMOptimizer

# Create MLP
model = MLP_CEM()

# Initial parameters
initial_params = model.get_parameters()

# CEM optimizer
cem_optimizer = CEMOptimizer(initial_parameters = initial_params)

# Population
population = cem_optimizer.sample_population()

# Rewards for testing
rewards = torch.tensor([1.5, 2.0, 0.5, 3.0, 1.2, 4.0, 0.8, 2.5, 1.0, 3.5])

# Update parameters based on rewards
elites = cem_optimizer.update(population, rewards)

print("MLP parameters:", initial_params.shape)
print("Population:", population.shape)
print("Elites:", elites.shape)
print("Updated mean:", cem_optimizer.mean.shape)
print("Updated std:", cem_optimizer.std.shape)