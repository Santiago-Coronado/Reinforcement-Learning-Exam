from gymnasium.utils.env_checker import check_env
from gym_env import CVP_env

env = CVP_env(render_mode="human")

# Check Gymnasium compatibility
check_env(env)

# Initialize environment
obs, info = env.reset(seed=42)

print("Initial observation:", obs)
print("Initial info:", info)

# Execute random actions
for step in range(60):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep: {step + 1}")
    print("Action:", action)
    print("Observation:", obs)
    print("Reward:", reward)
    print("Info:", info)

    env.render()

    if terminated or truncated:
        break

# --Case change--
env = CVP_env(max_steps=100)

obs, info = env.reset(seed=42)

initial_case = info["current_case"]

for step in range(50):
    obs, reward, terminated, truncated, info = env.step(1)

print("\nInitial case:", initial_case)
print("Current case:", info["current_case"])
print("Case changed:", info["case_changed"])

# --If terminated or truncated--
env = CVP_env(max_steps=10)

obs, info = env.reset(seed=42)

for step in range(10):
    obs, reward, terminated, truncated, info = env.step(1)

print("\nTerminated:", terminated)
print("Truncated:", truncated)