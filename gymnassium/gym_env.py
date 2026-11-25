'''
Usamos el wrapper de gym demostrado en Python mostrado en clase:
https://github.com/Farama-Foundation/Gymnasium/blob/main/docs/introduction/create_custom_env.md
'''
from turtle import left, right
from typing import Optional
import numpy as np
import gymnasium as gym
from pathlib import Path # To handle numpy file paths

'''
En cada episodio (o durante la ejecución continua) seleccione aleatoriamente uno de los cuatro casos de sombreado.
Mantenga el caso activo durante un número de pasos (o un intervalo de tiempo simulado) configurable, tras el cual cambie de forma aleatoria a otro caso.
Exponga como acción la corriente 
propuesta por el agente (o un incremento de corriente, según el diseño que elijan).
Devuelva como observación  el valor de la Potencia.
Calcule la recompensa en función de la potencia obtenida (o de una señal derivada de ella).
No proporcione al agente el valor del máximo real de la curva (ground-truth).
calcula el gradiente de la potencia con respecto a la corriente y devuelva este valor como parte de la observación. 
acciones se van a definir como incrementos de corriente, por ejemplo, aumentar o disminuir la corriente en un valor fijo.
transformar corriente potencia 
no le damos la forma, muestreamos e inicializamos los metodos que llamaran a esta clase, para que se pueda usar en el PPO.
'''
EPISODES = 300
SEED = 42
BATCH = 50 # step for changing case

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, size: int = 5, render_mode: Optional[str] = None):
        self.size = size
        self.episodes = EPISODES
        self.seed = SEED
        self.batch = BATCH
        self.render_mode = render_mode
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._current_case = np.random.choice([0, 1, 2, 3])  # Randomly select one of the four cases
        self._current_current = np.random.uniform(0, 1)  # Randomly select a current value
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [row, col] coordinates
                "current_case": gym.spaces.Discrete(4),  # Current case (0, 1, 2, or 3)
                "current_value": gym.spaces.Box(0, 1, shape=(), dtype=float),  # Current value
                "current_gradient": gym.spaces.Box(-1, 1, shape=(), dtype=float),  # Current gradient
            }
        )
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }
        self._current_gradient = 0.0  # Initialize the gradient

    def _get_obs(self):
        return {"agent": self._agent_location.copy(), "current_case": self._current_case, "current_value": self._current_current, "current_gradient": self._current_gradient}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        return self._get_obs(), self._get_info()

    def step(self, action):
        # every step, we need to check if we need to change the case/dataset
        for step in range(BATCH):
            # Perform the action
            self._current_current = np.clip(self._current_current + (action - 1) * 0.1, 0, 1)  # Adjust the current value
            self._current_power = self._current_current ** 2  # Calculate the power
            self._current_case = np.random.choice([0, 1, 2, 3])  # Randomly select one of the four cases
            self._current_gradient = (self._current_current - 0.5) * 2  # Calculate the gradient
            direction = self._action_to_direction[action]
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            for row in range(self.size):
                line = ""
                for col in range(self.size):
                    if np.array_equal([row, col], self._agent_location):
                        line += "A "
                    elif np.array_equal([row, col], self._target_location):
                        line += "T "
                    else:
                        line += ". "
                print(line)
            print()

if __name__ == "__main__":
    for episode in range (EPISODES):
        # inicializar el ambiente con su agente
        env = GridWorldEnv(size=5, render_mode="human")
        # resetear el ambiente para obtener la primera observación
        obs, info = env.reset()
        # imprimir la observación inicial y la información adicional
        print("Initial Observation:", obs)
        # printar la información adicional
        print("Initial Info:", info)
        # renderizar el ambiente
        env.render()
        # imprimir la acción tomada y la observación resultante después de un paso
        action = env.action_space.sample()  # tomar una acción aleatoria
        obs, reward, terminated, truncated, info = env.step(action)
        print("Action Taken:", action)
        print("Observation after Step:", obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
    print("Info after Step:", info) 

class CVP_env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Build the environment
    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 200):
        self.episodes = EPISODES
        self.seed = SEED
        self.batch = BATCH
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Each step, the current is incremented by 0.1
        self.current_increment = 0.1

        # Load the cases from the numpy files
        data_path = Path(__file__).resolve().parent.parent / "src"

        self.cases = []
        for i in range(4):
            data = np.load(data_path / f"caso_{i}.npy")

            # voltage, current, power
            curve = data[:, [1, 2]] # We only need current and power for the environment

            # Sort the curve by current to interpolate power values
            curve = curve[np.argsort(curve[:, 0])]
            
            self.cases.append(curve)

        # Observation space: current, power and gradient
        self.observation_space = gym.spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (3,),
            dtype = np.float32
        )

        # Actions: decrease, maintain or increase current
        self.action_space = gym.spaces.Discrete(3)

        # Initialize environment variables
        self._current_case = 0
        self._current_current = 0.0
        self._current_power = 0.0
        self._current_gradient = 0.0
        self._step_count = 0
        self._steps_since_change = 0

    # Get the power
    def _get_power(self, current):
        # Get the active case curve
        curve = self.cases[self._current_case]

        # Interpolate the power value based on the current
        power = np.interp(current, curve[:, 0], curve[:, 1])
        return power

    # Get the gradient
    def _get_gradient(self):
        # Get the active case curve
        curve = self.cases[self._current_case]

        # Delta for gradient calculation (initially set to 0.001, can be adjusted)
        delta = 0.001

        # Limit the range of the gradient calculation
        left = max(curve[0, 0], self._current_current - delta)
        right = min(curve[-1, 0], self._current_current + delta)

        # If the left and right points are the same, return 0
        if right == left:
            return 0.0

        power_left = self._get_power(left)
        power_right = self._get_power(right)

        return (power_right - power_left) / (right - left)

    # Get the observation
    def _get_obs(self):
        return np.array([
            self._current_current,
            self._current_power,
            self._current_gradient
        ], dtype=np.float32)

    # Get aditional information about the environment
    def _get_info(self):
        return {
            "current_case": self._current_case,
            "current": self._current_current,
            "power": self._current_power,
            "gradient": self._current_gradient
        }

    # Reset the environment
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)

        # Select ramdomly one of the cases
        self._current_case = int(self.np_random.integers(0, 4))

        # Select the active curve
        curve = self.cases[self._current_case]

        self._current_current = float(self.np_random.uniform(curve[0, 0], curve[-1, 0]))

        # Reset episode counters
        self._step_count = 0
        self._steps_since_change = 0

        # Calculate initial power and gradient
        self._current_power = self._get_power(self._current_current)
        self._current_gradient = self._get_gradient()

        return self._get_obs(), self._get_info()

    # Step the environment
    def step(self, action):
        # Convert the selected action into a current adjustment
        current_change = (int(action) - 1) * self.current_increment

        # Get the active case curve
        curve = self.cases[self._current_case]

        # Update current
        # np.clip is used to keep the current within the curve boundaries
        self._current_current = float(np.clip(
            self._current_current + current_change,
            curve[0, 0],
            curve[-1, 0]
        ))

        # Calculate power and gradient after the action
        self._current_power = self._get_power(self._current_current)
        self._current_gradient = self._get_gradient()

        # Obtain reward
        # We don't use the maximum of the curve to calculate the reward nor to stop the episode
        reward = self._current_power

        # Save the evaluated state before changing the case
        # When the curve is changed, the power is recalculated with the new condition
        # But, the reward remains to the previous value
        # This can help to register the previous rewards before the case change
        evaluated_case = self._current_case
        evaluated_power = self._current_power
        evaluated_current = self._current_current

        # Update the step count
        self._step_count += 1
        self._steps_since_change += 1

        # Change the shading case each BATCH
        case_changed = False

        if self._steps_since_change >= self.batch:
            # Select a different case from the current one
            possible_cases = [
                i for i in range(4)
                if i != self._current_case
            ]

            self._current_case = int(
                self.np_random.choice(possible_cases)
            )

            self._steps_since_change = 0
            case_changed = True

            # Adjust current to the limits of the new curve
            curve = self.cases[self._current_case]

            self._current_current = float(np.clip(
                self._current_current,
                curve[0, 0],
                curve[-1, 0]
            ))

            # Recalculate power and gradient for the new case
            self._current_power = self._get_power(self._current_current)
            self._current_gradient = self._get_gradient()

        # Episode ends when reached the maximum number of steps
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = self._get_info()

        # Additional information for evaluation and visualization
        info["evaluated_case"] = evaluated_case
        info["evaluated_power"] = evaluated_power
        info["evaluated_current"] = evaluated_current
        info["case_changed"] = case_changed

        return self._get_obs(), reward, terminated, truncated, info

    # Render the environment (For now, only simple vision,
    # we should implement more implementations like Matplotlib to visualize the environment and to generate GIF)
    def render(self):
        if self.render_mode == "human":
            print(f"Active case: {self._current_case}")
            print(f"Current: {self._current_current:.3f} A")
            print(f"Power: {self._current_power:.3f} W")
            print(f"Gradient: {self._current_gradient:.3f}")