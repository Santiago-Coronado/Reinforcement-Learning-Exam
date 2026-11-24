'''
Usamos el wrapper de gym demostrado en Python mostrado en clase:
https://github.com/Farama-Foundation/Gymnasium/blob/main/docs/introduction/create_custom_env.md
'''
from typing import Optional
import numpy as np
import gymnasium as gym

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