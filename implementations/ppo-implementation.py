import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(42)
torch.manual_seed(42)


class BackboneNetwork(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred


class PPOAgent:
    def __init__(self, input_size=3, hidden_size=64, output_size=3, dropout=0.1, lr=3e-4, gamma=0.99, gae_lambda=0.95, epsilon=0.2, entropy_coef=0.01, device="cpu"):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef

        actor = BackboneNetwork(input_size, hidden_size, output_size, dropout)
        critic = BackboneNetwork(input_size, hidden_size, 1, dropout)
        self.model = ActorCritic(actor, critic).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, observation, deterministic=False):
        self.model.eval()
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_pred, value_pred = self.model(obs_tensor)
            probabilities = torch.softmax(action_pred, dim=1)
            dist = Categorical(probabilities)
            action = torch.argmax(probabilities, dim=1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value_pred.item()

    def calculate_returns(self, rewards, values, next_values, dones, terminated):
        advantages = []
        gae = 0.0
        for r, v, nv, d, term in zip(reversed(rewards), reversed(values), reversed(next_values), reversed(dones), reversed(terminated)):
            bootstrap = 0.0 if term else 1.0
            cont = 0.0 if d else 1.0
            delta = r + self.gamma * nv * bootstrap - v
            gae = delta + self.gamma * self.gae_lambda * cont * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def calculate_advantages(self, returns, values):
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def calculate_surrogate_loss(self, actions_log_probability_old, actions_log_probability_new, advantages):
        advantages = advantages.detach()
        policy_ratio = (actions_log_probability_new - actions_log_probability_old).exp()
        surrogate_loss_1 = policy_ratio * advantages
        surrogate_loss_2 = torch.clamp(policy_ratio, min=1.0 - self.epsilon, max=1.0 + self.epsilon) * advantages
        return torch.min(surrogate_loss_1, surrogate_loss_2)

    def calculate_losses(self, surrogate_loss, entropy, returns, value_pred):
        entropy_bonus = self.entropy_coef * entropy
        policy_loss = -(surrogate_loss + entropy_bonus).mean()
        value_loss = f.smooth_l1_loss(returns, value_pred).mean()
        return policy_loss, value_loss

    def forward_pass(self, env, steps_per_update, state, info):
        states, actions, actions_log_probability, values, rewards, next_values, dones, terminated_list = [], [], [], [], [], [], [], []
        completed_rewards = []
        episode_reward = 0.0

        for _ in range(steps_per_update):
            action, log_prob, value = self.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminated:
                next_val = 0.0
            else:
                with torch.no_grad():
                    _, next_val_pred = self.model(torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0))
                    next_val = next_val_pred.item()

            states.append(state)
            actions.append(action)
            actions_log_probability.append(log_prob)
            values.append(value)
            rewards.append(float(reward))
            next_values.append(next_val)
            dones.append(done)
            terminated_list.append(terminated)

            episode_reward += reward
            state = next_state

            if done:
                completed_rewards.append(episode_reward)
                episode_reward = 0.0
                state, info = env.reset()

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(actions_log_probability, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)

        returns = self.calculate_returns(rewards, values, next_values, dones, terminated_list)
        advantages = self.calculate_advantages(returns, values_t)

        return completed_rewards, states_t, actions_t, old_log_probs_t, advantages, returns, state, info

    def update_policy(self, states, actions, actions_log_probability_old, advantages, returns, ppo_steps=4, batch_size=128):
        dataset = TensorDataset(states, actions, actions_log_probability_old.detach(), advantages, returns)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        total_policy_loss, total_value_loss = 0.0, 0.0

        for _ in range(ppo_steps):
            for state, action, old_log_prob, advantage, return_val in loader:
                action_pred, value_pred = self.model(state)
                value_pred = value_pred.squeeze(-1)
                action_prob = f.softmax(action_pred, dim=-1)
                dist = Categorical(action_prob)
                entropy = dist.entropy()
                new_log_prob = dist.log_prob(action)

                surrogate_loss = self.calculate_surrogate_loss(old_log_prob, new_log_prob, advantage)
                policy_loss, value_loss = self.calculate_losses(surrogate_loss, entropy, return_val, value_pred)

                self.optimizer.zero_grad()
                policy_loss.backward()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

    def train(self, env, updates=100, steps_per_update=512, ppo_steps=4, batch_size=128):
        best_reward = -np.inf
        best_weights = None
        all_rewards = []
        history = []

        state, info = env.reset(seed=42)

        for update_idx in range(updates):
            completed_rewards, states, actions, old_log_probs, advantages, returns, state, info = self.forward_pass(
                env, steps_per_update, state, info
            )
            policy_loss, value_loss = self.update_policy(
                states, actions, old_log_probs, advantages, returns, ppo_steps=ppo_steps, batch_size=batch_size
            )

            all_rewards.extend(completed_rewards)
            recent_rewards = all_rewards[-10:] if all_rewards else [0.0]
            average_reward = float(np.mean(recent_rewards))
            generation_best = float(np.max(recent_rewards))

            if generation_best > best_reward and all_rewards:
                best_reward = generation_best
                best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}

            history.append({
                "update": update_idx + 1,
                "average_reward": average_reward,
                "best_reward": generation_best,
                "global_best": best_reward,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
            })

            print(
                f"Update {update_idx + 1}/{updates} | "
                f"Average Reward: {average_reward:.4f} | "
                f"Best Reward: {generation_best:.4f} | "
                f"Global Best: {best_reward:.4f}"
            )

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return history

    def evaluate_best_agent(model_agent, env, seed=123):
        observation, info = env.reset(seed=seed)
        episode_history = []

        episode_history.append({
            "step": 0,
            "case": info["current_case"],
            "current": info["current"],
            "power": info["power"],
            "reward": 0.0,
            "action": None,
            "case_changed": False,
        })
        terminated = False
        truncated = False
        step = 0

        while not terminated and not truncated:
            action, _, _ = model_agent.select_action(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1

            episode_history.append({
                "step": step,
                "case": info["evaluated_case"],
                "current": info["evaluated_current"],
                "power": info["evaluated_power"],
                "reward": reward,
                "action": action,
                "case_changed": info["case_changed"],
            })

        print(f"Evaluation completed. Total steps: {step}")
        print(f"Total reward: {sum(x['reward'] for x in episode_history):.4f}")

        actions = [x["action"] for x in episode_history if x["action"] is not None]
        unique_actions, counts = np.unique(actions, return_counts=True)

        print("\nAction distribution:")
        for action, count in zip(unique_actions, counts):
            print(f"Action {action}: {count} times ({count / len(actions) * 100:.2f}%)")

        return episode_history

    def create_gif(episode_history, env):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax_curve = axes[0]
        ax_progress = axes[1]
        reward_axis = ax_progress.twinx()

        all_currents = np.concatenate([curve[:, 0] for curve in env.cases])
        all_powers = np.concatenate([curve[:, 1] for curve in env.cases])

        x_min, x_max = all_currents.min(), all_currents.max()
        y_min, y_max = all_powers.min(), all_powers.max()

        def update(frame):
            data = episode_history[frame]
            case = data["case"]
            current = data["current"]
            power = data["power"]

            ax_curve.clear()
            ax_progress.clear()
            reward_axis.clear()

            curve = env.cases[case]

            ax_curve.plot(
                curve[:, 0], curve[:, 1],
                color="blue",
                label="P-I Curve",
                linewidth=2,
            )

            points = [
                (x["current"], x["power"])
                for x in episode_history[:frame + 1]
                if x["case"] == case and x["step"] > 0
            ]

            if points:
                points_array = np.array(points)
                ax_curve.scatter(
                    points_array[:, 0],
                    points_array[:, 1],
                    color="red",
                    label="Evaluated Points",
                    zorder=3,
                )

            ax_curve.scatter(
                current, power,
                color="green",
                marker="D",
                s=100,
                label="Current Position",
                zorder=4,
            )

            ax_curve.set_xlim(x_min, x_max)
            ax_curve.set_ylim(y_min, y_max * 1.1)
            ax_curve.set_title(f"Case {case} - Step {data['step']}")
            ax_curve.set_xlabel("Current I (A)")
            ax_curve.set_ylabel("Power P (W)")
            ax_curve.legend()
            ax_curve.grid(True)

            previous_data = episode_history[:frame + 1]

            steps = [x["step"] for x in previous_data]
            powers = [x["power"] for x in previous_data]
            rewards = [x["reward"] for x in previous_data]

            ax_progress.plot(steps, powers, label="Power (W)")
            reward_axis.plot(steps, rewards, color="orange", label="Reward")

            ax_progress.set_xlabel("Step")
            ax_progress.set_ylabel("Power (W)")
            reward_axis.set_ylabel("Reward")

            ax_progress.set_title("Power and Reward Progress")
            ax_progress.grid(True)

            lines1, labels1 = ax_progress.get_legend_handles_labels()
            lines2, labels2 = reward_axis.get_legend_handles_labels()

            ax_progress.legend(lines1 + lines2, labels1 + labels2, loc="best")

            action_text = "Initial" if data["action"] is None else str(data["action"])

            fig.suptitle(
                f"MLP-PPO | Case {case} | Step {data['step']} | "
                f"Current: {current:.3f} A | Power: {power:.3f} W | "
                f"Action: {action_text}"
            )

            plt.tight_layout()

        animation = FuncAnimation(
            fig,
            update,
            frames=len(episode_history),
            interval=300,
            repeat=True,
        )

        animation.save(
            "mlp_ppo.gif",
            writer=PillowWriter(fps=5),
        )

        plt.close(fig)
        print("GIF saved as mlp_ppo.gif")


if __name__ == "__main__":
    from pathlib import Path
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from gymnassium.gym_env import CVP_env

    env = CVP_env(max_steps=300)

    model_agent = PPOAgent(
        input_size=3,
        hidden_size=64,
        output_size=3,
        dropout=0.1,
        lr=0.001,
    )

    history = model_agent.train(env, updates=100, steps_per_update=512)

    episode_history = PPOAgent.evaluate_best_agent(
        model_agent,
        env,
        seed=123,
    )

    PPOAgent.create_gif(episode_history, env)

    env.close()