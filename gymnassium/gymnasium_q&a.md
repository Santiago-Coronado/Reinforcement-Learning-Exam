Worlflow derivado de: https://github.com/Farama-Foundation/Gymnasium/blob/main/docs/introduction/create_custom_env.md

Ask yourself these fundamental questions:

🎯 What skill should the agent learn?

Agent should be able to switch between 4 different environments with several optimization algorithms: reinforce, actor-critic, ppo, classic CEM, MLP-CEM.

These intervals will be predefined by episodes and steps. 

🎮 What actions can the agent take?

Action: current I (proposed I or incremental I)
Reward is power-dependent or a signal of the same.

🏆 How do we measure success?

Maximizing power

⏰ When should episodes end?

Depends on the method. TBD.