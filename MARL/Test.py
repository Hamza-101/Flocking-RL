import ray
from ray.rllib.algorithms.ppo import PPO
from env import FlockingEnv
from Settings import *
import os
import json

# Initialize Ray (if not already initialized)
ray.init(ignore_reinit_error=True)

# Load the trained MAPPO model
checkpoint_path = "/path/to/your/checkpoint"  # Update this with your actual checkpoint path
config = {
    "env": "flocking_env",
    "framework": "torch",
}

mappo_trainer = PPO(config=config)
mappo_trainer.restore(checkpoint_path)

# Load the environment
env = FlockingEnv()

# Number of episodes to run
num_episodes = 10
positions_directory = "./positions_data"  # Directory to save JSON files
os.makedirs(positions_directory, exist_ok=True)

for episode in range(num_episodes):
    observations = env.reset()
    done = {"__all__": False}
    episode_reward = 0
    timestep = 0
    reward_episode = []

    # Initialize dictionaries to store data
    positions_dict = {i: [] for i in range(len(env.agents))}
    velocities_dict = {i: [] for i in range(len(env.agents))}
    accelerations_dict = {i: [] for i in range(len(env.agents))}
    trajectory_dict = {i: [] for i in range(len(env.agents))}
    episode_rewards_dict = {}

    while not done["__all__"]:
        action_dict = {}

        # Compute actions for each agent
        for agent_id, obs in observations.items():
            action_dict[agent_id] = mappo_trainer.compute_single_action(observation=obs, policy_id=agent_id)

        # Apply actions to the environment
        observations, rewards, done, info = env.step(action_dict)

        # Accumulate rewards
        episode_reward += sum(rewards.values())

        # Store data for each agent
        for i, agent in enumerate(env.agents.values()):
            positions_dict[i].append(agent.position.tolist())
            velocity = agent.velocity.tolist()
            velocities_dict[i].append(velocity)
            acceleration = agent.acceleration.tolist()
            accelerations_dict[i].append(acceleration)
            trajectory_dict[i].append(agent.position.tolist())

        reward_episode.append(rewards)

        timestep += 1
        if timestep >= min(SimulationVariables["EvalTimeSteps"], 3000):
            break

    # Store rewards for the current episode
    episode_rewards_dict[str(episode)] = reward_episode

    # Save data into JSON files after each episode
    with open(os.path.join(positions_directory, f"Episode{episode}_positions.json"), 'w') as f:
        json.dump(positions_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_velocities.json"), 'w') as f:
        json.dump(velocities_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_accelerations.json"), 'w') as f:
        json.dump(accelerations_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_trajectory.json"), 'w') as f:
        json.dump(trajectory_dict, f, indent=4)

    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

# Shutdown Ray after testing
ray.shutdown()