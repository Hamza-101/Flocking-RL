import json
import matplotlib.pyplot as plt
from Params import *
# Initialize an empty list to store rewards for each episode
all_rewards = []

# Load rewards from the JSON files for each episode
for episode in range(1, RLVariables['Episodes']):
    with open(f'{Results["EpRewards"]}_Allignment_{episode}.json', 'r') as f:
        rewards = json.load(f)
        all_rewards.append(rewards)

# Plot the rewards over timesteps for each episode
for episode, rewards in enumerate(all_rewards, start=1):
    timestep = range(len(rewards))
    plt.plot(timestep, rewards, label=f"Episode {episode}")

plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward over Timesteps for Each Episode')
plt.legend()
plt.show()
