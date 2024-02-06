import json
import matplotlib.pyplot as plt
from Params import *
import os
# Initialize an empty list to store rewards for each episode
all_rewards = []
# Move to params
Logs=["AlignmentReward_log", "CohesionReward_log", "CollisionReward_log", "SeparationReward_log"]
  
# Modularize

# logs=readFiles(Logs)
def readFiles(logfiles):
    values = []
    for i in range(0, SimulationVariables['Episodes']):
        for j, file in enumerate(logfiles):
            values.append([])
            with open(os.path.join(Files["Flocking"], "Testing", "Rewards", "Components", f"Episode{i}", f"{file}.json",), 'r') as f:
                for line in f:
                    try:
                        reward = json.loads(line)
                        values[j].append(reward)
                    except json.JSONDecodeError:
                        print(f"Ignoring invalid JSON on line: {line}")
    return values

#Works
def rewardEpisodes():
    for episode in range(0, SimulationVariables["Episodes"]):
        episode_folder = os.path.join(Files["Flocking"], "Analysis", "Rewards", "Components", f"Episode{episode}")
        os.makedirs(episode_folder, exist_ok=True)
        
        all_rewards = []

        file_path = os.path.join(Files["Flocking"], "Testing", "Rewards", "Components", f"Episode{episode}", "Reward_Total_log.json")
        with open(file_path, 'r') as f:
            try:
                rewards = [float(line) for line in f]
                all_rewards.append(rewards)
            except ValueError:
                print(f"Ignoring invalid float value in file: {file_path}")

        # Plot the rewards over timesteps for each episode
        for _, rewards in enumerate(all_rewards, start=1):
            timestep = range(len(rewards))
            plt.plot(timestep, rewards, label=f"Episode {episode}")

        # Modularize
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.title(f'Reward over Timesteps for Episode {episode}')
        plt.legend()

        # Save the plot
        plt.savefig(os.path.join(episode_folder, 'Reward.png'))
        plt.clf()  # Clear the current figure for the next episode

def rewardBreakdownSingle():import os
import matplotlib.pyplot as plt

def rewardBreakdownSingle():
    figure_folder = f"Results\\Flocking\\Analysis\\Rewards\\Components"
    log_folder = f"Results\\Flocking\\Testing\\Rewards\\Components"

    # Ensure the directory exists, create if not
    os.makedirs(figure_folder, exist_ok=True)

    # Iterate over episodes
    for episode in range(0, SimulationVariables["Episodes"]):
        episode_folder = os.path.join(figure_folder, f"Episode{episode}")
        os.makedirs(episode_folder, exist_ok=True)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Iterate over log files
        for log_file in Logs:
            # Read data from the file
            with open(os.path.join(log_folder, f"Episode{episode}", f"{log_file}.json"), "r") as f:
                data = f.readlines()

            # Parse data and plot
            steps = range(1, len(data) + 1)
            rewards = [float(line.strip()) for line in data]
            ax.plot(steps, rewards, label=log_file.split("_")[0])

        # Set labels and title
        ax.set_xlabel("Steps")
        ax.set_ylabel("Reward")
        ax.set_title(f"Episode {episode}")
        ax.legend()

        # Save figure
        fig.savefig(os.path.join(episode_folder, "RewardsBreakdown.png"))

        # Close the figure
        plt.close(fig)

def plot_rewards_for_all_episodes(logfiles):
    for episode in range(SimulationVariables["Episodes"]):
        episode_folder = os.path.join(Files["Flocking"], "Testing", "Rewards", "Components", f"Episode{episode}")
        rewardBreakdownSingle(episode_folder, logfiles)

# Example usage:
# Logs = ["AlignmentReward_log", "CohesionReward_log", "CollisionReward_log", "SeparationReward_log"]
# plot_rewards_for_all_episodes(Logs)
readFiles(Logs)
# rewardBreakdown(Logs)
# rewardBreakdownSingle()
# plot_dynamics()
