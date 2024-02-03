import json
import matplotlib.pyplot as plt
from Params import *
import os
# Initialize an empty list to store rewards for each episode
all_rewards = []
# Move to params
Logs=["AlignmentReward_log", "CohesionReward_log", "CollisionReward_log", "SeparationReward_log", "Reward_Total_log"]
  
# Modularize

def readFiles(logfiles):
    values = []
    for i in range(0, SimulationVariables['Episodes']):
        for j, file in enumerate(logfiles):
            values.append([])
            with open(os.path.join(Files["Flocking"], "Testing", "Rewards", "Components", f"Episode_{i}", f"{file}.json",), 'r') as f:
                for line in f:
                    try:
                        reward = json.loads(line)
                        values[j].append(reward)
                    except json.JSONDecodeError:
                        print(f"Ignoring invalid JSON on line: {line}")
    return values

def rewardEpisodes():
    for episode in range(1, SimulationVariables["Episodes"]):
        episode_folder = f'{Files["Flocking"]}\\Analysis\\Rewards\\Other\\Episode{episode}'
        os.makedirs(episode_folder, exist_ok=True)
        
        all_rewards = []

        file_path = f'{Files["Flocking"]}\\Testing\\Rewards\\Other\\Episode_{episode}.json'
        with open(file_path, 'r') as f:
            rewards = json.load(f)
            all_rewards.append(rewards)

        # Plot the rewards over timesteps for each episode
        for _ , rewards in enumerate(all_rewards, start=1):
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

def rewardBreakdown(logfiles):
    files = readFiles(logfiles)

    # Plot all rewards together with different colors
    for i, rewards in enumerate(files):
        timestep = range(len(rewards))
        plt.plot(timestep, rewards, label=logfiles[i])

    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Reward over Timesteps for All Episodes')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f'{Files["Flocking"]}\\Analysis\\Rewards.png')

def rewardBreakdownSingle(logfiles):
    files = readFiles(logfiles)

    # Create a directory to save individual plots
    output_dir = f'{Files["Flocking"]}\\Analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save each reward log separately
    for i, rewards in enumerate(files):
        timestep = range(len(rewards))
        plt.plot(timestep, rewards)
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.title(f'Reward over Timesteps - {logfiles[i]}')
        plt.savefig(os.path.join(output_dir, f'{logfiles[i]}_Rewards.png'))
        plt.close()


def plot_dynamics(logfiles, output_dir):
    files = readFiles(logfiles)
    
    for logfile in files:
        with open(os.path.join(Files["Flocking"], "Testing", logfile), 'r') as f:
            data = json.load(f)

            for agent_id, agent_data in data.items():
                # Plot velocity
                plt.plot(agent_data['velocities_magnitude'], label=f'Agent {agent_id} - Velocity')
                
                # Plot acceleration
                plt.plot(agent_data['accelerations_magnitude'], label=f'Agent {agent_id} - Acceleration')
                
                plt.xlabel('Timestep')
                plt.ylabel('Magnitude')
                plt.title(f'Agent {agent_id} - Velocity and Acceleration - All Episodes')
                plt.legend()
                
                # Save the plot
                episode = int(logfile.split("_")[1].split(".")[0])
                output_file = os.path.join(output_dir, f'Agent_{agent_id}_Episode_{episode}_Velocity_Acceleration.png')
                plt.savefig(output_file)
                
                # Clear the plot for the next iteration
                plt.clf()

logs=readFiles(Logs)
# rewardEpisodes()
# rewardBreakdown()
# rewardBreakdownSingle()
# plot_dynamics()
