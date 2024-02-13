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
# Works
def readFiles(logfiles):
    values = []
    for i in range(0, SimulationVariables['Episodes']):
        for j, file in enumerate(logfiles):
            values.append([])
            file_path=os.path.join(Files["Flocking"], "Testing", "Rewards", "Components", f"Episode{i}", f"{file}.json") 
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        rewards = []
                        for line in f:
                            line = line.strip()  # Remove leading/trailing whitespace
                            if line:  # Skip empty lines
                                try:
                                    rewards.append(float(line))
                                except ValueError:
                                    print(f"Ignoring invalid float value: {line}")
                        all_rewards.append(rewards)
                    except Exception as e:
                        print(f"An error occurred while processing file: {file}")
                        print(e)
                 
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

        # Print length of rewards data for debugging
        print(f"Episode {episode} has {len(rewards)} timesteps.")

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

# Make it ok
def rewardBreakdownSingle(logfiles):
    for episode in range(0, SimulationVariables["Episodes"]):
        episode_folder = os.path.join(Files["Flocking"], "Analysis", "Rewards", "Components", f"Episode{episode}")
        os.makedirs(episode_folder, exist_ok=True)
        
        for log_file in logfiles:
            all_rewards = []

            file_path = os.path.join(Files["Flocking"], "Testing", "Rewards", "Components", f"Episode{episode}", f'{log_file}.json')
            with open(file_path, 'r') as f:
                try:
                    rewards = [float(line) for line in f]
                    all_rewards.append(rewards)
                except ValueError as e:
                    print(f"Ignoring invalid value in file {file_path}: {e}")

            # Plot and save each reward log separately
            for i, rewards in enumerate(all_rewards):
                timestep = range(len(rewards))
                plt.plot(timestep, rewards, label=logfiles[i])  # Use label for legend
                plt.xlabel('Timestep')
                plt.ylabel('Reward')
                plt.title(f'Reward over Timesteps - Episode {episode}')
                plt.legend()
                plt.savefig(os.path.join(episode_folder, f'{log_file}_Rewards.png'))
                plt.clf()
            # Clear the plot after saving all reward logs for this episode
            plt.close()


def plotDynamics():
    for episode in range(SimulationVariables["Episodes"]):
        accelerations_folder = os.path.join(Files["Flocking"], "Analysis", "Dynamics", "Accelerations", f"Episode{episode}")
        velocities_folder = os.path.join(Files["Flocking"], "Analysis", "Dynamics", "Velocities", f"Episode{episode}")
        os.makedirs(accelerations_folder, exist_ok=True)
        os.makedirs(velocities_folder, exist_ok=True)

        agent_dynamics = {i: {'velocities_magnitude': [], 'accelerations_magnitude': []} for i in range(10)}

        # Read accelerations data
        accelerations_file = os.path.join(Files['Flocking'], 'Testing', 'Dynamics', 'Accelerations', f'Episode_{episode}.json')
        with open(accelerations_file, 'r') as f:
            accelerations_data = json.load(f)
            # Ensure that accelerations_data is a list
            if isinstance(accelerations_data, list):
                for timestep_data in accelerations_data:
                    for agent_data in timestep_data['accelerations']:
                        agent_id = agent_data['agent_id']
                        acceleration_magnitude = agent_data['magnitude']
                        agent_dynamics[agent_id]['accelerations_magnitude'].append(acceleration_magnitude)
            else:
                print(f"Invalid data format in accelerations file for episode {episode}")

        # Plot and save accelerations for each agent
        for agent_id, data in agent_dynamics.items():
            plt.plot(data['accelerations_magnitude'], label=f'Agent {agent_id}', color=f'C{agent_id}')

        plt.xlabel('Timestep')
        plt.ylabel('Acceleration Magnitude')
        plt.title(f'Agent Accelerations - Episode {episode}')
        plt.legend()
        plt.savefig(os.path.join(accelerations_folder, f'Agent_Accelerations{episode}.png'))
        plt.clf()

        # Read velocities data
        velocities_file = os.path.join(Files['Flocking'], 'Testing', 'Dynamics', 'Velocities', f'Episode_{episode}.json')
        with open(velocities_file, 'r') as f:
            velocities_data = json.load(f)
            # Ensure that velocities_data is a list
            if isinstance(velocities_data, list):
                for timestep_data in velocities_data:
                    for agent_data in timestep_data['velocities']:
                        agent_id = agent_data['agent_id']
                        velocity_magnitude = agent_data['magnitude']
                        agent_dynamics[agent_id]['velocities_magnitude'].append(velocity_magnitude)
            else:
                print(f"Invalid data format in velocities file for episode {episode}")

        # Plot and save velocities for each agent
        for agent_id, data in agent_dynamics.items():
            plt.plot(data['velocities_magnitude'], label=f'Agent {agent_id}', color=f'C{agent_id}')

        plt.xlabel('Timestep')
        plt.ylabel('Velocity Magnitude')
        plt.title(f'Agent Velocities - Episode {episode}')
        plt.legend()
        plt.savefig(os.path.join(velocities_folder, f'Agent_Velocities{episode}.png'))
        plt.clf()



# plotDynamics()
        
rewardBreakdownSingle(Logs)
