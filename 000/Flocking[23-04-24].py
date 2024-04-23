
import os
import gym
import json
import numpy as np
import torch as th
from tqdm import tqdm
from Settings import *
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from scipy.signal import savgol_filter

# Activation Function -> Tanh
# Layers -> 8 Actor 8 Critic
# Neurons 512 Neurons per layer
# Input -> Positions and Velocity(Observation Space)
# Output -> Acceleration(Action Space)

# Neural Network Parameters
policy_kwargs = dict(
    activation_fn=th.nn.Tanh,  
    net_arch=[
        dict(pi=[512, 512, 512, 512, 512, 512, 512, 512], 
             vf=[512, 512, 512, 512, 512, 512, 512, 512]),
    ]
)


# Why reward less
# Continuous reward

# Move to other file

#Weights
class Agent:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

        # Initialize acceleration
        self.acceleration = np.round(np.random.uniform(-SimulationVariables["AccelerationUpperLimit"], SimulationVariables["AccelerationUpperLimit"], size=2), 2)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        
        # Calculate initial velocity based on the initial acceleration
        self.velocity = self.acceleration * SimulationVariables["dt"]
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    # See if can be moved
    def update(self, action):
        # Check if this correct
        self.acceleration += action
        # Update velocity
        self.velocity += self.acceleration * SimulationVariables["dt"]

        vel = np.linalg.norm(self.velocity)

        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity    

        self.position += self.velocity * SimulationVariables["dt"]
        return self.position, self.velocity
# Move to other file
class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

class FlockingEnv(gym.Env):
    def __init__(self):
        super(FlockingEnv, self).__init__()
        self.episode=0
        self.CTDE=False
        self.current_timestep = 0 
        self.reward_log = []
        self.counter=0

        self.agents = [Agent(position) for position in self.read_agent_locations()]
        
        min_action = np.array([-5, -5] * len(self.agents), dtype=np.float32)
        max_action = np.array([5, 5] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        min_obs = np.array([[-np.inf, -np.inf, -2.5, -2.5]] * len(self.agents), dtype=np.float32)
        max_obs = np.array([[np.inf, np.inf, 2.5, 2.5]] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)


    def step(self, actions):
        #   # Add Gaussian noise to actions
        noisy_actions = actions + np.random.normal(loc=0, scale=0.01, size=actions.shape)
        
        #Clip actions to action space bounds
        noisy_actions = np.clip(noisy_actions, self.action_space.low, self.action_space.high)

        # if(self.current_timestep % 400000 == 0):
        #     print(self.current_timestep)
        #     self.counter = self.counter + 1
        #     print("Counter", self.counter)

        self.current_timestep+=1
        reward=0
        done=False
        info={}

        observations = self.simulate_agents(actions)
        reward, out_of_flock = self.calculate_reward()
        
        #Validate this
        if (self.CTDE==False):
            # Terminal Conditions
            for agent in self.agents:
                if((self.check_collision(agent)) or (out_of_flock==True)):
                    done=True
                    env.reset()


        # if self.CTDE:

        #     log_path = os.path.join(Files['Flocking'], 'Testing', 'Rewards', 'Components', f"Episode{episode}")
        #     log_path = os.path.join(log_path, "Reward_Total_log.json")

        
        #     with open(log_path, 'a') as f:
        #         json.dump((round(reward, 2)), f, indent=4)
        #         f.write('\n')

        
        self.current_timestep = self.current_timestep + 1

        return observations, reward, done, info

        #CHECK RESET

    def close(self):
        print("Environment is closed. Cleanup complete.")
        
        #Does velocity make a difference
        #Observation Space
   
    def simulate_agents(self, actions):
        observations = []
        for i, agent in enumerate(self.agents):
            position, velocity = agent.update(actions[i])
            observations.append(np.concatenate([position, velocity]))  # Include velocity
        return observations
    
    def check_collision(self, agent):
        # See if correct
        # Collisions[idx].append(collisions_agent)

        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
        return False

    def reset(self):
        
        # seed_everything(SimulationVariables["Seed"])
        env.seed(SimulationVariables["Seed"])
        self.agents = [Agent(position) for position in self.read_agent_locations()]
        for agent in self.agents:
            agent.acceleration = np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)
            agent.velocity = agent.acceleration * SimulationVariables["dt"]               
        observation = self.get_observation()
        return observation   
   
    def get_observation(self):
        observation = np.zeros((len(self.agents), 4), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            observation[i] = [
                agent.position[0],
                agent.position[1],
                agent.velocity[0],  
                agent.velocity[1]
            ]
        return observation
   
    def calculate_reward(self):

        reward=0
       

        Collisions={}

        neighbor_positions = [[] for _ in range(len(self.agents))]  # Initialize empty lists for neighbor indices
        neighbor_velocities = [[] for _ in range(len(self.agents))]
        out_of_flock=False

        #WRITE COLLISION CODE
        for idx, _ in enumerate(self.agents):
            Collisions[idx] = []

            #Check this
        for _, agent in enumerate(self.agents): 
            neighbor_positions, neighbor_velocities=self.get_closest_neighbors(agent)
            
            # val, out_of_flock=self.calculate_combined_reward(agent, neighbor_positions, neighbor_velocities) #############################################################

            # reward+=val

            val, out_of_flock=self.reward(agent, neighbor_velocities, neighbor_positions)
            reward+=val

        return reward, out_of_flock

    # Find neighbor based on distance
    def get_closest_neighbors(self, agent):
        neighbor_positions=[]
        neighbor_velocities=[]

        # INDEX VS POSITION

        for i, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)

                if(self.CTDE==True):
                    if distance < SimulationVariables["NeighborhoodRadius"]:
                        neighbor_positions.append(other.position)
                        # print(other)
                        neighbor_velocities.append(other.velocity)
                        # print(other)

                else:  
                    neighbor_positions.append(other.position)
                    neighbor_velocities.append(other.velocity)

        return neighbor_positions, neighbor_velocities         

    def reward(self, agent, neighbor_velocities, neighbor_positions):
        total_reward=0
        outofflock=False

        if(len(neighbor_positions) > 0):
            for neighbor_position in neighbor_positions:
                distance = np.linalg.norm(agent.position - neighbor_position)
                    
                if distance < SimulationVariables["SafetyRadius"]:
                    # Large penalty to discourage agents from getting too close
                    total_reward -= 10 
                    
                elif SimulationVariables["SafetyRadius"] < distance < SimulationVariables["NeighborhoodRadius"]:
                    # Decaying exponential function for rewards
                    alpha = 0.99  # Adjust this parameter as needed
                    total_reward += np.exp(-alpha * distance)

            
            if (len(neighbor_velocities) > 0):
                average_velocity = np.mean(neighbor_velocities, axis=0)
                desired_orientation = average_velocity - agent.velocity   
                orientation_diff = np.arctan2(desired_orientation[1], desired_orientation[0]) - np.arctan2(agent.velocity[1], agent.velocity[0])

                if orientation_diff > np.pi:
                    orientation_diff -= 2 * np.pi
                elif orientation_diff < 0:
                    orientation_diff += 2 * np.pi

                alignment = 1 - np.abs(orientation_diff)

                if (alignment < 0.5):
                    total_reward -= 50 * (alignment)
                else:
                    total_reward += 50 * (alignment)

        else:
            # If no neighbors, encourage agent to find flock
            total_reward -=100
            outofflock=True

        return total_reward, outofflock
    
    def read_agent_locations(self):
        File = rf"{Results['InitPositions']}" + str(self.counter) + "\config.json"
        # print(File)
        with open(File, "r") as f:
            data = json.load(f)
        return data
    
    
def Log(episode, Rewards):
    episode_directory = os.path.join(Files['Flocking'], 'Testing', 'Rewards', 'Components', f"Episode{episode}")
    os.makedirs(episode_directory, exist_ok=True)

    for reward_type, reward_value in Rewards.items():
        log_path = os.path.join(episode_directory, f"{str(reward_type)}_log.json")
        with open(log_path, 'a') as f:
            f.write(json.dumps(round((reward_value), 2)) + '\n')

def delete_files(): 
    Paths = ["Results\Flocking\Testing\Dynamics\Accelerations", "Results\Flocking\Testing\Dynamics\Velocities", 
            "Results\Flocking\Testing\Rewards\Other"]

    Logs = ["AlignmentReward_log.json", "CohesionReward_log.json",
            "SeparationReward_log.json", "CollisionReward_log.json",
            "Reward_Total_log.json"]

    # Delete files based on Paths
    for Path in Paths:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], Path, f"Episode{episode}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    for log_file in Logs:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], "Testing", "Rewards", "Components", f"Episode{episode}", log_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")       

def generateResults1():
    with open(rf"{Results['EpisodalRewards']}" + ".json" , "r") as f:
        episode_rewards_dict = json.load(f)

    # Separate episodes based on total sum of rewards
    keys_above_threshold = []
    keys_below_threshold = []

    for episode, rewards in episode_rewards_dict.items():
        total_sum = sum(rewards)
        if total_sum > 1000000:
            keys_above_threshold.append(episode)
        else:
            keys_below_threshold.append(episode)

    # Plot and save episodes with total sum above 1000000
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    for episode in keys_above_threshold:
        rewards = episode_rewards_dict[episode]
        smoothed_rewards = savgol_filter(rewards, 21, 3)  # Adjust window length and polynomial order as needed
        plt.plot(range(1, len(smoothed_rewards) + 1), smoothed_rewards, label=f"Episode {episode}", alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"Rewards for Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("AboveThreshold.png", dpi=300)
    plt.close()

    # Plot and save episodes with total sum below 1000000
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    for episode in keys_below_threshold:
        rewards = episode_rewards_dict[episode]
        smoothed_rewards = savgol_filter(rewards, 21, 3)  # Adjust window length and polynomial order as needed
        plt.plot(range(1, len(smoothed_rewards) + 1), smoothed_rewards, label=f"Episode {episode}", alpha=0.7)

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"Rewards for Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("BelowThreshold.png", dpi=300)
    plt.close()

def generateResults2():
    with open(rf"{Results['EpisodalRewards']}.json", "r") as f:
        episode_rewards_dict = json.load(f)

    keys_above_threshold = []
    keys_below_threshold = []

    for episode, rewards in episode_rewards_dict.items():
        total_sum = sum(rewards)
        if total_sum > 1000000:
            keys_above_threshold.append(episode)
        else:
            keys_below_threshold.append(episode)

    plt.figure(figsize=(10, 6))

    for episode in keys_above_threshold:
        rewards = episode_rewards_dict[episode]
        smoothed_rewards = savgol_filter(rewards, 21, 3)
        plt.plot(range(1, len(smoothed_rewards) + 1), smoothed_rewards, label=f"Episode {episode}", alpha=0.7)

    for episode in keys_below_threshold:
        rewards = episode_rewards_dict[episode]
        smoothed_rewards = savgol_filter(rewards, 21, 3)
        plt.plot(range(1, len(smoothed_rewards) + 1), smoothed_rewards, label=f"Episode {episode}", alpha=0.7)

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"Rewards for {SimulationVariables['Episodes']} Episodes")
    plt.legend(title="Episode No.")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Combined.png", dpi=3000)

def generateAnalytics():

    for episode in range(SimulationVariables["Episodes"]):

        with open(os.path.join(rf"{Files['Flocking']}/Testing/Positions/Episode{episode}_velocities.json"), 'r') as f:
            velocities_dict = json.load(f)

        with open(os.path.join(rf"{Files['Flocking']}/Testing/Positions/Episode{episode}_accelerations.json"), 'r') as f:
            accelerations_dict = json.load(f)

        num_agents = len(velocities_dict)  # Assuming all agents have the same number of velocities

        # Initialize lists to store all agents' velocities and accelerations
        all_velocity_magnitudes = []
        all_acceleration_magnitudes = []

        for i in range(num_agents):
            velocities = np.array(velocities_dict[str(i)])
            accelerations = np.array(accelerations_dict[str(i)])

            # Calculate magnitudes
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

            # Append to the list of all agents' velocities and accelerations
            all_velocity_magnitudes.append(velocity_magnitudes)
            all_acceleration_magnitudes.append(acceleration_magnitudes)

        # Plot velocities for all agents in one file per episode
        plt.figure(figsize=(10, 5))
        for i, velocity_magnitudes in enumerate(all_velocity_magnitudes):
            plt.plot(velocity_magnitudes, label=f"Agent {i+1}")
        plt.title(f"All Agents' Velocities Magnitudes - Episode {episode}")
        plt.xlabel("Timestep")
        plt.ylabel("Velocity Magnitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Episode_{episode}_velocities.png")
        plt.close()

        # Plot accelerations for all agents in one file per episode
        plt.figure(figsize=(10, 5))
        for i, acceleration_magnitudes in enumerate(all_acceleration_magnitudes):
            plt.plot(acceleration_magnitudes, label=f"Agent {i+1}")
        plt.title(f"Accelerations {episode}")
        plt.xlabel("Timestep")
        plt.ylabel("Acceleration Magnitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Episode_{episode}_accelerations.png")
        plt.close()

        print(f"Plots for Episode {episode} - All Agents' velocities and accelerations magnitudes generated successfully.")

if os.path.exists(Results["Rewards"]):
    os.remove(Results["Rewards"])
    print(f"File {Results['Rewards']} has been deleted.")

def seed_everything(seed):
    # random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    env.seed(seed)
    env.action_space.seed(seed)


env=FlockingEnv()
# Explain this
seed_everything(SimulationVariables["Seed"])

# Model Training
# model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_Agents_tensorboard/", verbose=1)
# model.set_random_seed(SimulationVariables["ModelSeed"])
# model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"]) 
# model.save(rf"{Files['Flocking']}\\Models\\FlockingCombined")

# INcrease scale of plots


# Model Testing
env = FlockingEnv()
model = PPO.load(rf"{Files['Flocking']}\\Models\\FlockingCombined")

delete_files()
positions_directory = rf"{Files['Flocking']}/Testing/Episodes/"
os.makedirs(positions_directory, exist_ok=True)

env.counter=10
episode_rewards_dict = {}
positions_dict = {i: [] for i in range(len(env.agents))}

for episode in tqdm(range(0, SimulationVariables['Episodes'])):

    env.episode = episode
    print("Episode:", episode)
    env.CTDE = True
    obs = env.reset()
    done = False
    timestep = 0
    reward_episode = []
    reward_cumulative = 0
    positions_dict = {i: [] for i in range(len(env.agents))}
    velocities_dict = {i: [] for i in range(len(env.agents))}
    accelerations_dict = {i: [] for i in range(len(env.agents))}

    while timestep <= SimulationVariables["EvalTimeSteps"]:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        reward_episode.append(reward)
        
        for i, agent in enumerate(env.agents):
            positions_dict[i].append(agent.position.tolist())

            velocity_magnitude = np.linalg.norm(agent.velocity)
            velocities_dict[i].append(velocity_magnitude)

            acceleration_magnitude = np.linalg.norm(agent.acceleration)
            accelerations_dict[i].append(acceleration_magnitude)

        timestep += 1
        episode_rewards_dict[str(episode)] = reward_episode

    with open(f"{positions_directory}/Episode{episode}_positions.json", 'w') as f:
        json.dump(positions_dict, f, indent=4)
    with open(f"{positions_directory}/Episode{episode}_velocities.json", 'w') as f:
        json.dump(positions_dict, f, indent=4)
    with open(f"{positions_directory}/Episode{episode}_accelerations.json", 'w') as f:
        json.dump(positions_dict, f, indent=4)

    env.counter += 1
    print(sum(reward_episode))
    
with open(rf"{Results['EpisodalRewards']}" + ".json" , 'w') as f:
    json.dump(episode_rewards_dict, f, indent=4)

env.close()

print("Testing completed")

# Analytics
print("Generating Results")

generateResults1()
generateResults2()
generateAnalytics()

# >>> sb3.get_system_info()
