import os
import json
import gym
from gym import spaces
from Params import *
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Why reward less
# Continuous reward
# Exit on collision
# Centroid

# n_fall_steps = 10: Possibly the number of steps allowed for the drone to fall before the simulation ends.

# Move to other file

#Weights
class Agent:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        
        self.acceleration = np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)                # Acceleration Initialized to 0
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        
        self.velocity = np.zeros(2)      
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    # See if can be moved
    def update(self, action):
        
        self.acceleration += action

        # Update velocity
        self.velocity += self.acceleration * SimulationVariables["dt"]

        # Clip velocity to the maximum allowed
        vel = np.linalg.norm(self.velocity)

        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity    

        # Update position based on velocity
        self.position += self.velocity * SimulationVariables["dt"]
        return self.position
    
# Move to other file
class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

class FlockingEnv(gym.Env):
    def __init__(self):
        super(FlockingEnv, self).__init__()
        self.logs = ["Collision_log", "Cohesion_log", "Alignment_log", "Separation_log", "Reward_Total_log"]
        self.episode=0
        self.CTDE=False
        self.current_timestep = 0 
        self.reward_log = []
        self.counter=0
        self.agents = [Agent(position) for position in self.read_agent_locations(self.counter)]

        num_actions_per_agent = 11

        self.action_space = spaces.MultiDiscrete([num_actions_per_agent, num_actions_per_agent] * len(self.agents))
        # self.action_space = np.array(spaces.Discrete(11) * len(self.agents), dtype=np.float32)

        # Positions
        min_obs = np.array([[-np.inf, -np.inf]] * len(self.agents), dtype=np.float32)
        max_obs = np.array([[np.inf, np.inf]] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
      
    def step(self, actions):
        # print(actions)
        # print(actions)
        





        #CHECK THIS
        continuous_action_x = actions[::2] * 10 - 50  # x acceleration
        continuous_action_y = actions[1::2] * 10 - 50  # y acceleration

        # Combine x and y accelerations into a single array
        continuous_action = np.column_stack((continuous_action_x, continuous_action_y))
        







        
        self.current_timestep+=1
        reward=0
        done=False
        info={}
        Collisions={}
        observations = self.simulate_agents(continuous_action)

        alignment_reward, cohesion_reward, separation_reward, collision_reward, Collisions=self.calculate_reward()

        info["Collisions"]=Collisions
        info["CohesionReward"]=cohesion_reward
        info["AlignmentReward"]=alignment_reward
        info["CollisionReward"]=collision_reward         # Always negative
        info["SeparationReward"]=separation_reward

        #Do I need to change

        reward = info["SeparationReward"] + info["CollisionReward"] + info["AlignmentReward"] + info["CohesionReward"]
        
        # if(any(info["Collisions"])==True and self.CTDE==True):
        #     # print("Collision")
        #     done = True
        #     self.counter+=1
        
        if self.CTDE:
            # Make it better
            episode_directory = rf"{Files['Flocking']}\\Testing\\Rewards\\Components\\Episode{self.episode}"
            os.makedirs(episode_directory, exist_ok=True)

            log_path = os.path.join(episode_directory, "Reward_Total_log.json")

            with open(log_path, 'a') as f:
                json.dump(reward, f, indent=4)
                f.write('\n')

        return observations, reward, done, info

        #CHECK RESET

    def reset(self):
        # print("Reset", self.current_timestep)
        self.agents = [Agent(position) for position in self.read_agent_locations(self.counter)]

        for agent in self.agents:
            agent.acceleration=np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)
            agent.velocity=agent.acceleration * SimulationVariables["dt"]
            
        observation=self.get_observation()
        
        return observation
        
    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):
        observations = []
        for i, agent in enumerate(self.agents):
            new_position = agent.update(actions[i])
            observations.append(np.concatenate([new_position]))

        return observations
    
    def check_collision(self, agent):

        # See if correct
        
        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
        return False

    def get_observation(self):

        observation = np.zeros((len(self.agents), 2), dtype=np.float32)
        for i, agent in enumerate(self.agents):
                observation[i] = [
                        agent.position[0],
                        agent.position[1]
                    ]

        return observation.reshape((len(self.agents), 2))

    def calculate_reward(self):
      
        CohesionReward=0
        AlignmentReward=0
        CollisionReward=0
        SeparationReward=0

        Collisions={}

        neighbor_indices = [[] for _ in range(len(self.agents))]  # Initialize empty lists for neighbor indices
        neighbor_velocities = [[] for _ in range(len(self.agents))]

        for idx, _ in enumerate(self.agents):
            Collisions[idx] = []

        for idx, agent in enumerate(self.agents): 
            neighbor_indices, neighbor_velocities=self.get_closest_neighbors(agent)
            
            # Collision Penalty
            tmpVal, collisions_agent=self.CollisionReward(agent, neighbor_indices)
            CollisionReward+=tmpVal

            # Alignment Penalty
            AlignmentReward+=self.calculate_AlignmentReward(agent, neighbor_velocities)
            
            #Separation Reward
            ####################################
            ####################################
            ####################################Check this
            SeparationReward+=self.calculate_SeparationReward(agent, neighbor_indices)
            # print(SeparationReward)
            # Cohesion Reward
            CohesionReward+=self.calculate_CohesionReward(agent, neighbor_indices)

            Collisions[idx].append(collisions_agent)
            Rewards={
            "AlignmentReward": AlignmentReward, "CohesionReward" : CohesionReward, "SeparationReward": SeparationReward, "CollisionReward": CollisionReward,
            }            
            if(self.CTDE==True):
                Log(self.episode, Rewards)

        return AlignmentReward, CohesionReward, SeparationReward, CollisionReward, Collisions
     

    def get_closest_neighbors(self, agent):
        neighbor_indices = []
        neighbor_velocities = []
        
        for i, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)

                if(self.CTDE==True):
                    # print("CTDE")
                    if distance < SimulationVariables["NeighborhoodRadius"]:
                        neighbor_indices.append(i)
                        neighbor_velocities.append(other.velocity)

                else:
                    neighbor_indices.append(i)
                    neighbor_velocities.append(other.velocity)

        return neighbor_indices, neighbor_velocities         

    def CollisionReward(self, agent, neighbor_indices):
        #Every agents' total collision penalty every timestep
        Collision = False
        CollisionReward = 0
        distances=[]

        # Check neighbor indices
        neighbor_agents = [self.agents[idx] for idx in neighbor_indices]
        
        for n_agent in neighbor_agents:             #Use neighborhood or all
            distances.append(np.linalg.norm(agent.position - n_agent.position))

        for distance in distances:
            if distance < SimulationVariables["SafetyRadius"]:            
                CollisionReward -= 100
                Collision = True

        return CollisionReward, Collision
    
    def calculate_CohesionReward(self, agent, neighbor_indices):

        CohesionReward = 0

        if len(neighbor_indices) > 0:
        # Find why working
            center_of_mass = np.mean(neighbor_indices, axis=0)
            desired_position = center_of_mass - agent.position
            distance = np.linalg.norm(desired_position)
            if SimulationVariables["SafetyRadius"] < distance <= SimulationVariables["NeighborhoodRadius"]:
                CohesionReward = 1
            elif distance > SimulationVariables["NeighborhoodRadius"]:
                CohesionReward = -(1 * (distance - SimulationVariables["NeighborhoodRadius"]))
            
            else:
                CohesionReward = -1

        #   cohesion_reward = max(0, 1 - (distance - min_distance) / min_distance)

        # # Penalize agents for going beyond the boundary
        # if distance > SimulationVariables["NeighborhoodRadius"]:
        #     cohesion_reward -= (distance - max_boundary_distance) / max_boundary_distance

        return CohesionReward

    def calculate_SeparationReward(self, agent, neighbor_indices):
        SeparationReward=0

        #Check neighbor indices
        if len(neighbor_indices) > 0:
            for neighbor_position in neighbor_indices:
                # print(neighbor_position)
                relative_position = agent.position - neighbor_position
                distance = np.linalg.norm(relative_position)

                if distance >= SimulationVariables["SafetyRadius"]:
                    SeparationReward += 1
                elif (distance < SimulationVariables["SafetyRadius"]):
                    SeparationReward += -(1 * (distance - SimulationVariables["SafetyRadius"]))
        else:
            SeparationReward = -1
            
        return SeparationReward
    
    #FIX THIS
    def calculate_AlignmentReward(self, agent, neighbor_velocities):

        AlignmentReward = 0

        if (len(neighbor_velocities) > 0):
            desired_direction = self.alignment_calculation(agent, neighbor_velocities)
            orientation_diff = np.arctan2(desired_direction [1], desired_direction [0]) 
            - np.arctan2(agent.velocity[1], agent.velocity[0])

            if orientation_diff > np.pi:
                orientation_diff -= 2 * np.pi
            elif orientation_diff < 0:
                orientation_diff += 2 * np.pi

            AlignmentReward = 1 - np.abs(orientation_diff)

        return AlignmentReward 
    
    def alignment_calculation(self, agent, neighbor_velocities):
        #See if change needed
        if len(neighbor_velocities) > 0:
            average_velocity = np.mean(neighbor_velocities, axis=0)
            desired_orientation = average_velocity - agent.velocity
            return desired_orientation
        else:
            return np.zeros(2)
    
    def read_agent_locations(self, counter):
        File = rf"{Results['InitPositions']}"+str(counter)+"\config.json"
        print(File)
        with open(File, "r") as f:
            data = json.load(f)
        return data


def Log( episode, Rewards):
    episode_directory = os.path.join(Files['Flocking'], 'Testing', 'Rewards', 'Components', f"Episode_{episode}")
    os.makedirs(episode_directory, exist_ok=True)

    for reward_type, reward_value in Rewards.items():
        log_path = os.path.join(episode_directory, f"{str(reward_type)}_log.json")
        with open(log_path, 'a') as f:
            json.dump(reward_value, f, indent=4)
            f.write('\n')

def delete_files(): 
    Paths = ["Results\Flocking\Testing\Dynamics\Accelerations", "Results\Flocking\Testing\Dynamics\Velocities", "Results\Flocking\Testing\Rewards\Other"]

    for episode in range(0, 9):
        for file_path in env.logs:
            # Change this
            file_path = rf"{Files['Flocking']}\\Testing\\Rewards\\Components\\{episode}" + file_path + ".json"
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    for episode in range(0, 9):
        for Path in Paths:
            file_path = rf"{Files['Flocking']}" + Path + f"Episode{episode}" + ".json"
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    for episode in range(0, 9):
        for Path in Paths:
            file_path = rf"{Files['Flocking']}\\Testing\\Rewards\\Components\\Episode{episode}\\Reward_Total_log.json"
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")           

# Check if valid
if os.path.exists(Results["Rewards"]):
    os.remove(Results["Rewards"])
    print(f"File {Results['Rewards']} has been deleted.")

# Add directory
# ent_coef=0.5##
# callback = HalveLearningRateCallback()



env = DummyVecEnv([lambda: FlockingEnv()])


n_steps_per_update = 10000
n_updates = SimulationVariables["LearningTimeSteps"] // n_steps_per_update

model = PPO("MlpPolicy", env, tensorboard_log="./ppo_Agents_tensorboard/", verbose=1)
print("Default n_epochs:", model.n_epochs)
model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"])  # Adjust the multiplier

model.save(rf"{Files['Flocking']}\\Models\\Flocking")
env.close()



env = FlockingEnv()
model = PPO.load(rf"{Files['Flocking']}\\Models\\Flocking")


delete_files()


positions_directory = rf"{Files['Flocking']}/Testing/Positions/"
velocities_directory = rf"{Files['Flocking']}/Testing/Dynamics/Velocities/"
accelerations_directory = rf"{Files['Flocking']}/Testing/Dynamics/Accelerations/"

os.makedirs(positions_directory, exist_ok=True)
os.makedirs(velocities_directory, exist_ok=True)
os.makedirs(accelerations_directory, exist_ok=True)




for episode in range(0, SimulationVariables['Episodes']):
    env.episode=episode
    print("Episode:", episode)
    env.CTDE = True
    obs=env.reset()
    done=False
    reward_episode=[]
    agent_positions = {i: {'positions': [], 'velocities_magnitude': [], 'accelerations_magnitude': []} for i in range(len(env.agents))}
    timestep=0 

    while timestep <= SimulationVariables["EvalTimeSteps"] and not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        reward_episode.append(round(reward, 2))
        
        for i, agent in enumerate(env.agents):
            agent_positions[i]['positions'].append(agent.position.tolist())
            agent_positions[i]['velocities_magnitude'].append(np.linalg.norm(agent.velocity))
            agent_positions[i]['accelerations_magnitude'].append(np.linalg.norm(agent.acceleration))

        timestep += 1

    with open(rf"{Files['Flocking']}\\Testing\\Rewards\\Other\\Episode_{episode}.json", 'w') as f:
        json.dump(reward_episode, f, indent=4)

    with open(f"{positions_directory}Episode_{episode}.json", 'w') as f:
        json.dump(agent_positions, f, indent=4)

    with open(rf"{velocities_directory}Episode_{episode}.json", 'w') as f:
        velocities_data = {i: {'velocities_magnitude': agent_positions[i]['velocities_magnitude']} for i in range(len(env.agents))}
        json.dump(velocities_data, f, indent=4)

    with open(rf"{accelerations_directory}Episode_{episode}.json", 'w') as f:
        accelerations_data = {i: {'accelerations_magnitude': agent_positions[i]['accelerations_magnitude']} for i in range(len(env.agents))}
        json.dump(accelerations_data, f, indent=4)
    print("Reward:", sum(reward_episode))

env.close()
# CTDE
