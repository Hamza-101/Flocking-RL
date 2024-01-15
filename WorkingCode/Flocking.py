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

# Model Namw
NAME = rf'Flocking'

# Move to other file
class Agent:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        
        self.acceleration = np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)                # Acceleration Initialized to 0
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        
        self.velocity = self.acceleration * SimulationVariables["dt"]       
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    # See if can be moved
    def update(self, action):
        
        self.acceleration += action
        self.velocity += self.acceleration * SimulationVariables["dt"]

        vel = np.linalg.norm(self.velocity)

        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity

        self.position += self.velocity * SimulationVariables["dt"]

        return self.position
    
# Move to other file
class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

class FlockingEnv(gym.Env):
    def __init__(self):
        super(FlockingEnv, self).__init__()
        self.current_timestep = 0 
        self.reward_log = []

        self.agents = [Agent(position) for position in read_agent_locations()]

        ###Randomize it 
        # for agent in self.agents:
        #     agent.velocity = agent.velocity

        # Accelerations
        min_action = np.array([[-SimulationVariables["AccelerationUpperLimit"], -SimulationVariables["AccelerationUpperLimit"]]] * len(self.agents), dtype=np.float32) # AccelerationUpperLimit inverse
        max_action = np.array([[SimulationVariables["AccelerationUpperLimit"], SimulationVariables["AccelerationUpperLimit"]]] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        
        # Positions
        min_obs = np.array([[-np.inf, -np.inf]] * len(self.agents), dtype=np.float32)
        max_obs = np.array([[np.inf, np.inf]] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
      
    def step(self, actions):
        
        reward=0
        done=False
        info={}
        Collisions={}

        observations=self.simulate_agents(actions)

        alignment_reward, cohesion_reward, separation_reward, collision_penalty, Collisions=self.calculate_reward()

        info["Collisions"]=Collisions
        info["CohesionReward"]=cohesion_reward
        info["AlignmentReward"]=alignment_reward
        info["CollisionPenalty"]=collision_penalty          # Always negative
        info["SeparationReward"]=separation_reward

        reward = info["AlignmentReward"] + info["CohesionReward"] + info["CollisionPenalty"] + info["SeparationReward"]

        
        self.current_timestep+=1

        return observations, reward, done, info

        #CHECK RESET

    def reset(self):
        self.current_timestep = 0 
        self.agents = [Agent(position) for position in read_agent_locations()]

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
        Collisions={}

        CohesionReward=0
        AlignmentReward=0
        CollisionPenalty=0
        SeparationReward=0

        neighbor_indices = [[] for _ in range(len(self.agents))]  # Initialize empty lists for neighbor indices
        neighbor_velocities = [[] for _ in range(len(self.agents))]

        for idx, _ in enumerate(self.agents):
            Collisions[idx] = []

        for idx, agent in enumerate(self.agents): 
            neighbor_indices, neighbor_velocities=self.get_all_neighbor_positions(agent)
            
            # Collision Penalty
            tmpVal, collisions_agent=self.CollisionPenalty(agent, neighbor_indices)
            CollisionPenalty+=tmpVal

            # Alignment Penalty
            AlignmentReward+=self.calculate_AlignmentReward(agent, neighbor_velocities)
            
            #Separation Reward
            SeparationReward+=self.calculate_SeparationReward(agent, neighbor_indices)

            # Cohesion Reward
            CohesionReward+=self.calculate_CohesionReward(agent, neighbor_indices)

            Collisions[idx].append(collisions_agent)

        return AlignmentReward, CohesionReward, SeparationReward, CollisionPenalty, Collisions

    def get_all_neighbor_positions(self, agent):
        neighbor_indices = []
        neighbor_velocities = []

        for i, other in enumerate(self.agents):
            if agent != other:
                neighbor_indices.append(i)
                neighbor_velocities.append(other.velocity)

        return neighbor_indices, neighbor_velocities         

    def get_closest_neighbors(self, agent, max_distance):
        neighbor_indices = []
        neighbor_velocities = []

        for i, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)
                if distance < max_distance:
                    neighbor_indices.append(i)
                    neighbor_velocities.append(other.velocity)

        return neighbor_indices, neighbor_velocities         

    def CollisionPenalty(self, agent, neighbor_indices):
        #Every agents' total collision penalty every timestep
        Collision = False
        CollisionPenalty = 0
        distances=[]

        # Check neighbor indices
        neighbor_agents = [self.agents[idx] for idx in neighbor_indices]
        
        for n_agent in neighbor_agents:             #Use neighborhood or all
            distances.append(np.linalg.norm(agent.position - n_agent.position))

        for distance in distances:
            if distance < SimulationVariables["SafetyRadius"]:            
                CollisionPenalty -= 1000
                Collision = True

        return CollisionPenalty, Collision
    
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
    
####################Call Back Learning rate#####################

# class HalveLearningRateCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(HalveLearningRateCallback, self).__init__(verbose)
#         self.learning_rate_divisor = 2
#         self.halving_step = 2048  # Timestep to halve the learning rate
#         self.halved = False
        
#     def _on_step(self):
#         if self.num_timesteps > self.halving_step:
#             self.model.policy.optimizer.param_groups[0]['lr'] *= 0.5  # Halve the learning rate
#             self.halved = True
#         return True  # Continue training

def read_agent_locations():
    with open(rf"{Results['InitPositions']}\config.json", "r") as f:
        data = json.load(f)
    return data

if os.path.exists(Results["Rewards"]):
    os.remove(Results["Rewards"])
    print(f"File {Results['Rewards']} has been deleted.")

# Add directory
# ent_coef=0.5##
# callback = HalveLearningRateCallback()


env = DummyVecEnv([lambda: FlockingEnv()])
model = PPO("MlpPolicy", env, tensorboard_log="./ppo_Agents_tensorboard/", verbose=1)

model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"])  # Adjust the multiplier

model.save(NAME)
env.close()



env = FlockingEnv()
model = PPO.load(NAME)

for episode in range(1, RLVariables['Episodes']):

    obs = env.reset()
    done = False
    reward = 0
    positions_dict = {i: [] for i in range(len(env.agents))}
    timestep = 0
    reward_log=[]

    while((timestep <= SimulationVariables["EvalTimeSteps"]) and (not done)):
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action) 
        reward_log.append(reward)

        print(reward)

        for i, agent in enumerate(env.agents):
            positions_dict[i].append(agent.position.tolist())
        # with open(rf'RewardTesting_Episode{episode}.json', 'w') as f:
        #        json.dump(reward_log, f, indent=4)
        timestep = timestep + 1
      
        
    
    with open(rf'Flocking{episode}.json', 'w') as f:        #Add to params file
        json.dump(positions_dict, f, indent=4)

env.close()