import os
import json
import gym
from gym import spaces
from Params import *
import math
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import freeze_support
from pathlib import Path

class Agent:
   
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

        self.acceleration = np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)
        self.velocity = np.zeros(2)

        self.max_velocity = SimulationVariables["VelocityUpperLimit"]
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]


    def update(self, action):
        self.acceleration += action
        self.velocity += self.acceleration * SimulationVariables["dt"]

        vel = np.linalg.norm(self.velocity)
        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity

        self.position += self.velocity * SimulationVariables["dt"]

        return self.position
    
class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

class FlockingEnv(gym.Env):
    def __init__(self):
        super(FlockingEnv, self).__init__()
        self.current_timestep = 0 
        self.reward_log = []

        self.agents = [Agent(position) for position in read_agent_locations()[:SimulationVariables["SimAgents"]]]

        for agent in self.agents:
            agent.velocity = agent.acceleration * SimulationVariables["dt"]

        min_action = np.array([[-SimulationVariables["VelocityUpperLimit"], -SimulationVariables["VelocityUpperLimit"]]] * len(self.agents), dtype=np.float32)
        max_action = np.array([[SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"]]] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        min_obs = np.array([[-np.inf, -np.inf, -SimulationVariables["AccelerationUpperLimit"], -SimulationVariables["AccelerationUpperLimit"]]] * len(self.agents), dtype=np.float32)
        max_obs = np.array([[np.inf, np.inf, SimulationVariables["AccelerationUpperLimit"], SimulationVariables["AccelerationUpperLimit"]]] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

        
    def step(self, actions):
        info={}
        reward = 0
        done = False 
        Collisions = {}

        observations=self.simulate_agents(actions)

        alignment_reward, collision_penalty, cohesion_reward, Collisions  = self.calculate_reward()
        info["Collisions"]=Collisions

        # info["CohesionReward"]=cohesion_reward
        info["AlignmentReward"]=alignment_reward
        # # # info["VelocityMatchingReward"]=velocity_matching_reward
        # info["CollisionPenalty"]=collision_penalty
        # + info["CohesionReward"] + info["CollisionPenalty"]

        reward =  info["AlignmentReward"]
        # if any(info["Collisions"]):
        #     # print("Collision detected! Exiting the loop.")
        #     done = True
        # print("collision", info["Collisions"])

        with open('Rewards.txt', 'a+') as f:
            f.write("-------------------------\n")
            f.write(f"{reward}\n")
            f.write("-------------------------\n")


        with open('Results.txt', 'a+') as f:
            f.write("-------------------------\n")
            f.write(f"{self.current_timestep}\n")
            f.write(f"{observations}\n")
            f.write(f"{info}\n")
            f.write(f"{reward}\n")
            f.write("-------------------------\n")
        
        self.current_timestep+=1

        return observations, reward, done, info

        #CHECK RESET

    def reset(self):
        self.current_timestep = 0 
        self.agents = [Agent(position) for position in read_agent_locations()[:SimulationVariables["SimAgents"]]]

        for agent in self.agents:
            agent.acceleration = np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)
            agent.velocity = np.zeros(2)
        observation = self.get_observation()
        return observation
        
    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):
        observations = []
        for i, agent in enumerate(self.agents):
            new_position = agent.update(actions[i])
            observations.append(np.concatenate([new_position, agent.velocity]))

        return observations
    
    def check_collision(self, agent):
        
        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
        return False

    def get_observation(self):
            observation = np.zeros((len(self.agents), 4), dtype=np.float32)

            for i, agent in enumerate(self.agents):
                observation[i] = [
                    agent.position[0],
                    agent.position[1],
                    agent.velocity[0],
                    agent.velocity[1]
                ]

            return observation.reshape((len(self.agents), 4))



    def calculate_reward(self):
        CohesionReward = 0
        AlignmentReward = 0
       
        CollisionPenalty = 0
        neighbor_indices = [[] for _ in range(len(self.agents))]  # Initialize empty lists for neighbor indices
        neighbor_velocities = [[] for _ in range(len(self.agents))]
        

        Collisions = {}
        for idx, _ in enumerate(self.agents):
            Collisions[idx] = []

        # Finding neighbors information
        for idx, agent in enumerate(self.agents):
            neighbor_indices, neighbor_velocities = self.get_closest_neighbors(agent, SimulationVariables["NeighborhoodRadius"])
            # CollisionPenalty
            CollisionPenalty , collisions_agent = self.CollisionPenalty(agent, neighbor_indices)

            # CohesionReward = self.calculate_CohesionReward(agent, neighbor_indices)
            AlignmentReward = self.calculate_AlignmentReward(agent, neighbor_velocities)
            
            Collisions[idx].append(collisions_agent)
            CohesionReward = self.calculate_CohesionReward(agent, neighbor_velocities)

        return AlignmentReward, CollisionPenalty, CohesionReward, Collisions
    
    def alignment_calculation(self, agent, neighbor_velocities):
        if len(neighbor_velocities) > 0:
            average_velocity = np.mean(neighbor_velocities, axis=0)
            desired_orientation = average_velocity - agent.velocity
            return desired_orientation
        else:
            return np.zeros(2)
            
    def get_closest_neighbors(self, agent, max_distance):
        #Fix this
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

        neighbor_agents = [self.agents[idx] for idx in neighbor_indices]
        for n_agent in neighbor_agents:  
            distance = np.linalg.norm(agent.position - n_agent.position)
            distances.append(distance)

        for distance in distances:
            if distance < SimulationVariables["SafetyRadius"]:            
                CollisionPenalty -= 1000
                Collision = True

        return CollisionPenalty, Collision
    
    def calculate_CohesionReward(self, agent, neighbor_indices):
        # Every agent's total cohesion penalty every timestep
        CohesionReward = 0
        distances = []

        # Find why working
        for n_position in neighbor_indices:
            distance = np.linalg.norm(agent.position - n_position)
            distances.append(distance)

        for distance in distances:                
            if  distance <= SimulationVariables["NeighborhoodRadius"]:
                CohesionReward += 500 
            elif (distance > SimulationVariables["NeighborhoodRadius"]):
                CohesionReward -= 1000 * (distance - SimulationVariables["NeighborhoodRadius"])  # Incrementally decrease rewards
                
        return CohesionReward
    
    def calculate_AlignmentReward(self, agent, neighbor_velocities):
        AlignmentReward = 0
        min_reward = -200
        max_reward = 200

        desired_velocity = self.alignment_calculation(agent, neighbor_velocities)
        orientation_diff = np.arctan2(desired_velocity[1], desired_velocity[0]) - np.arctan2(agent.velocity[1], agent.velocity[0])

        # Normalize orientation_diff between [0, π]
        orientation_diff = (orientation_diff + np.pi) % (2 * np.pi) - np.pi
        if orientation_diff < 0:
            orientation_diff += 2 * np.pi


        AlignmentReward = ((max_reward - min_reward) / np.pi) * orientation_diff + max_reward

        return AlignmentReward
    
class HalveLearningRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(HalveLearningRateCallback, self).__init__(verbose)
        self.learning_rate_divisor = 2
        self.halving_step = 2048  # Timestep to halve the learning rate
        self.halved = False
        
    def _on_step(self):
        if self.num_timesteps > self.halving_step:
            self.model.policy.optimizer.param_groups[0]['lr'] *= 0.5  # Halve the learning rate
            self.halved = True
        return True  # Continue training

def read_agent_locations():
    with open(rf"{Results['InitPositions']}\config.json", "r") as f:
        data = json.load(f)
    return data

def make_env():
    return FlockingEnv()

def epochLog():
    return

if os.path.exists(Results["Rewards"]):
    os.remove(Results["Rewards"])
    print(f"File {Results['Rewards']} has been deleted.")

Name = rf'Agents_Allignment{SimulationVariables["SimAgents"]}_PPO_{SimulationVariables["LearningTimeSteps"]}'
env = DummyVecEnv([lambda: FlockingEnv()])
model = PPO("MlpPolicy", env,  tensorboard_log="./ppo_Agents_tensorboard/", verbose=1)
model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"])  # Adjust the multiplier

# Save the model
model.save(Name)
env.close()

# Load the model
env = FlockingEnv()
model = PPO.load(Name)

# Run for 10 episodes
for episode in range(1, RLVariables['Episodes']):

    obs = env.reset()
    done = False
    
    reward = 0
    positions_dict = {i: [] for i in range(len(env.agents))}
    timestep = 0
    reward_log=[]
    print("Episode", episode)

    # Completion condition
    while((timestep <= SimulationVariables["EvalTimeSteps"]) and (not done)):
      
        action, state = model.predict(obs)
        
        obs, reward, done, info = env.step(action) 
        reward_log.append(reward)
        print(reward)
        for i, agent in enumerate(env.agents):
            positions_dict[i].append(agent.position.tolist())
        with open(rf'{Results["EpRewards"]}_Allignment_{episode}.json', 'w') as f:
               json.dump(reward_log, f, indent=4)
        timestep = timestep + 1
          
    with open(rf'agent_positionsTestAllignment_{episode}.json', 'w') as f:        
        json.dump(positions_dict, f, indent=4)


env.close()
