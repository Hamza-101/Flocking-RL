import os
import json
import gym
from gym import spaces
from Params import *
import math
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import freeze_support
from pathlib import Path
from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3.common.noise import NormalActionNoise

############################################################
############################################################
##############Make Environment more agent based#############
############################################################
############################################################

class EpisodeCounterCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeCounterCallback, self).__init__(verbose)
        self.timestep_counter=0
        
    def _on_step(self) -> bool:
        # This method needs to be implemented, but we don't use it in this callback
        # You can simply return True
        return True

    def _on_rollout_start(self) -> None:
        # Increment the episode counter at the start of each rollout
        # if(self.timestep_counter%100==0):
        #     self.increment_file()
       return

class StoreState(BaseCallback):
    def __init__(self, env, verbose=0):
        super(StoreState, self).__init__(verbose)
        self.env = env
        self.states_on_collision = []

    def _on_step(self) -> bool:
        # Check for collisions
        collision = any(self.env.envs[0].check_collision(agent) for agent in self.env.envs[0].agents)

        # If collision occurs, store the current state
        if collision:
            current_state = self.env.envs[0].get_observation()
            self.states_on_collision.append(current_state)

        return True

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


# n_fall_steps=10: Possibly the number of steps allowed for the drone to fall before the simulation ends.
class Agent:
   
    def __init__(self, position):
        self.position = np.array(position, dtype=float)  
        self.acceleration = np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)                # Acceleration Initialized to 0
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        
        self.velocity = self.acceleration * SimulationVariables["dt"]       
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

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
        self.current_timestep=0 
        self.reward_log=[]
        self.counter=0
        self.agents=[Agent(position) for position in self.read_agent_locations(self.counter)]
        for agent in self.agents:
            agent.velocity = np.zeros(2) ###CHANGE 

        min_action = np.array([[-1, -1]] * len(self.agents), dtype=np.float32)
        max_action = np.array([[1, 1]] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        min_obs = np.array([[-np.inf, -np.inf]] * len(self.agents), dtype=np.float32)
        max_obs = np.array([[np.inf, np.inf]] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        
    def step(self, actions):
        # Increase counter every 2000 timesteps
        # self.current_timestep+=1
        # print(f"Episode Counter: {self.current_timestep}")
        self.current_timestep+=1
        info={}
        reward = 0
        done = False 
        Collisions = {}
        
        observations=self.simulate_agents(actions)

        separation_reward, alignment_reward, cohesion_reward, collision_penalty, Collisions = self.calculate_reward()

        info["Collisions"] = Collisions
        info["CohesionReward"]= cohesion_reward
        info["SeparationReward"] = separation_reward
        info["AlignmentReward"] = alignment_reward
        info["CollisionPenalty"] = collision_penalty
        
       
        reward = info["AlignmentReward"] + info["CohesionReward"] + info["CollisionPenalty"] + info["SeparationReward"]

        # if any(info["Collisions"]):
        #     done = True
          
        #      # If collision occurs, store the current state
        #     print("collision")

        if any(self.current_timestep==100000):
            done = True
          
            self.counter+=1
            print(self.counter)
            env.reset() #lkdsfklsdjf


        # Implement the termination condition based on your criteria
            
            #CHECK
        # self.current_timestep+=1
        # print("Step", self.current_timestep)
        return observations, reward, done, info

        #CHECK RESET

    def reset(self):

        self.agents = [Agent(position) for position in self.read_agent_locations(self.counter)]
        
        for agent in self.agents:
            agent.acceleration=np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)
            agent.velocity=agent.acceleration * SimulationVariables["dt"]
        observation = self.get_observation()
        # print(observation)
        
        # print("Reset", self.current_timestep)
     
        return observation
        
    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):
        observations = []
        for i, agent in enumerate(self.agents):
            new_position = agent.update(actions[i])
            observations.append(new_position)

        return observations
    
    def check_collision(self, agent):
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
                        agent.position[1],
                    ]
             

            return observation.reshape((len(self.agents), 2))

    def calculate_reward(self):
        CohesionReward=0
        AlignmentReward=0
        SeparationReward=0
        CollisionPenalty = 0

        neighbor_indices = [[] for _ in range(len(self.agents))]  # Initialize empty lists for neighbor indices
        neighbor_velocities = [[] for _ in range(len(self.agents))]
        Collisions = {}
        for idx, _ in enumerate(self.agents):
            Collisions[idx] = []

        # Finding neighbors information
        for idx, agent in enumerate(self.agents):
            neighbor_indices, neighbor_velocities = self.get_closest_neighbors(agent, SimulationVariables["NeighborhoodRadius"])

            CollisionPenalty , collisions_agent = self.CollisionPenalty(agent, neighbor_indices)
            CohesionReward += self.calculate_CohesionReward(agent, neighbor_indices)
            SeparationReward += self.calculate_SeparationReward(agent, neighbor_indices)
            AlignmentReward += self.calculate_AlignmentReward(agent, neighbor_velocities)

            Collisions[idx].append(collisions_agent)
               

        return SeparationReward, AlignmentReward, CohesionReward, CollisionPenalty, Collisions
    
    def alignment_calculation(self, agent, neighbor_velocities):
        if len(neighbor_velocities) > 0:
            average_velocity = np.mean(neighbor_velocities, axis=0)
            desired_velocity = average_velocity - agent.velocity
            return desired_velocity
        else:
            return np.zeros(2)
            
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

    def read_agent_locations(self, counter):
        File = rf"{Results['InitPositions']}"+ str(counter) +"\config.json"
        print(File)
        with open(File, "r") as f:
            data = json.load(f)
        return data

    



if os.path.exists(Results["Rewards"]):
    os.remove(Results["Rewards"])
    print(f"File {Results['Rewards']} has been deleted.")

env = DummyVecEnv([lambda: FlockingEnv()])
buffer_size = int(1e6)  # Adjust the buffer size as needed
ErrorHandler = StoreState(env=env)

n_actions=env.action_space.shape[-1]
noise_std=0.1
action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))

stored_states=ErrorHandler.states_on_collision
episode_counter_callback=EpisodeCounterCallback()

print("Stored States on Collision:", stored_states)

model = DDPG("MlpPolicy", env, buffer_size=buffer_size, tensorboard_log="./ddpg_Agents_tensorboard/", verbose=1)

n_steps_per_update = 100000
n_updates = SimulationVariables["LearningTimeSteps"] // n_steps_per_update
print(n_updates)
model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"], n_updates=n_updates)  # Adjust the multiplier
    
    #, callback=ErrorHandler
model.save("DDPG")
env.close()



# # # Load the model
# env = FlockingEnv()
# model = DDPG.load("DDPG")
# # buffer=model.load_replay_buffer("ddpg_replay_buffer")
# print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer")


# # Run for 10 episodes
# for episode in range(1, RLVariables['Episodes']):
    
#     obs = env.reset(episode)
#     done = False
    
#     total_reward = 0
#     positions_dict = {i: [] for i in range(len(env.agents))}
#     timestep = 0
#     reward_log=[]

#     print("Episode", episode)

#     # Completion condition
#     while((timestep <= SimulationVariables["EvalTimeSteps"]) and (not done)):
      
#         action, _states = model.predict(obs)
        
#         next_obs, reward, done, _ = env.step(action) ######### env.step() #Add condition to exit on collision
#         total_reward+=reward

#         reward_log.append(reward)
#         print(reward)

#         for i, agent in enumerate(env.agents):
#             positions_dict[i].append(agent.position.tolist())
#         with open(rf'RewardTesting_Episode{episode}.json', 'w') as f:
#                json.dump(reward_log, f, indent=4)
      
#         timestep = timestep + 1
  
#     # print(reward_log)
    
#     with open(rf'Flocking_new_{episode}.json', 'w') as f:        #Add to params file
#         json.dump(positions_dict, f, indent=4)


# env.close()