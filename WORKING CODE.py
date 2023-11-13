import json
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Params import *

class Agent:
   
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityInit"], SimulationVariables["VelocityInit"], size=2), 2)
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    def update(self, action):
        self.velocity += action

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
        self.agents = [Agent(position) for position in read_agent_locations()]

        for agent in self.agents:
            agent.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityInit"], SimulationVariables["VelocityInit"], size=2), 2)

        min_action = np.array([[-5, -5]] * len(self.agents), dtype=np.float32)
        max_action = np.array([[5, 5]] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        min_obs = np.array([[-np.inf, -np.inf]] * len(self.agents), dtype=np.float32)
        max_obs = np.array([[np.inf, np.inf]] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        
    def step(self, actions):
        info={}
        reward = 0
        done = False 
    
        observations=self.simulate_agents(actions)

        reward, cohesion_reward, separation_reward, collision_penalty = self.calculate_reward()
        info["Timestep"]=self.current_timestep
        info["Reward"]=reward
        info["CohesionReward"]=cohesion_reward
        info["SeparationReward"]=separation_reward
        info["CollisionPenalty"]=collision_penalty

        if(collision_penalty==1000):    #Exit on collision
            print("Agent Collided")
            done=True

         # Implement the termination condition based on your criteria

        with open('Results.txt', 'a+') as f:
            f.write("-------------------------\n")
            f.write(f"{self.current_timestep}\n")
            f.write(f"{observations}\n")
            f.write(f"{info}\n")
            f.write("-------------------------\n")
        self.current_timestep+=1
        
        return observations, reward, done, info
            
    def reset(self):
        self.current_timestep = 0 
        agent_positions = read_agent_locations()
        self.agents = [Agent(position) for position in agent_positions]
        for agent in self.agents:
            agent.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityInit"], SimulationVariables["VelocityInit"], size=2), 2)
        observation = self.get_observation()
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
                if isinstance(agent, Agent):
                    observation[i] = [
                        agent.position[0],
                        agent.position[1],
                    ]
                elif isinstance(agent, tuple):
                    observation[i] = [
                        agent[0],
                        agent[1],
                    ]

            return observation.reshape((len(self.agents), 2))
    
    # def calculate_reward(self):
    #     reward=0

    #     for agent in self.agents:
    #         #############Fix it##############
    #         neighbor_indices = self.get_closest_neighbors(agent, SimulationVariables["NeighborhoodRadius"])

    #         # Get arrays for neighbor positions and velocities
    #         neighbor_positions = np.array([agent.position for agent in self.agents])[neighbor_indices]
    #         neighbor_velocities = np.array([agent.velocity for agent in self.agents])[neighbor_indices]

            
    #         # Reward for cohesion
    #         cohesion_reward = 0.0
    #         if len(neighbor_indices) > 0:
    #             center_of_mass = np.mean(neighbor_positions, axis=0)
    #             cohesion_reward = np.linalg.norm(center_of_mass - agent.position)

    #         # Reward for separation
    #         separation_reward = 0.0
    #         # for neighbor_index in neighbor_indices:
    #         #     neighbor_position = self.agents[neighbor_index].position
    #         #     relative_position = self.position - neighbor_position
    #         #     distance = np.linalg.norm(relative_position)
    #         #     separation_reward += distance

    #         separation_reward = np.sum(
    #         np.linalg.norm(agent.position - neighbor_positions, axis=1)
    #         )

    #         # Reward for velocity matching
    #         velocity_matching_reward = 0.0
    #         average_velocity = np.mean(neighbor_velocities, axis=0)
    #         velocity_matching_reward = np.linalg.norm(average_velocity - agent.velocity)

    #         collision_penalty=0.0                
    #         if(self.check_collision(agent)==True):
    #             collision_penalty -= 1000
    #             # print(self.current_timestep)
    #         else:
    #             collision_penalty -= 0
            

    #     reward = (
    #         cohesion_reward +
    #         separation_reward +
    #         collision_penalty +
    #         velocity_matching_reward
    #     )

    #     return reward, cohesion_reward, separation_reward, collision_penalty
    

    # Better reward function
    def calculate_reward(self):
        total_reward = 0
        cohesion_reward = 0
        separation_reward = 0
        collision_penalty = 0
        velocity_matching_reward = 0

        for agent in self.agents:
            for other in self.agents:
                if agent != other:
                    distance = np.linalg.norm(agent.position - other.position)

                    # if distance <= 50:
                    #     cohesion_reward += 5

                    if distance < SimulationVariables["NeighborhoodRadius"]:
                        separation_reward -= 100

                    velocity_matching_reward += np.linalg.norm(np.mean([other.velocity for other in self.agents], axis=0) - agent.velocity)

                   
                    if distance < SimulationVariables["SafetyRadius"]:
                        collision_penalty -= 1000

    
        total_reward = separation_reward + velocity_matching_reward + collision_penalty

        # print(f"Total: {total_reward}, Cohesion: {cohesion_reward}, Separation: {separation_reward}, Velocity Matching: {velocity_matching_reward}, Collision: {collision_penalty}")

        return total_reward, cohesion_reward, separation_reward, collision_penalty

    def get_closest_neighbors(self, agent, max_distance):
        neighbor_indices = []

        for i, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)
                if distance < max_distance:
                    neighbor_indices.append(i)

        return neighbor_indices

def read_agent_locations():
    with open(rf"{Results['InitPositions']}\config.json", "r") as f:
        data = json.load(f)
        print("File loaded")
    return data


env = DummyVecEnv([lambda: FlockingEnv()])

model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log = "./ppo_Agents_tensorboard/")
model.learn(total_timesteps = SimulationVariables["TimeSteps"]*100)
model.save("PPO")

env.close()

# *2000

env = FlockingEnv()
model = PPO.load("PPO")
# model.set_random_seed(random_seed)

obs = env.reset()


#Add episodes and reward for every episode output

done = False
total_reward = 0
positions_dict = {i: [] for i in range(len(env.agents))}  
timestep=0

while ((timestep<=1000) or not done):
    # actions = np.random.uniform(-5, 5, size=(len(env.agents), 2))
    action, _ = model.predict(obs)

    obs, reward, done, info = env.step(action)
    total_reward += reward

    for i, agent in enumerate(env.agents):
        positions_dict[i].append(agent.position.tolist())

    if timestep>= 2000:
        done = True
    timestep+=1
    

with open('agent_positionsTest.json', 'w') as f:
    json.dump(positions_dict, f, indent=4)

print("Total Accumulated Reward:", total_reward)

env.close()
