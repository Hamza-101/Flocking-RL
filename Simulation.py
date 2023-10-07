import json
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from Params import *

class Agent:

    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([-(SimulationVariables["VelocityInit"]), SimulationVariables["VelocityInit"]], dtype=float)
        self.acceleration = np.array([-(SimulationVariables["AccelerationInit"]), SimulationVariables["AccelerationInit"]], dtype=float)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    def update(self, agents, dt):
        positions = np.array([agent.position for agent in agents])
        velocities = np.array([agent.velocity for agent in agents])

        neighbor_indices = self.get_closest_neighbors(agents, SimulationVariables["NeighborhoodRadius"])
        self.flock(neighbor_indices, positions, velocities)
        self.velocity += self.acceleration * dt

        vel = np.linalg.norm(self.velocity)
        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity

        self.position += self.velocity * dt

    def flock(self, neighbor_indices, positions, velocities):
        alignment = self.align(neighbor_indices, velocities)
        cohesion = self.cohere(neighbor_indices, positions)
        separation = self.separate(neighbor_indices, positions)

        total_force = (
            ((ReynoldsVariables["w_alignment"]) * alignment) +
            ((ReynoldsVariables["w_cohesion"]) * cohesion) +
            ((ReynoldsVariables["w_separation"]) * separation)
        )

        self.acceleration = np.clip(total_force, 0, self.max_acceleration)

    def align(self, neighbor_indices, velocities):
        if len(neighbor_indices) > 0:
            neighbor_velocities = velocities[neighbor_indices]
            average_velocity = np.mean(neighbor_velocities, axis=0)
            desired_velocity = average_velocity - self.velocity
            return desired_velocity
        else:
            return np.zeros(2)

    def cohere(self, neighbor_indices, positions):
        if len(neighbor_indices) > 0:
            neighbor_positions = positions[neighbor_indices]
            center_of_mass = np.mean(neighbor_positions, axis=0)
            desired_direction = center_of_mass - self.position
            return desired_direction
        else:
            return np.zeros(2)

    def separate(self, neighbor_indices, positions):
        if len(neighbor_indices) > 0:
            neighbor_positions = positions[neighbor_indices]
            separation_force = np.zeros(2)

            for neighbor_position in neighbor_positions:
                relative_position = self.position - neighbor_position
                distance = np.linalg.norm(relative_position)

                if distance > 0:
                    separation_force += (relative_position / (distance * distance))

            return separation_force
        else:
            return np.zeros(2)

    def get_closest_neighbors(self, agents, max_distance):
        neighbor_indices = []

        for i, agent in enumerate(agents):
            if agent != self:
                distance = np.linalg.norm(agent.position - self.position)
                if distance < max_distance:
                    neighbor_indices.append(i)

        return neighbor_indices

class Encoder(json.JSONEncoder):
        def default(self, obj):
            return json.JSONEncoder.default(self, obj)
     
class FlockingEnv(gym.Env):
    def __init__(self, Agents):
        super(FlockingEnv, self).__init__()
        self.agents=Agents
        self.NumAgents=len(Agents)
        self.CollisionThreshold = 0
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.NumAgents * 4,), dtype=np.float32)
        self.RewardLimits = (-float('inf'), float('inf'))
        self.reset()

    def step(self, action):
        total_reward = 0
        for Agent in self.agents:
            total_reward += self.calculate_reward(Agent)
        
        done = False  # Implement your termination condition if needed
        info = {}
        observation = self.get_observation()
    
        return observation, total_reward, done, info
        
    def reset(self): 
        agent_locations = read_agent_locations()
        for agent, position in zip(self.agents, agent_locations):
            agent.position = np.array(position, dtype=float)

        observation = self.get_observation()
        observation = np.reshape(observation, (-1,))
        
        return observation


    def get_observation(self):
        observation = np.zeros((len(self.agents), 4), dtype=np.float32)

        for i, agent in enumerate(self.agents):
            observation[i] = [
                agent.position[0],
                agent.position[1],
                agent.velocity[0] / agent.max_velocity,
                agent.velocity[1] / agent.max_velocity
            ]
        return observation.flatten()

    def calculate_reward(self, agent):
        cohesion_reward = 0
        separation_reward = 0
        collision_penalty = 0

        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                
                # Cohesion Reward
                if distance <= 50:
                    cohesion_reward += 1

                # Separation Reward
                separation_distance = 20
                if distance < separation_distance:
                    separation_reward -= 1

                # Collision Penalty
                if distance < self.CollisionThreshold:
                    collision_penalty -= 10

        total_reward = cohesion_reward + separation_reward + collision_penalty

        return total_reward

def read_agent_locations():
    with open(rf"{Results['InitPositions']}\config.json", "r") as f:
        data = json.load(f)
        print("File loaded")
    return data

def simulate_agents(agents, num_timesteps=1000):
    agent_data = {id(agent): [] for agent in agents}

    for timestep in range(num_timesteps):
        for agent in agents:
            agent_id = id(agent)
            agent.update(agents, SimulationVariables["dt"])

            agent_data_full = {
                "x": round(agent.position[0], 2),
                "y": round(agent.position[1], 2)
            }

            agent_data[agent_id].append(agent_data_full)

    print("Simulation Complete")
    save_agent_data(agent_data)

def save_agent_data(agent_data):

    try:
        with open(rf"{Results['InitPositions']}\run.json", "r") as f:
            all_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_data = {}

    for agent_id, data_list in agent_data.items():
        if agent_id not in all_data:
            all_data[agent_id] = []
        all_data[agent_id] += data_list

    with open(rf"{Results['InitPositions']}\run.json", "w") as f:
        json.dump(all_data, f, cls = Encoder)

agent_locations = read_agent_locations()
agents = [Agent(position) for position in agent_locations]

env = DummyVecEnv([lambda: FlockingEnv(agents)])

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_Agents_tensorboard/")
model.learn(total_timesteps=SimulationVariables["TimeSteps"])

model.save("PPO")

env.close()