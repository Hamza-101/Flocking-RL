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
        self.positions_history = []

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

        self.positions_history.append(self.position.tolist())

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
    def __init__(self, agents):
        super(FlockingEnv, self).__init__()
        self.Agents = agents
        self.NumAgents = SimulationVariables["SimAgents"]  
        self.CollisionThreshold = 0  # Set collision threshold 
        self.RewardLimits = (-float('inf'), float('inf'))
        self.action_space = spaces.Discrete(8)  
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.NumAgents * 2,), dtype=np.float32)  # Match observation_space to your defined space
        self.reset()

    def reset(self):
        agent_locations = read_agent_locations()
        agents = [Agent(position) for position in agent_locations]
        self.Agents = agents  # Replace the current agent list with the new one
        return self.get_observation()

    def step(self, action):
        reward = 0
        for Agent in self.Agents:
            reward += self.calculate_reward(Agent)
            # Return the next observation, reward, done flag, and additional info
            done = False  # Implement your termination condition if needed
            info = {}
            return self.get_observation(), reward, done, info

    def get_observation(self):
        scale_factor = 1000.0  # Adjust this value based on the scale of your simulation
        state = []
        for agent in self.Agents:
            state.extend([agent.position[0] / scale_factor, agent.position[1] / scale_factor])
        return np.array(state, dtype = np.float32)

    def render(self, mode = 'human'):
        pass

    def close(self):
        # Implement any cleanup logic here
        pass

    def calculate_reward(self, agent):
        cohesion_reward = 0
        separation_reward = 0
        collision_penalty = 0

        for other in self.Agents:
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
    with open(rf"{SimulationVariables['TrainFile']}\config.json", "r") as f:
        data = json.load(f)
        print("File loaded")
    return data

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

    #Recheck this
    # Define a custom callback for monitoring training progress (optional)

def custom_callback(locals_, globals_):
    if locals_['total_timesteps'] % 1000 == 0:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Step: {locals_['total_timesteps']}, Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
    if locals_['total_timesteps'] % 1000 == 0:
        agent_data = {id(agent): agent.positions_history for agent in agents}
        save_agent_data(agent_data, f"training_step_{locals_['total_timesteps']}")

        
agent_locations = read_agent_locations()
agents = [Agent(position) for position in agent_locations]

env = DummyVecEnv([lambda: FlockingEnv(agents)])

model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log = "./ppo_Agents_tensorboard/")
model.learn(total_timesteps = SimulationVariables["TimeSteps"], callback = custom_callback)

model.save("PPO")

env.close()