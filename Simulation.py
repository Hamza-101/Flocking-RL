import json
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from Params import *

#####################Reynolds vs RL
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

    # def align(self, neighbor_indices, velocities):
    #     if len(neighbor_indices) > 0:
    #         neighbor_velocities = velocities[neighbor_indices]
    #         average_velocity = np.mean(neighbor_velocities, axis=0)
    #         desired_velocity = average_velocity - self.velocity
    #         return desired_velocity
    #     else:
    #         return np.zeros(2)

    # def cohere(self, neighbor_indices, positions):
    #     if len(neighbor_indices) > 0:
    #         neighbor_positions = positions[neighbor_indices]
    #         center_of_mass = np.mean(neighbor_positions, axis=0)
    #         desired_direction = center_of_mass - self.position
    #         return desired_direction
    #     else:
    #         return np.zeros(2)

    # def separate(self, neighbor_indices, positions):
    #     if len(neighbor_indices) > 0:
    #         neighbor_positions = positions[neighbor_indices]
    #         separation_force = np.zeros(2)

    #         for neighbor_position in neighbor_positions:
    #             relative_position = self.position - neighbor_position
    #             distance = np.linalg.norm(relative_position)

    #             if distance > 0:
    #                 separation_force += (relative_position / (distance * distance))

    #         return separation_force
    #     else:
    #         return np.zeros(2)

    # def get_closest_neighbors(self, agents, max_distance):
    #     neighbor_indices = []

    #     for i, agent in enumerate(agents):
    #         if agent != self:
    #             distance = np.linalg.norm(agent.position - self.position)
    #             if distance < max_distance:
    #                 neighbor_indices.append(i)

    #     return neighbor_indices 
    
    
class Encoder(json.JSONEncoder):
        def default(self, obj):
            return json.JSONEncoder.default(self, obj)
     
class FlockingEnv(gym.Env):
    def __init__(self, agents): 
        super(FlockingEnv, self).__init__()
        self.current_timestep = 0
        self.flocking_agents = agents

        #Check what to use velocity and or acceleration [5, 5]
        min_obs = np.array([[-np.inf, -np.inf]] * len(self.flocking_agents), dtype=np.float32)
        max_obs = np.array([[np.inf, np.inf]] * len(self.flocking_agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

        #
        min_action = np.array([[-5, -5]] * len(self.flocking_agents), dtype=np.float32)
        max_action = np.array([[5, 5]] * len(self.flocking_agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)  # Control over velocity in x and y axes
    
        print("Action space: ", self.action_space)
        print("Observation space: ", self.observation_space)
    
    def step(self, actions): 
        done, observations = simulate_agents(actions)
        print(observations)
        reward=self.calculate_reward()

        print("Code.Observations: ", observations)
            
        return observations, reward, done

    def check_collision(self, agent):
        for other in self.flocking_agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
        return False

    def reset(self):
        agent_locations = read_agent_locations()
        agents =[Agent(position) for position in agent_locations]
        self.__init__(agents)

    def calculate_reward(self):
        cohesion_reward = 0
        separation_reward = 0
        collision_penalty = 0

        for agent in self.flocking_agents:
            for other in self.flocking_agents:
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
                    if distance < SimulationVariables["SafetyRadius"]:
                        collision_penalty -= 10

        total_reward = cohesion_reward + separation_reward + collision_penalty

        print(f"Agent Rewards and Penalties: Cohesion {cohesion_reward}, Separation {separation_reward}, Collision {collision_penalty}")

        return total_reward
    
    def get_observation(self):
        observation_agents = np.array([[agent.position[0], agent.position[1], agent.velocity[0], agent.velocity[1]] for agent in self.flocking_agents], dtype=np.float32)
        return observation_agents
    
    def calculate_future_direction(self):
        future_agents = [Agent(agent.position) for agent in self.flocking_agents]

        future_positions = []
        future_velocities = []

        for agent in future_agents:
            agent.update(SimulationVariables["dt"], future_agents, np.zeros(2))
            future_positions.append(agent.position)
            future_velocities.append(agent.velocity)

        return future_positions, future_velocities
    
    # def remove_fatality(self):
 

def read_agent_locations():
    with open(rf"{Results['InitPositions']}\config.json", "r") as f:
        data = json.load(f)
        print("File loaded")
    return data

def simulate_agents(agents):
    agent_data = {id(agent): [] for agent in agents}

    for agent in agents:
            # agent_id = id(agent)
            agent.update(agents, SimulationVariables["dt"])

            # agent_data_full = {
            #     "x": round(agent.position[0], 2),
            #     "y": round(agent.position[1], 2)
            # }

            # agent_data[agent_id].append(agent_data_full)

    print("Simulation Complete")


agent_locations = read_agent_locations()
agents = [Agent(position) for position in agent_locations]


env = DummyVecEnv([lambda: FlockingEnv(agents)])

actions = np.array([]) 
observations = np.array([]) 
rewards = np.array([])

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_Agents_tensorboard/")
# model.learn(total_timesteps=SimulationVariables["TimeSteps"])
# model.save("PPO")
