import json
import numpy as np
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from Settings import *
import os
import torch as th

# Plotting functions

class Agent:
    def __init__(self, position):
        self.position = {
            "x": float(position[0]), 
            "y": float(position[1])
        }
        
        # Initialize acceleration as a dictionary with x and y components
        self.acceleration = {
            "x": 0.0,
            "y": 0.0
        }
        
        # Initialize other agent properties
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        
        # Randomly assign initial velocity as a dictionary with x and y components
        self.velocity = {
            "x": round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"]), 2),
            "y": round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"]), 2)
        }
        
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    def update(self, action):
        # Update acceleration with the action
        self.acceleration["x"] += action[0]
        self.acceleration["y"] += action[1]
        
        # Clip acceleration to the max limits
        self.acceleration["x"] = np.clip(self.acceleration["x"], -self.max_acceleration, self.max_acceleration)
        self.acceleration["y"] = np.clip(self.acceleration["y"], -self.max_acceleration, self.max_acceleration)
        
        # Update velocity based on acceleration
        self.velocity["x"] += self.acceleration["x"] * SimulationVariables["dt"]
        self.velocity["y"] += self.acceleration["y"] * SimulationVariables["dt"]
        
        # Calculate the magnitude of the velocity
        vel = np.linalg.norm([self.velocity["x"], self.velocity["y"]])
        
        # Normalize and limit the velocity if it exceeds the maximum
        if vel > self.max_velocity:
            self.velocity["x"] = (self.velocity["x"] / vel) * self.max_velocity
            self.velocity["y"] = (self.velocity["y"] / vel) * self.max_velocity
        
        # Update position based on the current velocity
        self.position["x"] += self.velocity["x"] * SimulationVariables["dt"]
        self.position["y"] += self.velocity["y"] * SimulationVariables["dt"]
        
        return np.array([self.position["x"], self.position["y"]]), np.array([self.velocity["x"], self.velocity["y"]])

class FlockingEnv(MultiAgentEnv):
    def __init__(self):
        super(FlockingEnv, self).__init__()
        self.episode = 0
        self.counter = 1
        self.CTDE = False
        self.current_timestep = 0
        self.reward_log = []

        # Reading agent locations and creating agents
        agent_locations = self.read_agent_locations()

        # Initialize agents using the positions from the dictionary values
        self.agents = [Agent(position) for position in agent_locations.values()]

        # If agent IDs are still needed, you can create them, otherwise omit this
        self.agents_ids = [f"agent_{i}" for i in range(len(self.agents))]

        # Per-agent action and observation space
        # Action space as a dictionary for each agent

        self.action_space = gym.spaces.Dict({
            i: gym.spaces.Box(low=np.array([-5, -5], dtype=np.float32),
                        high=np.array([5, 5], dtype=np.float32),
                        dtype=np.float32)
            for i in range(1, len(agent_locations) + 1)  # Keys correspond to agent indices
        })

        self.observation_space = gym.spaces.Dict({
            i: gym.spaces.Box(
                low=np.array([-np.inf, -np.inf, -SimulationVariables["VelocityUpperLimit"], -SimulationVariables["VelocityUpperLimit"]], dtype=np.float32),
                high=np.array([np.inf, np.inf, SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"]], dtype=np.float32),
                dtype=np.float32
            )
            for i in range(1, len(agent_locations) + 1)  # Keys correspond to agent indices
        })
        
    def step(self, action_dict):
        observations = {}
        rewards = {}
        dones = {"__all__": False}
        infos = {}

        # Apply noisy actions and prepare actions for agents
        actions = {}
        for agent_id, action in action_dict.items():
            # Apply noise to actions
            noisy_action = action + np.random.normal(loc=0, scale=0.5, size=action.shape)
            action = np.clip(noisy_action, self.action_space[agent_id].low, self.action_space[agent_id].high)
            
            # Store the processed action in a dictionary
            actions[agent_id] = action

        # Call simulate_agents to update the agents and get observations
        observations = self.simulate_agents(actions)

        # Calculate rewards and check for termination conditions
        for agent_id in self.agents_ids:
            reward, out_of_flock = self.calculate_reward(self.agents[agent_id - 1])
            rewards[agent_id] = reward

            # If there's a collision or agent is out of flock, mark the episode as done
            if self.check_collision(self.agents[agent_id - 1]) or out_of_flock:
                dones["__all__"] = True  # End episode if a collision occurs
                self.reset()

        # Update the time step count
        self.current_timestep += 1

        return observations, rewards, dones, infos

    def reset(self):
        self.seed(SimulationVariables["Seed"]) ############

        # Reset all agents
        agent_locations = self.read_agent_locations()

        # Initialize agents using the positions from the dictionary values
        self.agents = [Agent(position) for position in agent_locations.values()]

        # If agent IDs are still needed, you can create them, otherwise omit this
        self.agents_ids = [f"agent_{i}" for i in range(len(self.agents))]

        # Initialize observations for each agent
        observations = {}
        for agent_id, agent in self.agents.items():
             # Initialize acceleration as a dictionary with x and y components
            self.acceleration = {
                "x": 0.0,
                "y": 0.0
            }

            self.velocity = {
            "x": round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"]), 2),
            "y": round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"]), 2)
            }

            # Observation function

            # Randomly assign initial velocity as a dictionary with x and y components
            # Initialize the observation for each agent
            observations[agent_id] = np.array([
                agent.position["x"],
                agent.position["y"],
                agent.velocity["x"],
                agent.velocity["y"]
            ])

        self.current_timestep = 0  # Reset time step count

        return observations

    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):
        observations = {}  # Initialize an empty dictionary for observations

        for agent_id in self.agents_ids:
            # Retrieve the action for the current agent from the actions dictionary
            action = actions[agent_id]

            # Update the agent using the action
            position, velocity = self.agents[agent_id - 1].update(action)  # Adjusting index if IDs start from 1

            # Create an observation pair by extracting x and y from the dictionaries
            observation_pair = np.array([
                position["x"], 
                position["y"], 
                velocity["x"], 
                velocity["y"]
            ])

            # Store the observation in the dictionary using agent_id as the key
            observations[agent_id] = observation_pair

        return observations  # Return the observations dictionary
    
    def check_collision(self, agent):
        
        for other in self.agents:
            if agent != other:
                # Calculate the distance using the x and y components of the position
                distance = np.linalg.norm([
                    agent.position["x"] - other.position["x"],
                    agent.position["y"] - other.position["y"]
                ])
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
                    
        return False


    def get_closest_neighbors(self, agent):
        neighbor_positions = []
        neighbor_velocities = []

        for other in self.agents:
            if agent != other:
                # Calculate the distance using the x and y components of the position
                distance = np.linalg.norm([
                    other.position["x"] - agent.position["x"],
                    other.position["y"] - agent.position["y"]
                ])

                if self.CTDE:
                    if distance < SimulationVariables["NeighborhoodRadius"]:
                        neighbor_positions.append(other.position)
                        neighbor_velocities.append(other.velocity)

                else:  
                    neighbor_positions.append(other.position)
                    neighbor_velocities.append(other.velocity)

        return neighbor_positions, neighbor_velocities    

    def calculate_reward(self, agent):
        reward = 0
        out_of_flock = False

        # Get closest neighbors for the agent
        neighbor_positions, neighbor_velocities = self.get_closest_neighbors(agent)

        # Calculate reward for the agent based on neighbors' positions and velocities
        reward, out_of_flock = self.reward(agent, neighbor_velocities, neighbor_positions)

        return reward, out_of_flock

    def reward(self, agent, neighbor_velocities, neighbor_positions):
        CohesionReward = 0
        AlignmentReward = 0
        total_reward = 0
        outofflock = False
        midpoint = (SimulationVariables["SafetyRadius"] + SimulationVariables["NeighborhoodRadius"]) / 2

        if len(neighbor_positions) > 0:
            for neighbor_position in neighbor_positions:
                # Accessing x and y from the neighbor position dictionary
                distance = np.linalg.norm([
                    agent.position["x"] - neighbor_position["x"],
                    agent.position["y"] - neighbor_position["y"]
                ])

                if distance <= SimulationVariables["SafetyRadius"]:
                    CohesionReward += -10
                elif SimulationVariables["SafetyRadius"] < distance <= midpoint:
                    CohesionReward += (10 / (midpoint - SimulationVariables["SafetyRadius"])) * (distance - SimulationVariables["SafetyRadius"])
                elif midpoint < distance <= SimulationVariables["NeighborhoodRadius"]:
                    CohesionReward += 10 - (10 / (SimulationVariables["NeighborhoodRadius"] - midpoint)) * (distance - midpoint)

            # Calculate average velocity using the dictionary structure
            if neighbor_velocities:
                average_velocity = np.mean(
                    [np.array([v["x"], v["y"]]) for v in neighbor_velocities], axis=0
                )
            else:
                average_velocity = np.zeros(2)

            dot_product = np.dot(average_velocity, np.array([agent.velocity["x"], agent.velocity["y"]]))
            norm_product = np.linalg.norm(average_velocity) * np.linalg.norm([agent.velocity["x"], agent.velocity["y"]])

            if norm_product == 0:
                cos_angle = 1.0
            else:
                cos_angle = dot_product / norm_product

            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            orientation_diff = np.arccos(cos_angle)

            alignment = (orientation_diff / np.pi)
            AlignmentReward = -20 * alignment + 10  

        else:
            CohesionReward -= 10
            outofflock = True

        total_reward = CohesionReward + AlignmentReward

        return total_reward, outofflock


    def read_agent_locations(self):
        # Build file path dynamically
        File = rf"{Results['InitPositions']}" + str(self.counter) + "\config.json"

        # Read the data from the JSON file
        with open(File, "r") as f:
            data = json.load(f)

        # Convert list of locations to a dictionary with indices starting from 1
        indexed_data = {i+1: loc for i, loc in enumerate(data)}

        return indexed_data

# Move to other file
    def seed(self, seed=SimulationVariables["Seed"]):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)  # Ensure NumPy uses the seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.backends.cudnn.deterministic = True
        return [seed]