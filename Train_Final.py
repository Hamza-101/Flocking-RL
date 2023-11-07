import json
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Params import *
import gc
class Agent:
   
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityInit"], SimulationVariables["VelocityInit"], size=2), 2)
        self.acceleration = np.array([-(SimulationVariables["AccelerationInit"]), SimulationVariables["AccelerationInit"]], dtype=float)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
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
       
        observations=self.simulate_agents(actions)

        reward, cohesion_reward, separation_reward, collision_penalty = self.calculate_reward()
        info["Timestep"]=self.current_timestep
        info["Reward"]=reward
        info["CohesionReward"]=cohesion_reward
        info["SeparationReward"]=separation_reward
        info["CollisionPenalty"]=collision_penalty

        done = False  # Implement the termination condition based on your criteria

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
        self.agents = [Agent(position) for position in read_agent_locations()]
        for agent in self.agents:
            agent.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityInit"], SimulationVariables["VelocityInit"], size=2), 2)
        done = False
        return done
        
    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):
        observations = []
        for i, agent in enumerate(self.agents):
            new_position = agent.update(actions[i])
            observations.append(new_position)

        return observations
    
    def check_collision(self, agent):
        for other in self.flocking_agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
        return False

    def get_observation(self):
        observation = np.zeros((len(self.agents), 2), dtype=np.float32)

        for agent in enumerate(self.agents):
            observation[agent] = [
                agent.position[0],
                agent.position[1],
            ]
        return observation.flatten()
    
    def calculate_reward(self):
        total_reward = 0
        cohesion_reward = 0
        separation_reward = 0
        collision_penalty = 0

        for agent in self.agents:
            for other in self.agents:
                if agent != other:
                    distance = np.linalg.norm(agent.position - other.position)

                    # Cohesion Reward
                    if distance <= 50:
                        cohesion_reward += 5

                    # Separation Reward
                    if distance < SimulationVariables["NeighborhoodRadius"]:
                        separation_reward -= 5

                    # Collision Penalty
                    if distance < SimulationVariables["SafetyRadius"]:
                        collision_penalty -= 100

                        

        total_reward = cohesion_reward + separation_reward + collision_penalty

        # print(f"Total: {total_reward} Cohesion {cohesion_reward}, Separation {separation_reward}, Collision {collision_penalty}")

        return total_reward, cohesion_reward, separation_reward, collision_penalty
   

def read_agent_locations():
    with open(rf"{Results['InitPositions']}\config.json", "r") as f:
        data = json.load(f)
        print("File loaded")
    return data



env = DummyVecEnv([lambda: FlockingEnv()])

model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log = "./ppo_Agents_tensorboard/")
model.learn(total_timesteps = SimulationVariables["TimeSteps"])
model.save("PPO")

env.close()

 
# Initialize the environment
env = FlockingEnv()

# Run a sample test episode
obs = env.reset()
done = False
total_reward = 0
positions_dict = {i: [] for i in range(len(env.agents))}  # Initialize empty lists for each agent ID

while not done:
    # Assuming that the action space is within the specified limits
    actions = np.random.uniform(-5, 5, size=(len(env.agents), 2))
    new_obs, reward, done, info = env.step(actions)
    total_reward += reward

    # Store the positions of all agents at each timestep
    for i, agent in enumerate(env.agents):
        positions_dict[i].append(agent.position.tolist())

    # Print the timestep information
    print("Timestep:", info["Timestep"])
    print("Reward:", info["Reward"])
    print("Cohesion Reward:", info["CohesionReward"])
    print("Separation Reward:", info["SeparationReward"])
    print("Collision Penalty:", info["CollisionPenalty"])

    # Print the updated positions of the agents

    # Assume the episode ends after a certain number of timesteps for testing purposes
    if info["Timestep"] >= 1000:
        done = True

# Save the positions to a JSON file
with open('agent_positions.json', 'w') as f:
    json.dump(positions_dict, f, indent=4)

# Print the total accumulated reward
print("Total Accumulated Reward:", total_reward)

# Close the environment after testing
env.close()
