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
        # for agent in self.agents:
        #     print(agent.position)
        #     print(agent.velocity)

    def step(self, actions):
        info={}
        reward = 0
       
        
        normalized_actions = self.action_mapping(actions)
        observations=self.simulate_agents(normalized_actions)

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
        # Reset the positions and velocities of the agents to their initial values
        for agent, initial_position in zip(self.agents, read_agent_locations()):
            agent.position = np.array(initial_position, dtype=float)
            agent.velocity = np.array([-(SimulationVariables["VelocityInit"]), SimulationVariables["VelocityInit"]], dtype=float)
            agent.acceleration = np.array([-(SimulationVariables["AccelerationInit"]), SimulationVariables["AccelerationInit"]], dtype=float)

        # Return the initial observation after resetting the environment
        obs = np.array([[agent.position[0], agent.position[1]] for agent in self.agents])

        return obs
    
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

    #Not dividing by 10
    def action_mapping(self, actions):

        velocities = []
        for action in actions:  
            action_step_x = round((action[0]), 2)
            action_step_y = round((action[1]), 2)
            velocities.append(np.array([action_step_x, action_step_y]))

        return velocities
    
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


if __name__ == "__main__":

    env = DummyVecEnv([lambda: FlockingEnv()])

    model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log = "./ppo_Agents_tensorboard/")
    model.learn(total_timesteps = SimulationVariables["TimeSteps"])
    model.save("PPO")

    env.close()
