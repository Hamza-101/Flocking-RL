from stable_baselines3 import PPO
from Params import *
from Simulation import Agent, FlockingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import json

def read_agent_locations():
    with open(rf"{SimulationVariables['TestFile']}\config.json", "r") as f:
        data = json.load(f)
        print("File loaded")
    return data

loaded_model = PPO.load("PPO")

agent_locations = read_agent_locations()
agents = [Agent(position) for position in agent_locations]

env = DummyVecEnv([lambda: FlockingEnv(agents)])

obs = env.reset()
done = False
while not done:
    action, _ = loaded_model.predict(obs)
    obs, reward, done, _ = env.step(action)

# Close the Gym environment
env.close()