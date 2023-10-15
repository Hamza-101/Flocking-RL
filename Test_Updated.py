import json
from stable_baselines3 import PPO
from Params import *
from Simulation import FlockingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib import pyplot as plt

def read_agent_locations():
    with open(rf"{SimulationVariables['TestFile']}\config.json", "r") as f:
        data = json.load(f)
        print("File loaded")
    return data

# Load the trained model
model = PPO.load("PPO")

# Create an environment
env = DummyVecEnv([lambda: FlockingEnv()])

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward} - Std reward: {std_reward}")

# Visualize the behavior of the agents
obs = env.reset()
for i in range(1000):  # Run for 1000 steps
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

# Save the evaluation results
with open('evaluation_results.txt', 'w') as f:
    f.write(f"Mean reward: {mean_reward} - Std reward: {std_reward}")
