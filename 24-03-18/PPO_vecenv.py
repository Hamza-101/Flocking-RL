import os
import json
import gym
from gym import spaces
from Settings import *
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th

# Why reward less
# Continuous reward
# Exit on collision
# Centroid


# Move to other file



#Weights
class Agent:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

        # Initialize acceleration
        self.acceleration = np.round(np.random.uniform(-SimulationVariables["AccelerationUpperLimit"], SimulationVariables["AccelerationUpperLimit"], size=2), 2)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        
        # Calculate initial velocity based on the initial acceleration
        self.velocity = self.acceleration * SimulationVariables["dt"]
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    # See if can be moved
    def update(self, action):
        

        # Check if this correct
        self.acceleration += action


        # Update velocity
        self.velocity += self.acceleration * SimulationVariables["dt"]

        vel = np.linalg.norm(self.velocity)

        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity    

        self.position += self.velocity * SimulationVariables["dt"]
        return self.position, self.velocity

# Move to other file
class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

class FlockingEnv(gym.Env):
    def __init__(self, position, ID):
        super(FlockingEnv, self).__init__()
        self.episode=0
        self.CTDE=False
        self.current_timestep = 0 
        self.reward_log = []
        self.counter=0
        self.env_id=ID

        self.agents = [Agent(position)]
        
        # Accelerations
        min_action = np.array([-5, -5], dtype=np.float32)
        max_action = np.array([5, 5], dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        # Positions
        min_obs = np.array([[-np.inf, -np.inf, -2.5, -2.5]] * len(self.agents), dtype=np.float32)
        max_obs = np.array([[np.inf, np.inf, 2.5, 2.5]] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)


    def step(self, actions):
        
        # if(self.current_timestep % 400000 == 0):
        #     print(self.current_timestep)
        #     self.counter = self.counter + 1
        #     print("Counter", self.counter)

        self.current_timestep+=1
        reward=0
        done=False
        info={}

        observations = self.simulate_agents(actions)

        reward, out_of_flock =  self.calculate_reward()
        if(self.CTDE==False):
            # Terminal Conditions
            for agent in self.agents:
                if((self.check_collision(agent)) or (out_of_flock==True)):
                    done=True


        # if self.CTDE:

        #     log_path = os.path.join(Files['Flocking'], 'Testing', 'Rewards', 'Components', f"Episode{episode}")
        #     log_path = os.path.join(log_path, "Reward_Total_log.json")

        
        #     with open(log_path, 'a') as f:
        #         json.dump((round(reward, 2)), f, indent=4)
        #         f.write('\n')

        
        self.current_timestep = self.current_timestep + 1

        return observations, reward, done, info

        #CHECK RESET

    def close(self):
        print("Environment is closed. Cleanup complete.")
        
        #Does velocity make a difference
        #Observation Space
   
    def simulate_agents(self, actions):
        observations = []
        for i, agent in enumerate(self.agents):
            position, velocity = agent.update(actions[i])
            observations.append(np.concatenate([position, velocity]))  # Include velocity
        return observations
    
    def check_collision(self, agent):
        # See if correct
        # Collisions[idx].append(collisions_agent)

        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
        return False

    def reset(self):

        
        boid_positions =  self.read_agent_locations() # Implement this function according to your file format

        self.agents = [Agent(boid_positions[self.env_id])]

        for agent in self.agents:
            agent.acceleration = np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)
            agent.velocity = agent.acceleration * SimulationVariables["dt"]
                
        observation = self.get_observation()

        return observation   
   
    def get_observation(self):
        observation = np.zeros((len(self.agents), 4), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            observation[i] = [
                agent.position[0],
                agent.position[1],
                agent.velocity[0],  
                agent.velocity[1]
            ]
        return observation

     
    def calculate_reward(self):

        collision_reward=0
        reward=0

        Collisions={}

        neighbor_positions = [[] for _ in range(len(self.agents))]  # Initialize empty lists for neighbor indices
        neighbor_velocities = [[] for _ in range(len(self.agents))]
        out_of_flock=False

        #WRITE COLLISION CODE
        for idx, _ in enumerate(self.agents):
            Collisions[idx] = []

        for _, agent in enumerate(self.agents): 
            neighbor_positions, neighbor_velocities=self.get_closest_neighbors(agent)
            
            reward, out_of_flock=self.calculate_combined_reward(agent, neighbor_positions, neighbor_velocities) #############################################################
            # collision_reward += self.CollisionRewardCal(agent, neighbor_positions)

        return reward, out_of_flock
    
    #REMOVED COLLISION
    def CollisionRewardCal(self, agent, neighbor_positions):
        CollisionReward = 0
        distances=[]

        # Check neighbor indices
        
        for position in neighbor_positions:             #Use neighborhood or all
            distance = np.linalg.norm(agent.position - position)
            if distance < SimulationVariables["SafetyRadius"]:            
                CollisionReward -= 1000

        return CollisionReward
    
    # Find neighbor based on distance
    def get_closest_neighbors(self, agent):
        neighbor_positions=[]
        neighbor_velocities=[]

        # INDEX VS POSITION

        for i, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)
                if(self.CTDE==True):
                    # print("CTDE")
                    if distance < SimulationVariables["NeighborhoodRadius"]:
                        neighbor_agent = self.agents[i]
                        neighbor_positions.append(neighbor_agent.position)
                        neighbor_velocities.append(other.velocity)
                        
                else:  
                    neighbor_agent= self.agents[i]
                    neighbor_positions.append(neighbor_agent.position)
                    neighbor_velocities.append(other.velocity)

        return neighbor_positions, neighbor_velocities         

    def calculate_combined_reward(self, agent, neighbor_positions, neighbor_velocities):
        total_reward = 0  
        out_of_flock = False

        if neighbor_positions:  # Check if there are neighbors
            # Cohesion: Encourage agents to stay close to each other
            center_of_mass = np.mean(neighbor_positions, axis=0)
            cohesion_reward = -0.1 * np.linalg.norm(agent.position - center_of_mass)
            total_reward += cohesion_reward

            # Alignment: Encourage agents to align their velocities with neighbors
            if neighbor_velocities:
                avg_neighbor_velocity = np.mean(neighbor_velocities, axis=0)
                alignment_reward = np.dot(agent.velocity, avg_neighbor_velocity)
                total_reward += alignment_reward

            # Collision Avoidance: Penalize getting too close to neighbors
            for neighbor_position in neighbor_positions:
                distance = np.linalg.norm(agent.position - neighbor_position)
                if distance < SimulationVariables["SafetyRadius"]:
                    total_reward += -50 * (1 - (distance / SimulationVariables["SafetyRadius"]))

            # In Flock Reward
            # total_reward += 100  # Reward for being in the flock

        else:
            # If no neighbors, encourage agent to find flock
            total_reward = -500
            out_of_flock = True

        return total_reward, out_of_flock
    
    def read_agent_locations(self):
        File = rf"{Results['InitPositions']}" + str(self.counter) + "\config.json"
        # print(File)
        with open(File, "r") as f:
            data = json.load(f)
        return data
    
    
    

def read_agent_locations(counter):
    File = rf"{Results['InitPositions']}" + str(counter) + "\config.json"

    with open(File, "r") as f:
        data = json.load(f)
    return data
    

def Log(episode, Rewards):
    episode_directory = os.path.join(Files['Flocking'], 'Testing', 'Rewards', 'Components', f"Episode{episode}")
    os.makedirs(episode_directory, exist_ok=True)

    for reward_type, reward_value in Rewards.items():
        log_path = os.path.join(episode_directory, f"{str(reward_type)}_log.json")
        with open(log_path, 'a') as f:
            f.write(json.dumps(round((reward_value), 2)) + '\n')

def delete_files(): 
    Paths = ["Results\Flocking\Testing\Dynamics\Accelerations", "Results\Flocking\Testing\Dynamics\Velocities", 
            "Results\Flocking\Testing\Rewards\Other"]

    Logs = ["AlignmentReward_log.json", "CohesionReward_log.json",
            "SeparationReward_log.json", "CollisionReward_log.json",
            "Reward_Total_log.json"]

    # Delete files based on Paths
    for Path in Paths:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], Path, f"Episode{episode}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                # print(f"File {file_path} has been deleted.")

    # Delete files based on Logs
    for log_file in Logs:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], "Testing", "Rewards", "Components", f"Episode{episode}", log_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                # print(f"File {file_path} has been deleted.")       

# Check if valid
if os.path.exists(Results["Rewards"]):
    os.remove(Results["Rewards"])
    print(f"File {Results['Rewards']} has been deleted.")


n_steps_per_update = 10000
n_updates = SimulationVariables["LearningTimeSteps"] // n_steps_per_update
        
policy_kwargs = dict(
    activation_fn=th.nn.Tanh,  # Replace with desired activation function if needed
    net_arch=[
        # Define layers for both pi (actor) and vf (critic) networks
        dict(pi=[512, 512, 512, 512, 512, 512, 512, 512], vf=[512, 512, 512, 512, 512, 512, 512, 512]),
    ]
)



random_seed=23

boid_positions = read_agent_locations(SimulationVariables["counter"]) 

envs = [lambda i=i: lambda: FlockingEnv(boid_positions[i], i) for i in range(SimulationVariables["SimAgents"])]
vec_env = DummyVecEnv([env() for env in envs])

model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_Agents_tensorboard/", verbose=1)


model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"]) 
model.save(rf"{Files['Flocking']}\\Models\\Flocking2")



# Testing
env = FlockingEnv()
model = PPO.load(rf"{Files['Flocking']}\\Models\\Flocking2")



delete_files()

positions_directory = rf"{Files['Flocking']}/Testing/Positions/"
velocities_directory = rf"{Files['Flocking']}/Testing/Dynamics/Velocities/"
accelerations_directory = rf"{Files['Flocking']}/Testing/Dynamics/Accelerations/"

os.makedirs(positions_directory, exist_ok=True)
os.makedirs(velocities_directory, exist_ok=True)
os.makedirs(accelerations_directory, exist_ok=True)
env.counter=15

for episode in range(0, SimulationVariables['Episodes']):
    env.episode=episode
    print("Episode:", episode)
    env.CTDE=True
    obs=env.reset()
    print(env.counter)
    done=False
    timestep=0 
    reward_episode=[]
    reward_curr=0

    positions_dict = {i: [] for i in range(len(env.agents))}

    while timestep <= SimulationVariables["EvalTimeSteps"]:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        reward_episode.append(round(reward, 2))

        for i, agent in enumerate(env.agents):
            positions_dict[i].append(agent.position.tolist())

        timestep += 1
    with open(rf"{positions_directory}/Episode_{episode}.json", 'w') as f:
        json.dump(positions_dict, f, indent=4)

    print(sum(reward_episode))
    env.counter+=1

env.close()
