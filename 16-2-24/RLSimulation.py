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
    def __init__(self):
        super(FlockingEnv, self).__init__()
        self.episode=0
        self.CTDE=False
        self.current_timestep = 0 
        self.reward_log = []
        self.counter=0

        self.agents = [Agent(position) for position in self.read_agent_locations()]
        
        # Accelerations
        min_action = np.array([-5, -5] * len(self.agents), dtype=np.float32)
        max_action = np.array([5, 5] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        # Positions
        min_obs = np.array([[-np.inf, -np.inf, -2.5, -2.5]] * len(self.agents), dtype=np.float32)
        max_obs = np.array([[np.inf, np.inf, 2.5, 2.5]] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)


    def step(self, actions):
        
        # if(self.current_timestep % 100000 == 0):
        #     print(self.current_timestep)
        #     self.counter = self.counter + 1
        #     print("Counter", self.counter)

        self.current_timestep+=1
        reward=0
        done=False
        info={}
        Collisions={}
        observations = self.simulate_agents(actions)

        # Collisions

        alignment_reward, cohesion_reward, separation_reward, collision_reward, Collisions=self.calculate_reward()

        info["Collisions"]=Collisions
        info["CohesionReward"]=cohesion_reward
        info["AlignmentReward"]=alignment_reward
        info["CollisionReward"]=collision_reward         # Always negative
        info["SeparationReward"]=separation_reward
        #Do I need to change

        reward = info["SeparationReward"] + info["AlignmentReward"] + info["CohesionReward"] + info["CollisionReward"]

        # if any(info["Collisions"]):
        #     done = True
        #     observations=self.reset()

        if self.CTDE:

            log_path = os.path.join(Files['Flocking'], 'Testing', 'Rewards', 'Components', f"Episode{episode}")
            log_path = os.path.join(log_path, "Reward_Total_log.json")

        
            with open(log_path, 'a') as f:
                json.dump((round(reward, 2)), f, indent=4)
                f.write('\n')

        
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
        self.agents = [Agent(position) for position in self.read_agent_locations()]

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
    
        CohesionReward=0
        AlignmentReward=0
        SeparationReward=0
        CollisionReward=0

        Collisions={}

        neighbor_positions = [[] for _ in range(len(self.agents))]  # Initialize empty lists for neighbor indices
        neighbor_velocities = [[] for _ in range(len(self.agents))]

        #WRITE COLLISION CODE
        for idx, _ in enumerate(self.agents):
            Collisions[idx] = []

        for _, agent in enumerate(self.agents): 
            neighbor_positions, neighbor_velocities=self.get_closest_neighbors(agent)

            tmpVal, collisions_agent=self.CollisionReward(agent, neighbor_positions)
            AlignmentReward += self.calculate_AlignmentReward(agent, neighbor_velocities)            
            SeparationReward += self.calculate_SeparationReward(agent, neighbor_positions)
            CohesionReward += self.calculate_CohesionReward(agent, neighbor_positions)

            CollisionReward += tmpVal

            Collisions[idx].append(collisions_agent)

            # Testing-Partial Observation
            if(self.CTDE==True):
                Rewards = {
                    "AlignmentReward": AlignmentReward, 
                    "CohesionReward" : CohesionReward,
                    "SeparationReward": SeparationReward,
                    "CollisionReward": CollisionReward,
                }      

                Log(self.episode, Rewards)

        return AlignmentReward, CohesionReward, SeparationReward, CollisionReward, Collisions
     
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

    def CollisionReward(self, agent, neighbor_positions):
        Collision = False
        CollisionReward = 0
        distances=[]

        # Check neighbor indices
        
        for position in neighbor_positions:             #Use neighborhood or all
            distance = np.linalg.norm(agent.position - position)
            if distance < SimulationVariables["SafetyRadius"]:            
                CollisionReward -= 1000
                Collision = True

        return CollisionReward, Collision
    
    def calculate_CohesionReward(self, agent, neighbor_positions):

        CohesionReward=0
        if(len(neighbor_positions) > 0):
            # Find why working
            center_of_mass = np.mean(neighbor_positions, axis=0)
            # print("Center of mass", center_of_mass)
            desired_position = center_of_mass - agent.position
            distance = np.linalg.norm(desired_position)
            if SimulationVariables["SafetyRadius"] < distance <= SimulationVariables["NeighborhoodRadius"]:
                CohesionReward = 1
            elif distance > SimulationVariables["NeighborhoodRadius"]:
                CohesionReward = -(1 * (distance - SimulationVariables["NeighborhoodRadius"]))
                
        # else:
        #     CohesionReward = -1 * 

        return CohesionReward

    def calculate_SeparationReward(self, agent, neighbor_positions):
        SeparationReward=0

        #Check neighbor indices
        if len(neighbor_positions) > 0:
            for neighbor_position in neighbor_positions:

                relative_position = agent.position - neighbor_position
                distance = np.linalg.norm(relative_position)

                if distance >= SimulationVariables["SafetyRadius"]:
                    SeparationReward += 1
                elif (distance < SimulationVariables["SafetyRadius"]):
                    SeparationReward += -(1 * (distance - SimulationVariables["SafetyRadius"]))
            
        return SeparationReward
    
    # Test THIS
    def calculate_AlignmentReward(self, agent, neighbor_velocities):

        AlignmentReward=0

        if (len(neighbor_velocities) > 0):
            average_velocity = np.mean(neighbor_velocities, axis=0)
            
            desired_orientation = average_velocity - agent.velocity   
            orientation_diff = np.arctan2(
            desired_orientation[1], desired_orientation [0]) 
            - np.arctan2(agent.velocity[1], agent.velocity[0])

            if orientation_diff > np.pi:
                orientation_diff -= 2 * np.pi
            elif orientation_diff < 0:
                orientation_diff += 2 * np.pi

            AlignmentReward = 1 - np.abs(orientation_diff)

        return AlignmentReward 
    
    def read_agent_locations(self):
        File = rf"{Results['InitPositions']}" + str(self.counter) + "\config.json"
        # print(File)
        with open(File, "r") as f:
            data = json.load(f)
        return data
    
    def seed(self, seed=None):
        np.random.seed(seed)  
 


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

env =  FlockingEnv()

n_steps_per_update = 10000
n_updates = SimulationVariables["LearningTimeSteps"] // n_steps_per_update
        
policy_kwargs = dict(
    activation_fn=th.nn.ReLU,  # Replace with desired activation function if needed
    net_arch=[
        # Define layers for both pi (actor) and vf (critic) networks
        dict(pi=[512, 512, 512, 512, 512, 512, 512, 512], vf=[512, 512, 512, 512, 512, 512, 512, 512]),
    ]
)


random_seed = 23
env.seed(random_seed)
print("Random Seed:", env.seed)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_Agents_tensorboard/", verbose=1)

model.set_random_seed(random_seed)
model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"]) 
model.save(rf"{Files['Flocking']}\\Models\\Flocking")



# Testing
env = FlockingEnv()
model = PPO.load(rf"{Files['Flocking']}\\Models\\Flocking")

delete_files()

positions_directory = rf"{Files['Flocking']}/Testing/Positions/"
velocities_directory = rf"{Files['Flocking']}/Testing/Dynamics/Velocities/"
accelerations_directory = rf"{Files['Flocking']}/Testing/Dynamics/Accelerations/"

os.makedirs(positions_directory, exist_ok=True)
os.makedirs(velocities_directory, exist_ok=True)
os.makedirs(accelerations_directory, exist_ok=True)

for episode in range(0, SimulationVariables['Episodes']):
    env.episode=episode
    print("Episode:", episode)
    env.CTDE=True
    obs=env.reset()
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
    print(reward_curr)
    with open(f"{positions_directory}Episode_{episode}.json", 'w') as f:
        json.dump(positions_dict, f, indent=4)

env.close()