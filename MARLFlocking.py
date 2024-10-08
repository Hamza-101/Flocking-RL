
import os
import gym
import json
import numpy as np
import torch as th
from tqdm import tqdm
from Settings import *
from gym import spaces
import matplotlib.pyplot as plt
import torch.nn.functional as F
from stable_baselines3 import PPO
from scipy.signal import savgol_filter
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

# Ensure the model only uses CPU
th.cuda.is_available = lambda: True

# Neural Network Parameters
policy_kwargs = dict(
    activation_fn=th.nn.Tanh,  # Using ReLU activation function
    net_arch=[dict(pi=[512, 512, 512, 512, 512, 512, 512, 512], 
                   vf=[512, 512, 512, 512, 512, 512, 512, 512])]  # Separate networks for policy (pi) and value function (vf)
)

class TQDMProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        if self.pbar:
            # Update progress bar with the number of timesteps
            self.pbar.update(self.model.num_timesteps - self.pbar.n)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

class LossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossCallback, self).__init__(verbose)
        self.loss_threshold = 2000

    def _on_step(self) -> bool:
        
        # if(self.current_timestep > (SimulationVariables["LearningTimesteps"]/2)):
        if len(self.model.ep_info_buffer) >= 1000:
            recent_losses = [ep_info['loss'] for ep_info in self.model.ep_info_buffer[-1000:]]
            average_loss = np.mean(recent_losses)

            if average_loss < self.loss_threshold:
                print(f"Stopping training because average loss ({average_loss}) is below threshold.")
                return False  

        return True


# Custom actor network per agent
class CustomActor(th.nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomActor, self).__init__()
        self.fc1 = th.nn.Linear(observation_space.shape[0], 64)
        self.fc2 = th.nn.Linear(64, 64)

        # Update action head based on action space type
        if isinstance(action_space, spaces.Box):  # Continuous action space
            self.action_head = th.nn.Linear(64, action_space.shape[0])
        elif isinstance(action_space, spaces.Discrete):  # Discrete action space
            self.action_head = th.nn.Linear(64, action_space.n)
        else:
            raise NotImplementedError("Action space type not supported")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        return action_logits

# Shared critic network
class SharedCritic(th.nn.Module):
    def __init__(self, observation_space):
        super(SharedCritic, self).__init__()
        self.fc1 = th.nn.Linear(observation_space.shape[0], 64)
        self.fc2 = th.nn.Linear(64, 64)
        self.value_head = th.nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value

# Custom Actor-Critic policy for PPO with different actors per agent and shared critic
class CustomMultiAgentPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule):
        super(CustomMultiAgentPolicy, self).__init__(observation_space, action_space, lr_schedule)
        
        # Create different actors for each agent (assuming max 3 agents here, adjust as necessary)
        self.actors = [CustomActor(observation_space, action_space) for _ in range(3)]

        # Shared critic
        self.critic = SharedCritic(observation_space)

    def forward(self, obs, agent_id):
        if agent_id < len(self.actors):
            action_logits = self.actors[agent_id](obs)
        else:
            raise ValueError("Agent ID out of range")
        return action_logits

    def critic_forward(self, obs):
        return self.critic(obs)


#Multiple random initialization
class Agent:
    def __init__(self, position, agent_id):
        
        #Update reset
        self.agent_id = agent_id  

        self.position = np.array(position, dtype=float)

        # Random initialization of velocity and initializing acceleration to null
        self.acceleration = np.zeros(2)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]

        self.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)       
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    def update(self, action):

        #SEE THIS
        self.acceleration += action
        
        self.acceleration=np.clip(self.acceleration, -(SimulationVariables["AccelerationUpperLimit"]), SimulationVariables["AccelerationUpperLimit"])

        self.velocity += self.acceleration * SimulationVariables["dt"]
        vel = np.linalg.norm(self.velocity)
        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity    
        self.position += self.velocity * SimulationVariables["dt"]

        return self.position, self.velocity

class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

# 3 Agents
class FlockingEnv(gym.Env):
    def __init__(self, *args, use_sde=False, **kwargs):
        super(FlockingEnv, self).__init__(*args, **kwargs)
        self.use_sde = use_sde

        self.episode=0
        self.counter=3602
        self.CTDE=False
        self.current_timestep = 0
        self.reward_log = []
        self.np_random, _ = gym.utils.seeding.np_random(None)

        self.agents = [Agent(position, agent_id=f"agent_{i}") 
               for i, position in enumerate(self.read_agent_locations())]
        print(self.agents)

        # Use settings file in actions and observations
        min_action = np.array([-5, -5] * len(self.agents), dtype=np.float32)
        max_action = np.array([5, 5] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        #Check this
        min_obs = np.array([-np.inf, -np.inf, -2.5, -2.5] * len(self.agents), dtype=np.float32)
        max_obs = np.array([np.inf, np.inf, 2.5, 2.5] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def step(self, actions):
        training_rewards = {}

        noisy_actions = actions + np.random.normal(loc=0, scale=0.5, size=actions.shape)
        actions = np.clip(noisy_actions, self.action_space.low, self.action_space.high)

        self.current_timestep += 1
        reward = 0
        done = False
        info = {}
        
        observations = self.simulate_agents(actions)
        reward, out_of_flock = self.calculate_reward()

        if not self.CTDE:
            for idx in range(len(self.agents)):
                if self.check_collision(idx) or out_of_flock:
                    done = True
                    env.reset() 
                    break 

        with open("training_rewards.json", "w") as f:
            json.dump(training_rewards, f)

        return observations, reward, done, info

    def reset(self):
        env.seed(SimulationVariables["Seed"])
        self.agents = [Agent(position, agent_id=f"agent_{i}") 
                   for i, position in enumerate(self.read_agent_locations())]


        for agent in self.agents:
            agent.acceleration = np.zeros(2)
            agent.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)             

        observation = self.get_observation().flatten()
        
        ################################
        self.current_timestep = 0  # Reset time step count
        ################################
        return observation   

    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):
        observations = []  # Initialize an empty 1D array
        actions_reshaped = actions.reshape(((SimulationVariables["SimAgents"]), 2))
        for i, agent in enumerate(self.agents):
            position, velocity = agent.update(actions_reshaped[i])
            observation_pair = np.concatenate([position, velocity])
            observations = np.concatenate([observations, observation_pair])  # Concatenate each pair directly

        return observations
    
    def check_collision(self, agent_idx):
        agent = self.agents[agent_idx]
        for other_idx, other in enumerate(self.agents):
            if agent_idx != other_idx:  # Compare indices
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
        return False

    def get_observation(self):
        observations = np.zeros((len(self.agents), 4), dtype=np.float32)
    
        for i, agent in enumerate(self.agents):
            observations[i] = [
                agent.position[0],
                agent.position[1],
                agent.velocity[0],
                agent.velocity[1]
            ]

        # Reshape the observation into 1D                    
        return observations
   
    def get_closest_neighbors(self, agent_idx):
        agent = self.agents[agent_idx]
        neighbor_positions = []
        neighbor_velocities = []
        
        for other_idx, other in enumerate(self.agents):
            if agent_idx != other_idx:  # Compare indices instead of agents
                distance = np.linalg.norm(other.position - agent.position)

                if self.CTDE:
                    if distance < SimulationVariables["NeighborhoodRadius"]:
                        neighbor_positions.append(other.position)
                        neighbor_velocities.append(other.velocity)
                else:
                    neighbor_positions.append(other.position)
                    neighbor_velocities.append(other.velocity)

        return neighbor_positions, neighbor_velocities       
   
    def calculate_reward(self):
        reward = 0
        Collisions = {}

        # Initialize neighbor data structures based on the number of agents
        neighbor_positions = [[] for _ in range(len(self.agents))] 
        neighbor_velocities = [[] for _ in range(len(self.agents))]
        out_of_flock = False

        for idx in range(len(self.agents)):
            Collisions[idx] = []  

        for idx, agent in enumerate(self.agents): 
            neighbor_positions, neighbor_velocities = self.get_closest_neighbors(idx)
            val, out_of_flock = self.reward(agent, neighbor_velocities, neighbor_positions)
            reward += val

        return reward, out_of_flock

    def reward(self, agent, neighbor_velocities, neighbor_positions):
        CohesionReward = 0
        AlignmentReward = 0
        total_reward = 0
        outofflock = False
        midpoint = (SimulationVariables["SafetyRadius"] + SimulationVariables["NeighborhoodRadius"]) / 2

        if len(neighbor_positions) > 0:
            for neighbor_position in neighbor_positions:

                distance = np.linalg.norm(agent.position - neighbor_position)
                if distance <= SimulationVariables["SafetyRadius"]:
                    CohesionReward += 0
                elif SimulationVariables["SafetyRadius"] < distance <= midpoint:
                    CohesionReward += (10 / (midpoint - SimulationVariables["SafetyRadius"])) * (distance - SimulationVariables["SafetyRadius"])
                elif midpoint < distance <= SimulationVariables["NeighborhoodRadius"]:
                    CohesionReward += 10 - (10 / (SimulationVariables["NeighborhoodRadius"] - midpoint)) * (distance - midpoint)
      
                average_velocity = np.mean(neighbor_velocities, axis = 0)
                dot_product = np.dot(average_velocity, agent.velocity)
                norm_product = np.linalg.norm(average_velocity) * np.linalg.norm(agent.velocity)

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

        return total_reward, outofflock     #See if this needs to be changed individual out of flockness monitored

    def read_agent_locations(self):
        File = rf"{Results['InitPositions']}" + str(self.counter) + "\config.json"
        with open(File, "r") as f:
            data = json.load(f)

        return data

    def seed(self, seed=SimulationVariables["Seed"]):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]
#------------------------

def delete_files(): 
    Paths = ["Results\Flocking\Testing\Dynamics\Accelerations", "Results\Flocking\Testing\Dynamics\Velocities", 
            "Results\Flocking\Testing\Rewards\Other"]

    Logs = ["AlignmentReward_log.json", "CohesionReward_log.json",
            "SeparationReward_log.json", "CollisionReward_log.json",
            "Reward_Total_log.json"]

    for Path in Paths:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], Path, f"Episode{episode}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    for log_file in Logs:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], "Testing", "Rewards", "Components", f"Episode{episode}", log_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")       

def generateCombined():
    with open(rf"{Results['EpisodalRewards']}.json", "r") as f:
        episode_rewards_dict = json.load(f)

    keys_above_threshold = []
    keys_below_threshold = []

    for episode, rewards in episode_rewards_dict.items():
        total_sum = sum(rewards)
        if total_sum > 1000000:
            keys_above_threshold.append(episode)
        else:
            keys_below_threshold.append(episode)

    plt.figure(figsize=(10, 6))
    plt.clf()

    #Fix this
    for episode in keys_above_threshold:
        rewards = episode_rewards_dict[episode]
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"Episode {episode}", alpha=0.7)

    for episode in keys_below_threshold:
        rewards = episode_rewards_dict[episode]
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"Episode {episode}", alpha=0.7)

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"Rewards for Episodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Output.png", dpi=300)

def generateVelocity():
    # Loop through episodes
    for episode in range(0, SimulationVariables["Episodes"]):
        velocities_dict = {}

        with open(os.path.join(positions_directory, f"Episode{episode}_velocities.json"), 'r') as f:
            episode_velocities = json.load(f)

        for agent_id in range(len(env.agents)):
            velocities_dict.setdefault(agent_id, []).extend(episode_velocities.get(str(agent_id), []))

        plt.figure(figsize=(10, 5))
        plt.clf()  
        for agent_id in range(len(env.agents)):
            agent_velocities = np.array(velocities_dict[agent_id])
            # agent_velocities = savgol_filter(agent_velocities, window_length=5, polyorder=3, axis=0)
            velocities_magnitude = np.sqrt(agent_velocities[:, 0]**2 + agent_velocities[:, 1]**2)  
      
            plt.plot(velocities_magnitude, label=f"Agent {agent_id+1}")

        plt.title(f"Velocity - Episode {episode}")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity Magnitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Episode{episode}_SmoothedVelocity.png")

def generateAcceleration():
    for episode in range(0, SimulationVariables["Episodes"]):
        with open(os.path.join(positions_directory, f"Episode{episode}_accelerations.json"), 'r') as f:
            episode_accelerations = json.load(f)

        plt.figure(figsize=(10, 5))
        plt.clf()

        for agent_id in range(len(env.agents)):

            agent_accelerations = np.array(episode_accelerations[str(agent_id)])

            smoothed_accelerations=np.sqrt(agent_accelerations[:, 0]**2 + agent_accelerations[:, 1]**2)
            # print(smoothed_accelerations)

            smoothed_accelerations = savgol_filter(smoothed_accelerations, window_length=3, polyorder=2, axis=0)
            accelerations_magnitude = np.clip(smoothed_accelerations, 0, 5) 

            plt.plot(accelerations_magnitude, label=f"Agent {agent_id+1}")

        plt.title(f"Acceleration - Episode {episode}")
        plt.xlabel("Time Step")
        plt.ylabel("Acceleration Magnitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Episode{episode}_SmoothedAcceleration.png")

#------------------------

if os.path.exists(Results["Rewards"]):
    os.remove(Results["Rewards"])
    print(f"File {Results['Rewards']} has been deleted.")

if os.path.exists("training_rewards.json"):
    os.remove("training_rewards.json")
    print(f"File training_rewards has been deleted.")    

def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    env.seed(seed)
    env.action_space.seed(seed)

env=FlockingEnv()
seed_everything(SimulationVariables["Seed"])
loss_callback = LossCallback()


# # Model Trainingfrom stable_baselines3 import PPO

# Model setup with corrected use_sde handling
model = PPO(
    "MlpPolicy",  # Using the custom policy if needed
    FlockingEnv(use_sde=True),  # Ensure the environment uses 'use_sde' if needed
    policy_kwargs=policy_kwargs,  # Neural network structure defined earlier
    verbose=1,
    tensorboard_log="./ppo_flocking_tensorboard/",
    use_sde=True,  # Enable state-dependent exploration (SDE)
    sde_sample_freq=4,  # SDE sampling frequency, adjust as per requirement
)

model.set_random_seed(SimulationVariables["ModelSeed"])
progress_callback = TQDMProgressCallback(total_timesteps=SimulationVariables["LearningTimeSteps"])
model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"], callback=[progress_callback, loss_callback])
model.save(rf"{Files['Flocking']}\\Models\\FlockingCombinedNew")

positions_directory = rf"{Files['Flocking']}/Testing/Episodes/"
os.makedirs(positions_directory, exist_ok=True)
