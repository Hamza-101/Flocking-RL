import os
import json
import gym
from gym import spaces
from Params import *
import math
import numpy as np
import random
from multiprocessing import freeze_support
from pathlib import Path
from tensorflow.keras import layers, models, optimizers
from collections import deque
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.callbacks import *
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


############################################################
############################################################
##############Make Environment more agent based#############
############################################################
############################################################

# class StoreState(BaseCallback):
#     def __init__(self, env, verbose=0):
#         super(StoreState, self).__init__(verbose)
#         self.env = env
#         self.states_on_collision = []

#     def _on_step(self) -> bool:
#         # Check for collisions
#         collision = any(self.env.envs[0].check_collision(agent) for agent in self.env.envs[0].agents)

#         # If collision occurs, store the current state
#         if collision:
#             current_state = self.env.envs[0].get_observation()
#             self.states_on_collision.append(current_state)

#         return True

# class HalveLearningRateCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(HalveLearningRateCallback, self).__init__(verbose)
#         self.learning_rate_divisor = 2
#         self.halving_step = 2048  # Timestep to halve the learning rate
#         self.halved = False
        
#     def _on_step(self):
#         if self.num_timesteps > self.halving_step:
#             self.model.policy.optimizer.param_groups[0]['lr'] *= 0.5  # Halve the learning rate
#             self.halved = True
#         return True  # Continue training

# n_fall_steps=10: Possibly the number of steps allowed for the drone to fall before the simulation ends.
class Agent:
   
    def __init__(self, position):
        self.position = np.array(position, dtype=float)  
        self.acceleration = np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)                # Acceleration Initialized to 0
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        
        self.velocity = self.acceleration * SimulationVariables["dt"]       
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    def update(self, action):
        self.acceleration += action
        self.velocity += self.acceleration * SimulationVariables["dt"]

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
        self.current_timestep=0 
        self.reward_log = []
        self.counter=0
        self.agents = [Agent(position) for position in self.read_agent_locations(self.counter)]
        self.episode_length=10000           #Episode length


        for agent in self.agents:
            agent.velocity = np.zeros(2) ###CHANGE 

        min_action = np.array([[-1, -1]] * len(self.agents), dtype=np.float32)
        max_action = np.array([[1, 1]] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        min_obs = np.array([[-np.inf, -np.inf]] * len(self.agents), dtype=np.float32)
        max_obs = np.array([[np.inf, np.inf]] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        
    def step(self, actions):
        info={}
        reward = 0
        done = False 
        Collisions = {}
        
        observations=self.simulate_agents(actions)

        separation_reward, alignment_reward, cohesion_reward, collision_penalty, Collisions = self.calculate_reward()

        info["Collisions"]=Collisions
        info["CohesionReward"]= cohesion_reward
        info["SeparationReward"] = separation_reward
        info["AlignmentReward"] = alignment_reward
        info["CollisionPenalty"] = collision_penalty
        
       
        reward = info["AlignmentReward"] + info["CohesionReward"] + info["CollisionPenalty"] + info["SeparationReward"]

        if any(info["Collisions"]):
            done = True
             # If collision occurs, store the current state
            print("collision")


        # Implement the termination condition based on your criteria
        self.current_timestep+=1
        return observations, reward, done, info

        #CHECK RESET

    def reset(self):
        
        self.current_timestep += 1

        # Increase counter every 2000 timesteps
        if self.current_timestep % 2000 == 0:
            self.counter += 1
            
        self.current_timestep = 0 
        self.agents = [Agent(position) for position in self.read_agent_locations(self.counter)]

        for agent in self.agents:
            agent.acceleration=np.round(np.random.uniform(-SimulationVariables["AccelerationInit"], SimulationVariables["AccelerationInit"], size=2), 2)
            agent.velocity=agent.acceleration * SimulationVariables["dt"]
            print(agent.position)
        observation = self.get_observation()
       
        return observation
        
    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):
        observations = []
        for i, agent in enumerate(self.agents):
            new_position = agent.update(actions[i])
            observations.append(new_position)

        return observations
    
    def check_collision(self, agent):
        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
        return False

    def get_observation(self):
            observation = np.zeros((len(self.agents), 2), dtype=np.float32)

            for i, agent in enumerate(self.agents):

                    observation[i] = [
                        agent.position[0],
                        agent.position[1],
                    ]
             

            return observation.reshape((len(self.agents), 2))

    def calculate_reward(self):
        CohesionReward=0
        AlignmentReward=0
        SeparationReward=0
        CollisionPenalty = 0

        neighbor_indices = [[] for _ in range(len(self.agents))]  # Initialize empty lists for neighbor indices
        neighbor_velocities = [[] for _ in range(len(self.agents))]
        Collisions = {}
        for idx, _ in enumerate(self.agents):
            Collisions[idx] = []

        # Finding neighbors information
        for idx, agent in enumerate(self.agents):
            neighbor_indices, neighbor_velocities = self.get_closest_neighbors(agent, SimulationVariables["NeighborhoodRadius"])

            CollisionPenalty , collisions_agent = self.CollisionPenalty(agent, neighbor_indices)
            CohesionReward += self.calculate_CohesionReward(agent, neighbor_indices)
            SeparationReward += self.calculate_SeparationReward(agent, neighbor_indices)
            AlignmentReward += self.calculate_AlignmentReward(agent, neighbor_velocities)

            Collisions[idx].append(collisions_agent)
               

        return SeparationReward, AlignmentReward, CohesionReward, CollisionPenalty, Collisions
    
    def alignment_calculation(self, agent, neighbor_velocities):
        if len(neighbor_velocities) > 0:
            average_velocity = np.mean(neighbor_velocities, axis=0)
            desired_velocity = average_velocity - agent.velocity
            return desired_velocity
        else:
            return np.zeros(2)
            
    def get_closest_neighbors(self, agent, max_distance):
        neighbor_indices = []
        neighbor_velocities = []

        for i, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)
                if distance < max_distance:
                    neighbor_indices.append(i)
                    neighbor_velocities.append(other.velocity)

        return neighbor_indices, neighbor_velocities         

    def CollisionPenalty(self, agent, neighbor_indices):
        #Every agents' total collision penalty every timestep
        Collision = False
        CollisionPenalty = 0
        distances=[]

        # Check neighbor indices
        neighbor_agents = [self.agents[idx] for idx in neighbor_indices]
        
        for n_agent in neighbor_agents:             #Use neighborhood or all
            distances.append(np.linalg.norm(agent.position - n_agent.position))

        for distance in distances:
            if distance < SimulationVariables["SafetyRadius"]:            
                CollisionPenalty -= 1000
                Collision = True

        return CollisionPenalty, Collision
    
    def calculate_CohesionReward(self, agent, neighbor_indices):

        CohesionReward = 0

        if len(neighbor_indices) > 0:
        # Find why working
            center_of_mass = np.mean(neighbor_indices, axis=0)
            desired_position = center_of_mass - agent.position
            distance = np.linalg.norm(desired_position)
            if SimulationVariables["SafetyRadius"] < distance <= SimulationVariables["NeighborhoodRadius"]:
                CohesionReward = 1
            elif distance > SimulationVariables["NeighborhoodRadius"]:
                CohesionReward = -(1 * (distance - SimulationVariables["NeighborhoodRadius"]))
            
            else:
                CohesionReward = -1

        #   cohesion_reward = max(0, 1 - (distance - min_distance) / min_distance)

        # # Penalize agents for going beyond the boundary
        # if distance > SimulationVariables["NeighborhoodRadius"]:
        #     cohesion_reward -= (distance - max_boundary_distance) / max_boundary_distance

        return CohesionReward

    def calculate_SeparationReward(self, agent, neighbor_indices):
        SeparationReward=0

        #Check neighbor indices
        if len(neighbor_indices) > 0:
            for neighbor_position in neighbor_indices:
                # print(neighbor_position)
                relative_position = agent.position - neighbor_position
                distance = np.linalg.norm(relative_position)

                if distance >= SimulationVariables["SafetyRadius"]:
                    SeparationReward += 1
                elif (distance < SimulationVariables["SafetyRadius"]):
                    SeparationReward += -(1 * (distance - SimulationVariables["SafetyRadius"]))
        else:
            SeparationReward = -1
            
        return SeparationReward
    
    #FIX THIS
    def calculate_AlignmentReward(self, agent, neighbor_velocities):

        AlignmentReward = 0

        if (len(neighbor_velocities) > 0):
            desired_direction = self.alignment_calculation(agent, neighbor_velocities)
            orientation_diff = np.arctan2(desired_direction [1], desired_direction [0]) 
            - np.arctan2(agent.velocity[1], agent.velocity[0])

            if orientation_diff > np.pi:
                orientation_diff -= 2 * np.pi
            elif orientation_diff < 0:
                orientation_diff += 2 * np.pi

            AlignmentReward = 1 - np.abs(orientation_diff)

        return AlignmentReward 

    def read_agent_locations(self, counter):
        File = rf"{Results['InitPositions']}"+ str(counter) +"\config.json"
        print(File)
        with open(File, "r") as f:
            data = json.load(f)
        return data


if os.path.exists(Results["Rewards"]):
    os.remove(Results["Rewards"])
    print(f"File {Results['Rewards']} has been deleted.")



env=FlockingEnv()

# envs = [make_boid_env() for _ in range(num_envs)]  # Create your boid envs
# vec_env = VectorizedEnv(envs)

#-------------------------------------------------------------------------

states = env.observation_space.shape
actions = env.action_space

def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=500000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

model = build_model(states, actions)


model.summary()


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

dqn.save_weights('dqn_weights.h5f', overwrite=True)

del model
del dqn
del env

env = FlockingEnv()
actions = env.action_space.n
states = env.observation_space.shape[0]
model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('dqn_weights.h5f')

_ = dqn.test(env, nb_episodes=5, visualize=True)


# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# action_high = env.action_space.high[0]
# action_low = env.action_space.low[0]



# class OUActionNoise:
#     def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
#         self.theta = theta
#         self.mean = mean
#         self.std_dev = std_deviation
#         self.dt = dt
#         self.x_initial = x_initial
#         self.reset()

#     def __call__(self):
#         # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
#         x = (
#             self.x_prev
#             + self.theta * (self.mean - self.x_prev) * self.dt
#             + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
#         )
#         # Store x into x_prev
#         # Makes next noise dependent on current one
#         self.x_prev = x
#         return x

#     def reset(self):
#         if self.x_initial is not None:
#             self.x_prev = self.x_initial
#         else:
#             self.x_prev = np.zeros_like(self.mean)


# # Actor Model
# def build_actor():
#     model = models.Sequential([
#         layers.InputLayer(input_shape=(state_dim,)),
#         layers.Dense(400, activation='relu'),
#         layers.Dense(300, activation='relu'),
#         layers.Dense(action_dim, activation='tanh')  # tanh for scaling between -1 and 1
#     ])
#     model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')
#     return model

# # Critic Model
# def build_critic():
#     state_input = layers.Input(shape=(state_dim,))
#     action_input = layers.Input(shape=(action_dim,))
    
#     state_branch = layers.Dense(400, activation='relu')(state_input)
#     action_branch = layers.Dense(300, activation='relu')(action_input)
    
#     merged = layers.Concatenate()([state_branch, action_branch])
#     merged = layers.Dense(300, activation='relu')(merged)
#     output = layers.Dense(1)(merged)
    
#     model = models.Model(inputs=[state_input, action_input], outputs=output)
#     model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')
#     return model

# class DDPGAgent:
#     def __init__(self, action_dim, mean, std_deviation):
#         self.actor = build_actor()
#         self.target_actor = build_actor()
#         self.critic = build_critic()
#         self.target_critic = build_critic()

#         self.target_actor.set_weights(self.actor.get_weights())
#         self.target_critic.set_weights(self.critic.get_weights())

#         self.replay_buffer = deque(maxlen=DDPGParam["buffer_size"])
#         self.noise_process = OUActionNoise(mean, std_deviation)

#     def select_action(self, state):
#         action = self.actor.predict(state.reshape(1, -1))[0]
#         action += DDPGParam["exploration_noise"] * self.noise_process.sample()
#         return np.clip(action, action_low, action_high)

#     def remember(self, state, action, reward, next_state, done):
#         self.replay_buffer.append((state, action, reward, next_state, done))

#     def replay(self):
#         if len(self.replay_buffer) < DDPGParam["batch_size"]:
#             return

#         minibatch = np.array(random.sample(self.replay_buffer, DDPGParam["batch_size"]))
#         states = np.vstack(minibatch[:, 0])
#         actions = np.vstack(minibatch[:, 1])
#         rewards = minibatch[:, 2]
#         next_states = np.vstack(minibatch[:, 3])
#         dones = minibatch[:, 4]

#         # Update critic
#         next_actions = self.target_actor.predict(next_states)
#         target_q_values = self.target_critic.predict([next_states, next_actions])
#         y = rewards + DDPGParam["gamma"] * (1 - dones) * target_q_values.flatten()
#         self.critic.fit([states, actions], y, epochs=1, verbose=0)

#         # Update actor policy using the sampled gradient
#         actions_for_grad = self.actor.predict(states)
#         grads = self.critic.gradient([states, actions_for_grad])[0]
#         self.actor.train_on_batch(states, grads)

#         # Soft update target networks
#         actor_weights = np.array(self.actor.get_weights())
#         target_actor_weights = np.array(self.target_actor.get_weights())
#         self.target_actor.set_weights(DDPGParam["tau"] * actor_weights + (1 - DDPGParam["tau"]) * target_actor_weights)

#         critic_weights = np.array(self.critic.get_weights())
#         target_critic_weights = np.array(self.target_critic.get_weights())
#         self.target_critic.set_weights(DDPGParam["tau"] * critic_weights + (1 - DDPGParam["tau"]) * target_critic_weights)


# agent = DDPGAgent(action_dim=action_dim, mean=np.zeros(action_dim), std_deviation=np.ones(action_dim))

# for episode in range(DDPGParam["Episodes"]):
#     state = env.reset()
#     total_reward = 0

#     while True:
#         action = agent.select_action(state)
#         next_state, reward, done, _ = env.step(action)
#         agent.remember(state, action, reward, next_state, done)
#         agent.replay()

#         total_reward += reward
#         state = next_state

#         if done:
#             break

#     print(f"Episode: {episode + 1}, Total Reward: {total_reward}")





# # # Run for 10 episodes
# # for episode in range(1, RLVariables['Episodes']):
    
# #     obs = env.reset(episode)
# #     done = False
    
# #     total_reward = 0
# #     positions_dict = {i: [] for i in range(len(env.agents))}
# #     timestep = 0
# #     reward_log=[]

# #     print("Episode", episode)

# #     # Completion condition
# #     while((timestep <= SimulationVariables["EvalTimeSteps"]) and (not done)):
      
# #         action, _states = model.predict(obs)
        
# #         next_obs, reward, done, _ = env.step(action) ######### env.step() #Add condition to exit on collision
# #         total_reward+=reward

# #         reward_log.append(reward)
# #         print(reward)

# #         for i, agent in enumerate(env.agents):
# #             positions_dict[i].append(agent.position.tolist())
# #         with open(rf'RewardTesting_Episode{episode}.json', 'w') as f:
# #                json.dump(reward_log, f, indent=4)
      
# #         timestep = timestep + 1
  
# #     # print(reward_log)
    
# #     with open(rf'Flocking_new_{episode}.json', 'w') as f:        #Add to params file
# #         json.dump(positions_dict, f, indent=4)


# env.close()