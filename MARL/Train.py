import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from env import FlockingEnv
from Settings import *
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from collections import deque
import numpy as np
# Config and kwargs not being used 

class LossCallback(DefaultCallbacks):
    def __init__(self, loss_threshold=2000):
        super().__init__()
        self.loss_threshold = loss_threshold

    def on_train_result(self, *, trainer, result, **kwargs):
        # Assuming "policy_loss" or similar is in the result dict
        # Adjust this key according to your setup
        if 'policy_loss' in result['info']:
            policy_loss = result['info']['policy_loss']
            # Stopping if the average policy loss is below threshold
            if policy_loss < self.loss_threshold:
                print(f"Stopping training as loss ({policy_loss}) is below {self.loss_threshold}")
                result['done'] = True  # This will signal to stop the training
# Custom callback to handle replay buffer
class ReplayBufferCallback(DefaultCallbacks):
    def __init__(self, replay_buffer):
        super().__init__()
        self.replay_buffer = replay_buffer

    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policies, postprocessed_batch, **kwargs):
        # Extract data from the postprocessed batch
        observations = postprocessed_batch["obs"]
        actions = postprocessed_batch["actions"]
        rewards = postprocessed_batch["rewards"]
        next_observations = postprocessed_batch["new_obs"]
        dones = postprocessed_batch["dones"]

        for obs, action, reward, next_obs, done in zip(observations, actions, rewards, next_observations, dones):
            self.replay_buffer.add(obs, action, reward, next_obs, done)

class ReplayBuffer:
    def __init__(self, capacity, observation_space, action_space):
        self.capacity = capacity
        self.observation_space = observation_space
        self.action_space = action_space
        self.buffer = deque(maxlen=capacity)
        self.size = 0

    def add(self, observation, action, reward, next_observation, done):
        self.buffer.append((observation, action, reward, next_observation, done))
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = [self.buffer[i] for i in indices]
        observations, actions, rewards, next_observations, dones = zip(*batch)

        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_observations': np.array(next_observations),
            'dones': np.array(dones)
        }

    def __len__(self):
        return self.size

ray.init()

# Create an instance of the environment to access spaces
env = FlockingEnv()

replay_buffer = ReplayBuffer(
    capacity=100000,  # Adjust capacity as needed
    observation_space=env.observation_space,
    action_space=env.action_space
)

# Register the environment if not already registered
env_name = "flocking_env"
tune.register_env(env_name, lambda config: FlockingEnv())

policy_ids = [f"agent_{i}" for i in range(3)]  # Adjust the number based on the number of agents

config = PPOConfig() \
    .environment(env_name) \
    .framework("torch") \
    .multi_agent(
        policies={
            policy_id: (
                None,
                env.observation_space,
                env.action_space,
                {
                    "model": {
                        "vf_share_layers": True,
                        "fcnet_hiddens": [512, 512, 512, 512, 512, 512, 512, 512],
                        "fcnet_activation": "tanh",
                    },
                }
            )
            for policy_id in policy_ids
        },
        policy_mapping_fn=lambda agent_id, **kwargs: agent_id,
    ) \
    .rollouts(num_rollout_workers=2) \
    .training(
        train_batch_size=2048,
        sgd_minibatch_size=64,
        num_sgd_iter=10
    )


# Example of running the training
tune.run(
    "PPO",
    config=config.to_dict(),  # Convert the config to a dictionary
    stop={"timesteps_total": SimulationVariables["LearningTimeSteps"]},  # Train for 2 million timesteps
    # callbacks=[ReplayBufferCallback(replay_buffer)]  # Add custom callback
)

# Shutdown Ray
ray.shutdown()

# tune.run(
#     PPO,
#     config=config,  # No need to convert to dict
#     stop={"episode_reward_mean": 100},  # Adjust this based on your environment
# )

# tune.run(
#     PPO,
#     config=config,
#     stop={"training_iteration": 100},  # Optional stop condition if loss threshold isn't hit
#     callbacks=[LossCallback(loss_threshold=2000)]
# )
