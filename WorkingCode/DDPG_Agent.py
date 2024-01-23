
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque

# Environment
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_high = env.action_space.high[0]
action_low = env.action_space.low[0]

# Hyperparameters
gamma = 0.999999
tau = 0.001
buffer_size = 100000
batch_size = 512
exploration_noise = 0.1

# Actor Model
def build_actor():
    model = models.Sequential([
        layers.InputLayer(input_shape=(state_dim,)),
        layers.Dense(400, activation='relu'),
        layers.Dense(300, activation='relu'),
        layers.Dense(action_dim, activation='tanh')  # tanh for scaling between -1 and 1
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

# Critic Model
def build_critic():
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    
    state_branch = layers.Dense(400, activation='relu')(state_input)
    action_branch = layers.Dense(300, activation='relu')(action_input)
    
    merged = layers.Concatenate()([state_branch, action_branch])
    merged = layers.Dense(300, activation='relu')(merged)
    output = layers.Dense(1)(merged)
    
    model = models.Model(inputs=[state_input, action_input], outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

# Ornstein-Uhlenbeck Process for Exploration Noise
class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.3):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

# DDPG Agent
class DDPGAgent:
    def __init__(self):
        self.actor = build_actor()
        self.target_actor = build_actor()
        self.critic = build_critic()
        self.target_critic = build_critic()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.replay_buffer = deque(maxlen=buffer_size)
        self.noise_process = OrnsteinUhlenbeckProcess(action_dim)

    def select_action(self, state):
        action = self.actor.predict(state.reshape(1, -1))[0]
        action += exploration_noise * self.noise_process.sample()
        return np.clip(action, action_low, action_high)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < batch_size:
            return

        minibatch = np.array(random.sample(self.replay_buffer, batch_size))
        states = np.vstack(minibatch[:, 0])
        actions = np.vstack(minibatch[:, 1])
        rewards = minibatch[:, 2]
        next_states = np.vstack(minibatch[:, 3])
        dones = minibatch[:, 4]

        # Update critic
        next_actions = self.target_actor.predict(next_states)
        target_q_values = self.target_critic.predict([next_states, next_actions])
        y = rewards + gamma * (1 - dones) * target_q_values.flatten()
        self.critic.fit([states, actions], y, epochs=1, verbose=0)

        # Update actor policy using the sampled gradient
        actions_for_grad = self.actor.predict(states)
        grads = self.critic.gradient([states, actions_for_grad])[0]
        self.actor.train_on_batch(states, grads)

        # Soft update target networks
        actor_weights = np.array(self.actor.get_weights())
        target_actor_weights = np.array(self.target_actor.get_weights())
        self.target_actor.set_weights(tau * actor_weights + (1 - tau) * target_actor_weights)

        critic_weights = np.array(self.critic.get_weights())
        target_critic_weights = np.array(self.target_critic.get_weights())
        self.target_critic.set_weights(tau * critic_weights + (1 - tau) * target_critic_weights)


# Training
agent = DDPGAgent()
num_episodes = 1000

# for episode in range(num_episodes):
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