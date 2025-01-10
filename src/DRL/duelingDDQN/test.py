import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Value of state
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # Advantage of each action
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Replay Buffer
class ReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def add(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Agent
class DuelingDDQNAgent:
    def __init__(self, state_dim, action_dim, buffer_size=100000, batch_size=64, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = DuelingDQN(state_dim, action_dim)
        self.target_network = DuelingDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample mini-batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute Q-target using target network
        with torch.no_grad():
            next_q_values = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_targets = self.target_network(next_states).gather(1, next_q_values)
            q_targets = rewards + self.gamma * next_q_targets * (1 - dones)

        # Compute Q-prediction using Q-network
        q_predictions = self.q_network(states).gather(1, actions)

        # Compute loss and update
        loss = nn.MSELoss()(q_predictions, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Main training loop
def train_dueling_ddqn():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DuelingDDQNAgent(state_dim, action_dim)

    episodes = 300
    update_target_every = 10

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            agent.update()
        
        if episode % update_target_every == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    env.close()

# Train the agent
train_dueling_ddqn()
